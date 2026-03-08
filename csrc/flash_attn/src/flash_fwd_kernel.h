/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
#include "philox_unpack.cuh" // For at::cuda::philox::unpack

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"
#include "dropout.h"
#include "rotary.h"

namespace FLASH_NAMESPACE {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ElementAccum, typename Params, int kBlockM, bool Is_even_MN>
__forceinline__ __device__ auto get_lse_tile(const Params &params, const int bidb, const int bidh, const int m_block, const BlockInfo</*Varlen=*/!Is_even_MN> &binfo) {
        // When params.unpadded_lse is false, LSE is written as (b, h, seqlen_q) - this is non-variable seqlen path.
        // Otherwise, when params.seqlenq_ngroups_swapped is true, it is written as (h, seqlen_q, b) to account for seqlen_q <-> h swapping trick.
        // Otherwise, it's written as (h, b, seqlen_q).
        const bool varlen_q = params.unpadded_lse && !params.seqlenq_ngroups_swapped;
        auto lse_offset = varlen_q ? binfo.q_offset(params.seqlen_q, 1, bidb) : 0;
        auto gmem_ptr_lse = make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + lse_offset);

        auto lse_shape = varlen_q ? make_shape(1, params.h, params.total_q) : make_shape(params.b, params.h, params.seqlen_q);
        auto lse_stride = params.seqlenq_ngroups_swapped ? make_stride(1, params.seqlen_q * params.b, params.b) : (
            params.unpadded_lse ? make_stride(params.h * params.total_q, params.total_q, 1) :  make_stride(params.h * params.seqlen_q, params.seqlen_q, 1)
            );

        auto lse_layout = make_layout(lse_shape, lse_stride);
        Tensor mLSE = make_tensor(gmem_ptr_lse, lse_layout);
        auto mLSE_slice = varlen_q ? mLSE(0, bidh, _) : mLSE(bidb, bidh, _);
        return local_tile(mLSE_slice, Shape<Int<kBlockM>>{}, make_coord(m_block));
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_1rowblock(const Params &params, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    static_assert(kBlockM % kNWarps == 0, "warp-stationary requires blockM divisible by nWarps");
    constexpr int kWarpRows = kBlockM / kNWarps;
    constexpr int kBlockRowStride = kBlockM / kWarpRows;
    using MMA_Atom_Arch = typename Kernel_traits::MMA_Atom_Arch;
    using TiledMmaQK = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<_1, _1, _1>>,
        Tile<Int<kWarpRows>, _16, _4>
    >;
    using TiledMmaPV = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<_1, _1, _1>>,
        Tile<Int<kWarpRows>, _16, _4>
    >;
    const int warp_id = tidx / 32;
    const int lane_id = tidx % 32;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    FLASH_NAMESPACE::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;
    const int rows_valid = binfo.actual_seqlen_q - m_block * kBlockM;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        const int rows_this_block = binfo.actual_seqlen_q - m_block * kBlockM;
        const int warp_row_base = warp_id * kWarpRows;
        Tensor gO_warp = local_tile(gO, Shape<Int<kWarpRows>, Int<kHeadDim>>{},
                                    make_coord(warp_id, 0));
        TiledMmaPV tiled_mma_pv_early;
        auto thr_mma_pv_early = tiled_mma_pv_early.get_thread_slice(lane_id);
        Tensor caccO_early = make_identity_tensor(Shape<Int<kWarpRows>, Int<kHeadDim>>{});
        Tensor taccOcO_early = thr_mma_pv_early.partition_C(caccO_early);
        #pragma unroll
        for (int i = 0; i < size(taccOcO_early); ++i) {
            const int row_local = get<0>(taccOcO_early(i));
            const int col = get<1>(taccOcO_early(i));
            const int row_global = warp_row_base + row_local;
            if (row_global < rows_this_block && (Is_even_K || col < params.d)) {
                gO_warp(row_local, col) = Element(0);
            }
        }
        Tensor taccOcO_logical_early = make_tensor(taccOcO_early.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(taccOcO_early.layout()));
        Tensor taccOcO_row_early = taccOcO_logical_early(_, 0);
        if (get<1>(taccOcO_row_early(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(taccOcO_row_early); ++mi) {
                const int row_global = warp_row_base + get<0>(taccOcO_row_early(mi));
                if (row_global < rows_this_block) { gLSE(row_global) = INFINITY; }
            }
        }
        return;
    }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                          + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                          + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
    Tensor sP = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutP{});
    Tensor sQ_warp = local_tile(sQ, Shape<Int<kWarpRows>, Int<kHeadDim>>{},
                                make_coord(warp_id, 0));
    Tensor sP_warp = local_tile(sP, Shape<Int<kWarpRows>, Int<kBlockN>>{},
                                make_coord(warp_id, 0));
    Tensor gP_warp = local_tile(gP, Shape<Int<kWarpRows>, Int<kBlockN>>{},
                                make_coord(warp_id, 0));

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    TiledMmaQK tiled_mma_qk;
    TiledMmaPV tiled_mma_pv;
    auto thr_mma_qk = tiled_mma_qk.get_thread_slice(lane_id);
    auto thr_mma_pv = tiled_mma_pv.get_thread_slice(lane_id);
    Tensor tSrQ  = thr_mma_qk.partition_fragment_A(sQ_warp);                   // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma_qk.partition_fragment_B(sK);                        // (MMA,MMA_N,MMA_K)
    Tensor tOsQ  = thr_mma_qk.partition_A(sQ_warp);                            // (MMA,MMA_M,MMA_K)
    Tensor tOsK  = thr_mma_qk.partition_B(sK);                                 // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma_pv.partition_fragment_B(sVtNoSwizzle);             // (MMA, MMA_K,MMA_N)
    Tensor tOsVtWarp = thr_mma_pv.partition_B(sVt);                            // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma_qk.partition_C(gP_warp);
    Tensor cS = make_identity_tensor(Shape<Int<kWarpRows>, Int<kBlockN>>{});      // (BLK_M, BLK_N) -> (row, col)
    Tensor tScS = thr_mma_qk.partition_C(cS);                                      // (MMA, MMA_M, MMA_N) -> (row, col)
    Tensor tScS_row = make_tensor(tScS.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tScS.layout()))(_, 0);

    Tensor acc_o = partition_fragment_C(tiled_mma_pv, Shape<Int<kWarpRows>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K
    Tensor caccO = make_identity_tensor(Shape<Int<kWarpRows>, Int<kHeadDim>>{});                 // (BLK_M, BLK_K) -> (row, col)
    Tensor taccOcO = thr_mma_pv.partition_C(caccO);                                                // (MMA, MMA_M, MMA_K) -> (row, col)

    //
    // Copy Atom retiling
    //

    typename Kernel_traits::TiledMma tiled_mma_copy;
    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_copy);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    if constexpr (Is_even_MN) {
        FLASH_NAMESPACE::copy<true, Is_even_K>(
            gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
    } else {
        FLASH_NAMESPACE::copy<false, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
    }

    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    if constexpr (!Is_even_MN) {
        const int sk_rows = size<0>(sK);
        const int sk_cols = size<1>(sK);
        for (int idx = tidx; idx < sk_rows * sk_cols; idx += Kernel_traits::kNThreads) {
            sK(idx / sk_cols, idx % sk_cols) = Element(0);
        }
        __syncthreads();
    }
    if constexpr (Is_even_MN) {
        FLASH_NAMESPACE::copy<true, Is_even_K>(
            gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
    } else {
        FLASH_NAMESPACE::copy<false, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
    }

    clear(acc_o);

    FLASH_NAMESPACE::Softmax<2 * size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    FLASH_NAMESPACE::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma_qk, Shape<Int<kWarpRows>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        __syncthreads();

        // Advance gV
        if (masking_step > 0) {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
        } else {
            if constexpr (!Is_even_MN) {
                const int sv_rows = size<0>(sV);
                const int sv_cols = size<1>(sV);
                for (int idx = tidx; idx < sv_rows * sv_cols; idx += Kernel_traits::kNThreads) {
                    sV(idx / sv_cols, idx % sv_cols) = Element(0);
                }
                __syncthreads();
            }
            // Clear the smem tiles to account for predicated off loads
            FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
            );
        }

        #pragma unroll
        for (int i = 0; i < size<2>(tSrQ); ++i) {
            cute::copy(tOsQ(_, _, i), tSrQ(_, _, i));
            cute::copy(tOsK(_, _, i), tSrK(_, _, i));
            cute::gemm(tiled_mma_qk, tSrQ(_, _, i), tSrK(_, _, i), acc_s);
        }
        if constexpr (Is_softcap){
            FLASH_NAMESPACE::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask_idx<Is_causal, Is_even_MN>(
            acc_s, tScS, n_block * kBlockN, m_block * kBlockM + warp_id * kWarpRows, 0
        );

        __syncthreads();
        if (n_block > n_block_min) {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
        }

        // TODO: when we have key_padding_mask we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/(Is_causal || Is_local || !Is_even_MN)>(acc_s, acc_o, params.scale_softmax_log2, tScS_row, taccOcO, tScS)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/(Is_causal || Is_local || !Is_even_MN)>(acc_s, acc_o, params.scale_softmax_log2, tScS_row, taccOcO, tScS);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
        if constexpr (!Is_even_MN) {
            #pragma unroll
            for (int i = 0; i < size(rP); ++i) {
                if (get<0>(tScS(i)) + warp_id * kWarpRows >= rows_valid) { rP(i) = Element(0); }
            }
        }
        int block_row_idx = m_block * kBlockRowStride + warp_id;
        int block_col_idx = n_block * (kBlockN / 32);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                rP_drop, block_row_idx, block_col_idx, kBlockRowStride
            );
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.apply_dropout(rP, block_row_idx, block_col_idx, kBlockRowStride);
        }

        if constexpr (kBlockN == 64 && (kWarpRows == 16 || kWarpRows == 32)) {
            Tensor tOrP = thr_mma_pv.partition_fragment_A(sP_warp);
            const int lane_group = lane_id & 0x10;
            const int lane_parity = lane_id & 0x1;
            #pragma unroll
            for (int i = 0; i < size(tOrP); ++i) {
                const int src_lane = lane_group | lane_parity | (((i >> 1) & 0x1) << 1);
                const int group = i >> 2;
                int perm = 0;
                if constexpr (kWarpRows == 16) {
                    perm = (group & ~0x3) | (((group & 0x1) << 1) | ((group & 0x2) >> 1));
                } else {
                    perm = (group & ~0x7) | (((group & 0x3) << 1) | ((group & 0x4) >> 2));
                }
                const int base_idx = (perm << 2) + (i & 0x1);
                const float src0 = static_cast<float>(rP(base_idx + 0));
                const float src1 = static_cast<float>(rP(base_idx + 2));
                const float got0 = __shfl_sync(0xffffffffu, src0, src_lane);
                const float got1 = __shfl_sync(0xffffffffu, src1, src_lane);
                tOrP(i) = Element(4.f * ((lane_id & 0x2) ? got1 : got0));
            }
            FLASH_NAMESPACE::gemm_rs(
                acc_o, tOrP, tOrVt, tOsVt, tiled_mma_pv, smem_tiled_copy_V, smem_thr_copy_V
            );
        } else {
            Tensor tOrP = make_tensor(rP.data(), FLASH_NAMESPACE::convert_layout_acc_Aregs<TiledMmaPV>(rP.layout()));
            FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma_pv, smem_tiled_copy_V, smem_thr_copy_V);
        }

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma_qk, Shape<Int<kWarpRows>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        __syncthreads();
        FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);

        #pragma unroll
        for (int i = 0; i < size<2>(tSrQ); ++i) {
            cute::copy(tOsQ(_, _, i), tSrQ(_, _, i));
            cute::copy(tOsK(_, _, i), tSrK(_, _, i));
            cute::gemm(tiled_mma_qk, tSrQ(_, _, i), tSrK(_, _, i), acc_s);
        }
        if constexpr (Is_softcap){
            FLASH_NAMESPACE::apply_softcap(acc_s, params.softcap);
        }

        __syncthreads();
        if (n_block > n_block_min) {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
        }
        
        mask.template apply_mask_idx</*Causal_mask=*/false>(
            acc_s, tScS, n_block * kBlockN, m_block * kBlockM + warp_id * kWarpRows, 0
        );

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/(Is_local || !Is_even_MN)>(acc_s, acc_o, params.scale_softmax_log2, tScS_row, taccOcO, tScS);

        Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
        if constexpr (!Is_even_MN) {
            #pragma unroll
            for (int i = 0; i < size(rP); ++i) {
                if (get<0>(tScS(i)) + warp_id * kWarpRows >= rows_valid) { rP(i) = Element(0); }
            }
        }
        int block_row_idx = m_block * kBlockRowStride + warp_id;
        int block_col_idx = n_block * (kBlockN / 32);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                rP_drop, block_row_idx, block_col_idx, kBlockRowStride
            );
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.apply_dropout(rP, block_row_idx, block_col_idx, kBlockRowStride);
        }

        if constexpr (kBlockN == 64 && (kWarpRows == 16 || kWarpRows == 32)) {
            Tensor tOrP = thr_mma_pv.partition_fragment_A(sP_warp);
            const int lane_group = lane_id & 0x10;
            const int lane_parity = lane_id & 0x1;
            #pragma unroll
            for (int i = 0; i < size(tOrP); ++i) {
                const int src_lane = lane_group | lane_parity | (((i >> 1) & 0x1) << 1);
                const int group = i >> 2;
                int perm = 0;
                if constexpr (kWarpRows == 16) {
                    perm = (group & ~0x3) | (((group & 0x1) << 1) | ((group & 0x2) >> 1));
                } else {
                    perm = (group & ~0x7) | (((group & 0x3) << 1) | ((group & 0x4) >> 2));
                }
                const int base_idx = (perm << 2) + (i & 0x1);
                const float src0 = static_cast<float>(rP(base_idx + 0));
                const float src1 = static_cast<float>(rP(base_idx + 2));
                const float got0 = __shfl_sync(0xffffffffu, src0, src_lane);
                const float got1 = __shfl_sync(0xffffffffu, src1, src_lane);
                tOrP(i) = Element(4.f * ((lane_id & 0x2) ? got1 : got0));
            }
            FLASH_NAMESPACE::gemm_rs(
                acc_o, tOrP, tOrVt, tOsVt, tiled_mma_pv, smem_tiled_copy_V, smem_thr_copy_V
            );
        } else {
            Tensor tOrP = make_tensor(rP.data(), FLASH_NAMESPACE::convert_layout_acc_Aregs<TiledMmaPV>(rP.layout()));
            FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma_pv, smem_tiled_copy_V, smem_thr_copy_V);
        }
    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout, tScS_row, taccOcO);

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = FLASH_NAMESPACE::convert_type<Element>(acc_o);
    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                          + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);
    const int rows_this_block = binfo.actual_seqlen_q - m_block * kBlockM;
    const int warp_row_base = warp_id * kWarpRows;
    Tensor gO_warp = local_tile(gO, Shape<Int<kWarpRows>, Int<kHeadDim>>{},
                                make_coord(warp_id, 0));
    #pragma unroll
    for (int i = 0; i < size(rO); ++i) {
        const int row_local = get<0>(taccOcO(i));
        const int col = get<1>(taccOcO(i));
        const int row_global = warp_row_base + row_local;
        if (row_global < rows_this_block && (Is_even_K || col < params.d)) {
            gO_warp(row_local, col) = rO(i);
        }
    }

    // 将 taccOcO 混沌的物理布局，套上你写的转换器，变成干净的 (Rows, Cols) 二维视图
    Tensor taccOcO_logical = make_tensor(taccOcO.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(taccOcO.layout()));
    
    // 我们只需要行坐标来写 LSE，所以直接切片：取逻辑上的第 0 列 (_, 0)
    Tensor taccOcO_row = taccOcO_logical(_, 0);

    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
    if (get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row_global = warp_row_base + get<0>(taccOcO_row(mi));
            if (row_global < rows_this_block) {
                gLSE(row_global) = lse(mi);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv(const Params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx, const int num_n_splits) {

}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    // We want the fwd and bwd to generate the same dropout pattern (RNG), without restricting
    // them to have the same number of threads or have to traverse the attention matrix
    // in the same order.
    // In the Philox RNG, we use the offset to store the batch, head, and the lane id
    // (within a warp). We use the subsequence to store the location of the 16 x 32 blocks within
    // the attention matrix. This way, as long as we have the batch, head, and the location of
    // the 16 x 32 block within the attention matrix, we can generate the exact same dropout pattern.

    FLASH_NAMESPACE::compute_attn_1rowblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, typename Params>
inline __device__ void compute_attn_splitkv(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = Split ? blockIdx.z / params.h : blockIdx.y;
    // The block index for the head.
    const int bidh = Split ? blockIdx.z - bidb * params.h : blockIdx.z;
    const int n_split_idx = Split ? blockIdx.y : 0;
    const int num_n_splits = Split ? gridDim.y : 1;
    FLASH_NAMESPACE::compute_attn_1rowblock_splitkv<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV>(params, bidb, bidh, m_block, n_split_idx, num_n_splits);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, int kBlockM, int Log_max_splits, bool Is_even_K, typename Params>
inline __device__ void combine_attn_seqk_parallel(const Params &params) {

}

} // namespace FLASH_NAMESPACE
