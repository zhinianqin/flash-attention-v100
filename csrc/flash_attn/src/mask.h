/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once
#include "namespace_config.h"

#include <cute/tensor.hpp>

namespace FLASH_NAMESPACE {

using namespace cute;

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask(Tensor<Engine, Layout> &tensor, const int max_seqlen_k,
                                  const int col_idx_offset_ = 0) {
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    #pragma unroll
    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        const int col_idx_base = col_idx_offset + nj * 8;
        #pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
            const int col_idx = col_idx_base + j;
            if (col_idx >= max_seqlen_k) {
                // Without the "make_coord" we get wrong results
                #pragma unroll
                for (int mi = 0; mi < size<0>(tensor); ++mi) {
                    tensor(mi, make_coord(j, nj)) = -INFINITY;
                }
            }
        }
    }
}

template <bool HasWSLeft=true, typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask_local(Tensor<Engine, Layout> &tensor, const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset,
                                        const int max_seqlen_q, const int warp_row_stride,
                                        const int window_size_left, const int window_size_right) {
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    #pragma unroll
    for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
        const int row_idx_base = row_idx_offset + mi * warp_row_stride;
        #pragma unroll
        for (int i = 0; i < size<0, 0>(tensor); ++i) {
            const int row_idx = row_idx_base + i * 8;
            const int col_idx_limit_left = std::max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left);
            const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right);
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const int col_idx_base = col_idx_offset + nj * 8;
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const int col_idx = col_idx_base + j;
                    if (col_idx >= col_idx_limit_right || (HasWSLeft && col_idx < col_idx_limit_left)) {
                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                    }
                }
            }
            // if (cute::thread0()) {
            //     printf("mi = %d, i = %d, row_idx = %d, max_seqlen_k = %d\n", mi, i, row_idx, max_seqlen_k);
            //     print(tensor(make_coord(i, mi), _));
            //     // print(tensor(_, j + nj * size<1, 0>(tensor)));
            // }
        }
    }
}

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask_causal(Tensor<Engine, Layout> &tensor, const int col_idx_offset_,
                                         const int max_seqlen_k, const int row_idx_offset,
                                         const int max_seqlen_q, const int warp_row_stride) {
    // Causal masking is equivalent to local masking with window_size_left = infinity and window_size_right = 0
    apply_mask_local</*HasWSLeft=*/false>(tensor, col_idx_offset_, max_seqlen_k, row_idx_offset,
                                          max_seqlen_q, warp_row_stride, -1, 0);
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void apply_mask_causal_w_idx(
    Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &idx_rowcol,
    const int col_idx_offset_, const int max_seqlen_k, const int row_idx_offset)
{
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 2, "Only support 2D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(tensor) == size<0>(idx_rowcol));
    CUTE_STATIC_ASSERT_V(size<1>(tensor) == size<1>(idx_rowcol));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        const int col_idx_limit = std::min(max_seqlen_k, 1 + row_idx_offset + get<0>(idx_rowcol(mi, 0)));
        #pragma unroll
        for (int ni = 0; ni < size<1, 1>(tensor); ++ni) {
            if (col_idx_offset_ + get<1>(idx_rowcol(0, ni)) >= col_idx_limit) {
                tensor(mi, ni) = -INFINITY;
            }
        }
        // if (cute::thread0()) {
        //     printf("ni = %d, j = %d, col_idx = %d, max_seqlen_k = %d\n", ni, j, col_idx, max_seqlen_k);
        //     print(tensor(_, make_coord(j, ni)));
        //     // print(tensor(_, j + ni * size<1, 0>(tensor)));
        // }
    }
}

template <bool Is_causal, bool Is_local, bool Has_alibi>
struct Mask {

    const int max_seqlen_k, max_seqlen_q;
    const int window_size_left, window_size_right;
    const float alibi_slope;

    __forceinline__ __device__ Mask(const int max_seqlen_k, const int max_seqlen_q,
                                    const int window_size_left, const int window_size_right,
                                    const float alibi_slope=0.f)
        : max_seqlen_k(max_seqlen_k)
        , max_seqlen_q(max_seqlen_q)
        , window_size_left(window_size_left)
        , window_size_right(window_size_right)
        , alibi_slope(!Has_alibi ? 0.0 : alibi_slope) {
    };

    // Causal_mask: whether this particular iteration needs causal masking
    template <bool Causal_mask=false, bool Is_even_MN=true, typename Engine, typename Layout, typename EngineIdx, typename LayoutIdx>
    __forceinline__ __device__ void apply_mask(Tensor<Engine, Layout> &tensor_,
                                               Tensor<EngineIdx, LayoutIdx> const &idx_tensor,
                                               const int col_idx_offset_,
                                               const int row_idx_offset,
                                               const int warp_row_stride) {
        static_assert(!(Causal_mask && Is_local), "Cannot be both causal and local");
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        static_assert(decltype(size<0>(tensor_))::value == 8, "First dimension must be 8");
        static constexpr bool Need_masking = Has_alibi || Causal_mask || Is_local || !Is_even_MN;
        static constexpr int kMmaRows = decltype(size<1>(tensor_))::value;
        if constexpr (Need_masking) {
            if constexpr (kMmaRows == 4) {
                (void)warp_row_stride;
                CUTE_STATIC_ASSERT_V(size(tensor_) == size(idx_tensor));
                #pragma unroll 1
                for (int i = 0; i < size(tensor_); ++i) {
                    const int row_idx = row_idx_offset + int(get<0>(idx_tensor(i)));
                    const int col_idx = col_idx_offset_ + int(get<1>(idx_tensor(i)));
                    if constexpr (Has_alibi) {
                        if constexpr (Is_causal) {
                            tensor_(i) += alibi_slope * col_idx;
                        } else {
                            tensor_(i) -= alibi_slope * abs(row_idx + max_seqlen_k - max_seqlen_q - col_idx);
                        }
                    }
                    bool mask_out = false;
                    if constexpr (Causal_mask || Is_local) {
                        const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right);
                        if constexpr (Is_local) {
                            const int col_idx_limit_left = std::max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left);
                            mask_out = col_idx >= col_idx_limit_right || col_idx < col_idx_limit_left;
                        } else {
                            mask_out = col_idx >= col_idx_limit_right;
                        }
                    } else if constexpr (!Is_even_MN) {
                        mask_out = col_idx >= max_seqlen_k;
                    }
                    if (mask_out) {
                        tensor_(i) = -INFINITY;
                    }
                }
            } else {
                // Deprecated sparse path keeps the existing row/col reshaped implementation.
                Tensor tensor = make_tensor(tensor_.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tensor_.layout()));
                Tensor idx_tensor_rowcol = make_tensor(idx_tensor.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(idx_tensor.layout()));
                static constexpr bool Col_idx_only = !(Has_alibi && !Is_causal) && !Is_local && !Causal_mask;
                if constexpr (Col_idx_only) {
                    #pragma unroll
                    for (int n = 0; n < size<1, 2>(tensor); ++n) {
                        #pragma unroll
                        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                            #pragma unroll
                            for (int j = 0; j < size<1, 0>(tensor); ++j) {
                                const int col_idx = col_idx_offset_ + get<1>(idx_tensor_rowcol(make_coord(0, 0), make_coord(j, nj, n)));
                                #pragma unroll
                                for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                                    #pragma unroll
                                    for (int i = 0; i < size<0, 0>(tensor); ++i) {
                                        auto coord = make_coord(make_coord(i, mi), make_coord(j, nj, n));
                                        if constexpr (Has_alibi) {
                                            tensor(coord) += alibi_slope * col_idx;
                                        }
                                        if constexpr (!Is_even_MN) {
                                            if (col_idx >= max_seqlen_k) {
                                                tensor(coord) = -INFINITY;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    #pragma unroll
                    for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                        #pragma unroll
                        for (int i = 0; i < size<0, 0>(tensor); ++i) {
                            const int row_idx = row_idx_offset + get<0>(idx_tensor_rowcol(make_coord(i, mi), make_coord(0, 0, 0)));
                            const int col_idx_limit_left = std::max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left);
                            const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right);
                            #pragma unroll
                            for (int n = 0; n < size<1, 2>(tensor); ++n) {
                                #pragma unroll
                                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                                    #pragma unroll
                                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                                        const int col_idx = col_idx_offset_ + get<1>(idx_tensor_rowcol(make_coord(i, mi), make_coord(j, nj, n)));
                                        auto coord = make_coord(make_coord(i, mi), make_coord(j, nj, n));
                                        if constexpr (Has_alibi) {
                                            if constexpr (Is_causal) {
                                                tensor(coord) += alibi_slope * col_idx;
                                            } else {
                                                tensor(coord) -= alibi_slope * abs(row_idx + max_seqlen_k - max_seqlen_q - col_idx);
                                            }
                                        }
                                        if constexpr (Causal_mask) {
                                            if (col_idx >= col_idx_limit_right) {
                                                tensor(coord) = -INFINITY;
                                            }
                                        }
                                        if constexpr (Is_local) {
                                            if (col_idx >= col_idx_limit_right || col_idx < col_idx_limit_left) {
                                                tensor(coord) = -INFINITY;
                                            }
                                        }
                                        if constexpr (!Causal_mask && !Is_local && !Is_even_MN) {
                                            if (col_idx >= max_seqlen_k) {
                                                tensor(coord) = -INFINITY;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    template <bool Causal_mask=false, bool Is_even_MN=true, typename Engine, typename Layout, typename EngineIdx, typename LayoutIdx>
    __forceinline__ __device__ void apply_mask_idx(Tensor<Engine, Layout> &tensor_,
                                                   Tensor<EngineIdx, LayoutIdx> const &idx_tensor,
                                                   const int col_idx_offset_,
                                                   const int row_idx_offset,
                                                   const int warp_row_stride) {
        static_assert(!(Causal_mask && Is_local), "Cannot be both causal and local");
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        static constexpr bool Need_masking = Has_alibi || Causal_mask || Is_local || !Is_even_MN;
        if constexpr (Need_masking) {
            CUTE_STATIC_ASSERT_V(size(tensor_) == size(idx_tensor));
            #pragma unroll
            for (int i = 0; i < size(tensor_); ++i) {
                const int row_idx = row_idx_offset + get<0>(idx_tensor(i));
                const int col_idx = col_idx_offset_ + get<1>(idx_tensor(i));
                const int col_idx_limit_left = std::max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left);
                const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right);
                if constexpr (Has_alibi) {
                    if constexpr (Is_causal) {
                        tensor_(i) += alibi_slope * col_idx;
                    } else {
                        tensor_(i) -= alibi_slope * abs(row_idx + max_seqlen_k - max_seqlen_q - col_idx);
                    }
                }
                if constexpr (Causal_mask) {
                    if (col_idx >= col_idx_limit_right) { tensor_(i) = -INFINITY; }
                }
                if constexpr (Is_local) {
                    if (col_idx >= col_idx_limit_right || col_idx < col_idx_limit_left) { tensor_(i) = -INFINITY; }
                }
                if constexpr (!Causal_mask && !Is_local && !Is_even_MN) {
                    // Varlen/非整块路径需要同时屏蔽越界的行与列，避免无效行参与 softmax。
                    if (col_idx >= max_seqlen_k || row_idx >= max_seqlen_q) { tensor_(i) = -INFINITY; }
                }
            }
        }
    }
};

} // namespace FLASH_NAMESPACE
