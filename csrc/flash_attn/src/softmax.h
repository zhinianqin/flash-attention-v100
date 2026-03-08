/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>

#include "namespace_config.h"
#include "philox.cuh"
#include "utils.h"

namespace FLASH_NAMESPACE {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Operator, typename TensorRowIdx>
__device__ __forceinline__ float row_keyed_allreduce(float x, TensorRowIdx const &row_idx, const int mi, Operator &op) {
    int key = get<0>(row_idx(mi));
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float x_peer = __shfl_xor_sync(uint32_t(-1), x, offset);
        int key_peer = __shfl_xor_sync(uint32_t(-1), key, offset);
        if (key_peer == key) { x = op(x, x_peer); }
    }
    return x;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNRows>
struct Softmax {

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;

    __forceinline__ __device__ Softmax() {};

    template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1, typename TensorRowIdx, typename TensorAccOIdx, typename TensorScoreIdx>
    __forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2, TensorRowIdx const &row_idx, TensorAccOIdx const &acc_o_idx, TensorScoreIdx const &score_idx) {
        MaxOp<float> max_op;
        if (Is_first) {
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) { row_max(mi) = -INFINITY; }
            #pragma unroll
            for (int i = 0; i < size(acc_s); ++i) {
                const int row = int(get<0>(score_idx(i)));
                #pragma unroll
                for (int mi = 0; mi < size(row_max); ++mi) {
                    if (row == int(get<0>(row_idx(mi)))) { row_max(mi) = max_op(row_max(mi), float(acc_s(i))); }
                }
            }
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) { row_max(mi) = row_keyed_allreduce(row_max(mi), row_idx, mi, max_op); }
            #pragma unroll
            for (int mi = 0; mi < size(row_sum); ++mi) { row_sum(mi) = 0.f; }
            #pragma unroll
            for (int i = 0; i < size(acc_s); ++i) {
                const int row = int(get<0>(score_idx(i)));
                float row_max_i = 0.f;
                bool found = false;
                #pragma unroll
                for (int mi = 0; mi < size(row_max); ++mi) {
                    if (row == int(get<0>(row_idx(mi)))) {
                        float rm = row_max(mi);
                        row_max_i = !Check_inf ? rm : (rm == -INFINITY ? 0.0f : rm);
                        found = true;
                    }
                }
                if (!found) { continue; }
                const float score_i = float(acc_s(i));
                const float p = exp2f(score_i * softmax_scale_log2 - row_max_i * softmax_scale_log2);
                acc_s(i) = p;
                #pragma unroll
                for (int mi = 0; mi < size(row_sum); ++mi) {
                    if (row == int(get<0>(row_idx(mi)))) { row_sum(mi) += p; }
                }
            }
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            Tensor scores_max_cur = make_fragment_like(row_max);
            #pragma unroll
            for (int mi = 0; mi < size(scores_max_cur); ++mi) { scores_max_cur(mi) = -INFINITY; }
            #pragma unroll
            for (int i = 0; i < size(acc_s); ++i) {
                const int row = int(get<0>(score_idx(i)));
                #pragma unroll
                for (int mi = 0; mi < size(scores_max_cur); ++mi) {
                    if (row == int(get<0>(row_idx(mi)))) { scores_max_cur(mi) = max_op(scores_max_cur(mi), float(acc_s(i))); }
                }
            }
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                const float block_max = row_keyed_allreduce(scores_max_cur(mi), row_idx, mi, max_op);
                row_max(mi) = max_op(row_max(mi), block_max);
            }
            Tensor scores_scale_row = make_fragment_like(row_max);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                scores_scale_row(mi) = scores_scale;
                row_sum(mi) *= scores_scale;
            }
            #pragma unroll
            for (int i = 0; i < size(acc_o); ++i) {
                const int row = get<0>(acc_o_idx(i));
                float scale = 1.f;
                #pragma unroll
                for (int mi = 0; mi < size(row_max); ++mi) {
                    if (row == get<0>(row_idx(mi))) { scale = scores_scale_row(mi); }
                }
                acc_o(i) *= scale;
            }
            Tensor scores_sum_cur = make_fragment_like(row_sum);
            #pragma unroll
            for (int mi = 0; mi < size(scores_sum_cur); ++mi) { scores_sum_cur(mi) = 0.f; }
            #pragma unroll
            for (int i = 0; i < size(acc_s); ++i) {
                const int row = int(get<0>(score_idx(i)));
                float row_max_i = 0.f;
                bool found = false;
                #pragma unroll
                for (int mi = 0; mi < size(row_max); ++mi) {
                    if (row == int(get<0>(row_idx(mi)))) {
                        float rm = row_max(mi);
                        row_max_i = !Check_inf ? rm : (rm == -INFINITY ? 0.0f : rm);
                        found = true;
                    }
                }
                if (!found) { continue; }
                const float score_i = float(acc_s(i));
                const float p = exp2f(score_i * softmax_scale_log2 - row_max_i * softmax_scale_log2);
                acc_s(i) = p;
                #pragma unroll
                for (int mi = 0; mi < size(scores_sum_cur); ++mi) {
                    if (row == int(get<0>(row_idx(mi)))) { scores_sum_cur(mi) += p; }
                }
            }
            #pragma unroll
            for (int mi = 0; mi < size(row_sum); ++mi) { row_sum(mi) += scores_sum_cur(mi); }
        }
    };

    template<bool Is_dropout=false, bool Split=false, typename Tensor0, typename TensorRowIdx, typename TensorAccOIdx>
    __forceinline__ __device__ TensorT normalize_softmax_lse(Tensor0 &acc_o, float softmax_scale, float rp_dropout, TensorRowIdx const &row_idx, TensorAccOIdx const &acc_o_idx) {
        SumOp<float> sum_op;
        #pragma unroll
        for (int mi = 0; mi < size(row_sum); ++mi) {
            row_sum(mi) = row_keyed_allreduce(row_sum(mi), row_idx, mi, sum_op);
        }
        TensorT lse = make_fragment_like(row_sum);
        TensorT row_scale = make_fragment_like(row_sum);
        #pragma unroll
        for (int mi = 0; mi < size(row_sum); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * softmax_scale + __logf(sum);
            row_scale(mi) = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
        }
        #pragma unroll
        for (int i = 0; i < size(acc_o); ++i) {
            const int row = get<0>(acc_o_idx(i));
            float scale = 1.f;
            #pragma unroll
            for (int mi = 0; mi < size(row_scale); ++mi) {
                if (row == get<0>(row_idx(mi))) {
                    scale = row_scale(mi);
                }
            }
            acc_o(i) *= scale;
        }
        return lse;
    };
};

}  // namespace FLASH_NAMESPACE
