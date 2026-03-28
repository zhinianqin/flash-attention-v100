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

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++) {
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max) {
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum) {
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
}

template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
#ifdef UNFUSE_FMA
            tensor(mi, ni) = exp2f(__fmul_rn(tensor(mi, ni), scale) - max_scaled);
#else
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
#endif
        }
    }
}

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

    template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
        Tensor scores = make_tensor(acc_s.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == kNRows);
        if (Is_first) {
            FLASH_NAMESPACE::template reduce_max</*zero_init=*/true>(scores, row_max);
            if constexpr (Check_inf) {
                #pragma unroll
                for (int mi = 0; mi < size(row_max); ++mi) {
                    if (row_max(mi) == -INFINITY) { row_max(mi) = 0.0f; }
                }
            }
            FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            FLASH_NAMESPACE::reduce_sum</*zero_init=*/true>(scores, row_sum);
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            FLASH_NAMESPACE::template reduce_max</*zero_init=*/false>(scores, row_max);
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
            static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale;
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
            }
            FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            FLASH_NAMESPACE::reduce_sum</*zero_init=*/false>(scores, row_sum);
        }
    };

    template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1, typename TensorRowIdx, typename TensorAccOIdx, typename TensorScoreIdx>
    __forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2, TensorRowIdx const &row_idx, TensorAccOIdx const &acc_o_idx, TensorScoreIdx const &score_idx) {
        MaxOp<float> max_op;
        if constexpr (kNRows == 8) {
            if (Is_first) {
                #pragma unroll 1
                for (int mi = 0; mi < size(row_max); ++mi) {
                    const int row = int(get<0>(row_idx(mi)));
                    float scores_max = -INFINITY;
                    #pragma unroll 1
                    for (int i = 0; i < size(acc_s); ++i) {
                        if (row == int(get<0>(score_idx(i)))) {
                            scores_max = max_op(scores_max, float(acc_s(i)));
                        }
                    }
                    scores_max = row_keyed_allreduce(scores_max, row_idx, mi, max_op);
                    row_max(mi) = Check_inf && scores_max == -INFINITY ? 0.0f : scores_max;
                    const float max_scaled = scores_max == -INFINITY ? 0.0f : scores_max * softmax_scale_log2;
                    float scores_sum = 0.0f;
                    #pragma unroll 1
                    for (int i = 0; i < size(acc_s); ++i) {
                        if (row != int(get<0>(score_idx(i)))) { continue; }
#ifdef UNFUSE_FMA
                        const float p = exp2f(__fmul_rn(float(acc_s(i)), softmax_scale_log2) - max_scaled);
#else
                        const float p = exp2f(float(acc_s(i)) * softmax_scale_log2 - max_scaled);
#endif
                        acc_s(i) = p;
                        scores_sum += p;
                    }
                    row_sum(mi) = scores_sum;
                }
            } else {
                #pragma unroll 1
                for (int mi = 0; mi < size(row_max); ++mi) {
                    const int row = int(get<0>(row_idx(mi)));
                    const float scores_max_prev = row_max(mi);
                    float scores_max_cur = -INFINITY;
                    #pragma unroll 1
                    for (int i = 0; i < size(acc_s); ++i) {
                        if (row == int(get<0>(score_idx(i)))) {
                            scores_max_cur = max_op(scores_max_cur, float(acc_s(i)));
                        }
                    }
                    const float block_max = row_keyed_allreduce(scores_max_cur, row_idx, mi, max_op);
                    row_max(mi) = max_op(scores_max_prev, block_max);
                    const float scores_max = !Check_inf
                        ? row_max(mi)
                        : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                    const float scores_scale = exp2f((scores_max_prev - scores_max) * softmax_scale_log2);
                    row_sum(mi) *= scores_scale;
                    #pragma unroll 1
                    for (int i = 0; i < size(acc_o); ++i) {
                        if (row == int(get<0>(acc_o_idx(i)))) {
                            acc_o(i) *= scores_scale;
                        }
                    }
                    const float max_scaled = row_max(mi) == -INFINITY ? 0.0f : row_max(mi) * softmax_scale_log2;
                    float scores_sum_cur = 0.0f;
                    #pragma unroll 1
                    for (int i = 0; i < size(acc_s); ++i) {
                        if (row != int(get<0>(score_idx(i)))) { continue; }
#ifdef UNFUSE_FMA
                        const float p = exp2f(__fmul_rn(float(acc_s(i)), softmax_scale_log2) - max_scaled);
#else
                        const float p = exp2f(float(acc_s(i)) * softmax_scale_log2 - max_scaled);
#endif
                        acc_s(i) = p;
                        scores_sum_cur += p;
                    }
                    row_sum(mi) += scores_sum_cur;
                }
            }
        } else {
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
        }
    };

    template<bool Is_dropout=false, bool Split=false, typename Tensor0>
    __forceinline__ __device__ TensorT normalize_softmax_lse(Tensor0 &acc_o, float softmax_scale, float rp_dropout=1.0f) {
        SumOp<float> sum_op;
        quad_allreduce_(row_sum, row_sum, sum_op);
        TensorT lse = make_fragment_like(row_sum);
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * softmax_scale + __logf(sum);
            float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scale; }
        }
        return lse;
    };

    template<bool Is_dropout=false, bool Split=false, typename Tensor0, typename TensorRowIdx, typename TensorAccOIdx>
    __forceinline__ __device__ TensorT normalize_softmax_lse(Tensor0 &acc_o, float softmax_scale, float rp_dropout, TensorRowIdx const &row_idx, TensorAccOIdx const &acc_o_idx) {
        SumOp<float> sum_op;
        if constexpr (kNRows == 8) {
            #pragma unroll 1
            for (int mi = 0; mi < size(row_sum); ++mi) {
                row_sum(mi) = row_keyed_allreduce(row_sum(mi), row_idx, mi, sum_op);
            }
            TensorT lse = make_fragment_like(row_sum);
            #pragma unroll 1
            for (int mi = 0; mi < size(row_sum); ++mi) {
                const int row = int(get<0>(row_idx(mi)));
                const float sum = row_sum(mi);
                const float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
                lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * softmax_scale + __logf(sum);
                const float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
                #pragma unroll 1
                for (int i = 0; i < size(acc_o); ++i) {
                    if (row == int(get<0>(acc_o_idx(i)))) {
                        acc_o(i) *= scale;
                    }
                }
            }
            return lse;
        } else {
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
        }
    };
};

}  // namespace FLASH_NAMESPACE
