/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

using namespace cute;

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_>
struct Flash_fwd_kernel_traits  {
    using Element = cutlass::half_t;
    using ElementAccum = float;
    using index_t = int64_t;
    using MMA_Atom_Arch = MMA_Atom<SM70_8x8x4_F32F16F16F32_TN>;
    using SmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, Element>;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    using TiledMma = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<_4, Int<kNWarps>, _1>>, 
        Tile<_32, Int<kNWarps * 8>, _4>
    >;

    static constexpr int kSmemSize = 96 * 1024;
};

// Is_V_in_regs is an option to reduce smem usage, but will increase register pressue.
// No_double_buffer is another option to reduce smem usage, but will slow things down.
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
         int AtomLayoutMSdP_=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=2,
         bool Is_V_in_regs_=false, bool No_double_buffer_=false, typename elem_type=cutlass::half_t>
struct Flash_bwd_kernel_traits {

};

////////////////////////////////////////////////////////////////////////////////////////////////////
