#include <iostream>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm70.hpp>

using namespace cute;

__global__ void print_sm70_layout() {
    if (threadIdx.x == 0) {
        // 1. 定义 SM70 FP32 累加器原子 (M=8, N=8, K=4)
        using MMA_Atom_Arch = MMA_Atom<SM70_8x8x4_F32F16F16F32_TN>;

        using TiledMma = TiledMMA<
            MMA_Atom_Arch,
            Layout<Shape<_4,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
            Tile<_32, _8, _4>>;

        TiledMma mma;
        printf("=== TiledMMA Configuration ===\n");
        print(mma);

        auto acc = partition_fragment_C(mma, Shape<_64, _64>{});
        printf("\n=== Accumulator Layout (acc.layout()) ===\n");
        print(acc.layout());

        auto acc2 = partition_fragment_C(mma, Shape<_128, _128>{});
        printf("\n=== Accumulator Layout (acc2.layout()) ===\n");
        print(acc2.layout());

        using TiledMma2 = TiledMMA<
            MMA_Atom_Arch,
            Layout<Shape<_4,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
            Tile<_32, _16, _4>>;

        TiledMma2 mma2;
        printf("=== TiledMMA2 Configuration ===\n");
        print(mma2);

        auto acc3 = partition_fragment_C(mma2, Shape<_64, _64>{});
        printf("\n=== Accumulator Layout (acc3.layout()) ===\n");
        print(acc3.layout());

        auto acc4 = partition_fragment_C(mma2, Shape<_128, _128>{});
        printf("\n=== Accumulator Layout (acc4.layout()) ===\n");
        print(acc4.layout());


        using TiledMma3 = TiledMMA<
            MMA_Atom_Arch,
            Layout<Shape<_8,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
            Tile<_64, _8, _4>>;

        TiledMma3 mma3;
        printf("\n\n=== TiledMma3 Configuration ===\n");
        print(mma3);

        auto acc5 = partition_fragment_C(mma3, Shape<_128, _128>{});
        printf("\n=== Accumulator Layout (acc5.layout()) ===\n");
        print(acc5.layout());

        using TiledMma4 = TiledMMA<
            MMA_Atom_Arch,
            Layout<Shape<_8,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
            Tile<_64, _16, _4>>;

        TiledMma4 mma4;
        printf("\n\n=== TiledMma4 Configuration ===\n");
        print(mma4);

        auto acc6 = partition_fragment_C(mma4, Shape<_128, _128>{});
        printf("\n=== Accumulator Layout (acc6.layout()) ===\n");
        print(acc6.layout());

        printf("\n");
    }
}

int main() {
    print_sm70_layout<<<1, 128>>>();
    cudaDeviceSynchronize();
    return 0;
}