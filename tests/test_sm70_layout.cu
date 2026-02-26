#include <iostream>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm70.hpp>

using namespace cute;

__global__ void print_sm70_layout() {
    if (threadIdx.x == 0) {
        // 1. 定义 SM70 FP32 累加器原子 (M=8, N=8, K=4)
        using mma_atom = MMA_Atom<SM70_8x8x4_F32F16F16F32_TN>;

        // 2. 设计 64x64 的 Tile，使用 128 个线程
        using tiled_mma = TiledMMA<
            mma_atom,
            Layout<Shape<_4, _4, _1>>,  // Atom 排布: 16 Atoms * 8 Threads = 128 Threads
            Tile<_64, _64, _4>          // 总体 Tile 大小 (BlockM=64, BlockN=64, K=4)
        >;

        tiled_mma mma;

        // 3. 修正：使用全局函数分配针对 64x64 Tile 的累加器 C (fp32)
        auto acc = partition_fragment_C(mma, Shape<_64, _64>{});

        printf("=== TiledMMA Configuration ===\n");
        print(mma);
        printf("\n\n=== Accumulator Layout (acc.layout()) ===\n");
        print(acc.layout());
        printf("\n");
    }
}

int main() {
    print_sm70_layout<<<1, 128>>>();
    cudaDeviceSynchronize();
    return 0;
}