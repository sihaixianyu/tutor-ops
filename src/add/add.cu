#include <stdlib.h>

#include <cuda_runtime.h>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include "error.h"
#include "macro.h"

__global__ void add(float* a, float* b, float* c, int N) {
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    c[idx] = a[idx] + b[idx];
}

int main() {
    constexpr auto N = 1024 * 1024 * 8;
    auto a_h = static_cast<float*>(malloc(N * sizeof(float)));
    auto b_h = static_cast<float*>(malloc(N * sizeof(float)));
    auto c_h = static_cast<float*>(malloc(N * sizeof(float)));
    for (auto i = 0; i < N; i++) {
        a_h[i] = i;
        b_h[i] = N - 1 - i;
    }

    auto a_d = static_cast<float*>(nullptr);
    auto b_d = static_cast<float*>(nullptr);
    auto c_d = static_cast<float*>(nullptr);
    CHECK_ERR(cudaMalloc((void**)&a_d, N * sizeof(float)));
    CHECK_ERR(cudaMalloc((void**)&b_d, N * sizeof(float)));
    CHECK_ERR(cudaMalloc((void**)&c_d, N * sizeof(float)));
    CHECK_ERR(cudaMemcpy(a_d, a_h, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_ERR(cudaMemcpy(b_d, b_h, N * sizeof(float), cudaMemcpyHostToDevice));

    auto block_size = 256;
    auto grid_size = CEIL(N, 256);
    add<<<grid_size, block_size>>>(a_d, b_d, c_d, N);
    cudaDeviceSynchronize();
    CHECK_ERR(cudaMemcpy(c_h, c_d, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (auto i = 0; i < N; i++) {
        if (c_h[i] != a_h[i] + b_h[i]) {
            fmt::println("Error at index {}: {} + {} != {}", i, a_h[i], b_h[i], c_h[i]);
            return -1;
        }
    }

    return 0;
}
