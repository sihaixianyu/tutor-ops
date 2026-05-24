#include <stdlib.h>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include "error.h"
#include "macro.h"

#define N 1024 * 8
#define BLOCK_SIZE 256
#define GRID_SIZE (CEIL(N, BLOCK_SIZE))

__global__ void reduce(float* x, float* y, int n) {
    auto tid = threadIdx.x;
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float buf[BLOCK_SIZE];
    if (idx < n) {
        buf[tid] = x[idx];
    } else {
        buf[tid] = 0;
    }
    __syncthreads();

    for (auto offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            buf[tid] += buf[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(y, buf[0]);
    }
}

int main() {
    constexpr auto kRatioTol = 1e-4f;
    constexpr auto kAddTol = 1e-8f;

    auto x = std::vector<float>(N);
    auto y = 0.0f;
    for (auto i = 0; i < N; i++) {
        x[i] = i;
    }

    auto x_d = static_cast<float*>(nullptr);
    auto y_d = static_cast<float*>(nullptr);
    CHECK_ERR(cudaMalloc((void**)&x_d, N * sizeof(float)));
    CHECK_ERR(cudaMalloc((void**)&y_d, sizeof(float)));
    CHECK_ERR(cudaMemcpy(x_d, x.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_ERR(cudaMemset(y_d, 0, sizeof(float)));

    reduce<<<GRID_SIZE, BLOCK_SIZE>>>(x_d, y_d, N);
    CHECK_ERR(cudaGetLastError());
    CHECK_ERR(cudaDeviceSynchronize());
    CHECK_ERR(cudaMemcpy(&y, y_d, sizeof(float), cudaMemcpyDeviceToHost));

    auto exp_sum = std::accumulate(x.begin(), x.end(), 0.0f);
    if (fabs(y - exp_sum) < kAddTol + kRatioTol * fabs(exp_sum)) {
        fmt::println("Ok: {} == {}", y, exp_sum);
    } else {
        fmt::println("Err: {} != {}", y, exp_sum);
        return -1;
    }

    return 0;
}
