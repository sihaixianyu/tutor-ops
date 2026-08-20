#include <algorithm>
#include <cmath>
#include <numeric>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

#include <fmt/core.h>

#include "calc.h"
#include "cmp.h"
#include "error.h"
#include "timer.h"

constexpr auto N = 1024 * 1024 * 8;
constexpr auto BLOCK_SIZE = 256;
constexpr auto REPEAT_TIME = 256;

namespace host {
auto sum_host(const std::vector<float>& x) -> std::tuple<float, float> {
    auto result = 0.0f;
    auto elapsed = util::time_cpu(REPEAT_TIME, [&]() { result = std::accumulate(x.begin(), x.end(), 0.0f); });
    return {result, elapsed};
}
}  // namespace host

namespace device {
// Launch a sum kernel (zeroing y first), time it, and read the accumulator back.
template <typename Kernel>
auto sum_device(Kernel kernel, float* x_d, float* y_d) -> std::tuple<float, float> {
    auto grid_size = util::ceil_div(N, BLOCK_SIZE);
    auto elapsed = util::time_cuda(REPEAT_TIME, [&]() {
        CHECK_ERR(cudaMemset(y_d, 0, sizeof(float)));
        kernel<<<grid_size, BLOCK_SIZE>>>(x_d, y_d, N);
        CHECK_ERR(cudaGetLastError());
        CHECK_ERR(cudaDeviceSynchronize());
    });

    auto result = 0.0f;
    CHECK_ERR(cudaMemcpy(&result, y_d, sizeof(float), cudaMemcpyDeviceToHost));
    return {result, elapsed};
}

// Atomic reduction: every thread adds its element into a single accumulator.
__global__ void sum(float* x, float* y, int n) {
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n) {
        atomicAdd(y, x[idx]);
    }
}

// Block reduction in shared memory, then one atomicAdd per block.
__global__ void sum_block(float* x, float* y, int n) {
    auto tid = threadIdx.x;
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float buf[BLOCK_SIZE];
    buf[tid] = (idx < n) ? x[idx] : 0.0f;
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

// Warp-shuffle reduction first (no shared), then shared for the warp-partial sums.
__global__ void sum_warp(float* x, float* y, int n) {
    __shared__ float buf[32];

    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    auto warp_id = threadIdx.x / warpSize;
    auto lane_id = threadIdx.x % warpSize;

    auto val = (idx < n) ? x[idx] : 0.0f;
    for (auto offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (lane_id == 0) {
        buf[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        auto warp_num = blockDim.x / warpSize;
        val = (lane_id < warp_num) ? buf[lane_id] : 0.0f;
#pragma unroll
        for (auto offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) {
            atomicAdd(y, val);
        }
    }
}
}  // namespace device

int main() {
    auto x = std::vector<float>(N);
    for (auto i = 0; i < N; i++) {
        x[i] = static_cast<float>(i);
    }

    auto x_d = static_cast<float*>(nullptr);
    auto y_d = static_cast<float*>(nullptr);
    CHECK_ERR(cudaMalloc((void**)&x_d, N * sizeof(float)));
    CHECK_ERR(cudaMalloc((void**)&y_d, sizeof(float)));
    CHECK_ERR(cudaMemcpy(x_d, x.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    auto [host_result, cpu_elapsed] = host::sum_host(x);
    fmt::println("[host] elapsed_time={} ms", cpu_elapsed);

    constexpr auto eps = 1e-3f;

    auto [device_result_naive, naive_elapsed] = device::sum_device(device::sum, x_d, y_d);
    auto naive_diff = std::fabs(device_result_naive - host_result);
    fmt::println("[device::naive] elapsed_time={} ms, result=[{}, {}], diff={}", naive_elapsed, host_result, device_result_naive, naive_diff);
    if (!util::cmp_scalar(device_result_naive, host_result, eps)) return -1;

    auto [device_result_block, block_elapsed] = device::sum_device(device::sum_block, x_d, y_d);
    auto block_diff = std::fabs(device_result_block - host_result);
    fmt::println("[device::block] elapsed_time={} ms, result=[{}, {}], diff={}", block_elapsed, host_result, device_result_block, block_diff);
    if (!util::cmp_scalar(device_result_block, host_result, eps)) return -1;

    auto [device_result_warp, warp_elapsed] = device::sum_device(device::sum_warp, x_d, y_d);
    auto warp_diff = std::fabs(device_result_warp - host_result);
    fmt::println("[device::warp] elapsed_time={} ms, result=[{}, {}], diff={}", warp_elapsed, host_result, device_result_warp, warp_diff);
    if (!util::cmp_scalar(device_result_warp, host_result, eps)) return -1;

    CHECK_ERR(cudaFree(x_d));
    CHECK_ERR(cudaFree(y_d));

    return 0;
}
