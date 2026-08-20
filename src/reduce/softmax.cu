#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <tuple>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#include <fmt/core.h>

#include "calc.h"
#include "cmp.h"
#include "error.h"
#include "timer.h"

constexpr auto N = 1024 * 1024 * 1;
constexpr auto BLOCK_SIZE = 1024;
constexpr auto REPEAT_TIME = 256;

namespace host {
auto softmax_host(const std::vector<float>& x) -> std::tuple<std::vector<float>, float> {
    auto result = std::vector<float>(N);
    auto elapsed = util::time_cpu(REPEAT_TIME, [&]() {
        auto max_val = *std::max_element(x.begin(), x.end());
        auto sum_exp = 0.0f;
        for (auto i = 0; i < N; i++) {
            sum_exp += std::exp(x[i] - max_val);
        }
        for (auto i = 0; i < N; i++) {
            result[i] = std::exp(x[i] - max_val) / sum_exp;
        }
    });
    return {std::move(result), elapsed};
}
}  // namespace host

namespace device {
__device__ static float atomic_max(float* address, float val);

// Launch the softmax pipeline (max -> sum -> softmax), time it, copy back.
template <typename Kernel, typename MaxKernel, typename SumKernel>
auto softmax_device(Kernel kernel, MaxKernel max_k, SumKernel sum_k, float* x_d, float* y_d, float* max_val,
                    float* sum_val) -> std::tuple<std::vector<float>, float> {
    auto grid_size = util::ceil_div(N, BLOCK_SIZE);
    auto block = dim3(BLOCK_SIZE, 1, 1);
    auto elapsed = util::time_cuda(REPEAT_TIME, [&]() {
        const auto flt_max = -FLT_MAX;
        CHECK_ERR(cudaMemcpy(max_val, &flt_max, sizeof(float), cudaMemcpyHostToDevice));
        CHECK_ERR(cudaMemset(sum_val, 0, sizeof(float)));

        max_k<<<grid_size, block>>>(x_d, max_val, N);
        sum_k<<<grid_size, block>>>(x_d, sum_val, max_val, N);
        kernel<<<grid_size, block>>>(x_d, y_d, sum_val, max_val, N);

        CHECK_ERR(cudaGetLastError());
        CHECK_ERR(cudaDeviceSynchronize());
    });

    auto result = std::vector<float>(N);
    CHECK_ERR(cudaMemcpy(result.data(), y_d, N * sizeof(float), cudaMemcpyDeviceToHost));
    return {std::move(result), elapsed};
}

// Numerically stable softmax: subtract the max, then divide by the sum of exps.
__global__ void softmax(float* x, float* y, float* sum_val, float* max_val, int n) {
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;

    y[idx] = expf(x[idx] - *max_val) / (*sum_val);
}

// Block reduction in shared memory for the global max (then atomic max).
__global__ void max_block(float* x, float* y, int n) {
    auto tid = threadIdx.x;
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float buf[BLOCK_SIZE];
    buf[tid] = (idx < n) ? x[idx] : -FLT_MAX;
    __syncthreads();

    for (auto offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            buf[tid] = fmaxf(buf[tid], buf[tid + offset]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomic_max(y, buf[0]);
    }
}

// Warp-shuffle max (then atomic max per block) for the global max.
__global__ void max_warp(float* x, float* y, int n) {
    __shared__ float buf[32];

    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    auto warp_id = threadIdx.x / warpSize;
    auto lane_id = threadIdx.x % warpSize;
    auto warp_num = blockDim.x / warpSize;

    auto val = (idx < n) ? x[idx] : -FLT_MAX;
    for (auto offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    if (lane_id == 0) {
        buf[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane_id < warp_num) ? buf[lane_id] : -FLT_MAX;
        for (auto offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        if (lane_id == 0) {
            atomic_max(y, val);
        }
    }
}

// Block reduction in shared memory for the sum of exps (then atomic add).
__global__ void sum_block(float* x, float* y, float* max, int n) {
    auto tid = threadIdx.x;
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float buf[BLOCK_SIZE];
    buf[tid] = (idx < n) ? expf(x[idx] - *max) : 0.0f;
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

// Warp-shuffle sum of exp(x - max), then atomic add for the denominator.
__global__ void sum_warp(float* x, float* y, float* max, int n) {
    __shared__ float buf[32];
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    auto warp_id = threadIdx.x / warpSize;
    auto lane_id = threadIdx.x % warpSize;
    auto warp_num = blockDim.x / warpSize;

    auto val = (idx < n) ? expf(x[idx] - *max) : 0.0f;
    for (auto offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (lane_id == 0) {
        buf[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane_id < warp_num) ? buf[lane_id] : 0.0f;
        for (auto offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) {
            atomicAdd(y, val);
        }
    }
}

// Atomic max for float via thread-safe CAS loop.
__device__ static float atomic_max(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i;
    int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}
}  // namespace device

int main() {
    auto x_h = std::vector<float>(N);
    auto y_h = std::vector<float>(N);

    auto x_d = static_cast<float*>(nullptr);
    auto y_d = static_cast<float*>(nullptr);
    auto max_val = static_cast<float*>(nullptr);
    auto sum_val = static_cast<float*>(nullptr);
    CHECK_ERR(cudaMalloc((void**)&x_d, N * sizeof(float)));
    CHECK_ERR(cudaMalloc((void**)&y_d, N * sizeof(float)));
    CHECK_ERR(cudaMalloc((void**)&max_val, sizeof(float)));
    CHECK_ERR(cudaMalloc((void**)&sum_val, sizeof(float)));

    std::iota(x_h.begin(), x_h.end(), 0.0f);
    CHECK_ERR(cudaMemcpy(x_d, x_h.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    auto [host_result, cpu_elapsed] = host::softmax_host(x_h);
    fmt::println("[host] elapsed_time={} ms", cpu_elapsed);

    constexpr auto eps = 1e-6f;

    auto [device_result_block, block_elapsed] =
        device::softmax_device(device::softmax, device::max_block, device::sum_block, x_d, y_d, max_val, sum_val);
    auto [block_ok, block_max_idx] = util::cmp_vec(host_result, device_result_block, eps);
    auto block_max_diff = std::fabs(host_result[block_max_idx] - device_result_block[block_max_idx]);
    fmt::println("[device::block] elapsed_time={} ms, result=[{}, {}], diff={:.10e}", block_elapsed,
                host_result[block_max_idx], device_result_block[block_max_idx], block_max_diff);
    if (!block_ok) return -1;

    auto [device_result_warp, warp_elapsed] =
        device::softmax_device(device::softmax, device::max_warp, device::sum_warp, x_d, y_d, max_val, sum_val);
    auto [warp_ok, warp_max_idx] = util::cmp_vec(host_result, device_result_warp, eps);
    auto warp_max_diff = std::fabs(host_result[warp_max_idx] - device_result_warp[warp_max_idx]);
    fmt::println("[device::warp] elapsed_time={} ms, result=[{}, {}], diff={:.10e}", warp_elapsed,
                host_result[warp_max_idx], device_result_warp[warp_max_idx], warp_max_diff);
    if (!warp_ok) return -1;

    CHECK_ERR(cudaFree(x_d));
    CHECK_ERR(cudaFree(y_d));
    CHECK_ERR(cudaFree(max_val));
    CHECK_ERR(cudaFree(sum_val));

    return 0;
}
