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

namespace {
__global__ void softmax(float* x, float* y, float* sum_val, float* max_val, int n);
__global__ void max_warp(float* x, float* y, int n);
__global__ void sum_warp(float* x, float* y, float* max_val, int n);
__device__ static float atomic_max(float* address, float val);

constexpr auto N = 1024 * 1024 * 1;
constexpr auto BLOCK_SIZE = 256;
constexpr auto REPEAT_TIME = 256;

auto softmax_cpu(const std::vector<float>& x) -> std::tuple<std::vector<float>, float> {
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

__global__ void softmax(float* x, float* y, float* sum_val, float* max_val, int n) {
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;

    y[idx] = expf(x[idx] - *max_val) / (*sum_val);
}

__global__ void max_warp(float* x, float* y, int n) {
    __shared__ float buf[32];

    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    auto warp_id = threadIdx.x / warpSize;
    auto lane_id = threadIdx.x % warpSize;

    auto val = (idx < n) ? x[idx] : -FLT_MAX;
    for (auto offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    if (lane_id == 0) {
        buf[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        auto warp_num = blockDim.x / warpSize;
        val = (lane_id < warp_num) ? buf[lane_id] : -FLT_MAX;
        for (auto offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        if (lane_id == 0) {
            atomic_max(y, val);
        }
    }
}

__global__ void sum_warp(float* x, float* y, float* max, int n) {
    __shared__ float buf[32];
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    auto warp_id = threadIdx.x / warpSize;
    auto lane_id = threadIdx.x % warpSize;

    auto val = (idx < n) ? expf(x[idx] - *max) : 0.0f;
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
        for (auto offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) {
            atomicAdd(y, val);
        }
    }
}

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

template <typename Kernel>
auto softmax_cuda(Kernel kernel, float* x_d, float* y_d, float* max_val, float* sum_val)
    -> std::tuple<std::vector<float>, float> {
    auto grid_size = util::ceil_div(N, BLOCK_SIZE);
    auto elapsed = util::time_cuda(REPEAT_TIME, [&]() {
        const auto flt_max = -FLT_MAX;
        CHECK_ERR(cudaMemcpy(max_val, &flt_max, sizeof(float), cudaMemcpyHostToDevice));
        CHECK_ERR(cudaMemset(sum_val, 0, sizeof(float)));

        max_warp<<<grid_size, BLOCK_SIZE>>>(x_d, max_val, N);
        sum_warp<<<grid_size, BLOCK_SIZE>>>(x_d, sum_val, max_val, N);
        kernel<<<grid_size, BLOCK_SIZE>>>(x_d, y_d, sum_val, max_val, N);

        CHECK_ERR(cudaGetLastError());
        CHECK_ERR(cudaDeviceSynchronize());
    });

    auto result = std::vector<float>(N);
    CHECK_ERR(cudaMemcpy(result.data(), y_d, N * sizeof(float), cudaMemcpyDeviceToHost));
    return {std::move(result), elapsed};
}
}  // namespace

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

    auto [cpu_result, cpu_elapsed] = softmax_cpu(x_h);
    auto [cuda_result, cuda_elapsed] = softmax_cuda(softmax, x_d, y_d, max_val, sum_val);
    fmt::println("cpu elapsed: {}, cuda elapsed: {}", cpu_elapsed, cuda_elapsed);

    constexpr auto eps = 1e-6f;
    auto [is_same, diffs] = util::cmp_vec(cpu_result, cuda_result, eps);
    if (is_same) {
        fmt::println("[softmax] ok tc_cpu={}, tc_cuda={}", cpu_elapsed, cuda_elapsed);
    } else {
        fmt::println("[softmax] err tc_cpu={}, tc_cuda={}", cpu_elapsed, cuda_elapsed);
    }

    auto max_diff = *std::max_element(diffs.begin(), diffs.end());
    fmt::println("max diff = {:.10e}", max_diff);

    CHECK_ERR(cudaFree(x_d));
    CHECK_ERR(cudaFree(y_d));
    CHECK_ERR(cudaFree(max_val));
    CHECK_ERR(cudaFree(sum_val));

    return 0;
}
