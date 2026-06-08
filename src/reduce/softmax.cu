#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <tuple>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#include <fmt/core.h>

#include "calc.h"
#include "error.h"
#include "timer.h"

namespace {
constexpr auto N = 1024 * 1024 * 8;
constexpr auto BLOCK_SIZE = 256;
constexpr auto REPEAT_TIME = 256;

__device__ static float atomic_max(float* addr, float val) {
    return 0;
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
            atomicMax(reinterpret_cast<unsigned int*>(y), __float_as_uint(val));
        }
    }
}

__global__ void sum_warp(float* x, float* y, float* max_val, int n) {
    __shared__ float buf[32];
}

__global__ void softmax(float* x, float* y, float* sum_exp, float* max_val, int n) {
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;

    y[idx] = expf(x[idx] - *max_val) / (*sum_exp);
}

auto host_softmax(const std::vector<float>& x) -> std::tuple<std::vector<float>, float> {
    auto result = std::vector<float>(N);
    auto elapsed = util::time_host(REPEAT_TIME, [&]() {
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

template <typename Kernel>
auto dev_softmax(Kernel kernel, float* x_d, float* y_d) -> std::tuple<std::vector<float>, float> {
    auto grid_size = util::ceil_div(N, BLOCK_SIZE);
    auto elapsed = util::time_device(REPEAT_TIME, [&]() {
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
    CHECK_ERR(cudaMalloc((void**)&x_d, N * sizeof(float)));
    CHECK_ERR(cudaMalloc((void**)&y_d, N * sizeof(float)));

    std::iota(x_h.begin(), x_h.end(), 0.0f);
    CHECK_ERR(cudaMemcpy(x_d, x_h.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    auto [host_result, host_elapsed] = host_softmax(x_h);
    auto run = [&, host_result = host_result, host_elapsed = host_elapsed](const char* name, auto kernel) -> bool {
        auto [y_h, device_elapsed] = dev_softmax(kernel, x_d, y_d);
        return true;
    };

    if (!run("softmax", softmax)) {
        return -1;
    }

    CHECK_ERR(cudaFree(x_d));
    CHECK_ERR(cudaFree(y_d));

    return 0;
}
