#include <cstdlib>
#include <numeric>
#include <tuple>
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

__global__ void sum(int* x, long long* y, int n) {
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n) {
        atomicAdd(reinterpret_cast<unsigned long long*>(y), static_cast<unsigned long long>(x[idx]));
    }
}

__global__ void sum_block(int* x, long long* y, int n) {
    auto tid = threadIdx.x;
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ long long buf[BLOCK_SIZE];
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
        atomicAdd(reinterpret_cast<unsigned long long*>(y), static_cast<unsigned long long>(buf[0]));
    }
}

__global__ void sum_warp(int* x, long long* y, int n) {
    __shared__ long long buf[32];

    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    auto warp_id = threadIdx.x / warpSize;
    auto lane_id = threadIdx.x % warpSize;

    auto val = (idx < n) ? x[idx] : 0;
    for (auto offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (lane_id == 0) {
        buf[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        auto warp_num = blockDim.x / warpSize;
        val = (lane_id < warp_num) ? buf[lane_id] : 0;
#pragma unroll
        for (auto offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) {
            atomicAdd(reinterpret_cast<unsigned long long*>(y), static_cast<unsigned long long>(val));
        }
    }
}

auto host_sum(const std::vector<int>& x) -> std::tuple<long long, float> {
    auto result = 0LL;
    auto elapsed = util::time_host(REPEAT_TIME, [&]() { result = std::accumulate(x.begin(), x.end(), 0LL); });
    return {result, elapsed};
}

template <typename Kernel>
auto dev_sum(Kernel kernel, int* x_d, long long* y_d) -> std::tuple<long long, float> {
    auto grid_size = util::ceil_div(N, BLOCK_SIZE);
    auto elapsed = util::time_device(REPEAT_TIME, [&]() {
        CHECK_ERR(cudaMemset(y_d, 0, sizeof(long long)));
        kernel<<<grid_size, BLOCK_SIZE>>>(x_d, y_d, N);
        CHECK_ERR(cudaGetLastError());
        CHECK_ERR(cudaDeviceSynchronize());
    });

    auto result = 0LL;
    CHECK_ERR(cudaMemcpy(&result, y_d, sizeof(long long), cudaMemcpyDeviceToHost));
    return {result, elapsed};
}
}  // namespace

int main() {
    auto x = std::vector<int>(N);
    for (auto i = 0; i < N; i++) {
        x[i] = i;
    }

    auto x_d = static_cast<int*>(nullptr);
    auto y_d = static_cast<long long*>(nullptr);
    CHECK_ERR(cudaMalloc((void**)&x_d, N * sizeof(int)));
    CHECK_ERR(cudaMalloc((void**)&y_d, sizeof(long long)));
    CHECK_ERR(cudaMemcpy(x_d, x.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    auto [host_result, host_elapsed] = host_sum(x);
    auto run = [&, host_expected = host_result, host_elapsed = host_elapsed](const char* name, auto kernel) -> bool {
        auto [y, device_elapsed] = dev_sum(kernel, x_d, y_d);

        if (std::llabs(y - host_expected) == 0) {
            fmt::println("{}:\n Ok[host={}, device={}]: {} == {}", name, host_elapsed, device_elapsed, y,
                         host_expected);
            return true;
        }

        fmt::println("{}:\n Err[host={}, device={}]: {} != {}", name, host_elapsed, device_elapsed, y, host_expected);
        return false;
    };

    if (!run("sum", sum)) {
        return -1;
    }
    if (!run("sum_block", sum_block)) {
        return -1;
    }
    if (!run("sum_warp", sum_warp)) {
        return -1;
    }

    CHECK_ERR(cudaFree(x_d));
    CHECK_ERR(cudaFree(y_d));

    return 0;
}
