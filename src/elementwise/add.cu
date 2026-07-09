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

__global__ void add(float* a, float* b, float* c, int n) {
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;

    c[idx] = a[idx] + b[idx];
}

__global__ void add_float4(float* a, float* b, float* c, int n) {
    auto idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx >= n)
        return;

    if (idx + 3 < n) {
        auto tmp_a = reinterpret_cast<float4*>(&(a[idx]))[0];
        auto tmp_b = reinterpret_cast<float4*>(&(b[idx]))[0];
        float4 tmp_c;
        tmp_c.x = tmp_a.x + tmp_b.x;
        tmp_c.y = tmp_a.y + tmp_b.y;
        tmp_c.z = tmp_a.z + tmp_b.z;
        tmp_c.w = tmp_a.w + tmp_b.w;
        reinterpret_cast<float4*>(&(c[idx]))[0] = tmp_c;
        return;
    }

    for (auto i = idx; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

auto host_add(const std::vector<float>& a, const std::vector<float>& b) -> std::tuple<std::vector<float>, float> {
    auto c = std::vector<float>(N);
    auto elapsed = util::time_cpu(REPEAT_TIME, [&]() {
        for (auto i = 0; i < N; i++) {
            c[i] = a[i] + b[i];
        }
    });
    return {c, elapsed};
}

template <typename Kernel>
auto dev_add(Kernel kernel, int grid_size, float* a_d, float* b_d, float* c_d)
    -> std::tuple<std::vector<float>, float> {
    auto elapsed = util::time_cuda(REPEAT_TIME, [&]() {
        kernel<<<grid_size, BLOCK_SIZE>>>(a_d, b_d, c_d, N);
        CHECK_ERR(cudaGetLastError());
        CHECK_ERR(cudaDeviceSynchronize());
    });

    auto result = std::vector<float>(N);
    CHECK_ERR(cudaMemcpy(result.data(), c_d, N * sizeof(float), cudaMemcpyDeviceToHost));
    return {result, elapsed};
}
}  // namespace

int main() {
    auto a_h = std::vector<float>(N);
    auto b_h = std::vector<float>(N);
    for (auto i = 0; i < N; i++) {
        a_h[i] = i;
        b_h[i] = N - 1 - i;
    }

    auto [host_result, host_elapsed] = host_add(a_h, b_h);

    auto a_d = static_cast<float*>(nullptr);
    auto b_d = static_cast<float*>(nullptr);
    auto c_d = static_cast<float*>(nullptr);
    CHECK_ERR(cudaMalloc((void**)&a_d, N * sizeof(float)));
    CHECK_ERR(cudaMalloc((void**)&b_d, N * sizeof(float)));
    CHECK_ERR(cudaMalloc((void**)&c_d, N * sizeof(float)));
    CHECK_ERR(cudaMemcpy(a_d, a_h.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_ERR(cudaMemcpy(b_d, b_h.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    auto run = [&, host_result = host_result, host_elapsed = host_elapsed](const char* name, auto kernel,
                                                                           int grid_size) -> bool {
        auto [dev_result, device_elapsed] = dev_add(kernel, grid_size, a_d, b_d, c_d);

        for (auto i = 0; i < N; i++) {
            if (dev_result[i] != host_result[i]) {
                fmt::println("{}:\n Err[host={}, device={}]: {} != {} at index {}", name, host_elapsed, device_elapsed,
                             dev_result[i], host_result[i], i);
                return false;
            }
        }

        fmt::println("{}:\n Ok[host={}, device={}]: {} == {}", name, host_elapsed, device_elapsed, dev_result[0],
                     host_result[0]);
        return true;
    };

    if (!run("add", add, util::ceil_div(N, BLOCK_SIZE))) {
        return -1;
    }

    auto grid_size = util::ceil_div(util::ceil_div(N, 4), BLOCK_SIZE);
    if (!run("add_float4", add_float4, grid_size)) {
        return -1;
    }

    CHECK_ERR(cudaFree(a_d));
    CHECK_ERR(cudaFree(b_d));
    CHECK_ERR(cudaFree(c_d));

    return 0;
}
