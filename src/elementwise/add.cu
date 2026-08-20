#include <cmath>
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
auto add_host(const std::vector<float>& a, const std::vector<float>& b) -> std::tuple<std::vector<float>, float> {
    auto c = std::vector<float>(N);
    auto elapsed = util::time_cpu(REPEAT_TIME, [&]() {
        for (auto i = 0; i < N; i++) {
            c[i] = a[i] + b[i];
        }
    });
    return {c, elapsed};
}
}  // namespace host

namespace device {
template <typename Kernel>
auto add_device(Kernel kernel, int grid_size, float* a_d, float* b_d, float* c_d)
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

// Elementwise add, one element per thread.
__global__ void add(float* a, float* b, float* c, int n) {
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;

    c[idx] = a[idx] + b[idx];
}

// Vectorized add: 4 elements per thread via float4 loads/stores.
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
}  // namespace device

int main() {
    auto a_h = std::vector<float>(N);
    auto b_h = std::vector<float>(N);
    for (auto i = 0; i < N; i++) {
        a_h[i] = i;
        b_h[i] = N - 1 - i;
    }

    auto [host_result, cpu_elapsed] = host::add_host(a_h, b_h);
    fmt::println("[host] elapsed_time={} ms", cpu_elapsed);

    auto a_d = static_cast<float*>(nullptr);
    auto b_d = static_cast<float*>(nullptr);
    auto c_d = static_cast<float*>(nullptr);
    CHECK_ERR(cudaMalloc((void**)&a_d, N * sizeof(float)));
    CHECK_ERR(cudaMalloc((void**)&b_d, N * sizeof(float)));
    CHECK_ERR(cudaMalloc((void**)&c_d, N * sizeof(float)));
    CHECK_ERR(cudaMemcpy(a_d, a_h.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_ERR(cudaMemcpy(b_d, b_h.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    auto [device_result_naive, naive_elapsed] = device::add_device(device::add, util::ceil_div(N, BLOCK_SIZE), a_d, b_d, c_d);
    constexpr auto eps = 1e-6f;
    auto [naive_ok, naive_max_idx] = util::cmp_vec(device_result_naive, host_result, eps);
    auto naive_max_diff = std::fabs(host_result[naive_max_idx] - device_result_naive[naive_max_idx]);
    fmt::println("[device::naive] elapsed_time={} ms, result=[{}, {}], diff={:.10e}", naive_elapsed,
                host_result[naive_max_idx], device_result_naive[naive_max_idx], naive_max_diff);
    if (!naive_ok) return -1;

    auto float4_grid_size = util::ceil_div(util::ceil_div(N, 4), BLOCK_SIZE);
    auto [device_result_float4, float4_elapsed] =
        device::add_device(device::add_float4, float4_grid_size, a_d, b_d, c_d);
    auto [float4_ok, float4_max_idx] = util::cmp_vec(device_result_float4, host_result, eps);
    auto float4_max_diff = std::fabs(host_result[float4_max_idx] - device_result_float4[float4_max_idx]);
    fmt::println("[device::float4] elapsed_time={} ms, result=[{}, {}], diff={:.10e}", float4_elapsed,
                host_result[float4_max_idx], device_result_float4[float4_max_idx], float4_max_diff);
    if (!float4_ok) return -1;

    CHECK_ERR(cudaFree(a_d));
    CHECK_ERR(cudaFree(b_d));
    CHECK_ERR(cudaFree(c_d));

    return 0;
}
