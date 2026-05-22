#include <stdlib.h>

#include <cuda_runtime.h>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include "util/err.h"
#include "util/common.h"

__global__ void vec_add_float4(float* a, float* b, float* c, int N) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx >= N)
        return;

    float4 tmp_a = FLOAT4(a[idx]);
    float4 tmp_b = FLOAT4(b[idx]);
    float4 tmp_c;
    tmp_c.x = tmp_a.x + tmp_b.x;
    tmp_c.y = tmp_a.y + tmp_b.y;
    tmp_c.z = tmp_a.z + tmp_b.z;
    tmp_c.w = tmp_a.w + tmp_b.w;
    FLOAT4(c[idx]) = tmp_c;
}

int main() {
    constexpr auto N = 8;
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

    auto block_size = 1024;
    auto grid_size = CEIL(CEIL(N, 4), 1024);
    vec_add_float4<<<grid_size, block_size>>>(a_d, b_d, c_d, N);

    CHECK_ERR(cudaMemcpy(c_h, c_d, N * sizeof(float), cudaMemcpyDeviceToHost));
    fmt::println("{}", "a_h:");
    fmt::print("{}\n", fmt::join(a_h, a_h + N, " "));
    fmt::println("{}", "b_h:");
    fmt::print("{}\n", fmt::join(b_h, b_h + N, " "));
    fmt::println("{}", "c_h:");
    fmt::print("{}\n", fmt::join(c_h, c_h + N, " "));

    return 0;
}