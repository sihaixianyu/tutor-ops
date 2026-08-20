#include <cmath>
#include <cstdlib>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

#include <fmt/core.h>

#include "calc.h"
#include "cmp.h"
#include "error.h"
#include "timer.h"

constexpr auto ROWS = 4096;
constexpr auto COLS = 8192;
constexpr auto N = ROWS * COLS;
constexpr auto REPEAT_TIME = 8;

namespace host {
auto transpose_host(const std::vector<float>& input, int rows, int cols) -> std::tuple<std::vector<float>, float> {
    auto result = std::vector<float>(input.size());
    auto elapsed = util::time_cpu(REPEAT_TIME, [&]() {
        for (auto row = 0; row < rows; ++row) {
            for (auto col = 0; col < cols; ++col) {
                result[col * rows + row] = input[row * cols + col];
            }
        }
    });
    return {std::move(result), elapsed};
}
}  // namespace host

namespace device {
constexpr auto TILE = 32;  // 32x32 = 1024 threads = one full block.

template <typename Kernel>
auto transpose_device(Kernel kernel, dim3 grid, dim3 block, const float* input, float* output)
    -> std::tuple<std::vector<float>, float> {
    auto elapsed = util::time_cuda(REPEAT_TIME, [&]() {
        kernel<<<grid, block>>>(input, output, ROWS, COLS);
        CHECK_ERR(cudaGetLastError());
        CHECK_ERR(cudaDeviceSynchronize());
    });

    auto result = std::vector<float>(N);
    CHECK_ERR(cudaMemcpy(result.data(), output, N * sizeof(float), cudaMemcpyDeviceToHost));
    return {std::move(result), elapsed};
}

// Baseline: each thread moves one element. Write output[col*m+row] is NOT coalesced.
__global__ void transpose_naive(const float* input, float* output, const uint m, const uint n) {
    const auto row = blockDim.y * blockIdx.y + threadIdx.y;
    const auto col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < m && col < n) {
        output[col * m + row] = input[row * n + col];  // map input[row][col] -> output[col][row]
    }
}

// Coalesce both reads and writes via shared memory. No padding (no bank opt).
__global__ void transpose_tile(const float* input, float* output, const uint m, const uint n) {
    __shared__ float tile_data[TILE][TILE + 1];

    // Load tile to shared (coalesced read), sync, then write transposed (coalesced write).
    const auto row = TILE * blockIdx.y + threadIdx.y;
    const auto col = TILE * blockIdx.x + threadIdx.x;
    if (row < m && col < n) {
        tile_data[threadIdx.y][threadIdx.x] = input[row * n + col];
    }
    __syncthreads();

    // Swap local indices -> consecutive threadIdx.x walk consecutive global addresses.
    const auto x = TILE * blockIdx.x + threadIdx.y;  // output row
    const auto y = TILE * blockIdx.y + threadIdx.x;  // output col
    if (x < n && y < m) {
        output[x * m + y] = tile_data[threadIdx.x][threadIdx.y];
    }
}
}  // namespace device

int main() {
    auto input = std::vector<float>(N);
    for (auto i = 0; i < N; i++) {
        input[i] = static_cast<float>(i);  // float(iota) stalls at 2^24; cast each i separately instead
    }

    float* input_d = nullptr;
    float* output_d = nullptr;
    CHECK_ERR(cudaMalloc((void**)&input_d, N * sizeof(float)));
    CHECK_ERR(cudaMalloc((void**)&output_d, N * sizeof(float)));
    CHECK_ERR(cudaMemcpy(input_d, input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    auto [host_result, cpu_elapsed] = host::transpose_host(input, ROWS, COLS);
    fmt::println("[host] elapsed_time={} ms", cpu_elapsed);

    const auto thread_per_block = 32;
    const auto naive_grid_x = util::ceil_div(COLS, thread_per_block);
    const auto naive_grid_y = util::ceil_div(ROWS, thread_per_block);
    auto naive_block = dim3(thread_per_block, thread_per_block, 1);
    auto naive_grid = dim3(naive_grid_x, naive_grid_y, 1);

    auto [device_result_naive, naive_elapsed] =
        device::transpose_device(device::transpose_naive, naive_grid, naive_block, input_d, output_d);
    constexpr auto eps = 1e-6f;
    auto [naive_ok, naive_max_idx] = util::cmp_vec(device_result_naive, host_result, eps);
    auto naive_max_diff = std::fabs(host_result[naive_max_idx] - device_result_naive[naive_max_idx]);
    fmt::println("[device::naive] elapsed_time={} ms, result=[{}, {}], diff={:.10e}", naive_elapsed,
                host_result[naive_max_idx], device_result_naive[naive_max_idx], naive_max_diff);
    if (!naive_ok) return -1;

    auto tile_block = dim3(device::TILE, device::TILE, 1);
    auto tile_grid = dim3(COLS / device::TILE, ROWS / device::TILE, 1);

    auto [device_result_tile, tile_elapsed] =
        device::transpose_device(device::transpose_tile, tile_grid, tile_block, input_d, output_d);
    auto [tile_ok, tile_max_idx] = util::cmp_vec(device_result_tile, host_result, eps);
    auto tile_max_diff = std::fabs(host_result[tile_max_idx] - device_result_tile[tile_max_idx]);
    fmt::println("[device::tile] elapsed_time={} ms, result=[{}, {}], diff={:.10e}", tile_elapsed,
                host_result[tile_max_idx], device_result_tile[tile_max_idx], tile_max_diff);
    if (!tile_ok) return -1;

    CHECK_ERR(cudaFree(input_d));
    CHECK_ERR(cudaFree(output_d));

    return 0;
}
