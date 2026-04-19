#include "add.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__global__ void add_kernel(const float* lhs, const float* rhs, float* out, std::size_t count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        out[index] = lhs[index] + rhs[index];
    }
}

}  // namespace

void add_vectors(const float* lhs, const float* rhs, float* out, std::size_t count) {
    if (count == 0) {
        return;
    }

    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((count + threads_per_block - 1) / threads_per_block);

    add_kernel<<<blocks, threads_per_block>>>(lhs, rhs, out, count);
    check_cuda(cudaGetLastError(), "kernel launch failed");
    check_cuda(cudaDeviceSynchronize(), "kernel execution failed");
}
