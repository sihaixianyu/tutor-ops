#include "add.cuh"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

void assert_cuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    std::abort();
  }
}

} // namespace

int main() {
  const cudaError_t runtime_status = cudaFree(nullptr);
  if (runtime_status == cudaErrorInsufficientDriver ||
      runtime_status == cudaErrorNoDevice) {
    std::cerr
        << "Skipping test_add: CUDA runtime is unavailable on this machine\n";
    return 77;
  }
  assert_cuda(runtime_status, "cudaFree");

  int device_count = 0;
  const cudaError_t device_status = cudaGetDeviceCount(&device_count);
  if (device_status == cudaErrorInsufficientDriver ||
      device_status == cudaErrorNoDevice || device_count == 0) {
    std::cerr
        << "Skipping test_add: CUDA runtime is unavailable on this machine\n";
    return 77;
  }
  assert_cuda(device_status, "cudaGetDeviceCount");

  constexpr std::size_t count = 8;
  const std::vector<float> lhs{1.0f,  2.5f, 3.0f, 4.0f,
                               -5.0f, 8.0f, 0.5f, 10.0f};
  const std::vector<float> rhs{2.0f, 1.5f,  -3.0f, 6.0f,
                               5.0f, -8.0f, 9.5f,  -2.0f};
  const std::vector<float> expected{3.0f, 4.0f, 0.0f,  10.0f,
                                    0.0f, 0.0f, 10.0f, 8.0f};

  float *device_lhs = nullptr;
  float *device_rhs = nullptr;
  float *device_out = nullptr;

  assert_cuda(cudaMalloc(&device_lhs, count * sizeof(float)), "cudaMalloc lhs");
  assert_cuda(cudaMalloc(&device_rhs, count * sizeof(float)), "cudaMalloc rhs");
  assert_cuda(cudaMalloc(&device_out, count * sizeof(float)), "cudaMalloc out");

  assert_cuda(cudaMemcpy(device_lhs, lhs.data(), count * sizeof(float),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy lhs");
  assert_cuda(cudaMemcpy(device_rhs, rhs.data(), count * sizeof(float),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy rhs");

  add_vectors(device_lhs, device_rhs, device_out, count);

  std::vector<float> result(count, 0.0f);
  assert_cuda(cudaMemcpy(result.data(), device_out, count * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy out");

  for (std::size_t i = 0; i < count; ++i) {
    assert(std::fabs(result[i] - expected[i]) < 1e-5f);
  }

  assert_cuda(cudaFree(device_lhs), "cudaFree lhs");
  assert_cuda(cudaFree(device_rhs), "cudaFree rhs");
  assert_cuda(cudaFree(device_out), "cudaFree out");

  std::cout << "test_add passed\n";
  return 0;
}
