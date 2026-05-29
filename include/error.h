#pragma once

#include <cuda_runtime.h>

#include <fmt/core.h>

#define CHECK_ERR(err) util::check_err(err, __FILE__, __LINE__)

namespace util {
inline auto check_err(cudaError_t error, const char* file, int line) -> void {
    if (error != cudaSuccess) {
        fmt::println("[ERROR] <{}:{}> Failed to exec cuda op due to '{}'.", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
}
}  // namespace util
