#pragma once

#include <cuda_runtime.h>

#include <fmt/core.h>

#define CHECK_ERR(err) check_cuda_err(err, __FILE__, __LINE__)

inline void check_cuda_err(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        fmt::println("[CUDA ERROR] at file {}(line {}):\n{}\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
};