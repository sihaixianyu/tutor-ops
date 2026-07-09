#pragma once

#include <chrono>

#include <cuda_runtime.h>

#include "error.h"

namespace util {
template <typename Func>
auto time_cpu(const int N, const Func& func) -> float {
    auto total_time = 0.0f;

    for (auto repeat = 0; repeat < N; repeat++) {
        auto start = std::chrono::steady_clock::now();
        func();
        auto stop = std::chrono::steady_clock::now();

        if (repeat > 0) {
            total_time += std::chrono::duration<float, std::milli>(stop - start).count();
        }
    }

    return N == 0 ? 0.0f : total_time;
}

template <typename Func>
auto time_cuda(const int N, const Func& func) -> float {
    float total_time = 0.0f;

    for (int repeat = 0; repeat < N; ++repeat) {
        cudaEvent_t start, stop;

        CHECK_ERR(cudaEventCreate(&start));
        CHECK_ERR(cudaEventCreate(&stop));

        CHECK_ERR(cudaEventRecord(start));

        func();

        CHECK_ERR(cudaEventRecord(stop));
        CHECK_ERR(cudaEventSynchronize(stop));

        float elapsed_time = 0.0f;

        CHECK_ERR(cudaEventElapsedTime(&elapsed_time, start, stop));

        if (repeat > 0) {
            total_time += elapsed_time;
        }

        CHECK_ERR(cudaEventDestroy(start));
        CHECK_ERR(cudaEventDestroy(stop));
    }

    return N == 0 ? 0.0f : total_time;
}
}  // namespace util
