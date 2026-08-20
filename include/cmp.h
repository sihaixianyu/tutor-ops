#pragma once

#include <cmath>
#include <tuple>
#include <vector>

namespace util {
template <typename T>
auto cmp_scalar(T a, T b, T eps) -> bool {
    auto scale = std::fmax(std::fabs(a), std::fabs(b));
    return std::fabs(a - b) <= eps * scale;
}

template <typename T>
auto cmp_vec(const std::vector<T>& a, const std::vector<T>& b, float eps) -> std::tuple<bool, std::size_t> {
    if (a.size() != b.size()) {
        return {false, 0};
    }

    auto is_same = true;
    auto max_idx = std::size_t{0};
    auto max_diff = T{0};
    for (auto i = 0; i < a.size(); i++) {
        auto diff = std::fabs(a[i] - b[i]);
        if (diff > eps) {
            is_same = false;
        }
        if (diff >= max_diff) {
            max_diff = diff;
            max_idx = static_cast<std::size_t>(i);
        }
    }

    return {is_same, max_idx};
}
}  // namespace util