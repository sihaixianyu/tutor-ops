#pragma once

#include <cmath>
#include <tuple>
#include <vector>

namespace util {
template <typename T>
auto cmp_vec(const std::vector<T>& a, const std::vector<T>& b, float eps) -> std::tuple<bool, std::vector<T>> {
    if (a.size() != b.size()) {
        return {false, {}};
    }

    auto diffs = std::vector<T>(a.size());
    for (auto i = 0; i < a.size(); i++) {
        diffs[i] = std::fabs(a[i] - b[i]);
    }

    auto is_same = true;
    for (auto diff : diffs) {
        if (diff > eps) {
            is_same = false;
            break;
        }
    }

    return {is_same, diffs};
}
}  // namespace util