#pragma once

namespace util {
template <typename T>
constexpr auto ceil_div(T a, T b) -> T {
    return (a + b - 1) / b;
}
}  // namespace util