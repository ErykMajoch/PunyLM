#pragma once
#include <cstdint>
#include <numeric>
#include <vector>

#include "punylm/core/device.h"

namespace punylm {
    using Shape = std::vector<int64_t>;

    inline int64_t numel(const Shape &s) {
        return std::accumulate(s.begin(), s.end(), int64_t{1}, std::multiplies<>());
    }

    inline int64_t last_dim(const Shape &s) { return s.empty() ? 1 : s.back(); }
    inline int64_t rows_of(const Shape &s) { return s.empty() ? 1 : numel(s) / s.back(); }
} // namespace punylm
