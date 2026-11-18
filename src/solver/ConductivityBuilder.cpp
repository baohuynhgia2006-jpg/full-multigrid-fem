#include "solver/ConductivityBuilder.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>

ConductivityBuilder::ConductivityBuilder(
    std::array<double, static_cast<size_t>(TissueClass::Count)> table
) : table_(table) {}

void ConductivityBuilder::assignFromLabels(
    const std::vector<uint8_t>& labels,
    Grid3D& sigma_h
) const {
    if (labels.size() != sigma_h.size()) {
        throw std::invalid_argument("Label buffer size does not match sigma_h grid size");
    }

    double* dst = sigma_h.raw();
    const size_t n = sigma_h.size();
    for (size_t idx = 0; idx < n; ++idx) {
        const uint8_t cls = labels[idx];
        dst[idx] = (cls < table_.size()) ? table_[cls] : 0.0;
    }
}

void ConductivityBuilder::assignFromPosteriors(
    const std::vector<float>& posteriors,
    Grid3D& sigma_h
) const {
    const size_t n = sigma_h.size();
    const size_t num_classes = table_.size();
    if (posteriors.size() != n * num_classes) {
        throw std::invalid_argument("Posterior buffer size mismatch");
    }

    double* dst = sigma_h.raw();
    for (size_t idx = 0; idx < n; ++idx) {
        double sig = 0.0;
        for (size_t c = 0; c < num_classes; ++c) {
            sig += static_cast<double>(posteriors[c * n + idx]) * table_[c];
        }
        dst[idx] = sig;
    }
}

void ConductivityBuilder::applyStaircaseSmoothing(Grid3D& sigma_h) {
    const double h_mm = sigma_h.h() * 1000.0;

    // Select squared integer radius matching Laakso & Hirata (2012) Section 3.2:
    // 2.0 mm -> 7 voxels (r^2 <= 1)
    // 1.0 mm -> 27 voxels (r^2 <= 3, full 3x3x3 cube)
    // 0.5 mm -> 93 voxels (r^2 <= 8, 5x5x5 cube with rounded corners)
    // 0.25 mm -> 895 voxels (r^2 <= 35, sphere)
    int max_r = 1;
    int max_d2 = 1;
    if (h_mm > 1.5) {
        max_r = 1; max_d2 = 1;   // 7 voxels
    } else if (h_mm > 0.75) {
        max_r = 1; max_d2 = 3;   // 27 voxels
    } else if (h_mm > 0.35) {
        max_r = 2; max_d2 = 8;   // 93 voxels
    } else {
        max_r = 6; max_d2 = 35;  // 895 voxels
    }

    std::vector<std::tuple<int, int, int>> offsets;
    for (int dz = -max_r; dz <= max_r; ++dz) {
        for (int dy = -max_r; dy <= max_r; ++dy) {
            for (int dx = -max_r; dx <= max_r; ++dx) {
                if (dx * dx + dy * dy + dz * dz <= max_d2) {
                    offsets.emplace_back(dx, dy, dz);
                }
            }
        }
    }

    const Grid3D src = sigma_h;
    const int nx = src.nx();
    const int ny = src.ny();
    const int nz = src.nz();
    const double inv_count = 1.0 / static_cast<double>(offsets.size());

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                double sum = 0.0;
                for (const auto& [dx, dy, dz] : offsets) {
                    const int sx = std::clamp(i + dx, 0, nx - 1);
                    const int sy = std::clamp(j + dy, 0, ny - 1);
                    const int sz = std::clamp(k + dz, 0, nz - 1);
                    sum += src(sx, sy, sz);
                }
                sigma_h(i, j, k) = sum * inv_count;
            }
        }
    }
}