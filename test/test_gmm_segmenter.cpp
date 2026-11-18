#include "segmentation/ITissueSegmenter.h"
#include "segmentation/SegmentationTypes.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

int main() {
    constexpr int N = 96;
    constexpr double h_mm = 2.0;
    const double center = 0.5 * (N - 1) * h_mm;

    // Radii in mm matching the 4-layer sphere benchmark + WM core
    constexpr double R_wm    = 52.0;
    constexpr double R_gm    = 65.0;
    constexpr double R_csf   = 70.0;
    constexpr double R_skull = 78.0;
    constexpr double R_scalp = 85.0;

    // Realistic relative T1-weighted signal intensities (bone & CSF both dark)
    constexpr std::array<double, 6> kT1Signal = {
        0.0,   // Air
        245.0, // Scalp
        45.0,  // Skull (dark cortical bone)
        85.0,  // CSF (dark fluid)
        210.0, // GrayMatter
        310.0  // WhiteMatter
    };

    const size_t total = static_cast<size_t>(N) * N * N;
    MriVolume mri;
    mri.nx = N;
    mri.ny = N;
    mri.nz = N;
    mri.spacing_mm = {h_mm, h_mm, h_mm};
    mri.intensities.resize(total, 0.0f);

    std::vector<uint8_t> gt_labels(total, 0);
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 10.0);

    for (int k = 0; k < N; ++k) {
        const double z = k * h_mm - center;
        for (int j = 0; j < N; ++j) {
            const double y = j * h_mm - center;
            for (int i = 0; i < N; ++i) {
                const double x = i * h_mm - center;
                const double r = std::sqrt(x * x + y * y + z * z);

                TissueClass cls = TissueClass::Air;
                if (r <= R_wm)         cls = TissueClass::WhiteMatter;
                else if (r <= R_gm)    cls = TissueClass::GrayMatter;
                else if (r <= R_csf)   cls = TissueClass::CSF;
                else if (r <= R_skull) cls = TissueClass::Skull;
                else if (r <= R_scalp) cls = TissueClass::Scalp;

                const size_t idx = static_cast<size_t>(i) + N * (static_cast<size_t>(j) + N * k);
                gt_labels[idx] = static_cast<uint8_t>(cls);

                if (cls == TissueClass::Air) {
                    mri.intensities[idx] = static_cast<float>(std::max(0.0, noise(rng) * 0.3));
                } else {
                    // Multiplicative B1 inhomogeneity gradient across head
                    const double b1 = 1.0 + 0.25 * (x / R_scalp) - 0.15 * (z / R_scalp);
                    const double val = b1 * kT1Signal[static_cast<size_t>(cls)] + noise(rng);
                    mri.intensities[idx] = static_cast<float>(std::max(1.0, val));
                }
            }
        }
    }

    auto segmenter = createSegmenter(SegmenterBackend::GmmSpatial, "");
    SegmentationResult seg = segmenter->segment(mri);

    assert(seg.nx == N && seg.ny == N && seg.nz == N);
    assert(seg.posteriors.size() == 6 * total);

    // Compute per-tissue Dice Similarity Coefficient (DSC)
    constexpr std::array<const char*, 6> kNames = {
        "Air", "Scalp", "Skull", "CSF", "GrayMatter", "WhiteMatter"
    };

    for (size_t c = 0; c < 6; ++c) {
        size_t tp = 0, fp = 0, fn = 0;
        for (size_t idx = 0; idx < total; ++idx) {
            const bool pred = (seg.labels[idx] == c);
            const bool truth = (gt_labels[idx] == c);
            if (pred && truth) tp++;
            else if (pred && !truth) fp++;
            else if (!pred && truth) fn++;
        }
        const double dsc = (2.0 * tp) / static_cast<double>(2 * tp + fp + fn);
        std::cout << "  " << kNames[c] << " Dice: " << dsc << "\n";
        assert(dsc > 0.85);
    }

    std::cout << "[PASS] " << segmenter->name() << " validated on noisy B1-corrupted T1 phantom\n";
    return 0;
}