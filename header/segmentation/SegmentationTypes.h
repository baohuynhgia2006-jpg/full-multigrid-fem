// header/segmentation/SegmentationTypes.h
#pragma once
#include <cstdint>
#include <vector>
#include <array>

enum class TissueClass : uint8_t {
    Air = 0,
    Scalp = 1,
    Skull = 2,
    CSF = 3,
    GrayMatter = 4,
    WhiteMatter = 5,
    Count = 6
};

struct MriVolume {
    int nx = 0, ny = 0, nz = 0;
    std::array<double, 3> spacing_mm{1.0, 1.0, 1.0}; // Native (dx, dy, dz)
    std::array<double, 16> affine{};                // NIfTI sform/qform matrix
    std::vector<float> intensities;                 // Flat (i + nx * (j + ny * k))
};

struct SegmentationResult {
    int nx = 0, ny = 0, nz = 0;
    std::array<double, 3> spacing_mm{1.0, 1.0, 1.0};
    std::vector<uint8_t> labels;                    // Hard argmax TissueClass per voxel
    std::vector<float> posteriors;                  // Optional: [Class][Voxel] probabilities
};