#pragma once
#include "../grid/Grid3D.h"
#include "../segmentation/SegmentationTypes.h"
#include <array>
#include <cstdint>
#include <vector>

class ConductivityBuilder {
public:
    // Default isotropic tissue conductivities at 3 kHz (in S/m)
    static constexpr std::array<double, static_cast<size_t>(TissueClass::Count)>
        kDefaultConductivities3kHz = {
            0.0,    // Air
            0.43,   // Scalp
            0.015,  // Skull
            1.79,   // CSF
            0.33,   // GrayMatter
            0.14    // WhiteMatter
        };

    explicit ConductivityBuilder(
        std::array<double, static_cast<size_t>(TissueClass::Count)> table = kDefaultConductivities3kHz
    );

    // Maps hard discrete labels onto voxel-centered conductivity grid sigma^h
    void assignFromLabels(const std::vector<uint8_t>& labels, Grid3D& sigma_h) const;

    // Maps soft posterior probabilities [Class][Voxel] onto sigma^h (partial-volume)
    void assignFromPosteriors(const std::vector<float>& posteriors, Grid3D& sigma_h) const;

    // Applies the resolution-dependent staircase smoothing filter (h in meters)
    static void applyStaircaseSmoothing(Grid3D& sigma_h);

private:
    std::array<double, static_cast<size_t>(TissueClass::Count)> table_;
};