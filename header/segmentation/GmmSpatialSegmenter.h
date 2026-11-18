#pragma once
#include "ITissueSegmenter.h"
#include <string>

struct GmmSegmenterConfig {
    int max_em_iterations = 25;      // EM convergence iterations
    double mrf_beta = 0.45;          // 3D 6-neighbor Potts smoothness weight
    double min_sigma = 0.015;        // Floor on normalized tissue standard deviation
    double convergence_tol = 1e-4;   // Max mean parameter shift stopping criterion
    bool estimate_bias_field = true; // Multiplicative B1 inhomogeneity correction
    int bias_smooth_radius = 12;     // Separable smoothing radius (in voxels) for B1 field
};

class GmmSpatialSegmenter : public ITissueSegmenter {
public:
    explicit GmmSpatialSegmenter(
        std::string atlas_path = "",
        GmmSegmenterConfig config = {}
    );

    SegmentationResult segment(const MriVolume& mri) const override;
    std::string name() const override { return "GmmSpatialSegmenter (EM-GMM + 3D MRF)"; }

private:
    std::string atlas_path_;
    GmmSegmenterConfig cfg_;
};