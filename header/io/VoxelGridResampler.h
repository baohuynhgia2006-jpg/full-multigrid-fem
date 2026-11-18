#pragma once
#include "../grid/Grid3D.h"
#include "../segmentation/SegmentationTypes.h"
#include <array>
#include <cstdint>
#include <vector>

struct SolverGridVolume {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    double h_m = 0.001;                      // Cubical voxel side length in meters
    std::array<double, 3> origin_m{};        // Corner (0,0,0) of padded grid in meters
    std::array<double, 16> affine_mm{};      // Updated NIfTI affine for saving outputs
    std::vector<uint8_t> labels;             // Padded discrete tissue labels
};

class VoxelGridResampler {
public:
    // Resamples continuous MRI intensities onto an isotropic grid (trilinear)
    static MriVolume resampleMriTrilinear(
        const MriVolume& src,
        double target_h_mm
    );

    // Resamples discrete tissue labels onto a uniform cubical FEM grid via sub-voxel
    // majority voting and pads with air voxels so (nx, ny, nz) are divisible by 2^(L-1).
    static SolverGridVolume resampleLabelsForSolver(
        const SegmentationResult& seg,
        const std::array<double, 16>& src_affine_mm,
        double target_h_mm,
        int num_levels,
        int min_air_pad_voxels = 2
    );

    // Removes any disconnected non-air islands after label resampling
    static void keepLargestHeadComponent(SolverGridVolume& vol);

    // Trilinearly interpolates nodal potential phi from a coarse grid onto a fine grid
    static void interpolateNodalField(
        const Grid3D& src_phi,
        const std::array<double, 3>& src_origin_m,
        Grid3D& dst_phi,
        const std::array<double, 3>& dst_origin_m
    );
};