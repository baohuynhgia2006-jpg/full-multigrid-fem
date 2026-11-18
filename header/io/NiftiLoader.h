#pragma once
#include "../segmentation/SegmentationTypes.h"
#include "../grid/Grid3D.h"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

enum class NiftiVersion : int {
    Nifti2 = 2,
    Nifti1 = 1
};

class NiftiLoader {
public:
    // Loads a 3D T1-weighted NIfTI-2 or NIfTI-1 volume (.nii, .nii.gz, or .hdr/.img)
    static MriVolume loadMri(const std::string& filepath);

    // Loads a 3D integer label NIfTI-2 or NIfTI-1 volume into SegmentationResult
    static SegmentationResult loadLabels(const std::string& filepath);

    // Saves a 3D uint8 label grid (defaults to NIfTI-2, falls back to NIfTI-1 if requested)
    static void saveLabels(
        const std::string& filepath,
        const std::vector<uint8_t>& labels,
        int nx, int ny, int nz,
        const std::array<double, 3>& spacing_mm,
        const std::array<double, 16>& affine_mm,
        NiftiVersion version = NiftiVersion::Nifti2
    );

    // Saves a 3D double scalar field (defaults to NIfTI-2, falls back to NIfTI-1 if requested)
    static void saveScalarGrid(
        const std::string& filepath,
        const Grid3D& grid,
        const std::array<double, 16>& affine_mm,
        NiftiVersion version = NiftiVersion::Nifti2
    );
};