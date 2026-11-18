#pragma once
#include "../segmentation/SegmentationTypes.h"
#include "../grid/Grid3D.h"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

enum class NiftiVersion : int {
    Nifti1 = 1,
    Nifti2 = 2
};

class NiftiLoader {
public:
    // Auto-detects and loads both NIfTI-2 (540B) and NIfTI-1 (348B) volumes (.nii, .nii.gz, .hdr/.img)
    static MriVolume loadMri(const std::string& filepath);

    // Auto-detects and loads both NIfTI-2 and NIfTI-1 integer label volumes
    static SegmentationResult loadLabels(const std::string& filepath);

    // Defaults to NIfTI-1; auto-promotes to NIfTI-2 if dimensions exceed 32,767 or NiftiVersion::Nifti2 is passed
    static void saveLabels(
        const std::string& filepath,
        const std::vector<uint8_t>& labels,
        int nx, int ny, int nz,
        const std::array<double, 3>& spacing_mm,
        const std::array<double, 16>& affine_mm,
        NiftiVersion version = NiftiVersion::Nifti1
    );

    // Defaults to NIfTI-1; auto-promotes to NIfTI-2 if dimensions exceed 32,767 or NiftiVersion::Nifti2 is passed
    static void saveScalarGrid(
        const std::string& filepath,
        const Grid3D& grid,
        const std::array<double, 16>& affine_mm,
        NiftiVersion version = NiftiVersion::Nifti1
    );
};