#include "io/NiftiLoader.h"
#include "io/VoxelGridResampler.h"
#include "grid/Grid3D.h"
#include "segmentation/SegmentationTypes.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int32_t readHeaderSizeBytes(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    assert(ifs.is_open());
    int32_t sizeof_hdr = 0;
    ifs.read(reinterpret_cast<char*>(&sizeof_hdr), sizeof(int32_t));
    return sizeof_hdr;
}

std::string readMagicBytes(const std::string& filepath, size_t offset, size_t len) {
    std::ifstream ifs(filepath, std::ios::binary);
    assert(ifs.is_open());
    ifs.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    std::string magic(len, '\0');
    ifs.read(magic.data(), static_cast<std::streamsize>(len));
    return magic;
}

bool approxEqual(double a, double b, double tol = 1e-5) {
    return std::abs(a - b) <= tol;
}

void testRoundTripVersion(NiftiVersion version, const fs::path& tmp_dir) {
    const std::string tag = (version == NiftiVersion::Nifti1) ? "nifti1" : "nifti2";
    const int nx = 24, ny = 20, nz = 12;
    const std::array<double, 3> spacing_mm = {0.5, 0.5, 1.0};

    // Non-trivial affine with sub-millimeter translation precision
    const std::array<double, 16> affine_mm = {
        -0.5,  0.0,  0.0,  12.345678,
         0.0,  0.5,  0.0, -24.567891,
         0.0,  0.0,  1.0,   6.789012,
         0.0,  0.0,  0.0,   1.0
    };

    const size_t n = static_cast<size_t>(nx) * ny * nz;
    std::vector<uint8_t> labels(n, static_cast<uint8_t>(TissueClass::Air));
    Grid3D scalar(nx, ny, nz, 0.001, 0.0);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const size_t idx = static_cast<size_t>(i) + nx * (static_cast<size_t>(j) + ny * k);
                if (i >= 4 && i < 20 && j >= 4 && j < 16 && k >= 2 && k < 10) {
                    labels[idx] = static_cast<uint8_t>(
                        (i >= 8 && i < 16) ? TissueClass::GrayMatter : TissueClass::CSF
                    );
                }
                scalar(i, j, k) = 0.25 * i - 0.1 * j + 1.5 * k;
            }
        }
    }

    // 1. Save & reload discrete tissue labels (.nii)
    const std::string lbl_path = (tmp_dir / (tag + "_labels.nii")).string();
    NiftiLoader::saveLabels(lbl_path, labels, nx, ny, nz, spacing_mm, affine_mm, version);

    if (version == NiftiVersion::Nifti1) {
        assert(readHeaderSizeBytes(lbl_path) == 348);
        assert(readMagicBytes(lbl_path, 344, 3) == "n+1");
    } else {
        assert(readHeaderSizeBytes(lbl_path) == 540);
        assert(readMagicBytes(lbl_path, 4, 3) == "n+2");
    }

    SegmentationResult loaded_lbl = NiftiLoader::loadLabels(lbl_path);
    assert(loaded_lbl.nx == nx && loaded_lbl.ny == ny && loaded_lbl.nz == nz);
    assert(approxEqual(loaded_lbl.spacing_mm[0], spacing_mm[0]));
    assert(approxEqual(loaded_lbl.spacing_mm[1], spacing_mm[1]));
    assert(approxEqual(loaded_lbl.spacing_mm[2], spacing_mm[2]));
    assert(loaded_lbl.labels == labels);

    // 2. Save & reload scalar field (.nii)
    const std::string scl_path = (tmp_dir / (tag + "_scalar.nii")).string();
    std::array<double, 16> iso_affine = affine_mm;
    iso_affine[0] = -1.0; iso_affine[5] = 1.0; iso_affine[10] = 1.0;

    NiftiLoader::saveScalarGrid(scl_path, scalar, iso_affine, version);
    MriVolume loaded_scl = NiftiLoader::loadMri(scl_path);
    assert(loaded_scl.nx == nx && loaded_scl.ny == ny && loaded_scl.nz == nz);

    const double aff_tol = (version == NiftiVersion::Nifti2) ? 1e-12 : 1e-5;
    for (size_t m = 0; m < 16; ++m) {
        assert(approxEqual(loaded_scl.affine[m], iso_affine[m], aff_tol));
    }
    for (size_t idx = 0; idx < n; ++idx) {
        assert(approxEqual(loaded_scl.intensities[idx], static_cast<float>(scalar.raw()[idx]), 1e-5));
    }

    fs::remove(lbl_path);
    fs::remove(scl_path);

#ifdef TMS_USE_ZLIB
    // 3. Compressed .nii.gz round-trip
    const std::string gz_path = (tmp_dir / (tag + "_labels.nii.gz")).string();
    NiftiLoader::saveLabels(gz_path, labels, nx, ny, nz, spacing_mm, affine_mm, version);
    SegmentationResult loaded_gz = NiftiLoader::loadLabels(gz_path);
    assert(loaded_gz.labels == labels);
    fs::remove(gz_path);
#endif

    // 4. Anisotropic-to-isotropic cubical resampling & multigrid padding
    constexpr int num_levels = 4; // divisor = 2^(4-1) = 8
    SolverGridVolume resampled = VoxelGridResampler::resampleLabelsForSolver(
        loaded_lbl, affine_mm, /*target_h_mm=*/1.0, num_levels, /*min_air_pad_voxels=*/2
    );

    const int divisor = 1 << (num_levels - 1);
    assert(resampled.nx % divisor == 0);
    assert(resampled.ny % divisor == 0);
    assert(resampled.nz % divisor == 0);
    assert(approxEqual(resampled.h_m, 0.001, 1e-9));

    // Exterior corner must remain Air, interior center must contain GrayMatter
    assert(resampled.labels[0] == static_cast<uint8_t>(TissueClass::Air));
    const size_t center_idx =
        static_cast<size_t>(resampled.nx / 2) +
        resampled.nx * (static_cast<size_t>(resampled.ny / 2) + resampled.ny * (resampled.nz / 2));
    assert(resampled.labels[center_idx] == static_cast<uint8_t>(TissueClass::GrayMatter));

    std::cout << "[PASS] " << tag << " round-trip & resampler ("
              << resampled.nx << "x" << resampled.ny << "x" << resampled.nz << " padded grid)\n";
}

} // namespace

int main() {
    const fs::path tmp_dir = fs::temp_directory_path();

    // Default output format: NIfTI-2 (540-byte header, "n+2")
    testRoundTripVersion(NiftiVersion::Nifti2, tmp_dir);

    // Explicit fallback: NIfTI-1 (348-byte header, "n+1")
    testRoundTripVersion(NiftiVersion::Nifti1, tmp_dir);

    return 0;
}