#include "io/VoxelGridResampler.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

int padUpToMultiple(int value, int divisor) {
    const int rem = value % divisor;
    return (rem == 0) ? value : (value + (divisor - rem));
}

inline float sampleTrilinear(const MriVolume& vol, double x, double y, double z) {
    if (x < 0.0 || y < 0.0 || z < 0.0 ||
        x > static_cast<double>(vol.nx - 1) ||
        y > static_cast<double>(vol.ny - 1) ||
        z > static_cast<double>(vol.nz - 1)) {
        return 0.0f;
    }

    const int i0 = static_cast<int>(std::floor(x));
    const int j0 = static_cast<int>(std::floor(y));
    const int k0 = static_cast<int>(std::floor(z));
    const int i1 = std::min(i0 + 1, vol.nx - 1);
    const int j1 = std::min(j0 + 1, vol.ny - 1);
    const int k1 = std::min(k0 + 1, vol.nz - 1);

    const double tx = x - i0;
    const double ty = y - j0;
    const double tz = z - k0;

    auto at = [&](int i, int j, int k) -> double {
        return vol.intensities[
            static_cast<size_t>(i) + vol.nx * (static_cast<size_t>(j) + vol.ny * k)
        ];
    };

    const double c00 = at(i0, j0, k0) * (1.0 - tx) + at(i1, j0, k0) * tx;
    const double c10 = at(i0, j1, k0) * (1.0 - tx) + at(i1, j1, k0) * tx;
    const double c01 = at(i0, j0, k1) * (1.0 - tx) + at(i1, j0, k1) * tx;
    const double c11 = at(i0, j1, k1) * (1.0 - tx) + at(i1, j1, k1) * tx;

    const double c0 = c00 * (1.0 - ty) + c10 * ty;
    const double c1 = c01 * (1.0 - ty) + c11 * ty;

    return static_cast<float>(c0 * (1.0 - tz) + c1 * tz);
}

} // namespace

MriVolume VoxelGridResampler::resampleMriTrilinear(
    const MriVolume& src,
    double target_h_mm
) {
    if (target_h_mm <= 0.0) {
        throw std::invalid_argument("target_h_mm must be > 0");
    }

    const double sx = src.spacing_mm[0] / target_h_mm;
    const double sy = src.spacing_mm[1] / target_h_mm;
    const double sz = src.spacing_mm[2] / target_h_mm;

    MriVolume dst;
    dst.nx = std::max(1, static_cast<int>(std::ceil(src.nx * sx)));
    dst.ny = std::max(1, static_cast<int>(std::ceil(src.ny * sy)));
    dst.nz = std::max(1, static_cast<int>(std::ceil(src.nz * sz)));
    dst.spacing_mm = {target_h_mm, target_h_mm, target_h_mm};
    dst.intensities.resize(static_cast<size_t>(dst.nx) * dst.ny * dst.nz, 0.0f);

    // Update affine scaling while preserving voxel (0,0,0) center alignment
    dst.affine = src.affine;
    const double inv_sx = target_h_mm / src.spacing_mm[0];
    const double inv_sy = target_h_mm / src.spacing_mm[1];
    const double inv_sz = target_h_mm / src.spacing_mm[2];
    for (int r = 0; r < 3; ++r) {
        dst.affine[r * 4 + 0] *= inv_sx;
        dst.affine[r * 4 + 1] *= inv_sy;
        dst.affine[r * 4 + 2] *= inv_sz;
    }

    for (int k = 0; k < dst.nz; ++k) {
        const double src_z = ((k + 0.5) * target_h_mm) / src.spacing_mm[2] - 0.5;
        for (int j = 0; j < dst.ny; ++j) {
            const double src_y = ((j + 0.5) * target_h_mm) / src.spacing_mm[1] - 0.5;
            for (int i = 0; i < dst.nx; ++i) {
                const double src_x = ((i + 0.5) * target_h_mm) / src.spacing_mm[0] - 0.5;
                dst.intensities[
                    static_cast<size_t>(i) + dst.nx * (static_cast<size_t>(j) + dst.ny * k)
                ] = sampleTrilinear(src, src_x, src_y, src_z);
            }
        }
    }

    return dst;
}

SolverGridVolume VoxelGridResampler::resampleLabelsForSolver(
    const SegmentationResult& seg,
    const std::array<double, 16>& src_affine_mm,
    double target_h_mm,
    int num_levels,
    int min_air_pad_voxels
) {
    if (target_h_mm <= 0.0 || num_levels < 1) {
        throw std::invalid_argument("Invalid target_h_mm or num_levels");
    }

    const int unpadded_nx = std::max(1, static_cast<int>(
        std::ceil((seg.nx * seg.spacing_mm[0]) / target_h_mm)
    ));
    const int unpadded_ny = std::max(1, static_cast<int>(
        std::ceil((seg.ny * seg.spacing_mm[1]) / target_h_mm)
    ));
    const int unpadded_nz = std::max(1, static_cast<int>(
        std::ceil((seg.nz * seg.spacing_mm[2]) / target_h_mm)
    ));

    const int divisor = 1 << (num_levels - 1);
    const int pad_min = std::max(1, min_air_pad_voxels);

    const int nx = padUpToMultiple(unpadded_nx + 2 * pad_min, divisor);
    const int ny = padUpToMultiple(unpadded_ny + 2 * pad_min, divisor);
    const int nz = padUpToMultiple(unpadded_nz + 2 * pad_min, divisor);

    // Center the anatomical volume inside the padded multigrid box
    const int off_x = (nx - unpadded_nx) / 2;
    const int off_y = (ny - unpadded_ny) / 2;
    const int off_z = (nz - unpadded_nz) / 2;

    SolverGridVolume out;
    out.nx = nx;
    out.ny = ny;
    out.nz = nz;
    out.h_m = target_h_mm * 1e-3;
    out.labels.assign(static_cast<size_t>(nx) * ny * nz, static_cast<uint8_t>(TissueClass::Air));

    // Sub-voxel offsets (0.25 and 0.75) for 2x2x2 majority voting inside each target voxel
    constexpr double sub_offsets[2] = {0.25, 0.75};
    constexpr size_t kNumClasses = static_cast<size_t>(TissueClass::Count);

    for (int k = 0; k < unpadded_nz; ++k) {
        for (int j = 0; j < unpadded_ny; ++j) {
            for (int i = 0; i < unpadded_nx; ++i) {
                std::array<int, kNumClasses> votes{};

                for (double dz : sub_offsets) {
                    const int sz = static_cast<int>(
                        std::floor(((k + dz) * target_h_mm) / seg.spacing_mm[2])
                    );
                    if (sz < 0 || sz >= seg.nz) {
                        votes[0]++;
                        continue;
                    }
                    for (double dy : sub_offsets) {
                        const int sy = static_cast<int>(
                            std::floor(((j + dy) * target_h_mm) / seg.spacing_mm[1])
                        );
                        if (sy < 0 || sy >= seg.ny) {
                            votes[0]++;
                            continue;
                        }
                        for (double dx : sub_offsets) {
                            const int sx = static_cast<int>(
                                std::floor(((i + dx) * target_h_mm) / seg.spacing_mm[0])
                            );
                            if (sx < 0 || sx >= seg.nx) {
                                votes[0]++;
                                continue;
                            }
                            const uint8_t cls = seg.labels[
                                static_cast<size_t>(sx) +
                                seg.nx * (static_cast<size_t>(sy) + seg.ny * sz)
                            ];
                            if (cls < kNumClasses) {
                                votes[cls]++;
                            }
                        }
                    }
                }

                uint8_t best_cls = 0;
                int best_votes = -1;
                for (size_t c = 0; c < kNumClasses; ++c) {
                    if (votes[c] > best_votes) {
                        best_votes = votes[c];
                        best_cls = static_cast<uint8_t>(c);
                    }
                }

                const size_t dst_idx =
                    static_cast<size_t>(i + off_x) +
                    nx * (static_cast<size_t>(j + off_y) + ny * (k + off_z));
                out.labels[dst_idx] = best_cls;
            }
        }
    }

    // Compute updated NIfTI affine (center of voxel (0,0,0) in padded grid)
    out.affine_mm = src_affine_mm;
    const double scale_x = target_h_mm / seg.spacing_mm[0];
    const double scale_y = target_h_mm / seg.spacing_mm[1];
    const double scale_z = target_h_mm / seg.spacing_mm[2];

    // Shift in source voxel coordinates from source (0,0,0) center to padded (0,0,0) center
    const double shift_src_i = (0.5 - off_x) * scale_x - 0.5;
    const double shift_src_j = (0.5 - off_y) * scale_y - 0.5;
    const double shift_src_k = (0.5 - off_z) * scale_z - 0.5;

    for (int r = 0; r < 3; ++r) {
        const double col0 = src_affine_mm[r * 4 + 0];
        const double col1 = src_affine_mm[r * 4 + 1];
        const double col2 = src_affine_mm[r * 4 + 2];
        const double t0   = src_affine_mm[r * 4 + 3];

        out.affine_mm[r * 4 + 0] = col0 * scale_x;
        out.affine_mm[r * 4 + 1] = col1 * scale_y;
        out.affine_mm[r * 4 + 2] = col2 * scale_z;
        out.affine_mm[r * 4 + 3] = t0 + col0 * shift_src_i + col1 * shift_src_j + col2 * shift_src_k;

        // World-space corner (0,0,0) of voxel (0,0,0) in meters
        const double corner_mm =
            out.affine_mm[r * 4 + 3] -
            0.5 * (out.affine_mm[r * 4 + 0] + out.affine_mm[r * 4 + 1] + out.affine_mm[r * 4 + 2]);
        out.origin_m[r] = corner_mm * 1e-3;
    }

    return out;
}