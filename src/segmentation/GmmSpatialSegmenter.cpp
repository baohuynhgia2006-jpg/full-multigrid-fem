#include "segmentation/GmmSpatialSegmenter.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <vector>

#ifdef TMS_USE_OPENMP
#include <omp.h>
#endif

namespace {

constexpr size_t kNumClasses = static_cast<size_t>(TissueClass::Count); // 6
constexpr size_t kFirstTissue = 1; // Scalp (1) .. WhiteMatter (5)
constexpr double kTwoPi = 6.28318530717958647692;

// Computes Otsu threshold over positive MRI intensities to separate head from background
float computeOtsuThreshold(const std::vector<float>& intensities) {
    float max_val = 0.0f;
    for (float v : intensities) {
        if (v > max_val) max_val = v;
    }
    if (max_val <= 0.0f) return 0.0f;

    constexpr int kBins = 256;
    std::array<uint64_t, kBins> hist{};
    uint64_t total = 0;

    const float scale = static_cast<float>(kBins - 1) / max_val;
    for (float v : intensities) {
        if (v > 0.0f) {
            const int bin = std::clamp(static_cast<int>(v * scale), 0, kBins - 1);
            hist[bin]++;
            total++;
        }
    }
    if (total == 0) return 0.0f;

    double sum_all = 0.0;
    for (int b = 0; b < kBins; ++b) {
        sum_all += static_cast<double>(b) * static_cast<double>(hist[b]);
    }

    double sum_bg = 0.0;
    uint64_t weight_bg = 0;
    double best_var = -1.0;
    int best_bin = 0;

    for (int b = 0; b < kBins; ++b) {
        weight_bg += hist[b];
        if (weight_bg == 0) continue;
        const uint64_t weight_fg = total - weight_bg;
        if (weight_fg == 0) break;

        sum_bg += static_cast<double>(b) * static_cast<double>(hist[b]);
        const double mean_bg = sum_bg / static_cast<double>(weight_bg);
        const double mean_fg = (sum_all - sum_bg) / static_cast<double>(weight_fg);
        const double diff = mean_bg - mean_fg;
        const double between_var =
            static_cast<double>(weight_bg) * static_cast<double>(weight_fg) * diff * diff;

        if (between_var > best_var) {
            best_var = between_var;
            best_bin = b;
        }
    }

    return (static_cast<float>(best_bin) + 0.5f) / scale;
}

// Retains only the largest 6-connected foreground component so disconnected background noise
// islands cannot create independent floating Neumann sub-domains in the FEM solver.
void keepLargestConnectedComponent(
    std::vector<uint8_t>& mask,
    int nx, int ny, int nz
) {
    const size_t n = static_cast<size_t>(nx) * ny * nz;
    const size_t stride_y = static_cast<size_t>(nx);
    const size_t stride_z = static_cast<size_t>(nx) * ny;

    std::vector<int32_t> comp_id(n, -1);
    std::vector<size_t> comp_sizes;
    std::vector<size_t> queue;
    queue.reserve(n / 4);

    for (size_t start = 0; start < n; ++start) {
        if (!mask[start] || comp_id[start] != -1) continue;

        const int32_t cid = static_cast<int32_t>(comp_sizes.size());
        comp_id[start] = cid;
        queue.clear();
        queue.push_back(start);

        size_t head = 0;
        while (head < queue.size()) {
            const size_t idx = queue[head++];
            const int i = static_cast<int>(idx % stride_y);
            const int j = static_cast<int>((idx / stride_y) % ny);
            const int k = static_cast<int>(idx / stride_z);

            auto tryVisit = [&](size_t n_idx) {
                if (mask[n_idx] && comp_id[n_idx] == -1) {
                    comp_id[n_idx] = cid;
                    queue.push_back(n_idx);
                }
            };

            if (i > 0)      tryVisit(idx - 1);
            if (i + 1 < nx) tryVisit(idx + 1);
            if (j > 0)      tryVisit(idx - stride_y);
            if (j + 1 < ny) tryVisit(idx + stride_y);
            if (k > 0)      tryVisit(idx - stride_z);
            if (k + 1 < nz) tryVisit(idx + stride_z);
        }

        comp_sizes.push_back(queue.size());
    }

    if (comp_sizes.empty()) return;

    const int32_t best_cid = static_cast<int32_t>(
        std::distance(comp_sizes.begin(), std::max_element(comp_sizes.begin(), comp_sizes.end()))
    );

    for (size_t idx = 0; idx < n; ++idx) {
        mask[idx] = (comp_id[idx] == best_cid) ? 1 : 0;
    }
}

// Flood-fills exterior air from the 6 volume boundary faces so dark interior skull/CSF
// remains inside head_mask while exterior background air is excluded.
std::vector<uint8_t> buildHeadMask(const MriVolume& mri) {
    const int nx = mri.nx;
    const int ny = mri.ny;
    const int nz = mri.nz;
    const size_t n = static_cast<size_t>(nx) * ny * nz;

    const float thresh = computeOtsuThreshold(mri.intensities);
    std::vector<uint8_t> exterior_air(n, 0);
    std::queue<size_t> q;

    auto tryEnqueue = [&](int i, int j, int k) {
        const size_t idx = static_cast<size_t>(i) + nx * (static_cast<size_t>(j) + ny * k);
        if (!exterior_air[idx] && mri.intensities[idx] <= thresh) {
            exterior_air[idx] = 1;
            q.push(idx);
        }
    };

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            tryEnqueue(0, j, k);
            tryEnqueue(nx - 1, j, k);
        }
    }
    for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < nx; ++i) {
            tryEnqueue(i, 0, k);
            tryEnqueue(i, ny - 1, k);
        }
    }
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            tryEnqueue(i, j, 0);
            tryEnqueue(i, j, nz - 1);
        }
    }

    const size_t stride_y = static_cast<size_t>(nx);
    const size_t stride_z = static_cast<size_t>(nx) * ny;

    while (!q.empty()) {
        const size_t idx = q.front();
        q.pop();

        const int i = static_cast<int>(idx % stride_y);
        const int j = static_cast<int>((idx / stride_y) % ny);
        const int k = static_cast<int>(idx / stride_z);

        if (i > 0)      tryEnqueue(i - 1, j, k);
        if (i + 1 < nx) tryEnqueue(i + 1, j, k);
        if (j > 0)      tryEnqueue(i, j - 1, k);
        if (j + 1 < ny) tryEnqueue(i, j + 1, k);
        if (k > 0)      tryEnqueue(i, j, k - 1);
        if (k + 1 < nz) tryEnqueue(i, j, k + 1);
    }

    std::vector<uint8_t> head_mask(n, 0);
    for (size_t idx = 0; idx < n; ++idx) {
        head_mask[idx] = exterior_air[idx] ? 0 : 1;
    }

    keepLargestConnectedComponent(head_mask, nx, ny, nz);
    return head_mask;
}

// Builds smooth ellipsoidal anatomical spatial priors for the 5 head tissues
// (Scalp, Skull, CSF, GrayMatter, WhiteMatter) anchored to the subject's head mask.
void buildEllipsoidalSpatialPriors(
    const MriVolume& mri,
    const std::vector<uint8_t>& head_mask,
    std::vector<float>& priors
) {
    const int nx = mri.nx;
    const int ny = mri.ny;
    const int nz = mri.nz;
    const size_t n = static_cast<size_t>(nx) * ny * nz;

    double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
    double sum_x2 = 0.0, sum_y2 = 0.0, sum_z2 = 0.0;
    size_t count = 0;

    for (int k = 0; k < nz; ++k) {
        const double z_mm = k * mri.spacing_mm[2];
        for (int j = 0; j < ny; ++j) {
            const double y_mm = j * mri.spacing_mm[1];
            for (int i = 0; i < nx; ++i) {
                const size_t idx = static_cast<size_t>(i) + nx * (static_cast<size_t>(j) + ny * k);
                if (!head_mask[idx]) continue;

                const double x_mm = i * mri.spacing_mm[0];
                sum_x += x_mm; sum_x2 += x_mm * x_mm;
                sum_y += y_mm; sum_y2 += y_mm * y_mm;
                sum_z += z_mm; sum_z2 += z_mm * z_mm;
                count++;
            }
        }
    }

    if (count == 0) {
        throw std::runtime_error("Head mask is empty; cannot segment MRI volume");
    }

    const double inv_c = 1.0 / static_cast<double>(count);
    const double cx = sum_x * inv_c;
    const double cy = sum_y * inv_c;
    const double cz = sum_z * inv_c;

    constexpr double kSqrt5 = 2.23606797749979;
    const double rx = std::max(10.0, kSqrt5 * std::sqrt(std::max(1.0, sum_x2 * inv_c - cx * cx)));
    const double ry = std::max(10.0, kSqrt5 * std::sqrt(std::max(1.0, sum_y2 * inv_c - cy * cy)));
    const double rz = std::max(10.0, kSqrt5 * std::sqrt(std::max(1.0, sum_z2 * inv_c - cz * cz)));

    // Normalized radial shell centers and widths (rho = 1.0 at scalp outer surface)
    // Index: 1=Scalp, 2=Skull, 3=CSF, 4=GrayMatter, 5=WhiteMatter
    constexpr std::array<double, kNumClasses> rho_center = {0.0, 0.96, 0.87, 0.79, 0.69, 0.35};
    constexpr std::array<double, kNumClasses> rho_width  = {1.0, 0.06, 0.05, 0.05, 0.09, 0.28};

    priors.assign(kNumClasses * n, 0.0f);

#ifdef TMS_USE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int k = 0; k < nz; ++k) {
        const double dz = (k * mri.spacing_mm[2] - cz) / rz;
        for (int j = 0; j < ny; ++j) {
            const double dy = (j * mri.spacing_mm[1] - cy) / ry;
            for (int i = 0; i < nx; ++i) {
                const size_t idx = static_cast<size_t>(i) + nx * (static_cast<size_t>(j) + ny * k);
                if (!head_mask[idx]) {
                    priors[0 * n + idx] = 1.0f;
                    continue;
                }

                const double dx = (i * mri.spacing_mm[0] - cx) / rx;
                const double rho = std::sqrt(dx * dx + dy * dy + dz * dz);

                double w_sum = 0.0;
                std::array<double, kNumClasses> w{};
                for (size_t c = kFirstTissue; c < kNumClasses; ++c) {
                    const double z_score = (rho - rho_center[c]) / rho_width[c];
                    w[c] = 1e-3 + std::exp(-0.5 * z_score * z_score);
                    w_sum += w[c];
                }

                const double inv_w = 1.0 / w_sum;
                for (size_t c = kFirstTissue; c < kNumClasses; ++c) {
                    priors[c * n + idx] = static_cast<float>(w[c] * inv_w);
                }
            }
        }
    }
}

// Fast 3D separable box-filter smoothing for low-frequency B1 bias-field estimation
void smoothFieldSeparable(
    std::vector<float>& field,
    int nx, int ny, int nz,
    int radius
) {
    if (radius <= 0) return;
    const size_t n = static_cast<size_t>(nx) * ny * nz;
    std::vector<float> tmp(n, 0.0f);

    // X pass
#ifdef TMS_USE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            const size_t row = nx * (static_cast<size_t>(j) + ny * k);
            for (int i = 0; i < nx; ++i) {
                const int i0 = std::max(0, i - radius);
                const int i1 = std::min(nx - 1, i + radius);
                double acc = 0.0;
                for (int x = i0; x <= i1; ++x) acc += field[row + x];
                tmp[row + i] = static_cast<float>(acc / static_cast<double>(i1 - i0 + 1));
            }
        }
    }

    // Y pass
#ifdef TMS_USE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int k = 0; k < nz; ++k) {
        const size_t slice = static_cast<size_t>(nx) * ny * k;
        for (int j = 0; j < ny; ++j) {
            const int j0 = std::max(0, j - radius);
            const int j1 = std::min(ny - 1, j + radius);
            const double inv_len = 1.0 / static_cast<double>(j1 - j0 + 1);
            for (int i = 0; i < nx; ++i) {
                double acc = 0.0;
                for (int y = j0; y <= j1; ++y) acc += tmp[slice + static_cast<size_t>(y) * nx + i];
                field[slice + static_cast<size_t>(j) * nx + i] = static_cast<float>(acc * inv_len);
            }
        }
    }

    // Z pass
#ifdef TMS_USE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int k = 0; k < nz; ++k) {
        const int k0 = std::max(0, k - radius);
        const int k1 = std::min(nz - 1, k + radius);
        const double inv_len = 1.0 / static_cast<double>(k1 - k0 + 1);
        const size_t slice_stride = static_cast<size_t>(nx) * ny;
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const size_t col = static_cast<size_t>(j) * nx + i;
                double acc = 0.0;
                for (int z = k0; z <= k1; ++z) acc += field[static_cast<size_t>(z) * slice_stride + col];
                tmp[static_cast<size_t>(k) * slice_stride + col] = static_cast<float>(acc * inv_len);
            }
        }
    }

    field.swap(tmp);
}

} // namespace

GmmSpatialSegmenter::GmmSpatialSegmenter(
    std::string atlas_path,
    GmmSegmenterConfig config
) : atlas_path_(std::move(atlas_path)), cfg_(config) {}

SegmentationResult GmmSpatialSegmenter::segment(const MriVolume& mri) const {
    const int nx = mri.nx;
    const int ny = mri.ny;
    const int nz = mri.nz;
    const size_t n = static_cast<size_t>(nx) * ny * nz;

    if (nx <= 0 || ny <= 0 || nz <= 0 || mri.intensities.size() != n) {
        throw std::invalid_argument("Invalid MriVolume dimensions or intensity buffer");
    }

    // 1. Separate exterior background air from the single connected head volume
    const std::vector<uint8_t> head_mask = buildHeadMask(mri);

    // 2. Normalize intensities inside head_mask using 98th percentile
    std::vector<float> head_vals;
    head_vals.reserve(n / 2);
    for (size_t idx = 0; idx < n; ++idx) {
        if (head_mask[idx] && mri.intensities[idx] > 0.0f) {
            head_vals.push_back(mri.intensities[idx]);
        }
    }
    if (head_vals.empty()) {
        throw std::runtime_error("No positive intensities found inside head mask");
    }

    const size_t p98_idx = static_cast<size_t>(0.98 * (head_vals.size() - 1));
    std::nth_element(head_vals.begin(), head_vals.begin() + p98_idx, head_vals.end());
    const float norm_scale = std::max(1e-6f, head_vals[p98_idx]);

    std::vector<float> y_norm(n, 0.0f);
    for (size_t idx = 0; idx < n; ++idx) {
        if (head_mask[idx]) {
            y_norm[idx] = std::max(0.0f, mri.intensities[idx] / norm_scale);
        }
    }

    // 3. Construct spatial tissue probability priors
    std::vector<float> atlas_priors;
    buildEllipsoidalSpatialPriors(mri, head_mask, atlas_priors);

    std::vector<float> gamma = atlas_priors;
    std::vector<float> bias_field(n, 1.0f);

    std::array<double, kNumClasses> mu{};
    std::array<double, kNumClasses> var{};
    const double min_var = cfg_.min_sigma * cfg_.min_sigma;

    const size_t stride_y = static_cast<size_t>(nx);
    const size_t stride_z = static_cast<size_t>(nx) * ny;

    // 4. Iterative Expectation-Maximization with 3D Mean-Field Potts MRF & Bias Correction
    for (int iter = 0; iter < cfg_.max_em_iterations; ++iter) {
        const std::array<double, kNumClasses> prev_mu = mu;

        // --- M-Step: Update class means and variances ---
        for (size_t c = kFirstTissue; c < kNumClasses; ++c) {
            double sum_w = 0.0;
            double sum_wy = 0.0;

#ifdef TMS_USE_OPENMP
            #pragma omp parallel for reduction(+:sum_w, sum_wy) schedule(static)
#endif
            for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(n); ++idx) {
                if (!head_mask[idx]) continue;
                const double w = gamma[c * n + idx];
                const double y_corr = static_cast<double>(y_norm[idx]) / bias_field[idx];
                sum_w += w;
                sum_wy += w * y_corr;
            }

            mu[c] = (sum_w > 1e-8) ? (sum_wy / sum_w) : (0.15 * c);

            double sum_w_sq = 0.0;
#ifdef TMS_USE_OPENMP
            #pragma omp parallel for reduction(+:sum_w_sq) schedule(static)
#endif
            for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(n); ++idx) {
                if (!head_mask[idx]) continue;
                const double w = gamma[c * n + idx];
                const double y_corr = static_cast<double>(y_norm[idx]) / bias_field[idx];
                const double diff = y_corr - mu[c];
                sum_w_sq += w * diff * diff;
            }

            var[c] = std::max(min_var, (sum_w > 1e-8) ? (sum_w_sq / sum_w) : 0.01);
        }

        // --- E-Step: Combine Atlas Prior + 3D 6-Neighbor Potts MRF + Gaussian Likelihood ---
        std::vector<float> next_gamma(kNumClasses * n, 0.0f);
        std::array<double, kNumClasses> inv_2var{};
        std::array<double, kNumClasses> norm_const{};
        for (size_t c = kFirstTissue; c < kNumClasses; ++c) {
            inv_2var[c] = 0.5 / var[c];
            norm_const[c] = 1.0 / std::sqrt(kTwoPi * var[c]);
        }

#ifdef TMS_USE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const size_t idx = static_cast<size_t>(i) + nx * (static_cast<size_t>(j) + ny * k);
                    if (!head_mask[idx]) {
                        next_gamma[0 * n + idx] = 1.0f;
                        continue;
                    }

                    const double y_corr = static_cast<double>(y_norm[idx]) / bias_field[idx];
                    double post_sum = 0.0;
                    std::array<double, kNumClasses> unnorm_post{};

                    for (size_t c = kFirstTissue; c < kNumClasses; ++c) {
                        double neigh_sum = 0.0;
                        const size_t base = c * n;
                        if (i > 0)      neigh_sum += gamma[base + idx - 1];
                        if (i + 1 < nx) neigh_sum += gamma[base + idx + 1];
                        if (j > 0)      neigh_sum += gamma[base + idx - stride_y];
                        if (j + 1 < ny) neigh_sum += gamma[base + idx + stride_y];
                        if (k > 0)      neigh_sum += gamma[base + idx - stride_z];
                        if (k + 1 < nz) neigh_sum += gamma[base + idx + stride_z];

                        const double spatial_prior =
                            static_cast<double>(atlas_priors[base + idx]) *
                            std::exp(cfg_.mrf_beta * neigh_sum);

                        const double diff = y_corr - mu[c];
                        const double likelihood =
                            norm_const[c] * std::exp(-diff * diff * inv_2var[c]);

                        unnorm_post[c] = spatial_prior * likelihood + 1e-30;
                        post_sum += unnorm_post[c];
                    }

                    const double inv_post = 1.0 / post_sum;
                    for (size_t c = kFirstTissue; c < kNumClasses; ++c) {
                        next_gamma[c * n + idx] = static_cast<float>(unnorm_post[c] * inv_post);
                    }
                }
            }
        }

        gamma.swap(next_gamma);

        // --- Multiplicative B1 Bias-Field Update ---
        if (cfg_.estimate_bias_field && iter >= 2 && (iter % 2 == 0)) {
            std::vector<float> num_field(n, 0.0f);
            std::vector<float> den_field(n, 0.0f);

#ifdef TMS_USE_OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(n); ++idx) {
                if (!head_mask[idx]) continue;
                double w_mu = 0.0;
                double w_mu2 = 0.0;
                for (size_t c = kFirstTissue; c < kNumClasses; ++c) {
                    const double w = static_cast<double>(gamma[c * n + idx]) / var[c];
                    w_mu += w * mu[c];
                    w_mu2 += w * mu[c] * mu[c];
                }
                num_field[idx] = static_cast<float>(w_mu * y_norm[idx]);
                den_field[idx] = static_cast<float>(w_mu2);
            }

            smoothFieldSeparable(num_field, nx, ny, nz, cfg_.bias_smooth_radius);
            smoothFieldSeparable(den_field, nx, ny, nz, cfg_.bias_smooth_radius);

            double bias_sum = 0.0;
            size_t head_count = 0;
            for (size_t idx = 0; idx < n; ++idx) {
                if (head_mask[idx] && den_field[idx] > 1e-6f) {
                    const float b = std::clamp(num_field[idx] / den_field[idx], 0.5f, 2.0f);
                    bias_field[idx] = b;
                    bias_sum += b;
                    head_count++;
                }
            }
            if (head_count > 0) {
                const float mean_bias = static_cast<float>(bias_sum / static_cast<double>(head_count));
                for (size_t idx = 0; idx < n; ++idx) {
                    if (head_mask[idx]) bias_field[idx] /= mean_bias;
                }
            }
        }

        if (iter > 3) {
            double max_delta = 0.0;
            for (size_t c = kFirstTissue; c < kNumClasses; ++c) {
                max_delta = std::max(max_delta, std::abs(mu[c] - prev_mu[c]));
            }
            if (max_delta < cfg_.convergence_tol) break;
        }
    }

    // 5. Extract hard argmax tissue labels
    SegmentationResult result;
    result.nx = nx;
    result.ny = ny;
    result.nz = nz;
    result.spacing_mm = mri.spacing_mm;
    result.labels.resize(n, static_cast<uint8_t>(TissueClass::Air));

#ifdef TMS_USE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(n); ++idx) {
        if (!head_mask[idx]) {
            result.labels[idx] = static_cast<uint8_t>(TissueClass::Air);
            continue;
        }

        uint8_t best_c = static_cast<uint8_t>(TissueClass::GrayMatter);
        float best_p = -1.0f;
        for (size_t c = kFirstTissue; c < kNumClasses; ++c) {
            const float p = gamma[c * n + idx];
            if (p > best_p) {
                best_p = p;
                best_c = static_cast<uint8_t>(c);
            }
        }
        result.labels[idx] = best_c;
    }

    result.posteriors = std::move(gamma);
    return result;
}