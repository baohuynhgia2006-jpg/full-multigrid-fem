#include "grid/MultigridHierarchy.h"
#include "solver/CoilModel.h"
#include "solver/ConductivityBuilder.h"
#include "solver/FemStencil.h"
#include "solver/FieldPostProcessor.h"
#include "solver/MultigridSolver.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

// Exact analytical induced E-field inside a concentric layered sphere (Sarvas / Ilvonen & Laakso)
std::array<double, 3> analyticalSphereDipoleE(
    const std::array<double, 3>& r,
    const std::array<double, 3>& r0,
    const std::array<double, 3>& m_prime
) {
    const std::array<double, 3> a = {r0[0] - r[0], r0[1] - r[1], r0[2] - r[2]};
    const double a_norm = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    const double r0_norm = std::sqrt(r0[0] * r0[0] + r0[1] * r0[1] + r0[2] * r0[2]);
    const double a_dot_r0 = a[0] * r0[0] + a[1] * r0[1] + a[2] * r0[2];

    const double F = a_norm * (r0_norm * a_norm + a_dot_r0);
    const double c1 = (a_norm * a_norm) / r0_norm + a_dot_r0 / a_norm + 2.0 * a_norm + 2.0 * r0_norm;
    const double c2 = a_norm + 2.0 * r0_norm + a_dot_r0 / a_norm;

    const std::array<double, 3> grad_r0_F = {
        c1 * r0[0] - c2 * r[0],
        c1 * r0[1] - c2 * r[1],
        c1 * r0[2] - c2 * r[2]
    };

    const double m_dot_gradF =
        m_prime[0] * grad_r0_F[0] +
        m_prime[1] * grad_r0_F[1] +
        m_prime[2] * grad_r0_F[2];

    // r x m'
    const std::array<double, 3> r_cross_m = {
        r[1] * m_prime[2] - r[2] * m_prime[1],
        r[2] * m_prime[0] - r[0] * m_prime[2],
        r[0] * m_prime[1] - r[1] * m_prime[0]
    };

    // r x r0
    const std::array<double, 3> r_cross_r0 = {
        r[1] * r0[2] - r[2] * r0[1],
        r[2] * r0[0] - r[0] * r0[2],
        r[0] * r0[1] - r[1] * r0[0]
    };

    const double prefactor = -1e-7 / (F * F);
    return {
        prefactor * (F * r_cross_m[0] - m_dot_gradF * r_cross_r0[0]),
        prefactor * (F * r_cross_m[1] - m_dot_gradF * r_cross_r0[1]),
        prefactor * (F * r_cross_m[2] - m_dot_gradF * r_cross_r0[2])
    };
}

} // namespace

int main() {
    // 4-layer concentric head sphere (in meters)
    constexpr double R_brain = 0.065;
    constexpr double R_csf   = 0.070;
    constexpr double R_skull = 0.078;
    constexpr double R_scalp = 0.085;

    // 96^3 voxels at h = 2 mm (192 mm cube, divisible by 2^(L-1) for L = 4)
    constexpr int N = 192;
    constexpr int num_levels = 4;
    constexpr double h = 0.001;
    constexpr double half_extent = 0.5 * N * h;
    const std::array<double, 3> origin = {-half_extent, -half_extent, -half_extent};

    // 1. Build synthetic 4-layer tissue label phantom
    std::vector<uint8_t> labels(static_cast<size_t>(N) * N * N, 0);
    for (int k = 0; k < N; ++k) {
        const double z = origin[2] + (k + 0.5) * h;
        for (int j = 0; j < N; ++j) {
            const double y = origin[1] + (j + 0.5) * h;
            for (int i = 0; i < N; ++i) {
                const double x = origin[0] + (i + 0.5) * h;
                const double r = std::sqrt(x * x + y * y + z * z);

                TissueClass cls = TissueClass::Air;
                if (r <= R_brain)      cls = TissueClass::GrayMatter;
                else if (r <= R_csf)   cls = TissueClass::CSF;
                else if (r <= R_skull) cls = TissueClass::Skull;
                else if (r <= R_scalp) cls = TissueClass::Scalp;

                labels[static_cast<size_t>(i) + N * (static_cast<size_t>(j) + N * k)] =
                    static_cast<uint8_t>(cls);
            }
        }
    }

    // 2. Initialize multigrid hierarchy & smooth conductivity
    MultigridHierarchy hierarchy(N, N, N, h, num_levels);
    ConductivityBuilder cond_builder;
    cond_builder.assignFromLabels(labels, hierarchy.level(0).sigma);
    ConductivityBuilder::applyStaircaseSmoothing(hierarchy.level(0).sigma);
    hierarchy.buildCoarseConductivities();

    // 3. Evaluate primary vector potential A0' from an off-axis magnetic dipole
    const std::array<double, 3> dipole_pos = {0.0, 0.0, 0.105};
    const std::array<double, 3> dipole_moment = {1.0, 0.5, 0.3};
    constexpr double dI_dt = 1.0e8; // 100 A/us typical TMS pulse

    CoilModel coil = CoilModel::createSingleDipole(dipole_moment);
    Vec3Grid A0_prime(N, N, N, h);
    coil.evaluateA0Prime(A0_prime, dI_dt, origin, dipole_pos);

    // 4. Assemble finest-grid FEM right-hand side f^h
    GridLevel& finest = hierarchy.level(0);
    for (int k = 1; k < N; ++k) {
        for (int j = 1; j < N; ++j) {
            for (int i = 1; i < N; ++i) {
                finest.f(i, j, k) = FemStencil::assembleNodalSource(
                    finest.sigma, A0_prime, i, j, k
                );
            }
        }
    }

    // 5. Solve via geometric multigrid V-cycles
    SolverConfig cfg;
    cfg.rel_residual_tol = 1e-5;
    cfg.max_vcycles = 60;
    MultigridSolver solver(cfg);

    const auto t0 = std::chrono::steady_clock::now();
    SolverStats stats = solver.solve(hierarchy);
    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 6. Compute E = -grad(phi) - A0' and compare against analytical solution in brain
    Vec3Grid E = FieldPostProcessor::computeElectricField(
        finest.phi, finest.sigma, A0_prime
    );

    const std::array<double, 3> m_prime = {
        dipole_moment[0] * dI_dt,
        dipole_moment[1] * dI_dt,
        dipole_moment[2] * dI_dt
    };

    std::vector<double> ref_mags;
    double max_err = 0.0;

    for (int k = 0; k < N; ++k) {
        const double z = origin[2] + (k + 0.5) * h;
        for (int j = 0; j < N; ++j) {
            const double y = origin[1] + (j + 0.5) * h;
            for (int i = 0; i < N; ++i) {
                const double x = origin[0] + (i + 0.5) * h;
                const double r = std::sqrt(x * x + y * y + z * z);

                // Evaluate inside interior brain region (away from staircase boundary)
                if (r <= R_brain - 3.0 * h) {
                    const auto E_ref = analyticalSphereDipoleE({x, y, z}, dipole_pos, m_prime);
                    const double ref_mag = std::sqrt(
                        E_ref[0] * E_ref[0] + E_ref[1] * E_ref[1] + E_ref[2] * E_ref[2]
                    );
                    ref_mags.push_back(ref_mag);

                    const double dx = E.x(i, j, k) - E_ref[0];
                    const double dy = E.y(i, j, k) - E_ref[1];
                    const double dz = E.z(i, j, k) - E_ref[2];
                    const double err = std::sqrt(dx * dx + dy * dy + dz * dz);
                    max_err = std::max(max_err, err);
                }
            }
        }
    }

    std::sort(ref_mags.begin(), ref_mags.end());
    const double E99 = ref_mags[static_cast<size_t>(0.99 * (ref_mags.size() - 1))];
    const double rel_e99_error = max_err / E99;

    std::cout << "Converged: " << (stats.converged ? "YES" : "NO") << "\n"
              << "V-cycles:  " << stats.iterations << "\n"
              << "Rel Res:   " << stats.final_rel_residual << "\n"
              << "Time:      " << elapsed_ms << " ms\n"
              << "Max |E - E_ref| / E_99: " << (rel_e99_error * 100.0) << " %\n";

    assert(stats.converged && stats.iterations <= 10);
    assert(rel_e99_error < 0.03); // < 3% discretization error at coarse 2 mm resolution
    return 0;
}