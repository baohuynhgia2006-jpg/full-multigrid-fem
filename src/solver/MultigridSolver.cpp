#include "solver/MultigridSolver.h"
#include "solver/FemStencil.h"
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#ifdef TMS_USE_OPENMP
#include <omp.h>
#endif

namespace {

double dotProduct(const Grid3D& a, const Grid3D& b) {
    const size_t n = a.size();
    const double* pa = a.raw();
    const double* pb = b.raw();
    double sum = 0.0;

#ifdef TMS_USE_OPENMP
    #pragma omp parallel for reduction(+:sum) schedule(static)
#endif
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
        sum += pa[i] * pb[i];
    }
    return sum;
}

} // namespace

MultigridSolver::MultigridSolver(SolverConfig config) : cfg_(config) {
#ifdef TMS_USE_OPENMP
    if (cfg_.num_threads > 0) {
        omp_set_num_threads(cfg_.num_threads);
    }
#endif
}

SolverStats MultigridSolver::solve(MultigridHierarchy& hierarchy) {
    return cfg_.use_mgcg ? solveMGCG(hierarchy) : solveStandaloneVCycle(hierarchy);
}

SolverStats MultigridSolver::solveStandaloneVCycle(MultigridHierarchy& hierarchy) {
    GridLevel& finest = hierarchy.level(0);
    if (!cfg_.use_initial_guess) {
        finest.phi.fill(0.0);
    }

    const double f_norm = finest.f.l2Norm();
    SolverStats stats{};
    if (f_norm == 0.0) {
        stats.converged = true;
        return stats;
    }

    for (int iter = 1; iter <= cfg_.max_vcycles; ++iter) {
        vCycle(hierarchy, 0);

        computeResidual(finest);
        const double rel_res = finest.r.l2Norm() / f_norm;

        stats.iterations = iter;
        stats.final_rel_residual = rel_res;

        if (cfg_.verbose) {
            std::cout << "  V-cycle " << iter << " | rel_res = " << rel_res << "\n";
        }

        if (rel_res <= cfg_.rel_residual_tol) {
            stats.converged = true;
            break;
        }
    }

    return stats;
}

SolverStats MultigridSolver::solveMGCG(MultigridHierarchy& hierarchy) {
    GridLevel& finest = hierarchy.level(0);
    const int nx = finest.phi.nx();
    const int ny = finest.phi.ny();
    const int nz = finest.phi.nz();
    const double h = finest.h;
    const size_t n = finest.phi.size();

    const double f_norm = finest.f.l2Norm();
    SolverStats stats{};
    if (f_norm == 0.0) {
        finest.phi.fill(0.0);
        stats.converged = true;
        return stats;
    }

    // Identify active tissue nodes (diag_weight > 0) for nullspace projection
    std::vector<uint8_t> active_nodes(n, 0);
    size_t active_count = 0;

    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                double off_diag = 0.0;
                double diag_w = 0.0;
                FemStencil::evaluateNode(finest.phi, finest.sigma, i, j, k, off_diag, diag_w);
                if (diag_w > 0.0) {
                    const size_t idx =
                        static_cast<size_t>(i) + nx * (static_cast<size_t>(j) + ny * k);
                    active_nodes[idx] = 1;
                    ++active_count;
                }
            }
        }
    }

    auto projectZeroMean = [&](Grid3D& g) {
        if (!cfg_.project_nullspace || active_count == 0) return;
        double* ptr = g.raw();
        double sum = 0.0;

#ifdef TMS_USE_OPENMP
        #pragma omp parallel for reduction(+:sum) schedule(static)
#endif
        for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(n); ++idx) {
            if (active_nodes[idx]) sum += ptr[idx];
        }

        const double mean = sum / static_cast<double>(active_count);

#ifdef TMS_USE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(n); ++idx) {
            if (active_nodes[idx]) ptr[idx] -= mean;
        }
    };

    Grid3D x(nx, ny, nz, h, 0.0);
    Grid3D r_cg(nx, ny, nz, h, 0.0);
    Grid3D p(nx, ny, nz, h, 0.0);
    Grid3D Ap(nx, ny, nz, h, 0.0);

    const Grid3D f_orig = finest.f;

    if (cfg_.use_initial_guess) {
        double* px = x.raw();
        const double* p_init = finest.phi.raw();
#ifdef TMS_USE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(n); ++idx) {
            px[idx] = active_nodes[idx] ? p_init[idx] : 0.0;
        }
        projectZeroMean(x);

        applyOperator(x, finest.sigma, Ap);
        double* pr = r_cg.raw();
        const double* pf = f_orig.raw();
        const double* pAp_ptr = Ap.raw();
#ifdef TMS_USE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(n); ++idx) {
            pr[idx] = active_nodes[idx] ? (pf[idx] - pAp_ptr[idx]) : 0.0;
        }
    } else {
        r_cg = f_orig;
    }

    projectZeroMean(r_cg);

    const double init_rel_res = r_cg.l2Norm() / f_norm;
    if (init_rel_res <= cfg_.rel_residual_tol) {
        finest.phi = x;
        finest.f = f_orig;
        computeResidual(finest);
        stats.final_rel_residual = init_rel_res;
        stats.converged = true;
        return stats;
    }

    // z_0 = P_perp * M^{-1} * r_0
    finest.f = r_cg;
    finest.phi.fill(0.0);
    vCycle(hierarchy, 0);
    projectZeroMean(finest.phi);
    p = finest.phi;

    double rz_old = dotProduct(r_cg, finest.phi);

    for (int iter = 1; iter <= cfg_.max_vcycles; ++iter) {
        applyOperator(p, finest.sigma, Ap);
        const double pAp = dotProduct(p, Ap);
        if (pAp <= 0.0) break;

        const double alpha = rz_old / pAp;

        double* px = x.raw();
        double* pr = r_cg.raw();
        const double* pp = p.raw();
        const double* pAp_ptr = Ap.raw();

#ifdef TMS_USE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(n); ++idx) {
            if (active_nodes[idx]) {
                px[idx] += alpha * pp[idx];
                pr[idx] -= alpha * pAp_ptr[idx];
            }
        }

        projectZeroMean(r_cg);

        const double rel_res = r_cg.l2Norm() / f_norm;
        stats.iterations = iter;
        stats.final_rel_residual = rel_res;

        if (cfg_.verbose) {
            std::cout << "  MGCG iter " << iter << " | rel_res = " << rel_res << "\n";
        }

        if (rel_res <= cfg_.rel_residual_tol) {
            stats.converged = true;
            break;
        }

        // Preconditioner step: z_{k+1} = P_perp * M^{-1} * r_{k+1}
        finest.f = r_cg;
        finest.phi.fill(0.0);
        vCycle(hierarchy, 0);
        projectZeroMean(finest.phi);

        const double rz_new = dotProduct(r_cg, finest.phi);
        const double beta = rz_new / rz_old;
        rz_old = rz_new;

        double* pp_mut = p.raw();
        const double* pz = finest.phi.raw();

#ifdef TMS_USE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(n); ++idx) {
            if (active_nodes[idx]) {
                pp_mut[idx] = pz[idx] + beta * pp_mut[idx];
            }
        }
    }

    finest.phi = x;
    finest.f = f_orig;
    computeResidual(finest);
    return stats;
}

void MultigridSolver::vCycle(MultigridHierarchy& hierarchy, int level_idx) {
    GridLevel& fine = hierarchy.level(level_idx);

    // Forward pre-smoothing
    smoothSOR(fine, cfg_.pre_smooth_steps, /*reverse_order=*/false);
    computeResidual(fine);

    GridLevel& coarse = hierarchy.level(level_idx + 1);
    restrictResidual(fine, coarse);
    coarse.phi.fill(0.0);

    const int coarsest_idx = hierarchy.numLevels() - 1;
    if (level_idx + 1 == coarsest_idx) {
        // Symmetric coarsest-grid solve (forward + backward sweeps)
        smoothSOR(coarse, cfg_.coarsest_smooth_steps, /*reverse_order=*/false);
        smoothSOR(coarse, cfg_.coarsest_smooth_steps, /*reverse_order=*/true);
    } else {
        vCycle(hierarchy, level_idx + 1);
    }

    prolongateAndCorrect(coarse, fine);

    // Adjoint backward post-smoothing
    smoothSOR(fine, cfg_.post_smooth_steps, /*reverse_order=*/true);
}

void MultigridSolver::smoothSOR(GridLevel& lvl, int num_sweeps, bool reverse_order) {
    const int nx = lvl.phi.nx();
    const int ny = lvl.phi.ny();
    const int nz = lvl.phi.nz();
    const double h2 = lvl.h * lvl.h;

#ifdef TMS_USE_OPENMP
    if (lvl.phi.size() >= cfg_.omp_min_nodes) {
        const double omega = cfg_.omega_multicolor;
        const double one_minus_omega = 1.0 - omega;

        for (int sweep = 0; sweep < num_sweeps; ++sweep) {
            for (int c_idx = 0; c_idx < 8; ++c_idx) {
                const int color = reverse_order ? (7 - c_idx) : c_idx;
                const int ci = color & 1;
                const int cj = (color >> 1) & 1;
                const int ck = (color >> 2) & 1;

                #pragma omp parallel for schedule(static)
                for (int k = 1 + ck; k < nz - 1; k += 2) {
                    for (int j = 1 + cj; j < ny - 1; j += 2) {
                        for (int i = 1 + ci; i < nx - 1; i += 2) {
                            double off_diag_sum = 0.0;
                            double diag_weight = 0.0;
                            FemStencil::evaluateNode(
                                lvl.phi, lvl.sigma, i, j, k, off_diag_sum, diag_weight
                            );

                            if (diag_weight <= 0.0) continue;

                            const double gs_target =
                                (h2 * lvl.f(i, j, k) + off_diag_sum) / diag_weight;
                            lvl.phi(i, j, k) =
                                one_minus_omega * lvl.phi(i, j, k) + omega * gs_target;
                        }
                    }
                }
            }
        }
        return;
    }
#endif

    // Sequential sweep (symmetric forward/backward ordering when use_mgcg = true)
    const double omega = cfg_.use_mgcg ? 1.0 : cfg_.omega_seq;
    const double one_minus_omega = 1.0 - omega;

    for (int sweep = 0; sweep < num_sweeps; ++sweep) {
        if (!reverse_order) {
            for (int k = 1; k < nz - 1; ++k) {
                for (int j = 1; j < ny - 1; ++j) {
                    for (int i = 1; i < nx - 1; ++i) {
                        double off_diag_sum = 0.0;
                        double diag_weight = 0.0;
                        FemStencil::evaluateNode(
                            lvl.phi, lvl.sigma, i, j, k, off_diag_sum, diag_weight
                        );
                        if (diag_weight <= 0.0) continue;

                        const double gs_target =
                            (h2 * lvl.f(i, j, k) + off_diag_sum) / diag_weight;
                        lvl.phi(i, j, k) =
                            one_minus_omega * lvl.phi(i, j, k) + omega * gs_target;
                    }
                }
            }
        } else {
            for (int k = nz - 2; k >= 1; --k) {
                for (int j = ny - 2; j >= 1; --j) {
                    for (int i = nx - 2; i >= 1; --i) {
                        double off_diag_sum = 0.0;
                        double diag_weight = 0.0;
                        FemStencil::evaluateNode(
                            lvl.phi, lvl.sigma, i, j, k, off_diag_sum, diag_weight
                        );
                        if (diag_weight <= 0.0) continue;

                        const double gs_target =
                            (h2 * lvl.f(i, j, k) + off_diag_sum) / diag_weight;
                        lvl.phi(i, j, k) =
                            one_minus_omega * lvl.phi(i, j, k) + omega * gs_target;
                    }
                }
            }
        }
    }
}

void MultigridSolver::applyOperator(const Grid3D& x, const Grid3D& sigma, Grid3D& Ax) {
    const int nx = x.nx();
    const int ny = x.ny();
    const int nz = x.nz();
    const double inv_h2 = 1.0 / (x.h() * x.h());

    Ax.fill(0.0);

#ifdef TMS_USE_OPENMP
    #pragma omp parallel for schedule(static) if(x.size() >= cfg_.omp_min_nodes)
#endif
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                double off_diag_sum = 0.0;
                double diag_weight = 0.0;
                FemStencil::evaluateNode(
                    x, sigma, i, j, k, off_diag_sum, diag_weight
                );

                if (diag_weight <= 0.0) continue;

                Ax(i, j, k) = inv_h2 * (diag_weight * x(i, j, k) - off_diag_sum);
            }
        }
    }
}

void MultigridSolver::computeResidual(GridLevel& lvl) {
    const int nx = lvl.phi.nx();
    const int ny = lvl.phi.ny();
    const int nz = lvl.phi.nz();
    const double inv_h2 = 1.0 / (lvl.h * lvl.h);

    lvl.r.fill(0.0);

#ifdef TMS_USE_OPENMP
    #pragma omp parallel for schedule(static) if(lvl.phi.size() >= cfg_.omp_min_nodes)
#endif
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                double off_diag_sum = 0.0;
                double diag_weight = 0.0;
                FemStencil::evaluateNode(
                    lvl.phi, lvl.sigma, i, j, k, off_diag_sum, diag_weight
                );

                if (diag_weight <= 0.0) continue;

                const double a_phi =
                    inv_h2 * (diag_weight * lvl.phi(i, j, k) - off_diag_sum);
                lvl.r(i, j, k) = lvl.f(i, j, k) - a_phi;
            }
        }
    }
}

void MultigridSolver::restrictResidual(const GridLevel& fine, GridLevel& coarse) {
    constexpr double w1d[3] = {1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0};

    const int cnx = coarse.f.nx();
    const int cny = coarse.f.ny();
    const int cnz = coarse.f.nz();

    coarse.f.fill(0.0);

#ifdef TMS_USE_OPENMP
    #pragma omp parallel for schedule(static) if(fine.r.size() >= cfg_.omp_min_nodes)
#endif
    for (int ck = 1; ck < cnz - 1; ++ck) {
        const int fk = 2 * ck;
        for (int cj = 1; cj < cny - 1; ++cj) {
            const int fj = 2 * cj;
            for (int ci = 1; ci < cnx - 1; ++ci) {
                const int fi = 2 * ci;

                double sum = 0.0;
                for (int dz = -1; dz <= 1; ++dz) {
                    const double wz = w1d[dz + 1];
                    for (int dy = -1; dy <= 1; ++dy) {
                        const double wyz = wz * w1d[dy + 1];
                        for (int dx = -1; dx <= 1; ++dx) {
                            const double w = wyz * w1d[dx + 1];
                            sum += w * fine.r(fi + dx, fj + dy, fk + dz);
                        }
                    }
                }
                coarse.f(ci, cj, ck) = sum;
            }
        }
    }
}

void MultigridSolver::prolongateAndCorrect(const GridLevel& coarse, GridLevel& fine) {
    const int fnx = fine.phi.nx();
    const int fny = fine.phi.ny();
    const int fnz = fine.phi.nz();

#ifdef TMS_USE_OPENMP
    #pragma omp parallel for schedule(static) if(fine.phi.size() >= cfg_.omp_min_nodes)
#endif
    for (int fk = 1; fk < fnz - 1; ++fk) {
        const int ck = fk / 2;
        const int pk = fk % 2;

        for (int fj = 1; fj < fny - 1; ++fj) {
            const int cj = fj / 2;
            const int pj = fj % 2;

            for (int fi = 1; fi < fnx - 1; ++fi) {
                const int ci = fi / 2;
                const int pi = fi % 2;

                double interp = 0.0;
                for (int dz = 0; dz <= pk; ++dz) {
                    for (int dy = 0; dy <= pj; ++dy) {
                        for (int dx = 0; dx <= pi; ++dx) {
                            interp += coarse.phi(ci + dx, cj + dy, ck + dz);
                        }
                    }
                }

                const double count = static_cast<double>((1 + pi) * (1 + pj) * (1 + pk));
                fine.phi(fi, fj, fk) += interp / count;
            }
        }
    }
}