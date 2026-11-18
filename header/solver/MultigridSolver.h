#pragma once
#include "../grid/MultigridHierarchy.h"
#include <cstddef>

struct SolverConfig {
    bool use_mgcg = true;             // True = V-cycle Preconditioned CG (recommended for OpenMP)
    double omega_seq = 1.4;           // Lexicographical SOR factor (used when use_mgcg = false)
    double omega_multicolor = 1.0;    // Symmetric 8-color Gauss-Seidel factor
    int pre_smooth_steps = 2;         // Symmetric pre-smoothing sweeps (nu_1 = 2)
    int post_smooth_steps = 2;        // Symmetric post-smoothing sweeps (nu_2 = 2)
    int coarsest_smooth_steps = 50;   // Symmetric forward+backward sweeps on level L-1
    double rel_residual_tol = 1e-5;   // < 1% E-field error threshold[cite: 12]
    int max_vcycles = 30;
    size_t omp_min_nodes = 32768;     // Minimum grid nodes (32^3) to activate OpenMP
    int num_threads = 0;              // 0 = use OpenMP runtime default
    bool verbose = true;              // Print relative residual per iteration
};

struct SolverStats {
    int iterations = 0;
    double final_rel_residual = 0.0;
    bool converged = false;
};

class MultigridSolver {
public:
    explicit MultigridSolver(SolverConfig config = {});

    SolverStats solve(MultigridHierarchy& hierarchy);

private:
    SolverConfig cfg_;

    SolverStats solveStandaloneVCycle(MultigridHierarchy& hierarchy);
    SolverStats solveMGCG(MultigridHierarchy& hierarchy);

    void vCycle(MultigridHierarchy& hierarchy, int level_idx);
    void smoothSOR(GridLevel& lvl, int num_sweeps, bool reverse_order);
    void computeResidual(GridLevel& lvl);
    void applyOperator(const Grid3D& x, const Grid3D& sigma, Grid3D& Ax);
    void restrictResidual(const GridLevel& fine, GridLevel& coarse);
    void prolongateAndCorrect(const GridLevel& coarse, GridLevel& fine);
};