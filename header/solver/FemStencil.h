#pragma once
#include "../grid/Grid3D.h"

namespace FemStencil {

    // Evaluates A*phi and diagonal term at interior node (i, j, k)
    // Returns {A_phi_val, diag_coeff} where diag_coeff = (8/3) * sigma_12345678 / h^2
    inline void evaluateNode(
        const Grid3D& phi,
        const Grid3D& sigma,
        int i, int j, int k,
        double& out_off_diag_sum, // -(1/6)*sum(sigma_ab*phi_ab) - (1/12)*sum(sigma_a*phi_a)
        double& out_diag_weight   // (8/3)*sigma_12345678
    ) {
        // Voxel conductivities around node (i, j, k): index = dx + 2*dy + 4*dz
        const double s1 = sigma(i - 1, j - 1, k - 1);
        const double s2 = sigma(i,     j - 1, k - 1);
        const double s3 = sigma(i - 1, j,     k - 1);
        const double s4 = sigma(i,     j,     k - 1);
        const double s5 = sigma(i - 1, j - 1, k);
        const double s6 = sigma(i,     j - 1, k);
        const double s7 = sigma(i - 1, j,     k);
        const double s8 = sigma(i,     j,     k);

        const double s_all = 0.125 * (s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8);
        out_diag_weight = (8.0 / 3.0) * s_all;

        // 12 edge-adjacent nodes (arithmetic average sigma_ab = 0.5 * (sa + sb))
        const double edge_sum =
            0.5 * (s1 + s2) * phi(i,     j - 1, k - 1) + // phi_12
            0.5 * (s3 + s4) * phi(i,     j + 1, k - 1) + // phi_34
            0.5 * (s5 + s6) * phi(i,     j - 1, k + 1) + // phi_56
            0.5 * (s7 + s8) * phi(i,     j + 1, k + 1) + // phi_78
            0.5 * (s1 + s3) * phi(i - 1, j,     k - 1) + // phi_13
            0.5 * (s2 + s4) * phi(i + 1, j,     k - 1) + // phi_24
            0.5 * (s5 + s7) * phi(i - 1, j,     k + 1) + // phi_57
            0.5 * (s6 + s8) * phi(i + 1, j,     k + 1) + // phi_68
            0.5 * (s1 + s5) * phi(i - 1, j - 1, k)     + // phi_15
            0.5 * (s2 + s6) * phi(i + 1, j - 1, k)     + // phi_26
            0.5 * (s3 + s7) * phi(i - 1, j + 1, k)     + // phi_37
            0.5 * (s4 + s8) * phi(i + 1, j + 1, k);      // phi_48

        // 8 corner-adjacent nodes
        const double corner_sum =
            s1 * phi(i - 1, j - 1, k - 1) +
            s2 * phi(i + 1, j - 1, k - 1) +
            s3 * phi(i - 1, j + 1, k - 1) +
            s4 * phi(i + 1, j + 1, k - 1) +
            s5 * phi(i - 1, j - 1, k + 1) +
            s6 * phi(i + 1, j - 1, k + 1) +
            s7 * phi(i - 1, j + 1, k + 1) +
            s8 * phi(i + 1, j + 1, k + 1);

        out_off_diag_sum = (1.0 / 6.0) * edge_sum + (1.0 / 12.0) * corner_sum;
    }

    // Mid-ordinate quadrature for f = integral(sigma * A0_prime . grad(psi) dV)
    double assembleNodalSource(
        const Grid3D& sigma,
        const Vec3Grid& A0_prime,
        int i, int j, int k
    );
}