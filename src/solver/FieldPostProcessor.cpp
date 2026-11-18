// src/solver/FieldPostProcessor.cpp
#include "solver/FieldPostProcessor.h"

namespace FieldPostProcessor {

Vec3Grid computeElectricField(
    const Grid3D& phi,
    const Grid3D& sigma,
    const Vec3Grid& A0_prime
) {
    const int nx = sigma.nx();
    const int ny = sigma.ny();
    const int nz = sigma.nz();
    const double h = sigma.h();
    const double inv_4h = 0.25 / h;

    Vec3Grid E(nx, ny, nz, h);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (sigma(i, j, k) <= 0.0) continue;

                // Trilinear gradient evaluated at voxel center (0.5, 0.5, 0.5)
                const double dphi_dx = inv_4h * (
                    (phi(i + 1, j,     k)     - phi(i, j,     k))     +
                    (phi(i + 1, j + 1, k)     - phi(i, j + 1, k))     +
                    (phi(i + 1, j,     k + 1) - phi(i, j,     k + 1)) +
                    (phi(i + 1, j + 1, k + 1) - phi(i, j + 1, k + 1))
                );

                const double dphi_dy = inv_4h * (
                    (phi(i,     j + 1, k)     - phi(i,     j, k))     +
                    (phi(i + 1, j + 1, k)     - phi(i + 1, j, k))     +
                    (phi(i,     j + 1, k + 1) - phi(i,     j, k + 1)) +
                    (phi(i + 1, j + 1, k + 1) - phi(i + 1, j, k + 1))
                );

                const double dphi_dz = inv_4h * (
                    (phi(i,     j,     k + 1) - phi(i,     j,     k)) +
                    (phi(i + 1, j,     k + 1) - phi(i + 1, j,     k)) +
                    (phi(i,     j + 1, k + 1) - phi(i,     j + 1, k)) +
                    (phi(i + 1, j + 1, k + 1) - phi(i + 1, j + 1, k))
                );

                E.x(i, j, k) = -dphi_dx - A0_prime.x(i, j, k);
                E.y(i, j, k) = -dphi_dy - A0_prime.y(i, j, k);
                E.z(i, j, k) = -dphi_dz - A0_prime.z(i, j, k);
            }
        }
    }

    return E;
}

} // namespace FieldPostProcessor