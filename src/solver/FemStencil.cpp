#include "solver/FemStencil.h"

namespace FemStencil {

double assembleNodalSource(
    const Grid3D& sigma,
    const Vec3Grid& A0_prime,
    int i, int j, int k
) {
    const double h = sigma.h();
    const double inv_4h = 0.25 / h;
    double sum = 0.0;

    for (int dz = -1; dz <= 0; ++dz) {
        const double sz = (dz == -1) ? 1.0 : -1.0;
        const int vz = k + dz;

        for (int dy = -1; dy <= 0; ++dy) {
            const double sy = (dy == -1) ? 1.0 : -1.0;
            const int vy = j + dy;

            for (int dx = -1; dx <= 0; ++dx) {
                const double sx = (dx == -1) ? 1.0 : -1.0;
                const int vx = i + dx;

                const double sig = sigma(vx, vy, vz);
                if (sig == 0.0) continue;

                const double dot =
                    sx * A0_prime.x(vx, vy, vz) +
                    sy * A0_prime.y(vx, vy, vz) +
                    sz * A0_prime.z(vx, vy, vz);

                sum += sig * dot;
            }
        }
    }

    // Weak form: int(sigma * grad(phi) . grad(psi)) = -int(sigma * A0' . grad(psi))
    return -inv_4h * sum;
}

} // namespace FemStencil