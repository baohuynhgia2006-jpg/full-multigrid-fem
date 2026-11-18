#include "grid/MultigridHierarchy.h"
#include <stdexcept>

MultigridHierarchy::MultigridHierarchy(
    int base_nx, int base_ny, int base_nz, double base_h, int num_levels
) {
    if (num_levels < 1) {
        throw std::invalid_argument("num_levels must be >= 1");
    }

    const int divisor = 1 << (num_levels - 1);
    if (base_nx % divisor != 0 || base_ny % divisor != 0 || base_nz % divisor != 0) {
        throw std::invalid_argument(
            "Base grid voxel dimensions must be divisible by 2^(num_levels - 1)"
        );
    }

    levels_.resize(num_levels);
    int nx = base_nx;
    int ny = base_ny;
    int nz = base_nz;
    double h = base_h;

    for (int l = 0; l < num_levels; ++l) {
        levels_[l].level_idx = l;
        levels_[l].h = h;
        levels_[l].sigma = Grid3D(nx, ny, nz, h, 0.0);
        levels_[l].phi   = Grid3D(nx + 1, ny + 1, nz + 1, h, 0.0);
        levels_[l].r     = Grid3D(nx + 1, ny + 1, nz + 1, h, 0.0);
        levels_[l].f     = Grid3D(nx + 1, ny + 1, nz + 1, h, 0.0);

        nx /= 2;
        ny /= 2;
        nz /= 2;
        h *= 2.0;
    }
}

void MultigridHierarchy::buildCoarseConductivities() {
    for (int l = 0; l + 1 < numLevels(); ++l) {
        const Grid3D& fine = levels_[l].sigma;
        Grid3D& coarse = levels_[l + 1].sigma;

        const int cnx = coarse.nx();
        const int cny = coarse.ny();
        const int cnz = coarse.nz();

        for (int k = 0; k < cnz; ++k) {
            const int fk = 2 * k;
            for (int j = 0; j < cny; ++j) {
                const int fj = 2 * j;
                for (int i = 0; i < cnx; ++i) {
                    const int fi = 2 * i;
                    const double sum =
                        fine(fi,     fj,     fk)     +
                        fine(fi + 1, fj,     fk)     +
                        fine(fi,     fj + 1, fk)     +
                        fine(fi + 1, fj + 1, fk)     +
                        fine(fi,     fj,     fk + 1) +
                        fine(fi + 1, fj,     fk + 1) +
                        fine(fi,     fj + 1, fk + 1) +
                        fine(fi + 1, fj + 1, fk + 1);

                    coarse(i, j, k) = 0.125 * sum;
                }
            }
        }
    }
}