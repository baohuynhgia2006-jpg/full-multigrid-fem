#pragma once
#include "../grid/Grid3D.h"
#include <array>
#include <vector>

struct MagneticDipole {
    std::array<double, 3> pos_m;       // Position in coil-local frame (meters)
    std::array<double, 3> moment_Am2;  // Magnetic dipole moment per unit current (m^2)
};

class CoilModel {
public:
    explicit CoilModel(std::vector<MagneticDipole> dipoles);

    // Idealized figure-8 coil: two 5 cm diameter circular loops, 314 dipoles each
    static CoilModel createFigure8Coil(
        double loop_radius_m = 0.025,
        int dipoles_per_loop = 314
    );

    // Single magnetic dipole at coil origin (used for analytical sphere benchmark)
    static CoilModel createSingleDipole(
        std::array<double, 3> moment_Am2 = {1.0, 0.0, 0.0}
    );

    // Evaluates A0' = (mu0 / 4pi) * (dI/dt) * sum( m_d x (r - r_d) / |r - r_d|^3 )
    // at voxel centers r = grid_origin_m + ((i+0.5)h, (j+0.5)h, (k+0.5)h)
    void evaluateA0Prime(
        Vec3Grid& A0_prime,
        double dI_dt,
        const std::array<double, 3>& grid_origin_m,
        const std::array<double, 3>& coil_pos_m,
        const std::array<double, 9>& coil_rot_row_major = {
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        }
    ) const;

    const std::vector<MagneticDipole>& dipoles() const { return dipoles_; }

private:
    std::vector<MagneticDipole> dipoles_;
};