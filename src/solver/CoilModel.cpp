#include "solver/CoilModel.h"
#include <cmath>
#include <utility>

CoilModel::CoilModel(std::vector<MagneticDipole> dipoles)
    : dipoles_(std::move(dipoles)) {}

CoilModel CoilModel::createFigure8Coil(double loop_radius_m, int dipoles_per_loop) {
    constexpr double kPi = 3.14159265358979323846;
    const double area = kPi * loop_radius_m * loop_radius_m;
    const double sub_moment = area / static_cast<double>(dipoles_per_loop);

    std::vector<MagneticDipole> dipoles;
    dipoles.reserve(static_cast<size_t>(2 * dipoles_per_loop));

    // Left loop (-x, +z moment) and right loop (+x, -z moment)
    for (int loop = 0; loop < 2; ++loop) {
        const double center_x = (loop == 0) ? -loop_radius_m : loop_radius_m;
        const double polarity = (loop == 0) ? 1.0 : -1.0;

        for (int n = 0; n < dipoles_per_loop; ++n) {
            const double theta = (2.0 * kPi * n) / dipoles_per_loop;
            const double x = center_x + loop_radius_m * std::cos(theta);
            const double y = loop_radius_m * std::sin(theta);
            dipoles.push_back({{x, y, 0.0}, {0.0, 0.0, polarity * sub_moment}});
        }
    }

    return CoilModel(std::move(dipoles));
}

CoilModel CoilModel::createSingleDipole(std::array<double, 3> moment_Am2) {
    return CoilModel({{{0.0, 0.0, 0.0}, moment_Am2}});
}

void CoilModel::evaluateA0Prime(
    Vec3Grid& A0_prime,
    double dI_dt,
    const std::array<double, 3>& grid_origin_m,
    const std::array<double, 3>& coil_pos_m,
    const std::array<double, 9>& R
) const {
    // Transform dipoles into world coordinates once
    std::vector<MagneticDipole> world_dipoles(dipoles_.size());
    for (size_t d = 0; d < dipoles_.size(); ++d) {
        const auto& p = dipoles_[d].pos_m;
        const auto& m = dipoles_[d].moment_Am2;

        world_dipoles[d].pos_m = {
            coil_pos_m[0] + R[0] * p[0] + R[1] * p[1] + R[2] * p[2],
            coil_pos_m[1] + R[3] * p[0] + R[4] * p[1] + R[5] * p[2],
            coil_pos_m[2] + R[6] * p[0] + R[7] * p[1] + R[8] * p[2]
        };
        world_dipoles[d].moment_Am2 = {
            R[0] * m[0] + R[1] * m[1] + R[2] * m[2],
            R[3] * m[0] + R[4] * m[1] + R[5] * m[2],
            R[6] * m[0] + R[7] * m[1] + R[8] * m[2]
        };
    }

    // mu0 / (4 * pi) = 1e-7 H/m
    const double scale = 1e-7 * dI_dt;
    const int nx = A0_prime.x.nx();
    const int ny = A0_prime.x.ny();
    const int nz = A0_prime.x.nz();
    const double h = A0_prime.x.h();

    for (int k = 0; k < nz; ++k) {
        const double rz = grid_origin_m[2] + (k + 0.5) * h;
        for (int j = 0; j < ny; ++j) {
            const double ry = grid_origin_m[1] + (j + 0.5) * h;
            for (int i = 0; i < nx; ++i) {
                const double rx = grid_origin_m[0] + (i + 0.5) * h;

                double ax = 0.0, ay = 0.0, az = 0.0;
                for (const auto& dip : world_dipoles) {
                    const double dx = rx - dip.pos_m[0];
                    const double dy = ry - dip.pos_m[1];
                    const double dz = rz - dip.pos_m[2];

                    const double r2 = dx * dx + dy * dy + dz * dz;
                    const double inv_r3 = 1.0 / (r2 * std::sqrt(r2));

                    // m x (r - r_d)
                    const double cx = dip.moment_Am2[1] * dz - dip.moment_Am2[2] * dy;
                    const double cy = dip.moment_Am2[2] * dx - dip.moment_Am2[0] * dz;
                    const double cz = dip.moment_Am2[0] * dy - dip.moment_Am2[1] * dx;

                    ax += cx * inv_r3;
                    ay += cy * inv_r3;
                    az += cz * inv_r3;
                }

                A0_prime.x(i, j, k) = scale * ax;
                A0_prime.y(i, j, k) = scale * ay;
                A0_prime.z(i, j, k) = scale * az;
            }
        }
    }
}