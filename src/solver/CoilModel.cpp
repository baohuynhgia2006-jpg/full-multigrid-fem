#include "solver/CoilModel.h"
#include <cmath>
#include <utility>
#include <fstream>
#include <sstream>
#include <stdexcept>

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

CoilModel CoilModel::createMagstim70Coil(int num_rings, int dipoles_per_ring) {
    constexpr double kPi = 3.14159265358979323846;
    constexpr double r_inner = 0.028;
    constexpr double r_outer = 0.043;
    constexpr double wing_offset = 0.044;

    std::vector<MagneticDipole> dipoles;
    dipoles.reserve(static_cast<size_t>(2 * num_rings * dipoles_per_ring));

    for (int loop = 0; loop < 2; ++loop) {
        const double center_x = (loop == 0) ? -wing_offset : wing_offset;
        const double polarity = (loop == 0) ? 1.0 : -1.0;

        for (int ring = 0; ring < num_rings; ++ring) {
            const double t = (num_rings > 1)
                ? static_cast<double>(ring) / static_cast<double>(num_rings - 1)
                : 0.5;
            const double r = r_inner + t * (r_outer - r_inner);
            const double area = kPi * r * r;
            const double sub_moment = area / static_cast<double>(dipoles_per_ring);

            for (int n = 0; n < dipoles_per_ring; ++n) {
                const double theta = (2.0 * kPi * n) / dipoles_per_ring;
                const double x = center_x + r * std::cos(theta);
                const double y = r * std::sin(theta);
                dipoles.push_back({{x, y, 0.0}, {0.0, 0.0, polarity * sub_moment}});
            }
        }
    }

    return CoilModel(std::move(dipoles));
}

CoilModel CoilModel::createCircularCoil(double loop_radius_m, int num_dipoles) {
    constexpr double kPi = 3.14159265358979323846;
    const double area = kPi * loop_radius_m * loop_radius_m;
    const double sub_moment = area / static_cast<double>(num_dipoles);

    std::vector<MagneticDipole> dipoles;
    dipoles.reserve(static_cast<size_t>(num_dipoles));

    for (int n = 0; n < num_dipoles; ++n) {
        const double theta = (2.0 * kPi * n) / num_dipoles;
        dipoles.push_back({
            {loop_radius_m * std::cos(theta), loop_radius_m * std::sin(theta), 0.0},
            {0.0, 0.0, sub_moment}
        });
    }

    return CoilModel(std::move(dipoles));
}

CoilModel CoilModel::loadFromCcdFile(const std::string& filepath) {
    std::ifstream ifs(filepath);
    if (!ifs) {
        throw std::runtime_error("Failed to open coil .ccd file: " + filepath);
    }

    std::vector<MagneticDipole> dipoles;
    std::string line;
    bool skipped_count_header = false;

    while (std::getline(ifs, line)) {
        // Strip comments starting with '#' or '%'
        const size_t comment_pos = line.find_first_of("#%");
        if (comment_pos != std::string::npos) {
            line.erase(comment_pos);
        }

        std::istringstream iss(line);
        std::vector<double> vals;
        double v = 0.0;
        while (iss >> v) {
            vals.push_back(v);
        }

        if (vals.empty()) continue;

        // Standard SimNIBS .ccd files have a single integer line giving the dipole count
        if (vals.size() == 1 && !skipped_count_header && dipoles.empty()) {
            skipped_count_header = true;
            dipoles.reserve(static_cast<size_t>(std::max(0.0, vals[0])));
            continue;
        }

        if (vals.size() >= 6) {
            dipoles.push_back({
                {vals[0], vals[1], vals[2]},
                {vals[3], vals[4], vals[5]}
            });
        }
    }

    if (dipoles.empty()) {
        throw std::runtime_error("No valid dipole rows (x y z mx my mz) found in: " + filepath);
    }

    return CoilModel(std::move(dipoles));
}

std::array<double, 9> CoilModel::rotationFromEulerDeg(
    double rx_deg, double ry_deg, double rz_deg
) {
    constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
    const double ax = rx_deg * kDegToRad;
    const double ay = ry_deg * kDegToRad;
    const double az = rz_deg * kDegToRad;

    const double cx = std::cos(ax), sx = std::sin(ax);
    const double cy = std::cos(ay), sy = std::sin(ay);
    const double cz = std::cos(az), sz = std::sin(az);

    // R = Rz(az) * Ry(ay) * Rx(ax)
    return {
        cz * cy,  cz * sy * sx - sz * cx,  cz * sy * cx + sz * sx,
        sz * cy,  sz * sy * sx + cz * cx,  sz * sy * cx - cz * sx,
        -sy,      cy * sx,                 cy * cx
    };
}

std::array<double, 9> CoilModel::rotationFromNormalAndYaw(
    const std::array<double, 3>& outward_normal,
    double yaw_deg
) {
    constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
    const double n_len = std::sqrt(
        outward_normal[0] * outward_normal[0] +
        outward_normal[1] * outward_normal[1] +
        outward_normal[2] * outward_normal[2]
    );
    std::array<double, 3> z_axis = {0.0, 0.0, 1.0};
    if (n_len > 1e-12) {
        z_axis = {
            outward_normal[0] / n_len,
            outward_normal[1] / n_len,
            outward_normal[2] / n_len
        };
    }

    // Pick reference vector not parallel to z_axis to construct tangent basis
    std::array<double, 3> ref = (std::abs(z_axis[1]) < 0.9)
        ? std::array<double, 3>{0.0, 1.0, 0.0}
        : std::array<double, 3>{1.0, 0.0, 0.0};

    // x0 = normalize(ref x z_axis)
    std::array<double, 3> x0 = {
        ref[1] * z_axis[2] - ref[2] * z_axis[1],
        ref[2] * z_axis[0] - ref[0] * z_axis[2],
        ref[0] * z_axis[1] - ref[1] * z_axis[0]
    };
    const double x0_len = std::sqrt(x0[0] * x0[0] + x0[1] * x0[1] + x0[2] * x0[2]);
    x0[0] /= x0_len; x0[1] /= x0_len; x0[2] /= x0_len;

    // y0 = z_axis x x0
    const std::array<double, 3> y0 = {
        z_axis[1] * x0[2] - z_axis[2] * x0[1],
        z_axis[2] * x0[0] - z_axis[0] * x0[2],
        z_axis[0] * x0[1] - z_axis[1] * x0[0]
    };

    const double yaw = yaw_deg * kDegToRad;
    const double cy = std::cos(yaw);
    const double sy = std::sin(yaw);

    const std::array<double, 3> x_axis = {
        cy * x0[0] + sy * y0[0],
        cy * x0[1] + sy * y0[1],
        cy * x0[2] + sy * y0[2]
    };
    const std::array<double, 3> y_axis = {
        -sy * x0[0] + cy * y0[0],
        -sy * x0[1] + cy * y0[1],
        -sy * x0[2] + cy * y0[2]
    };

    // Columns are [x_axis, y_axis, z_axis] stored in row-major order
    return {
        x_axis[0], y_axis[0], z_axis[0],
        x_axis[1], y_axis[1], z_axis[1],
        x_axis[2], y_axis[2], z_axis[2]
    };
}