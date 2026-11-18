#pragma once
#include "../grid/Grid3D.h"
#include <array>
#include <string>
#include <vector>

struct MagneticDipole {
    std::array<double, 3> pos_m;       // Position in coil-local frame (meters)
    std::array<double, 3> moment_Am2;  // Magnetic dipole moment per unit current (m^2)
};

class CoilModel {
public:
    explicit CoilModel(std::vector<MagneticDipole> dipoles);

    // Idealized figure-8 coil: two touching loops (default 5 cm diameter, 314 dipoles/loop)
    static CoilModel createFigure8Coil(
        double loop_radius_m = 0.025,
        int dipoles_per_loop = 314
    );

    // Multi-ring 70 mm figure-8 coil approximation (inner/outer winding radii 28-43 mm)
    static CoilModel createMagstim70Coil(int num_rings = 5, int dipoles_per_ring = 128);

    // Single circular loop coil
    static CoilModel createCircularCoil(
        double loop_radius_m = 0.045,
        int num_dipoles = 314
    );

    // Single magnetic dipole at coil origin
    static CoilModel createSingleDipole(
        std::array<double, 3> moment_Am2 = {1.0, 0.0, 0.0}
    );

    // Loads a magnetic dipole array from a .ccd file (rows of: x y z mx my mz)
    static CoilModel loadFromCcdFile(const std::string& filepath);

    // Builds a 3x3 row-major rotation matrix from Euler angles (rx, ry, rz) in degrees
    static std::array<double, 9> rotationFromEulerDeg(
        double rx_deg, double ry_deg, double rz_deg
    );

    // Builds a 3x3 row-major rotation matrix aligning coil +Z with outward_normal
    // and rotating the figure-8 handle in the tangent plane by yaw_deg
    static std::array<double, 9> rotationFromNormalAndYaw(
        const std::array<double, 3>& outward_normal,
        double yaw_deg = 45.0
    );

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