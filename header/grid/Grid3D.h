#pragma once
#include <vector>
#include <cstddef>
#include <algorithm>
#include <cmath>

class Grid3D {
public:
    Grid3D() = default;
    Grid3D(int nx, int ny, int nz, double h, double init_val = 0.0)
        : nx_(nx), ny_(ny), nz_(nz), h_(h),
          data_(static_cast<size_t>(nx) * ny * nz, init_val) {}

    inline double& operator()(int i, int j, int k) {
        return data_[static_cast<size_t>(i) + nx_ * (static_cast<size_t>(j) + ny_ * k)];
    }

    inline const double& operator()(int i, int j, int k) const {
        return data_[static_cast<size_t>(i) + nx_ * (static_cast<size_t>(j) + ny_ * k)];
    }

    void fill(double val) { std::fill(data_.begin(), data_.end(), val); }

    double l2Norm() const {
        double sum = 0.0;
        for (double v : data_) sum += v * v;
        return std::sqrt(sum);
    }

    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int nz() const { return nz_; }
    double h() const { return h_; }
    size_t size() const { return data_.size(); }
    double* raw() { return data_.data(); }
    const double* raw() const { return data_.data(); }

private:
    int nx_ = 0, ny_ = 0, nz_ = 0;
    double h_ = 1.0;
    std::vector<double> data_;
};

struct Vec3Grid {
    Grid3D x, y, z;
    Vec3Grid() = default;
    Vec3Grid(int nx, int ny, int nz, double h)
        : x(nx, ny, nz, h, 0.0), y(nx, ny, nz, h, 0.0), z(nx, ny, nz, h, 0.0) {}
};