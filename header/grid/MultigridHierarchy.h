#pragma once
#include "Grid3D.h"
#include <vector>

struct GridLevel {
    int level_idx = 0;
    double h = 1.0;
    Grid3D sigma; // Voxel-centered: [nx, ny, nz]
    Grid3D phi;   // Node-centered:  [nx + 1, ny + 1, nz + 1]
    Grid3D r;     // Node-centered:  [nx + 1, ny + 1, nz + 1]
    Grid3D f;     // Node-centered:  [nx + 1, ny + 1, nz + 1]
};

class MultigridHierarchy {
public:
    MultigridHierarchy(int base_nx, int base_ny, int base_nz, double base_h, int num_levels);

    void buildCoarseConductivities();

    int numLevels() const { return static_cast<int>(levels_.size()); }
    GridLevel& level(int l) { return levels_[l]; }
    const GridLevel& level(int l) const { return levels_[l]; }

private:
    std::vector<GridLevel> levels_;
};