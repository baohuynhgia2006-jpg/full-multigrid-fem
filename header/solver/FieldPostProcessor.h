// header/solver/FieldPostProcessor.h
#pragma once
#include "../grid/Grid3D.h"

namespace FieldPostProcessor {

Vec3Grid computeElectricField(
    const Grid3D& phi,
    const Grid3D& sigma,
    const Vec3Grid& A0_prime
);

} // namespace FieldPostProcessor