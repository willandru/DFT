#pragma once

#include "DFTData.h"

#include <vector>

TridiagonalMatrix buildKohnShamHamiltonian(
    const std::vector<double>& r,
    const std::vector<double>& effectivePotential,
    int l
);