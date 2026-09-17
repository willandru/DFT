#pragma once

#include "CartesianGrid.h"
#include "DFTData.h"

#include <vector>

TridiagonalMatrix buildKohnShamHamiltonian(
    const std::vector<double>& r,
    const std::vector<double>& effectivePotential,
    int l
);

std::vector<double> applyMolecularKohnShamHamiltonian(
    const CartesianGrid& grid,
    const std::vector<double>& effectivePotential,
    const std::vector<double>& psi
);