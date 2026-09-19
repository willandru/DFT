#pragma once

#include "CartesianGrid.h"

#include <vector>

std::vector<double> applyMolecularKohnShamHamiltonian(
    const CartesianGrid& grid,
    const std::vector<double>& effectivePotential,
    const std::vector<double>& psi
);