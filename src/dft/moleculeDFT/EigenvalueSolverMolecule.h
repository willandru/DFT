#pragma once

#include "CartesianGrid.h"
#include "DFTData.h"

#include <cstddef>
#include <vector>

MolecularOrbital solveMolecularOrbital(
    const CartesianGrid& grid,
    const std::vector<double>& effectivePotential,
    std::size_t orbitalIndex,
    SpinChannel spin,
    int electrons,
    const std::vector<MolecularOrbital>& previousOrbitals,
    const MolecularOrbital* initialGuess = nullptr,
    std::size_t maxIterations = 5000
);

std::vector<MolecularOrbital> solveMolecularOrbitals(
    const CartesianGrid& grid,
    const std::vector<double>& effectivePotential,
    std::size_t numberOfOrbitals,
    const std::vector<int>& occupations,
    SpinChannel spin,
    const std::vector<MolecularOrbital>& initialOrbitals = {},
    std::size_t maxIterations = 5000
);