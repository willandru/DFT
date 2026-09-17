#pragma once

#include "CartesianGrid.h"
#include "DFTData.h"

#include <cstddef>
#include <vector>

std::size_t countEigenvaluesBelow(
    const TridiagonalMatrix& matrix,
    double energy
);

double findEigenvalue(
    const TridiagonalMatrix& matrix,
    std::size_t index,
    double lowerBound,
    double upperBound
);

std::vector<double> solveEigenvector(
    const TridiagonalMatrix& matrix,
    double eigenvalue,
    std::size_t maxIterations = 1000
);

AtomicOrbital solveOrbital(
    const TridiagonalMatrix& matrix,
    const std::vector<double>& r,
    int n,
    int l,
    int electrons,
    std::size_t orbitalIndex
);

MolecularOrbital solveMolecularOrbital(
    const CartesianGrid& grid,
    const std::vector<double>& effectivePotential,
    std::size_t orbitalIndex,
    SpinChannel spin,
    int electrons,
    const std::vector<MolecularOrbital>& previousOrbitals,
    std::size_t maxIterations = 5000
);

std::vector<MolecularOrbital> solveMolecularOrbitals(
    const CartesianGrid& grid,
    const std::vector<double>& effectivePotential,
    std::size_t numberOfOrbitals,
    const std::vector<int>& occupations,
    SpinChannel spin,
    std::size_t maxIterations = 5000
);