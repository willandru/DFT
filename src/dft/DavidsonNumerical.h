#pragma once

#include "CartesianGrid.h"
#include "DFTData.h"

#include <cstddef>
#include <vector>

void davidsonOrthogonalizeAgainstOrbitals(
    std::vector<double>& v,
    const std::vector<MolecularOrbital>& orbitals,
    double dV
);

std::vector<double> davidsonBuildInitialVector(
    const CartesianGrid& grid,
    std::size_t orbitalIndex
);

std::vector<double> davidsonBuildHamiltonianDiagonal(
    const CartesianGrid& grid,
    const std::vector<double>& potential
);

std::vector<double> davidsonBuildCorrection(
    const std::vector<double>& residual,
    const std::vector<double>& diagonal,
    double eigenvalue
);

bool davidsonBuildIndependentVector(
    const CartesianGrid& grid,
    std::size_t orbitalIndex,
    const std::vector<MolecularOrbital>& orbitals,
    double dV,
    std::vector<double>& result
);

bool davidsonBuildInitialFromPrevious(
    const CartesianGrid& grid,
    const MolecularOrbital& initial,
    const std::vector<MolecularOrbital>& orbitals,
    double dV,
    std::vector<double>& result
);

bool davidsonAppendIndependentCorrection(
    std::vector<double>& correction,
    const std::vector<MolecularOrbital>& previousOrbitals,
    const std::vector<std::vector<double>>& basis,
    double dV,
    double subspaceTolerance
);