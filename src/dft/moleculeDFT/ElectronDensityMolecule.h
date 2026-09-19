#pragma once

#include "DFTData.h"

#include <vector>

std::vector<double> calculateMolecularSpinDensity(
    const std::vector<MolecularOrbital>& orbitals,
    SpinChannel spin
);

std::vector<double> calculateMolecularElectronDensity(
    const std::vector<MolecularOrbital>& orbitals
);

double integrateMolecularElectronDensity(
    const std::vector<double>& density,
    double dx,
    double dy,
    double dz
);