#pragma once

#include "DFTData.h"

#include <vector>

std::vector<double> calculateSpinDensity(
    const std::vector<double>& r,
    const std::vector<AtomicOrbital>& orbitals,
    SpinChannel spin
);

std::vector<double> calculateElectronDensity(
    const std::vector<double>& r,
    const std::vector<AtomicOrbital>& orbitals
);

double integrateElectronDensity(
    const std::vector<double>& r,
    const std::vector<double>& density
);

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