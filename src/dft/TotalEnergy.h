#pragma once

#include "DFTData.h"

#include <vector>

double calculateKineticEnergy(
    const std::vector<double>& r,
    const std::vector<AtomicOrbital>& orbitals
);

double calculateExternalEnergy(
    const std::vector<double>& r,
    const std::vector<double>& density,
    int Z
);

EnergyComponents calculateTotalEnergy(
    const std::vector<double>& r,
    const std::vector<double>& density,
    const std::vector<AtomicOrbital>& orbitals,
    const std::vector<double>& hartreePotential,
    int Z
);