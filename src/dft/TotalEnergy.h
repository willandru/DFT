#pragma once

#include "DFTData.h"
#include "XCFunctional.h"

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
    const XCFunctional& functional,
    const std::vector<double>& r,
    const std::vector<double>& alphaDensity,
    const std::vector<double>& betaDensity,
    const std::vector<AtomicOrbital>& orbitals,
    const std::vector<double>& hartreePotential,
    int Z
);