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