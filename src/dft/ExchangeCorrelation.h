#pragma once

#include "XCFunctional.h"

#include <vector>

std::vector<double> calculateSpinExchangeCorrelationPotential(
    const XCFunctional& functional,
    const std::vector<double>& r,
    const std::vector<double>& alphaDensity,
    const std::vector<double>& betaDensity,
    int spin
);

double calculateSpinExchangeCorrelationEnergy(
    const XCFunctional& functional,
    const std::vector<double>& r,
    const std::vector<double>& alphaDensity,
    const std::vector<double>& betaDensity
);