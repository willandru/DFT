#pragma once

#include <vector>

std::vector<double> calculateSpinExchangeCorrelationPotential(
    const std::vector<double>& alphaDensity,
    const std::vector<double>& betaDensity,
    int spin
);

double calculateSpinExchangeCorrelationEnergy(
    const std::vector<double>& r,
    const std::vector<double>& alphaDensity,
    const std::vector<double>& betaDensity
);