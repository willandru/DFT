#pragma once

#include "CartesianGrid.h"
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

std::vector<double> calculateMolecularSpinExchangeCorrelationPotential(
    const XCFunctional& functional,
    const CartesianGrid& grid,
    const std::vector<double>& alphaDensity,
    const std::vector<double>& betaDensity,
    int spin
);

double calculateMolecularExchangeCorrelationEnergy(
    const XCFunctional& functional,
    const CartesianGrid& grid,
    const std::vector<double>& alphaDensity,
    const std::vector<double>& betaDensity
);