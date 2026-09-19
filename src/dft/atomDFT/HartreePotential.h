#pragma once

#include <vector>

std::vector<double> calculateHartreePotential(
    const std::vector<double>& r,
    const std::vector<double>& density
);

double calculateHartreeEnergy(
    const std::vector<double>& r,
    const std::vector<double>& density,
    const std::vector<double>& hartreePotential
);