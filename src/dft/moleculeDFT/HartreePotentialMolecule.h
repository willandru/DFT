#pragma once

#include "CartesianGrid.h"

#include <vector>

std::vector<double> calculateHartreePotential(
    const CartesianGrid& grid,
    const std::vector<double>& density
);

double calculateHartreeEnergy(
    const CartesianGrid& grid,
    const std::vector<double>& density,
    const std::vector<double>& hartreePotential
);