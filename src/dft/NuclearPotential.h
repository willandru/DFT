#pragma once

#include "CartesianGrid.h"
#include "Molecule.h"

#include <cstddef>
#include <vector>

class NuclearPotential
{
public:
    static std::vector<double> calculate(
        const Molecule& molecule,
        const CartesianGrid& grid
    );

    static double calculateAtPoint(
        const Molecule& molecule,
        double x,
        double y,
        double z
    );
};