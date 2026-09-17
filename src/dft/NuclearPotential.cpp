#include "NuclearPotential.h"

#include <cmath>
#include <stdexcept>

std::vector<double> NuclearPotential::calculate(
    const Molecule& molecule,
    const CartesianGrid& grid
)
{
    std::vector<double> potential(grid.getSize(), 0.0);

    for (std::size_t k = 0; k < grid.getNz(); ++k)
    {
        const double z = grid.getZ(k);

        for (std::size_t j = 0; j < grid.getNy(); ++j)
        {
            const double y = grid.getY(j);

            for (std::size_t i = 0; i < grid.getNx(); ++i)
            {
                const double x = grid.getX(i);

                const std::size_t index = grid.getIndex(i, j, k);

                potential[index] =
                    calculateAtPoint(molecule, x, y, z);
            }
        }
    }

    return potential;
}

double NuclearPotential::calculateAtPoint(
    const Molecule& molecule,
    double x,
    double y,
    double z
)
{
    double potential = 0.0;

    for (const Molecule::Nucleus& nucleus : molecule.getNuclei())
    {
        const double dx =
            x - nucleus.position[0];

        const double dy =
            y - nucleus.position[1];

        const double dz =
            z - nucleus.position[2];

        const double distanceSquared =
            dx * dx +
            dy * dy +
            dz * dz;

        if (distanceSquared == 0.0)
        {
            throw std::runtime_error(
                "NuclearPotential is singular at a nuclear position."
            );
        }

        const double distance =
            std::sqrt(distanceSquared);

        potential -=
            static_cast<double>(nucleus.atomicNumber) / distance;
    }

    return potential;
}