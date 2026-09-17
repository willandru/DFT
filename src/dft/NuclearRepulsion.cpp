#include "NuclearRepulsion.h"

#include <cmath>
#include <stdexcept>

double NuclearRepulsion::calculate(const Molecule& molecule)
{
    const auto& nuclei = molecule.getNuclei();

    double energy = 0.0;

    for (std::size_t i = 0; i < nuclei.size(); ++i)
    {
        for (std::size_t j = i + 1; j < nuclei.size(); ++j)
        {
            const Molecule::Nucleus& nucleusA = nuclei[i];
            const Molecule::Nucleus& nucleusB = nuclei[j];

            const double dx =
                nucleusA.position[0] -
                nucleusB.position[0];

            const double dy =
                nucleusA.position[1] -
                nucleusB.position[1];

            const double dz =
                nucleusA.position[2] -
                nucleusB.position[2];

            const double distanceSquared =
                dx * dx +
                dy * dy +
                dz * dz;

            if (distanceSquared == 0.0)
            {
                throw std::runtime_error(
                    "NuclearRepulsion is undefined for coincident nuclei."
                );
            }

            const double distance =
                std::sqrt(distanceSquared);

            energy +=
                static_cast<double>(nucleusA.atomicNumber) *
                static_cast<double>(nucleusB.atomicNumber) /
                distance;
        }
    }

    return energy;
}