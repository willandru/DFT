#include "HartreePotentialMolecule.h"

#include "DFTConstants.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

constexpr std::size_t MAX_POISSON_ITERATIONS = 10000;
constexpr double POISSON_TOLERANCE = 1.0e-8;

double calculateTotalCharge(
    const CartesianGrid& grid,
    const std::vector<double>& density
)
{
    const double volumeElement =
        grid.getDx() *
        grid.getDy() *
        grid.getDz();

    double charge = 0.0;

    for (double value : density)
    {
        charge += value;
    }

    return charge * volumeElement;
}

double calculateBoundaryPotential(
    const CartesianGrid& grid,
    const std::vector<double>& density,
    std::size_t i,
    std::size_t j,
    std::size_t k
)
{
    const double totalCharge =
        calculateTotalCharge(
            grid,
            density
        );

    const double x = grid.getX(i);
    const double y = grid.getY(j);
    const double z = grid.getZ(k);

    double minimumDistanceSquared =
        std::numeric_limits<double>::infinity();

    const double distances[6] =
    {
        std::abs(x - grid.getXMin()),
        std::abs(x - grid.getXMax()),
        std::abs(y - grid.getYMin()),
        std::abs(y - grid.getYMax()),
        std::abs(z - grid.getZMin()),
        std::abs(z - grid.getZMax())
    };

    for (double distance : distances)
    {
        minimumDistanceSquared =
            std::min(
                minimumDistanceSquared,
                distance * distance
            );
    }

    if (minimumDistanceSquared <= 0.0)
    {
        return 0.0;
    }

    const double distance =
        std::sqrt(
            x * x +
            y * y +
            z * z
        );

    if (distance <= 0.0)
    {
        return 0.0;
    }

    return totalCharge / distance;
}

} // namespace

std::vector<double> calculateHartreePotential(
    const CartesianGrid& grid,
    const std::vector<double>& density
)
{
    if (density.size() != grid.getSize())
    {
        throw std::invalid_argument(
            "La densidad molecular y la malla cartesiana deben tener el mismo tamano."
        );
    }

    if (density.empty())
    {
        throw std::invalid_argument(
            "La densidad molecular no puede estar vacia."
        );
    }

    const std::size_t nx = grid.getNx();
    const std::size_t ny = grid.getNy();
    const std::size_t nz = grid.getNz();

    const double dx = grid.getDx();
    const double dy = grid.getDy();
    const double dz = grid.getDz();

    if (dx <= 0.0 ||
        dy <= 0.0 ||
        dz <= 0.0)
    {
        throw std::invalid_argument(
            "Los pasos de la malla cartesiana deben ser mayores que cero."
        );
    }

    const double inverseDxSquared =
        1.0 / (dx * dx);

    const double inverseDySquared =
        1.0 / (dy * dy);

    const double inverseDzSquared =
        1.0 / (dz * dz);

    const double denominator =
        2.0 *
        (
            inverseDxSquared +
            inverseDySquared +
            inverseDzSquared
        );

    std::vector<double> potential(
        grid.getSize(),
        0.0
    );

    /*
     * Boundary condition:
     *
     * V_H(r) -> Q / |r|
     *
     * far from the electronic density.
     *
     * Interior points are obtained by solving
     *
     * nabla^2 V_H = -4 pi rho
     *
     * using a finite-difference Jacobi iteration.
     */

    for (std::size_t k = 0; k < nz; ++k)
    {
        for (std::size_t j = 0; j < ny; ++j)
        {
            for (std::size_t i = 0; i < nx; ++i)
            {
                if (i == 0 ||
                    i + 1 == nx ||
                    j == 0 ||
                    j + 1 == ny ||
                    k == 0 ||
                    k + 1 == nz)
                {
                    const std::size_t index =
                        grid.getIndex(
                            i,
                            j,
                            k
                        );

                    potential[index] =
                        calculateBoundaryPotential(
                            grid,
                            density,
                            i,
                            j,
                            k
                        );
                }
            }
        }
    }

    std::vector<double> nextPotential =
        potential;

    for (std::size_t iteration = 0;
         iteration < MAX_POISSON_ITERATIONS;
         ++iteration)
    {
        double maximumDifference = 0.0;

        for (std::size_t k = 1; k + 1 < nz; ++k)
        {
            for (std::size_t j = 1; j + 1 < ny; ++j)
            {
                for (std::size_t i = 1; i + 1 < nx; ++i)
                {
                    const std::size_t index =
                        grid.getIndex(
                            i,
                            j,
                            k
                        );

                    const double xMinus =
                        potential[
                            grid.getIndex(
                                i - 1,
                                j,
                                k
                            )
                        ];

                    const double xPlus =
                        potential[
                            grid.getIndex(
                                i + 1,
                                j,
                                k
                            )
                        ];

                    const double yMinus =
                        potential[
                            grid.getIndex(
                                i,
                                j - 1,
                                k
                            )
                        ];

                    const double yPlus =
                        potential[
                            grid.getIndex(
                                i,
                                j + 1,
                                k
                            )
                        ];

                    const double zMinus =
                        potential[
                            grid.getIndex(
                                i,
                                j,
                                k - 1
                            )
                        ];

                    const double zPlus =
                        potential[
                            grid.getIndex(
                                i,
                                j,
                                k + 1
                            )
                        ];

                    const double source =
                        4.0 *
                        DFTConstants::PI *
                        density[index];

                    nextPotential[index] =
                        (
                            inverseDxSquared *
                            (xMinus + xPlus) +

                            inverseDySquared *
                            (yMinus + yPlus) +

                            inverseDzSquared *
                            (zMinus + zPlus) +

                            source
                        ) /
                        denominator;

                    maximumDifference =
                        std::max(
                            maximumDifference,
                            std::abs(
                                nextPotential[index] -
                                potential[index]
                            )
                        );
                }
            }
        }

        potential.swap(
            nextPotential
        );

        if (maximumDifference <
            POISSON_TOLERANCE)
        {
            break;
        }
    }

    return potential;
}

double calculateHartreeEnergy(
    const CartesianGrid& grid,
    const std::vector<double>& density,
    const std::vector<double>& hartreePotential
)
{
    if (density.size() != grid.getSize() ||
        hartreePotential.size() != grid.getSize())
    {
        throw std::invalid_argument(
            "La densidad, el potencial de Hartree y la malla deben tener el mismo tamano."
        );
    }

    if (density.empty())
    {
        throw std::invalid_argument(
            "La densidad molecular no puede estar vacia."
        );
    }

    const double dx = grid.getDx();
    const double dy = grid.getDy();
    const double dz = grid.getDz();

    if (dx <= 0.0 ||
        dy <= 0.0 ||
        dz <= 0.0)
    {
        throw std::invalid_argument(
            "Los pasos de la malla cartesiana deben ser mayores que cero."
        );
    }

    const double volumeElement =
        dx *
        dy *
        dz;

    double energy = 0.0;

    for (std::size_t i = 0;
         i < density.size();
         ++i)
    {
        energy +=
            density[i] *
            hartreePotential[i];
    }

    return 0.5 *
           energy *
           volumeElement;
}