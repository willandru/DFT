#include "DavidsonNumerical.h"

#include "DavidsonMath.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

constexpr double VECTOR_TOLERANCE_SQUARED = 1.0e-20;
constexpr double DAVIDSON_DENOMINATOR_TOLERANCE = 1.0e-10;

}


void davidsonOrthogonalizeAgainstOrbitals(
    std::vector<double>& v,
    const std::vector<MolecularOrbital>& orbitals,
    double dV
) {
    for (int pass = 0; pass < 2; ++pass) {

        for (const auto& orbital : orbitals) {

            const double orbitalNorm =
                davidsonNormSquared(
                    orbital.psi,
                    dV
                );

            if (!std::isfinite(orbitalNorm) ||
                orbitalNorm <=
                    VECTOR_TOLERANCE_SQUARED) {

                throw std::runtime_error(
                    "Un orbital molecular previo no es valido."
                );
            }

            const double projection =
                davidsonDot(
                    orbital.psi,
                    v,
                    dV
                ) / orbitalNorm;

            for (std::size_t i = 0;
                 i < v.size();
                 ++i) {

                v[i] -=
                    projection *
                    orbital.psi[i];
            }
        }
    }
}


std::vector<double> davidsonBuildInitialVector(
    const CartesianGrid& grid,
    std::size_t orbitalIndex
) {
    const std::size_t nx =
        grid.getNx();

    const std::size_t ny =
        grid.getNy();

    const std::size_t nz =
        grid.getNz();

    std::vector<double> v(
        grid.getSize(),
        0.0
    );

    const double cx =
        0.5 *
        (grid.getXMin() + grid.getXMax());

    const double cy =
        0.5 *
        (grid.getYMin() + grid.getYMax());

    const double cz =
        0.5 *
        (grid.getZMin() + grid.getZMax());

    const double sigma =
        0.8 +
        0.18 *
        static_cast<double>(orbitalIndex);

    const double factor =
        1.0 /
        (2.0 * sigma * sigma);

    const std::size_t mode =
        orbitalIndex % 9;

    for (std::size_t k = 0;
         k < nz;
         ++k) {

        const double z =
            grid.getZ(k) - cz;

        for (std::size_t j = 0;
             j < ny;
             ++j) {

            const double y =
                grid.getY(j) - cy;

            for (std::size_t i = 0;
                 i < nx;
                 ++i) {

                const double x =
                    grid.getX(i) - cx;

                const double r2 =
                    x * x +
                    y * y +
                    z * z;

                const double gaussian =
                    std::exp(-r2 * factor);

                double polynomial = 1.0;

                switch (mode) {

                case 1:
                    polynomial = x;
                    break;

                case 2:
                    polynomial = y;
                    break;

                case 3:
                    polynomial = z;
                    break;

                case 4:
                    polynomial = x * y;
                    break;

                case 5:
                    polynomial = x * z;
                    break;

                case 6:
                    polynomial = y * z;
                    break;

                case 7:
                    polynomial =
                        x * x -
                        y * y;
                    break;

                case 8:
                    polynomial =
                        2.0 * z * z -
                        x * x -
                        y * y;
                    break;

                default:
                    break;
                }

                const std::size_t index =
                    k * nx * ny +
                    j * nx +
                    i;

                v[index] =
                    polynomial *
                    gaussian;
            }
        }
    }

    return v;
}


std::vector<double> davidsonBuildHamiltonianDiagonal(
    const CartesianGrid& grid,
    const std::vector<double>& potential
) {
    const double dx =
        grid.getDx();

    const double dy =
        grid.getDy();

    const double dz =
        grid.getDz();

    if (dx <= 0.0 ||
        dy <= 0.0 ||
        dz <= 0.0) {

        throw std::invalid_argument(
            "Espaciamiento de malla invalido."
        );
    }

    const double kinetic =
        1.0 / (dx * dx) +
        1.0 / (dy * dy) +
        1.0 / (dz * dz);

    std::vector<double> diagonal(
        potential.size()
    );

    for (std::size_t i = 0;
         i < potential.size();
         ++i) {

        diagonal[i] =
            kinetic +
            potential[i];
    }

    return diagonal;
}


std::vector<double> davidsonBuildCorrection(
    const std::vector<double>& residual,
    const std::vector<double>& diagonal,
    double eigenvalue
) {
    std::vector<double> correction(
        residual.size(),
        0.0
    );

    for (std::size_t i = 0;
         i < residual.size();
         ++i) {

        const double denominator =
            eigenvalue -
            diagonal[i];

        if (!std::isfinite(denominator) ||
            std::abs(denominator) <
                DAVIDSON_DENOMINATOR_TOLERANCE) {

            correction[i] =
                residual[i];

        } else {

            correction[i] =
                residual[i] /
                denominator;
        }
    }

    return correction;
}


bool davidsonBuildIndependentVector(
    const CartesianGrid& grid,
    std::size_t orbitalIndex,
    const std::vector<MolecularOrbital>& orbitals,
    double dV,
    std::vector<double>& result
) {
    for (std::size_t seed = orbitalIndex;
         seed < orbitalIndex + 64;
         ++seed) {

        result =
            davidsonBuildInitialVector(
                grid,
                seed
            );

        davidsonOrthogonalizeAgainstOrbitals(
            result,
            orbitals,
            dV
        );

        if (!davidsonValidNorm(result, dV)) {
            continue;
        }

        davidsonNormalize(
            result,
            dV
        );

        return true;
    }

    return false;
}


bool davidsonBuildInitialFromPrevious(
    const CartesianGrid& grid,
    const MolecularOrbital& initial,
    const std::vector<MolecularOrbital>& orbitals,
    double dV,
    std::vector<double>& result
) {
    if (initial.psi.size() !=
        grid.getSize()) {

        return false;
    }

    result =
        initial.psi;

    for (double value : result) {

        if (!std::isfinite(value)) {
            return false;
        }
    }

    davidsonOrthogonalizeAgainstOrbitals(
        result,
        orbitals,
        dV
    );

    if (!davidsonValidNorm(result, dV)) {
        return false;
    }

    davidsonNormalize(
        result,
        dV
    );

    return true;
}


bool davidsonAppendIndependentCorrection(
    std::vector<double>& correction,
    const std::vector<MolecularOrbital>& previousOrbitals,
    const std::vector<std::vector<double>>& basis,
    double dV,
    double subspaceTolerance
) {
    davidsonOrthogonalizeAgainstOrbitals(
        correction,
        previousOrbitals,
        dV
    );

    davidsonOrthogonalize(
        correction,
        basis,
        dV
    );

    const double toleranceSquared =
        subspaceTolerance *
        subspaceTolerance;

    if (!davidsonValidNorm(
            correction,
            dV,
            toleranceSquared
        )) {

        return false;
    }

    davidsonNormalize(
        correction,
        dV,
        toleranceSquared
    );

    return true;
}