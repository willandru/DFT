#include "TotalEnergy.h"

#include "DFTConstants.h"
#include "ExchangeCorrelation.h"
#include "HartreePotential.h"
#include "KohnShamHamiltonian.h"
#include "NuclearRepulsion.h"

#include <cmath>
#include <stdexcept>
#include <vector>

double calculateKineticEnergy(
    const std::vector<double>& r,
    const std::vector<AtomicOrbital>& orbitals
) {
    if (r.size() < 2) {
        throw std::invalid_argument(
            "Se necesitan al menos dos puntos radiales."
        );
    }

    const double dr =
        r[1] - r[0];

    if (dr <= 0.0) {
        throw std::invalid_argument(
            "El paso radial debe ser mayor que cero."
        );
    }

    const double kineticDiagonal =
        1.0 / (dr * dr);

    const double kineticOffDiagonal =
        -0.5 / (dr * dr);

    double energy = 0.0;

    for (const AtomicOrbital& orbital : orbitals) {
        if (orbital.electrons <= 0) {
            continue;
        }

        if (orbital.u.size() != r.size()) {
            throw std::invalid_argument(
                "El orbital y la malla radial deben tener el mismo tamano."
            );
        }

        const double centrifugalCoefficient =
            0.5 *
            static_cast<double>(
                orbital.l *
                (orbital.l + 1)
            );

        double orbitalEnergy = 0.0;

        for (std::size_t i = 0;
             i < r.size();
             ++i) {

            if (r[i] <= 0.0) {
                throw std::invalid_argument(
                    "Los puntos radiales deben ser mayores que cero."
                );
            }

            double applied =
                (
                    kineticDiagonal +
                    centrifugalCoefficient /
                    (r[i] * r[i])
                ) *
                orbital.u[i];

            if (i > 0) {
                applied +=
                    kineticOffDiagonal *
                    orbital.u[i - 1];
            }

            if (i + 1 < r.size()) {
                applied +=
                    kineticOffDiagonal *
                    orbital.u[i + 1];
            }

            orbitalEnergy +=
                orbital.u[i] *
                applied;
        }

        energy +=
            static_cast<double>(
                orbital.electrons
            ) *
            orbitalEnergy *
            dr;
    }

    return energy;
}

double calculateExternalEnergy(
    const std::vector<double>& r,
    const std::vector<double>& density,
    int Z
) {
    if (r.size() != density.size()) {
        throw std::invalid_argument(
            "La malla radial y la densidad deben tener el mismo tamano."
        );
    }

    if (r.size() < 2) {
        throw std::invalid_argument(
            "Se necesitan al menos dos puntos radiales."
        );
    }

    if (Z <= 0) {
        throw std::invalid_argument(
            "El numero atomico debe ser mayor que cero."
        );
    }

    const double dr =
        r[1] - r[0];

    if (dr <= 0.0) {
        throw std::invalid_argument(
            "El paso radial debe ser mayor que cero."
        );
    }

    double energy = 0.0;

    for (std::size_t i = 0;
         i < r.size();
         ++i) {

        if (r[i] <= 0.0) {
            throw std::invalid_argument(
                "Los puntos radiales deben ser mayores que cero."
            );
        }

        const double integrand =
            -static_cast<double>(Z) *
            4.0 *
            DFTConstants::PI *
            r[i] *
            density[i];

        if (i == 0 ||
            i == r.size() - 1) {

            energy +=
                0.5 *
                integrand;
        }
        else {
            energy +=
                integrand;
        }
    }

    return energy * dr;
}

EnergyComponents calculateTotalEnergy(
    const XCFunctional& functional,
    const std::vector<double>& r,
    const std::vector<double>& alphaDensity,
    const std::vector<double>& betaDensity,
    const std::vector<AtomicOrbital>& orbitals,
    const std::vector<double>& hartreePotential,
    int Z
) {
    if (alphaDensity.size() != betaDensity.size()) {
        throw std::invalid_argument(
            "Las densidades alpha y beta deben tener el mismo tamano."
        );
    }

    if (r.size() != alphaDensity.size()) {
        throw std::invalid_argument(
            "La malla radial y las densidades deben tener el mismo tamano."
        );
    }

    std::vector<double> density(
        r.size(),
        0.0
    );

    for (std::size_t i = 0;
         i < r.size();
         ++i) {

        density[i] =
            alphaDensity[i] +
            betaDensity[i];
    }

    EnergyComponents energy;

    energy.kinetic =
        calculateKineticEnergy(
            r,
            orbitals
        );

    energy.external =
        calculateExternalEnergy(
            r,
            density,
            Z
        );

    energy.hartree =
        calculateHartreeEnergy(
            r,
            density,
            hartreePotential
        );

    energy.exchangeCorrelation =
        calculateSpinExchangeCorrelationEnergy(
            functional,
            r,
            alphaDensity,
            betaDensity
        );

    energy.total =
        energy.kinetic +
        energy.external +
        energy.hartree +
        energy.exchangeCorrelation;

    return energy;
}

double calculateMolecularKineticEnergy(
    const CartesianGrid& grid,
    const std::vector<MolecularOrbital>& orbitals
) {
    if (grid.getNx() < 2 ||
        grid.getNy() < 2 ||
        grid.getNz() < 2) {

        throw std::invalid_argument(
            "La malla cartesiana debe tener al menos dos puntos por dimension."
        );
    }

    const double dx = grid.getDx();
    const double dy = grid.getDy();
    const double dz = grid.getDz();

    if (dx <= 0.0 ||
        dy <= 0.0 ||
        dz <= 0.0) {

        throw std::invalid_argument(
            "Los pasos de la malla cartesiana deben ser mayores que cero."
        );
    }

    const std::size_t gridSize =
        grid.getSize();

    const double volumeElement =
        dx *
        dy *
        dz;

    double energy = 0.0;

    for (const MolecularOrbital& orbital : orbitals) {
        if (orbital.electrons <= 0) {
            continue;
        }

        if (orbital.psi.size() != gridSize) {
            throw std::invalid_argument(
                "El orbital molecular y la malla deben tener el mismo tamano."
            );
        }

        double orbitalEnergy = 0.0;

        for (std::size_t k = 0;
             k < grid.getNz();
             ++k) {

            for (std::size_t j = 0;
                 j < grid.getNy();
                 ++j) {

                for (std::size_t i = 0;
                     i < grid.getNx();
                     ++i) {

                    const std::size_t index =
                        grid.getIndex(i, j, k);

                    const double psi =
                        orbital.psi[index];

                    double laplacian =
                        0.0;

                    if (i > 0) {
                        const std::size_t indexPrevious =
                            grid.getIndex(
                                i - 1,
                                j,
                                k
                            );

                        laplacian +=
                            (
                                orbital.psi[indexPrevious] -
                                psi
                            ) /
                            (dx * dx);
                    }

                    if (i + 1 < grid.getNx()) {
                        const std::size_t indexNext =
                            grid.getIndex(
                                i + 1,
                                j,
                                k
                            );

                        laplacian +=
                            (
                                orbital.psi[indexNext] -
                                psi
                            ) /
                            (dx * dx);
                    }

                    if (j > 0) {
                        const std::size_t indexPrevious =
                            grid.getIndex(
                                i,
                                j - 1,
                                k
                            );

                        laplacian +=
                            (
                                orbital.psi[indexPrevious] -
                                psi
                            ) /
                            (dy * dy);
                    }

                    if (j + 1 < grid.getNy()) {
                        const std::size_t indexNext =
                            grid.getIndex(
                                i,
                                j + 1,
                                k
                            );

                        laplacian +=
                            (
                                orbital.psi[indexNext] -
                                psi
                            ) /
                            (dy * dy);
                    }

                    if (k > 0) {
                        const std::size_t indexPrevious =
                            grid.getIndex(
                                i,
                                j,
                                k - 1
                            );

                        laplacian +=
                            (
                                orbital.psi[indexPrevious] -
                                psi
                            ) /
                            (dz * dz);
                    }

                    if (k + 1 < grid.getNz()) {
                        const std::size_t indexNext =
                            grid.getIndex(
                                i,
                                j,
                                k + 1
                            );

                        laplacian +=
                            (
                                orbital.psi[indexNext] -
                                psi
                            ) /
                            (dz * dz);
                    }

                    const double kineticApplied =
                        -0.5 *
                        laplacian;

                    orbitalEnergy +=
                        psi *
                        kineticApplied *
                        volumeElement;
                }
            }
        }

        energy +=
            static_cast<double>(
                orbital.electrons
            ) *
            orbitalEnergy;
    }

    return energy;
}

double calculateMolecularExternalEnergy(
    const CartesianGrid& grid,
    const std::vector<double>& density,
    const std::vector<double>& nuclearPotential
) {
    const std::size_t gridSize =
        grid.getSize();

    if (density.size() != gridSize ||
        nuclearPotential.size() != gridSize) {

        throw std::invalid_argument(
            "La malla, la densidad y el potencial nuclear deben tener el mismo tamano."
        );
    }

    const double volumeElement =
        grid.getDx() *
        grid.getDy() *
        grid.getDz();

    double energy = 0.0;

    for (std::size_t index = 0;
         index < gridSize;
         ++index) {

        energy +=
            density[index] *
            nuclearPotential[index] *
            volumeElement;
    }

    return energy;
}

EnergyComponents calculateMolecularTotalEnergy(
    const XCFunctional& functional,
    const Molecule& molecule,
    const CartesianGrid& grid,
    const std::vector<double>& alphaDensity,
    const std::vector<double>& betaDensity,
    const std::vector<MolecularOrbital>& orbitals,
    const std::vector<double>& hartreePotential,
    const std::vector<double>& nuclearPotential
) {
    const std::size_t gridSize =
        grid.getSize();

    if (alphaDensity.size() != gridSize ||
        betaDensity.size() != gridSize) {

        throw std::invalid_argument(
            "Las densidades moleculares y la malla deben tener el mismo tamano."
        );
    }

    if (hartreePotential.size() != gridSize) {
        throw std::invalid_argument(
            "El potencial de Hartree y la malla deben tener el mismo tamano."
        );
    }

    if (nuclearPotential.size() != gridSize) {
        throw std::invalid_argument(
            "El potencial nuclear y la malla deben tener el mismo tamano."
        );
    }

    if (molecule.getNucleusCount() == 0) {
        throw std::invalid_argument(
            "La molecula debe contener al menos un nucleo."
        );
    }

    std::vector<double> density(
        gridSize,
        0.0
    );

    for (std::size_t index = 0;
         index < gridSize;
         ++index) {

        density[index] =
            alphaDensity[index] +
            betaDensity[index];
    }

    EnergyComponents energy;

    energy.kinetic =
        calculateMolecularKineticEnergy(
            grid,
            orbitals
        );

    energy.external =
        calculateMolecularExternalEnergy(
            grid,
            density,
            nuclearPotential
        );

    energy.hartree =
        calculateHartreeEnergy(
            grid,
            density,
            hartreePotential
        );

    energy.exchangeCorrelation =
        calculateMolecularExchangeCorrelationEnergy(
            functional,
            grid,
            alphaDensity,
            betaDensity
        );

    energy.nuclearRepulsion =
        NuclearRepulsion::calculate(
            molecule
        );

    energy.total =
        energy.kinetic +
        energy.external +
        energy.hartree +
        energy.exchangeCorrelation +
        energy.nuclearRepulsion;

    return energy;
}