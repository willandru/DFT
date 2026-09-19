#include "TotalEnergy.h"

#include "DFTConstants.h"
#include "ExchangeCorrelation.h"
#include "HartreePotential.h"

#include <cmath>
#include <stdexcept>

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
                0.5 * integrand;

        } else {

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