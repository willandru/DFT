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

    const double dr = r[1] - r[0];

    if (dr <= 0.0) {
        throw std::invalid_argument(
            "El paso radial debe ser mayor que cero."
        );
    }

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

        double integral = 0.0;

        for (std::size_t i = 1; i < r.size() - 1; ++i) {
            const double secondDerivative =
                (
                    orbital.u[i + 1] -
                    2.0 * orbital.u[i] +
                    orbital.u[i - 1]
                ) / (dr * dr);

            const double centrifugal =
                0.5 *
                static_cast<double>(
                    orbital.l * (orbital.l + 1)
                ) *
                orbital.u[i] /
                (r[i] * r[i]);

            const double value =
                orbital.u[i] *
                (
                    -0.5 * secondDerivative +
                    centrifugal
                );

            integral += value;
        }

        energy +=
            static_cast<double>(orbital.electrons) *
            integral *
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

    const double dr = r[1] - r[0];

    if (dr <= 0.0) {
        throw std::invalid_argument(
            "El paso radial debe ser mayor que cero."
        );
    }

    double energy = 0.0;

    for (std::size_t i = 0; i < r.size(); ++i) {
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

        if (i == 0 || i == r.size() - 1) {
            energy += 0.5 * integrand;
        } else {
            energy += integrand;
        }
    }

    return energy * dr;
}

EnergyComponents calculateTotalEnergy(
    const std::vector<double>& r,
    const std::vector<double>& density,
    const std::vector<AtomicOrbital>& orbitals,
    const std::vector<double>& hartreePotential,
    int Z
) {
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
        calculateExchangeCorrelationEnergy(
            r,
            density
        );

    energy.total =
        energy.kinetic +
        energy.external +
        energy.hartree +
        energy.exchangeCorrelation;

    return energy;
}