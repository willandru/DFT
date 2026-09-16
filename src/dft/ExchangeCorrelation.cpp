#include "ExchangeCorrelation.h"

#include "DFTConstants.h"

#include <cmath>
#include <stdexcept>

namespace {

double correlationEnergyPerElectronPZ81(double rs) {
    constexpr double A = 0.0311;
    constexpr double B = -0.0480;
    constexpr double C = 0.0020;
    constexpr double D = -0.0116;

    constexpr double gamma = -0.1423;
    constexpr double beta1 = 1.0529;
    constexpr double beta2 = 0.3334;

    if (rs < 1.0) {
        return
            A * std::log(rs) +
            B +
            C * rs * std::log(rs) +
            D * rs;
    }

    const double sqrtRs =
        std::sqrt(rs);

    return
        gamma /
        (
            1.0 +
            beta1 * sqrtRs +
            beta2 * rs
        );
}

double correlationPotentialPZ81(double rs) {
    constexpr double A = 0.0311;
    constexpr double C = 0.0020;
    constexpr double D = -0.0116;

    constexpr double gamma = -0.1423;
    constexpr double beta1 = 1.0529;
    constexpr double beta2 = 0.3334;

    const double epsilon =
        correlationEnergyPerElectronPZ81(rs);

    double derivative;

    if (rs < 1.0) {
        derivative =
            A / rs +
            C * (std::log(rs) + 1.0) +
            D;
    } else {
        const double sqrtRs =
            std::sqrt(rs);

        const double denominator =
            1.0 +
            beta1 * sqrtRs +
            beta2 * rs;

        const double derivativeDenominator =
            beta1 / (2.0 * sqrtRs) +
            beta2;

        derivative =
            -gamma *
            derivativeDenominator /
            (
                denominator *
                denominator
            );
    }

    return
        epsilon -
        rs * derivative / 3.0;
}

double correlationEnergyPerElectronFromDensity(
    double density
) {
    if (density <= DFTConstants::RHO_FLOOR) {
        return 0.0;
    }

    const double rs =
        std::cbrt(
            3.0 /
            (
                4.0 *
                DFTConstants::PI *
                density
            )
        );

    return correlationEnergyPerElectronPZ81(rs);
}

double correlationPotential(
    double density
) {
    if (density <= DFTConstants::RHO_FLOOR) {
        return 0.0;
    }

    const double rs =
        std::cbrt(
            3.0 /
            (
                4.0 *
                DFTConstants::PI *
                density
            )
        );

    return correlationPotentialPZ81(rs);
}

}

double exchangeEnergyDensity(
    double density
) {
    if (density <= DFTConstants::RHO_FLOOR) {
        return 0.0;
    }

    return
        -DFTConstants::EXCHANGE_COEFFICIENT *
        std::pow(
            density,
            4.0 / 3.0
        );
}

double correlationEnergyPerElectron(
    double density
) {
    return
        correlationEnergyPerElectronFromDensity(
            density
        );
}

double exchangeCorrelationEnergyPerElectron(
    double density
) {
    if (density <= DFTConstants::RHO_FLOOR) {
        return 0.0;
    }

    const double exchange =
        -DFTConstants::EXCHANGE_COEFFICIENT *
        std::cbrt(density);

    return
        exchange +
        correlationEnergyPerElectronFromDensity(
            density
        );
}

double exchangeCorrelationPotential(
    double density
) {
    if (density <= DFTConstants::RHO_FLOOR) {
        return 0.0;
    }

    const double exchangePotential =
        -std::cbrt(
            3.0 *
            density /
            DFTConstants::PI
        );

    return
        exchangePotential +
        correlationPotential(
            density
        );
}

std::vector<double> calculateExchangeCorrelationPotential(
    const std::vector<double>& density
) {
    std::vector<double> potential(
        density.size()
    );

    for (std::size_t i = 0;
         i < density.size();
         ++i) {

        potential[i] =
            exchangeCorrelationPotential(
                density[i]
            );
    }

    return potential;
}

double calculateExchangeCorrelationEnergy(
    const std::vector<double>& r,
    const std::vector<double>& density
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

        const double epsilonXC =
            exchangeCorrelationEnergyPerElectron(
                density[i]
            );

        const double integrand =
            4.0 *
            DFTConstants::PI *
            r[i] *
            r[i] *
            density[i] *
            epsilonXC;

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