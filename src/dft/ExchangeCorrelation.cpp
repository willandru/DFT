#include "ExchangeCorrelation.h"

#include "DFTConstants.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

struct CorrelationParameters {
    double A;
    double B;
    double C;
    double D;
    double gamma;
    double beta1;
    double beta2;
};

constexpr CorrelationParameters UNPOLARIZED = {
    0.0311,
    -0.0480,
    0.0020,
    -0.0116,
    -0.1423,
    1.0529,
    0.3334
};

constexpr CorrelationParameters POLARIZED = {
    0.01555,
    -0.0269,
    0.0007,
    -0.0048,
    -0.0843,
    1.3981,
    0.2611
};

double rsFromDensity(double density) {
    return std::cbrt(
        3.0 /
        (
            4.0 *
            DFTConstants::PI *
            density
        )
    );
}

double correlationEnergy(
    double rs,
    const CorrelationParameters& parameters
) {
    if (rs <= 0.0) {
        throw std::invalid_argument(
            "El parametro rs debe ser mayor que cero."
        );
    }

    if (rs < 1.0) {
        return
            parameters.A * std::log(rs) +
            parameters.B +
            parameters.C * rs * std::log(rs) +
            parameters.D * rs;
    }

    const double sqrtRs =
        std::sqrt(rs);

    const double denominator =
        1.0 +
        parameters.beta1 * sqrtRs +
        parameters.beta2 * rs;

    return
        parameters.gamma /
        denominator;
}

double correlationEnergyDerivativeRs(
    double rs,
    const CorrelationParameters& parameters
) {
    if (rs <= 0.0) {
        throw std::invalid_argument(
            "El parametro rs debe ser mayor que cero."
        );
    }

    if (rs < 1.0) {
        return
            parameters.A / rs +
            parameters.C * (std::log(rs) + 1.0) +
            parameters.D;
    }

    const double sqrtRs =
        std::sqrt(rs);

    const double denominator =
        1.0 +
        parameters.beta1 * sqrtRs +
        parameters.beta2 * rs;

    const double derivativeDenominator =
        parameters.beta1 /
        (2.0 * sqrtRs) +
        parameters.beta2;

    return
        -parameters.gamma *
        derivativeDenominator /
        (
            denominator *
            denominator
        );
}

double correlationPotential(
    double rs,
    const CorrelationParameters& parameters
) {
    const double epsilon =
        correlationEnergy(
            rs,
            parameters
        );

    const double derivative =
        correlationEnergyDerivativeRs(
            rs,
            parameters
        );

    return
        epsilon -
        rs * derivative / 3.0;
}

double spinInterpolation(double zeta) {
    const double z =
        std::clamp(
            zeta,
            -1.0,
            1.0
        );

    const double numerator =
        std::pow(
            1.0 + z,
            4.0 / 3.0
        ) +
        std::pow(
            1.0 - z,
            4.0 / 3.0
        ) -
        2.0;

    const double denominator =
        std::pow(
            2.0,
            4.0 / 3.0
        ) -
        2.0;

    return numerator / denominator;
}

double spinInterpolationDerivative(
    double zeta
) {
    const double z =
        std::clamp(
            zeta,
            -1.0,
            1.0
        );

    const double numerator =
        4.0 / 3.0 *
        (
            std::pow(
                1.0 + z,
                1.0 / 3.0
            ) -
            std::pow(
                1.0 - z,
                1.0 / 3.0
            )
        );

    const double denominator =
        std::pow(
            2.0,
            4.0 / 3.0
        ) -
        2.0;

    return numerator / denominator;
}

double exchangeEnergyPerElectron(
    double density
) {
    return
        -DFTConstants::EXCHANGE_COEFFICIENT *
        std::cbrt(density);
}

double exchangeEnergyPerElectronSpin(
    double density,
    double zeta
) {
    const double unpolarized =
        exchangeEnergyPerElectron(
            density
        );

    return
        unpolarized *
        (
            1.0 +
            spinInterpolation(zeta)
        );
}

double spinExchangeCorrelationEnergyPerElectron(
    double density,
    double zeta
) {
    const double epsilonExchange =
        exchangeEnergyPerElectronSpin(
            density,
            zeta
        );

    const double rs =
        rsFromDensity(density);

    const double epsilonCorrelation0 =
        correlationEnergy(
            rs,
            UNPOLARIZED
        );

    const double epsilonCorrelation1 =
        correlationEnergy(
            rs,
            POLARIZED
        );

    const double f =
        spinInterpolation(zeta);

    const double epsilonCorrelation =
        epsilonCorrelation0 +
        f *
        (
            epsilonCorrelation1 -
            epsilonCorrelation0
        );

    return
        epsilonExchange +
        epsilonCorrelation;
}

double spinExchangeCorrelationPotential(
    double density,
    double alphaDensity,
    double betaDensity,
    int spin
) {
    if (density <= DFTConstants::RHO_FLOOR) {
        return 0.0;
    }

    if (spin != 0 && spin != 1) {
        throw std::invalid_argument(
            "El canal de spin debe ser 0 (alpha) o 1 (beta)."
        );
    }

    const double alpha =
        std::max(
            alphaDensity,
            0.0
        );

    const double beta =
        std::max(
            betaDensity,
            0.0
        );

    const double zeta =
        std::clamp(
            (alpha - beta) / density,
            -1.0,
            1.0
        );

    const double rs =
        rsFromDensity(density);

    const double f =
        spinInterpolation(zeta);

    const double df =
        spinInterpolationDerivative(zeta);

    const double exchange =
        exchangeEnergyPerElectron(
            density
        );

    const double exchangeSpin =
        exchange *
        (1.0 + f);

    const double exchangeDerivativeRs =
        -exchangeSpin / rs;

    const double epsilonCorrelation0 =
        correlationEnergy(
            rs,
            UNPOLARIZED
        );

    const double epsilonCorrelation1 =
        correlationEnergy(
            rs,
            POLARIZED
        );

    const double derivativeCorrelation0 =
        correlationEnergyDerivativeRs(
            rs,
            UNPOLARIZED
        );

    const double derivativeCorrelation1 =
        correlationEnergyDerivativeRs(
            rs,
            POLARIZED
        );

    const double correlation =
        epsilonCorrelation0 +
        f *
        (
            epsilonCorrelation1 -
            epsilonCorrelation0
        );

    const double correlationDerivativeRs =
        derivativeCorrelation0 +
        f *
        (
            derivativeCorrelation1 -
            derivativeCorrelation0
        );

    const double epsilonXC =
        exchangeSpin +
        correlation;

    const double derivativeXCrs =
        exchangeDerivativeRs +
        correlationDerivativeRs;

    const double epsilonXCzeta =
        exchange * df +
        df *
        (
            epsilonCorrelation1 -
            epsilonCorrelation0
        );

    const double radialContribution =
        epsilonXC -
        rs *
        derivativeXCrs /
        3.0;

    if (spin == 0) {
        return
            radialContribution +
            (1.0 - zeta) *
            epsilonXCzeta;
    }

    return
        radialContribution -
        (1.0 + zeta) *
        epsilonXCzeta;
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
    if (density <= DFTConstants::RHO_FLOOR) {
        return 0.0;
    }

    const double rs =
        rsFromDensity(density);

    return correlationEnergy(
        rs,
        UNPOLARIZED
    );
}

double exchangeCorrelationEnergyPerElectron(
    double density
) {
    if (density <= DFTConstants::RHO_FLOOR) {
        return 0.0;
    }

    return
        exchangeEnergyPerElectron(
            density
        ) +
        correlationEnergyPerElectron(
            density
        );
}

double exchangeCorrelationPotential(
    double density
) {
    if (density <= DFTConstants::RHO_FLOOR) {
        return 0.0;
    }

    const double exchange =
        exchangeEnergyPerElectron(
            density
        );

    const double exchangePotential =
        4.0 / 3.0 *
        exchange;

    const double correlationPotentialValue =
        correlationPotential(
            rsFromDensity(density),
            UNPOLARIZED
        );

    return
        exchangePotential +
        correlationPotentialValue;
}

std::vector<double> calculateSpinExchangeCorrelationPotential(
    const std::vector<double>& alphaDensity,
    const std::vector<double>& betaDensity,
    int spin
) {
    if (spin != 0 && spin != 1) {
        throw std::invalid_argument(
            "El canal de spin debe ser 0 (alpha) o 1 (beta)."
        );
    }

    if (alphaDensity.size() != betaDensity.size()) {
        throw std::invalid_argument(
            "Las densidades alpha y beta deben tener el mismo tamano."
        );
    }

    std::vector<double> potential(
        alphaDensity.size(),
        0.0
    );

    for (std::size_t i = 0;
         i < alphaDensity.size();
         ++i) {

        const double density =
            alphaDensity[i] +
            betaDensity[i];

        potential[i] =
            spinExchangeCorrelationPotential(
                density,
                alphaDensity[i],
                betaDensity[i],
                spin
            );
    }

    return potential;
}

std::vector<double> calculateExchangeCorrelationPotential(
    const std::vector<double>& density
) {
    std::vector<double> potential(
        density.size(),
        0.0
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

        const double integrand =
            4.0 *
            DFTConstants::PI *
            r[i] *
            r[i] *
            density[i] *
            exchangeCorrelationEnergyPerElectron(
                density[i]
            );

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

double calculateSpinExchangeCorrelationEnergy(
    const std::vector<double>& r,
    const std::vector<double>& alphaDensity,
    const std::vector<double>& betaDensity
) {
    if (r.size() != alphaDensity.size() ||
        r.size() != betaDensity.size()) {

        throw std::invalid_argument(
            "La malla y las densidades spin deben tener el mismo tamano."
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

        const double density =
            alphaDensity[i] +
            betaDensity[i];

        double epsilonXC = 0.0;

        if (density > DFTConstants::RHO_FLOOR) {
            const double zeta =
                std::clamp(
                    (
                        alphaDensity[i] -
                        betaDensity[i]
                    ) / density,
                    -1.0,
                    1.0
                );

            epsilonXC =
                spinExchangeCorrelationEnergyPerElectron(
                    density,
                    zeta
                );
        }

        const double integrand =
            4.0 *
            DFTConstants::PI *
            r[i] *
            r[i] *
            density *
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