#include "PZ81.h"

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

double spinExchangeScaling(
    double zeta
) {
    const double z =
        std::clamp(
            zeta,
            -1.0,
            1.0
        );

    return
        (
            std::pow(
                1.0 + z,
                4.0 / 3.0
            ) +
            std::pow(
                1.0 - z,
                4.0 / 3.0
            )
        ) / 2.0;
}

double spinExchangeScalingDerivative(
    double zeta
) {
    const double z =
        std::clamp(
            zeta,
            -1.0,
            1.0
        );

    return
        2.0 / 3.0 *
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
}

double exchangeEnergyPerElectronSpin(
    double density,
    double zeta
) {
    return
        exchangeEnergyPerElectron(
            density
        ) *
        spinExchangeScaling(
            zeta
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

    const double exchange =
        exchangeEnergyPerElectron(
            density
        );

    const double exchangeScaling =
        spinExchangeScaling(
            zeta
        );

    const double exchangeScalingDerivative =
        spinExchangeScalingDerivative(
            zeta
        );

    const double exchangeSpin =
        exchange *
        exchangeScaling;

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

    const double f =
        spinInterpolation(zeta);

    const double df =
        spinInterpolationDerivative(zeta);

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
        exchange *
        exchangeScalingDerivative +
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

XCResult PZ81::evaluate(
    const XCInput& input
) const {
    const double alpha =
        std::max(
            input.alphaDensity,
            0.0
        );

    const double beta =
        std::max(
            input.betaDensity,
            0.0
        );

    const double density =
        alpha + beta;

    if (density <= DFTConstants::RHO_FLOOR) {
        return {
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        };
    }

    const double zeta =
        std::clamp(
            (alpha - beta) / density,
            -1.0,
            1.0
        );

    const double energyPerElectron =
        spinExchangeCorrelationEnergyPerElectron(
            density,
            zeta
        );

    const double potentialAlpha =
        spinExchangeCorrelationPotential(
            density,
            alpha,
            beta,
            0
        );

    const double potentialBeta =
        spinExchangeCorrelationPotential(
            density,
            alpha,
            beta,
            1
        );

    return {
        energyPerElectron,
        potentialAlpha,
        potentialBeta,
        0.0,
        0.0
    };
}