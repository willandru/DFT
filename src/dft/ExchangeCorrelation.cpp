#include "ExchangeCorrelation.h"

#include "DFTConstants.h"

#include <stdexcept>
#include <vector>

namespace {

double calculateRadialGradient(
    const std::vector<double>& r,
    const std::vector<double>& density,
    std::size_t i
) {
    const std::size_t size =
        r.size();

    if (i == 0) {
        return
            (
                density[1] -
                density[0]
            ) /
            (
                r[1] -
                r[0]
            );
    }

    if (i == size - 1) {
        return
            (
                density[size - 1] -
                density[size - 2]
            ) /
            (
                r[size - 1] -
                r[size - 2]
            );
    }

    return
        (
            density[i + 1] -
            density[i - 1]
        ) /
        (
            r[i + 1] -
            r[i - 1]
        );
}

double calculateRadialDivergence(
    const std::vector<double>& r,
    const std::vector<double>& coefficient,
    std::size_t i
) {
    const std::size_t size =
        r.size();

    if (size < 2) {
        throw std::invalid_argument(
            "Se necesitan al menos dos puntos radiales."
        );
    }

    if (r[i] == 0.0) {
        return 0.0;
    }

    if (i == 0) {
        const double radialValue0 =
            r[0] * r[0] * coefficient[0];

        const double radialValue1 =
            r[1] * r[1] * coefficient[1];

        const double derivative =
            (
                radialValue1 -
                radialValue0
            ) /
            (
                r[1] -
                r[0]
            );

        return
            derivative /
            (
                r[0] * r[0]
            );
    }

    if (i == size - 1) {
        const double radialValuePrevious =
            r[size - 2] *
            r[size - 2] *
            coefficient[size - 2];

        const double radialValueCurrent =
            r[size - 1] *
            r[size - 1] *
            coefficient[size - 1];

        const double derivative =
            (
                radialValueCurrent -
                radialValuePrevious
            ) /
            (
                r[size - 1] -
                r[size - 2]
            );

        return
            derivative /
            (
                r[size - 1] *
                r[size - 1]
            );
    }

    const double radialValuePrevious =
        r[i - 1] *
        r[i - 1] *
        coefficient[i - 1];

    const double radialValueNext =
        r[i + 1] *
        r[i + 1] *
        coefficient[i + 1];

    const double derivative =
        (
            radialValueNext -
            radialValuePrevious
        ) /
        (
            r[i + 1] -
            r[i - 1]
        );

    return
        derivative /
        (
            r[i] *
            r[i]
        );
}

}

std::vector<double> calculateSpinExchangeCorrelationPotential(
    const XCFunctional& functional,
    const std::vector<double>& r,
    const std::vector<double>& alphaDensity,
    const std::vector<double>& betaDensity,
    int spin
) {
    if (spin != 0 && spin != 1) {
        throw std::invalid_argument(
            "El canal de spin debe ser 0 (alpha) o 1 (beta)."
        );
    }

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

    std::vector<double> densityDerivative(
        alphaDensity.size(),
        0.0
    );

    std::vector<double> gradientCoefficient(
        alphaDensity.size(),
        0.0
    );

    for (std::size_t i = 0;
         i < alphaDensity.size();
         ++i) {

        XCInput input;

        input.alphaDensity =
            alphaDensity[i];

        input.betaDensity =
            betaDensity[i];

        input.alphaGradient =
            calculateRadialGradient(
                r,
                alphaDensity,
                i
            );

        input.betaGradient =
            calculateRadialGradient(
                r,
                betaDensity,
                i
            );

        const XCResult result =
            functional.evaluate(
                input
            );

        if (spin == 0) {
            densityDerivative[i] =
                result.potentialAlpha;

            gradientCoefficient[i] =
                result.gradientCoefficientAlpha;
        } else {
            densityDerivative[i] =
                result.potentialBeta;

            gradientCoefficient[i] =
                result.gradientCoefficientBeta;
        }
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

        if (density <= DFTConstants::RHO_FLOOR) {
            potential[i] = 0.0;
            continue;
        }

        if (r[i] == 0.0) {
            potential[i] =
                densityDerivative[i];
            continue;
        }

        const double divergence =
            calculateRadialDivergence(
                r,
                gradientCoefficient,
                i
            );

        potential[i] =
            densityDerivative[i] -
            divergence;
    }

    return potential;
}

double calculateSpinExchangeCorrelationEnergy(
    const XCFunctional& functional,
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

            XCInput input;

            input.alphaDensity =
                alphaDensity[i];

            input.betaDensity =
                betaDensity[i];

            input.alphaGradient =
                calculateRadialGradient(
                    r,
                    alphaDensity,
                    i
                );

            input.betaGradient =
                calculateRadialGradient(
                    r,
                    betaDensity,
                    i
                );

            const XCResult result =
                functional.evaluate(
                    input
                );

            epsilonXC =
                result.energyPerElectron;
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

            energy += integrand;
        }
    }

    return energy * dr;
}