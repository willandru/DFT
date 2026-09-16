#include "ExchangeCorrelation.h"

#include "DFTConstants.h"
#include "PZ81.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

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

    PZ81 functional;

    std::vector<double> potential(
        alphaDensity.size(),
        0.0
    );

    for (std::size_t i = 0;
         i < alphaDensity.size();
         ++i) {

        const XCResult result =
            functional.evaluate(
                alphaDensity[i],
                betaDensity[i]
            );

        if (spin == 0) {
            potential[i] =
                result.potentialAlpha;
        } else {
            potential[i] =
                result.potentialBeta;
        }
    }

    return potential;
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

    PZ81 functional;

    double energy = 0.0;

    for (std::size_t i = 0;
         i < r.size();
         ++i) {

        const double density =
            alphaDensity[i] +
            betaDensity[i];

        double epsilonXC = 0.0;

        if (density > DFTConstants::RHO_FLOOR) {

            const XCResult result =
                functional.evaluate(
                    alphaDensity[i],
                    betaDensity[i]
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