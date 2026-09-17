#include "ExchangeCorrelation.h"

#include "DFTConstants.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

double calculateRadialGradient(
    const std::vector<double>& r,
    const std::vector<double>& density,
    std::size_t i
) {
    const std::size_t size = r.size();

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
    const std::size_t size = r.size();

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
            r[0] *
            r[0] *
            coefficient[0];

        const double radialValue1 =
            r[1] *
            r[1] *
            coefficient[1];

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
                r[0] *
                r[0]
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

double calculateCartesianDerivative(
    const CartesianGrid& grid,
    const std::vector<double>& values,
    std::size_t i,
    std::size_t j,
    std::size_t k,
    int direction
) {
    const std::size_t nx = grid.getNx();
    const std::size_t ny = grid.getNy();
    const std::size_t nz = grid.getNz();

    if (direction == 0) {
        if (i == 0) {
            const std::size_t index0 =
                grid.getIndex(i, j, k);

            const std::size_t index1 =
                grid.getIndex(i + 1, j, k);

            return
                (
                    values[index1] -
                    values[index0]
                ) /
                grid.getDx();
        }

        if (i == nx - 1) {
            const std::size_t index0 =
                grid.getIndex(i - 1, j, k);

            const std::size_t index1 =
                grid.getIndex(i, j, k);

            return
                (
                    values[index1] -
                    values[index0]
                ) /
                grid.getDx();
        }

        const std::size_t indexPrevious =
            grid.getIndex(i - 1, j, k);

        const std::size_t indexNext =
            grid.getIndex(i + 1, j, k);

        return
            (
                values[indexNext] -
                values[indexPrevious]
            ) /
            (
                2.0 *
                grid.getDx()
            );
    }

    if (direction == 1) {
        if (j == 0) {
            const std::size_t index0 =
                grid.getIndex(i, j, k);

            const std::size_t index1 =
                grid.getIndex(i, j + 1, k);

            return
                (
                    values[index1] -
                    values[index0]
                ) /
                grid.getDy();
        }

        if (j == ny - 1) {
            const std::size_t index0 =
                grid.getIndex(i, j - 1, k);

            const std::size_t index1 =
                grid.getIndex(i, j, k);

            return
                (
                    values[index1] -
                    values[index0]
                ) /
                grid.getDy();
        }

        const std::size_t indexPrevious =
            grid.getIndex(i, j - 1, k);

        const std::size_t indexNext =
            grid.getIndex(i, j + 1, k);

        return
            (
                values[indexNext] -
                values[indexPrevious]
            ) /
            (
                2.0 *
                grid.getDy()
            );
    }

    if (direction == 2) {
        if (k == 0) {
            const std::size_t index0 =
                grid.getIndex(i, j, k);

            const std::size_t index1 =
                grid.getIndex(i, j, k + 1);

            return
                (
                    values[index1] -
                    values[index0]
                ) /
                grid.getDz();
        }

        if (k == nz - 1) {
            const std::size_t index0 =
                grid.getIndex(i, j, k - 1);

            const std::size_t index1 =
                grid.getIndex(i, j, k);

            return
                (
                    values[index1] -
                    values[index0]
                ) /
                grid.getDz();
        }

        const std::size_t indexPrevious =
            grid.getIndex(i, j, k - 1);

        const std::size_t indexNext =
            grid.getIndex(i, j, k + 1);

        return
            (
                values[indexNext] -
                values[indexPrevious]
            ) /
            (
                2.0 *
                grid.getDz()
            );
    }

    throw std::invalid_argument(
        "Direccion cartesiana invalida."
    );
}

double calculateGradientMagnitude(
    const CartesianGrid& grid,
    const std::vector<double>& density,
    std::size_t i,
    std::size_t j,
    std::size_t k
) {
    const double gradientX =
        calculateCartesianDerivative(
            grid,
            density,
            i,
            j,
            k,
            0
        );

    const double gradientY =
        calculateCartesianDerivative(
            grid,
            density,
            i,
            j,
            k,
            1
        );

    const double gradientZ =
        calculateCartesianDerivative(
            grid,
            density,
            i,
            j,
            k,
            2
        );

    return std::sqrt(
        gradientX * gradientX +
        gradientY * gradientY +
        gradientZ * gradientZ
    );
}

double calculateCartesianDivergence(
    const CartesianGrid& grid,
    const std::vector<double>& coefficient,
    const std::vector<double>& density,
    std::size_t i,
    std::size_t j,
    std::size_t k
) {
    const std::size_t nx = grid.getNx();
    const std::size_t ny = grid.getNy();
    const std::size_t nz = grid.getNz();

    const std::size_t index =
        grid.getIndex(i, j, k);

    const double coefficientCenter =
        coefficient[index];

    const double densityGradientX =
        calculateCartesianDerivative(
            grid,
            density,
            i,
            j,
            k,
            0
        );

    const double densityGradientY =
        calculateCartesianDerivative(
            grid,
            density,
            i,
            j,
            k,
            1
        );

    const double densityGradientZ =
        calculateCartesianDerivative(
            grid,
            density,
            i,
            j,
            k,
            2
        );

    const double fluxXCenter =
        coefficientCenter *
        densityGradientX;

    const double fluxYCenter =
        coefficientCenter *
        densityGradientY;

    const double fluxZCenter =
        coefficientCenter *
        densityGradientZ;

    double divergenceX = 0.0;
    double divergenceY = 0.0;
    double divergenceZ = 0.0;

    if (i == 0) {
        const std::size_t indexNext =
            grid.getIndex(i + 1, j, k);

        const double gradientXNext =
            calculateCartesianDerivative(
                grid,
                density,
                i + 1,
                j,
                k,
                0
            );

        const double fluxXNext =
            coefficient[indexNext] *
            gradientXNext;

        divergenceX =
            (
                fluxXNext -
                fluxXCenter
            ) /
            grid.getDx();
    }
    else if (i == nx - 1) {
        const std::size_t indexPrevious =
            grid.getIndex(i - 1, j, k);

        const double gradientXPrevious =
            calculateCartesianDerivative(
                grid,
                density,
                i - 1,
                j,
                k,
                0
            );

        const double fluxXPrevious =
            coefficient[indexPrevious] *
            gradientXPrevious;

        divergenceX =
            (
                fluxXCenter -
                fluxXPrevious
            ) /
            grid.getDx();
    }
    else {
        const std::size_t indexPrevious =
            grid.getIndex(i - 1, j, k);

        const std::size_t indexNext =
            grid.getIndex(i + 1, j, k);

        const double gradientXPrevious =
            calculateCartesianDerivative(
                grid,
                density,
                i - 1,
                j,
                k,
                0
            );

        const double gradientXNext =
            calculateCartesianDerivative(
                grid,
                density,
                i + 1,
                j,
                k,
                0
            );

        const double fluxXPrevious =
            coefficient[indexPrevious] *
            gradientXPrevious;

        const double fluxXNext =
            coefficient[indexNext] *
            gradientXNext;

        divergenceX =
            (
                fluxXNext -
                fluxXPrevious
            ) /
            (
                2.0 *
                grid.getDx()
            );
    }

    if (j == 0) {
        const std::size_t indexNext =
            grid.getIndex(i, j + 1, k);

        const double gradientYNext =
            calculateCartesianDerivative(
                grid,
                density,
                i,
                j + 1,
                k,
                1
            );

        const double fluxYNext =
            coefficient[indexNext] *
            gradientYNext;

        divergenceY =
            (
                fluxYNext -
                fluxYCenter
            ) /
            grid.getDy();
    }
    else if (j == ny - 1) {
        const std::size_t indexPrevious =
            grid.getIndex(i, j - 1, k);

        const double gradientYPrevious =
            calculateCartesianDerivative(
                grid,
                density,
                i,
                j - 1,
                k,
                1
            );

        const double fluxYPrevious =
            coefficient[indexPrevious] *
            gradientYPrevious;

        divergenceY =
            (
                fluxYCenter -
                fluxYPrevious
            ) /
            grid.getDy();
    }
    else {
        const std::size_t indexPrevious =
            grid.getIndex(i, j - 1, k);

        const std::size_t indexNext =
            grid.getIndex(i, j + 1, k);

        const double gradientYPrevious =
            calculateCartesianDerivative(
                grid,
                density,
                i,
                j - 1,
                k,
                1
            );

        const double gradientYNext =
            calculateCartesianDerivative(
                grid,
                density,
                i,
                j + 1,
                k,
                1
            );

        const double fluxYPrevious =
            coefficient[indexPrevious] *
            gradientYPrevious;

        const double fluxYNext =
            coefficient[indexNext] *
            gradientYNext;

        divergenceY =
            (
                fluxYNext -
                fluxYPrevious
            ) /
            (
                2.0 *
                grid.getDy()
            );
    }

    if (k == 0) {
        const std::size_t indexNext =
            grid.getIndex(i, j, k + 1);

        const double gradientZNext =
            calculateCartesianDerivative(
                grid,
                density,
                i,
                j,
                k + 1,
                2
            );

        const double fluxZNext =
            coefficient[indexNext] *
            gradientZNext;

        divergenceZ =
            (
                fluxZNext -
                fluxZCenter
            ) /
            grid.getDz();
    }
    else if (k == nz - 1) {
        const std::size_t indexPrevious =
            grid.getIndex(i, j, k - 1);

        const double gradientZPrevious =
            calculateCartesianDerivative(
                grid,
                density,
                i,
                j,
                k - 1,
                2
            );

        const double fluxZPrevious =
            coefficient[indexPrevious] *
            gradientZPrevious;

        divergenceZ =
            (
                fluxZCenter -
                fluxZPrevious
            ) /
            grid.getDz();
    }
    else {
        const std::size_t indexPrevious =
            grid.getIndex(i, j, k - 1);

        const std::size_t indexNext =
            grid.getIndex(i, j, k + 1);

        const double gradientZPrevious =
            calculateCartesianDerivative(
                grid,
                density,
                i,
                j,
                k - 1,
                2
            );

        const double gradientZNext =
            calculateCartesianDerivative(
                grid,
                density,
                i,
                j,
                k + 1,
                2
            );

        const double fluxZPrevious =
            coefficient[indexPrevious] *
            gradientZPrevious;

        const double fluxZNext =
            coefficient[indexNext] *
            gradientZNext;

        divergenceZ =
            (
                fluxZNext -
                fluxZPrevious
            ) /
            (
                2.0 *
                grid.getDz()
            );
    }

    return
        divergenceX +
        divergenceY +
        divergenceZ;
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
        }
        else {
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

std::vector<double> calculateMolecularSpinExchangeCorrelationPotential(
    const XCFunctional& functional,
    const CartesianGrid& grid,
    const std::vector<double>& alphaDensity,
    const std::vector<double>& betaDensity,
    int spin
) {
    if (spin != 0 && spin != 1) {
        throw std::invalid_argument(
            "El canal de spin debe ser 0 (alpha) o 1 (beta)."
        );
    }

    const std::size_t gridSize =
        grid.getSize();

    if (alphaDensity.size() != gridSize ||
        betaDensity.size() != gridSize) {

        throw std::invalid_argument(
            "La malla cartesiana y las densidades spin deben tener el mismo tamano."
        );
    }

    std::vector<double> densityGradientAlpha(
        gridSize,
        0.0
    );

    std::vector<double> densityGradientBeta(
        gridSize,
        0.0
    );

    std::vector<double> densityDerivative(
        gridSize,
        0.0
    );

    std::vector<double> gradientCoefficient(
        gridSize,
        0.0
    );

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

                const double density =
                    alphaDensity[index] +
                    betaDensity[index];

                if (density <= DFTConstants::RHO_FLOOR) {
                    continue;
                }

                densityGradientAlpha[index] =
                    calculateGradientMagnitude(
                        grid,
                        alphaDensity,
                        i,
                        j,
                        k
                    );

                densityGradientBeta[index] =
                    calculateGradientMagnitude(
                        grid,
                        betaDensity,
                        i,
                        j,
                        k
                    );

                XCInput input;

                input.alphaDensity =
                    alphaDensity[index];

                input.betaDensity =
                    betaDensity[index];

                input.alphaGradient =
                    densityGradientAlpha[index];

                input.betaGradient =
                    densityGradientBeta[index];

                const XCResult result =
                    functional.evaluate(
                        input
                    );

                if (spin == 0) {
                    densityDerivative[index] =
                        result.potentialAlpha;

                    gradientCoefficient[index] =
                        result.gradientCoefficientAlpha;
                }
                else {
                    densityDerivative[index] =
                        result.potentialBeta;

                    gradientCoefficient[index] =
                        result.gradientCoefficientBeta;
                }
            }
        }
    }

    std::vector<double> potential(
        gridSize,
        0.0
    );

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

                const double density =
                    alphaDensity[index] +
                    betaDensity[index];

                if (density <= DFTConstants::RHO_FLOOR) {
                    potential[index] = 0.0;
                    continue;
                }

                const double divergence =
                    calculateCartesianDivergence(
                        grid,
                        gradientCoefficient,
                        spin == 0
                            ? alphaDensity
                            : betaDensity,
                        i,
                        j,
                        k
                    );

                potential[index] =
                    densityDerivative[index] -
                    divergence;
            }
        }
    }

    return potential;
}

double calculateMolecularExchangeCorrelationEnergy(
    const XCFunctional& functional,
    const CartesianGrid& grid,
    const std::vector<double>& alphaDensity,
    const std::vector<double>& betaDensity
) {
    const std::size_t gridSize =
        grid.getSize();

    if (alphaDensity.size() != gridSize ||
        betaDensity.size() != gridSize) {

        throw std::invalid_argument(
            "La malla cartesiana y las densidades spin deben tener el mismo tamano."
        );
    }

    const double volumeElement =
        grid.getDx() *
        grid.getDy() *
        grid.getDz();

    double energy = 0.0;

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

                const double density =
                    alphaDensity[index] +
                    betaDensity[index];

                if (density <= DFTConstants::RHO_FLOOR) {
                    continue;
                }

                const double alphaGradient =
                    calculateGradientMagnitude(
                        grid,
                        alphaDensity,
                        i,
                        j,
                        k
                    );

                const double betaGradient =
                    calculateGradientMagnitude(
                        grid,
                        betaDensity,
                        i,
                        j,
                        k
                    );

                XCInput input;

                input.alphaDensity =
                    alphaDensity[index];

                input.betaDensity =
                    betaDensity[index];

                input.alphaGradient =
                    alphaGradient;

                input.betaGradient =
                    betaGradient;

                const XCResult result =
                    functional.evaluate(
                        input
                    );

                energy +=
                    density *
                    result.energyPerElectron *
                    volumeElement;
            }
        }
    }

    return energy;
}