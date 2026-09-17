#include "SelfConsistentField.h"

#include "DFTConstants.h"
#include "EigenvalueSolver.h"
#include "ElectronDensity.h"
#include "ExchangeCorrelation.h"
#include "HartreePotential.h"
#include "KohnShamHamiltonian.h"
#include "NumericalMethods.h"
#include "TotalEnergy.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

constexpr double MIN_MIXING = 0.10;
constexpr double MAX_MIXING = 0.50;
constexpr double MIXING_INCREASE = 1.10;
constexpr double MIXING_DECREASE = 0.50;
constexpr double OSCILLATION_FACTOR = 1.05;

std::vector<AtomicOrbital> solveSpinOrbitals(
    const std::vector<double>& r,
    const std::vector<double>& effectivePotential,
    const AtomicConfiguration& configuration,
    SpinChannel spin
) {
    std::vector<AtomicOrbital> orbitals;
    std::map<int, std::size_t> orbitalIndices;

    for (const ElectronicState& state :
         configuration.states) {

        const int electrons =
            spin == SpinChannel::Alpha
                ? state.alphaElectrons
                : state.betaElectrons;

        if (electrons <= 0) {
            continue;
        }

        const TridiagonalMatrix hamiltonian =
            buildKohnShamHamiltonian(
                r,
                effectivePotential,
                state.l
            );

        const std::size_t stateIndex =
            orbitalIndices[state.l]++;

        AtomicOrbital orbital =
            solveOrbital(
                hamiltonian,
                r,
                state.n,
                state.l,
                electrons,
                stateIndex
            );

        orbital.spin = spin;

        orbitals.push_back(
            std::move(orbital)
        );
    }

    return orbitals;
}

std::vector<AtomicOrbital> solveAllSpinOrbitals(
    const std::vector<double>& r,
    const std::vector<double>& alphaPotential,
    const std::vector<double>& betaPotential,
    const AtomicConfiguration& configuration
) {
    std::vector<AtomicOrbital> orbitals =
        solveSpinOrbitals(
            r,
            alphaPotential,
            configuration,
            SpinChannel::Alpha
        );

    const std::vector<AtomicOrbital> betaOrbitals =
        solveSpinOrbitals(
            r,
            betaPotential,
            configuration,
            SpinChannel::Beta
        );

    orbitals.insert(
        orbitals.end(),
        betaOrbitals.begin(),
        betaOrbitals.end()
    );

    return orbitals;
}

double calculateMaximumKSResidual(
    const std::vector<double>& r,
    const std::vector<double>& alphaPotential,
    const std::vector<double>& betaPotential,
    const std::vector<AtomicOrbital>& orbitals
) {
    if (r.size() < 2) {
        throw std::invalid_argument(
            "La malla radial debe contener al menos dos puntos."
        );
    }

    if (alphaPotential.size() != r.size() ||
        betaPotential.size() != r.size()) {

        throw std::invalid_argument(
            "Los potenciales spin y la malla radial deben tener el mismo tamano."
        );
    }

    const double dr =
        r[1] - r[0];

    if (dr <= 0.0) {
        throw std::invalid_argument(
            "El paso radial debe ser mayor que cero."
        );
    }

    double maximumResidual = 0.0;

    for (const AtomicOrbital& orbital : orbitals) {
        if (orbital.u.size() != r.size()) {
            throw std::invalid_argument(
                "El orbital y la malla radial deben tener el mismo tamano."
            );
        }

        const std::vector<double>& effectivePotential =
            orbital.spin == SpinChannel::Alpha
                ? alphaPotential
                : betaPotential;

        const TridiagonalMatrix hamiltonian =
            buildKohnShamHamiltonian(
                r,
                effectivePotential,
                orbital.l
            );

        double residualNorm = 0.0;

        for (std::size_t i = 0;
             i < r.size();
             ++i) {

            double value =
                hamiltonian.diagonal[i] *
                orbital.u[i];

            if (i > 0) {
                value +=
                    hamiltonian.lower[i - 1] *
                    orbital.u[i - 1];
            }

            if (i + 1 < r.size()) {
                value +=
                    hamiltonian.upper[i] *
                    orbital.u[i + 1];
            }

            const double residual =
                value -
                orbital.eigenvalue *
                orbital.u[i];

            residualNorm +=
                residual *
                residual;
        }

        residualNorm =
            std::sqrt(
                residualNorm * dr
            );

        maximumResidual =
            std::max(
                maximumResidual,
                residualNorm
            );
    }

    return maximumResidual;
}

void buildInitialSpinOrbitals(
    const std::vector<double>& r,
    const AtomicConfiguration& configuration,
    std::vector<AtomicOrbital>& alphaOrbitals,
    std::vector<AtomicOrbital>& betaOrbitals
) {
    alphaOrbitals.clear();
    betaOrbitals.clear();

    const double Z =
        static_cast<double>(
            configuration.Z
        );

    for (const ElectronicState& state :
         configuration.states) {

        const double exponent =
            std::max(
                0.15,
                Z /
                (
                    static_cast<double>(state.n) *
                    static_cast<double>(state.n)
                )
            );

        if (state.alphaElectrons > 0) {
            AtomicOrbital orbital;

            orbital.n = state.n;
            orbital.l = state.l;
            orbital.spin = SpinChannel::Alpha;
            orbital.electrons =
                state.alphaElectrons;

            orbital.u.resize(
                r.size()
            );

            for (std::size_t i = 0;
                 i < r.size();
                 ++i) {

                orbital.u[i] =
                    std::pow(
                        r[i],
                        state.l + 1
                    ) *
                    std::exp(
                        -exponent * r[i]
                    );
            }

            normalizeVector(
                orbital.u,
                r[1] - r[0]
            );

            alphaOrbitals.push_back(
                std::move(orbital)
            );
        }

        if (state.betaElectrons > 0) {
            AtomicOrbital orbital;

            orbital.n = state.n;
            orbital.l = state.l;
            orbital.spin = SpinChannel::Beta;
            orbital.electrons =
                state.betaElectrons;

            orbital.u.resize(
                r.size()
            );

            for (std::size_t i = 0;
                 i < r.size();
                 ++i) {

                orbital.u[i] =
                    std::pow(
                        r[i],
                        state.l + 1
                    ) *
                    std::exp(
                        -exponent * r[i]
                    );
            }

            normalizeVector(
                orbital.u,
                r[1] - r[0]
            );

            betaOrbitals.push_back(
                std::move(orbital)
            );
        }
    }
}

void buildInitialSpinDensities(
    const std::vector<double>& r,
    const AtomicConfiguration& configuration,
    std::vector<double>& alphaDensity,
    std::vector<double>& betaDensity
) {
    std::vector<AtomicOrbital> alphaOrbitals;
    std::vector<AtomicOrbital> betaOrbitals;

    buildInitialSpinOrbitals(
        r,
        configuration,
        alphaOrbitals,
        betaOrbitals
    );

    alphaDensity =
        calculateSpinDensity(
            r,
            alphaOrbitals,
            SpinChannel::Alpha
        );

    betaDensity =
        calculateSpinDensity(
            r,
            betaOrbitals,
            SpinChannel::Beta
        );
}

void buildEffectivePotentials(
    const std::vector<double>& r,
    const std::vector<double>& alphaDensity,
    const std::vector<double>& betaDensity,
    int Z,
    const XCFunctional& functional,
    std::vector<double>& alphaPotential,
    std::vector<double>& betaPotential
) {
    if (alphaDensity.size() != betaDensity.size() ||
        alphaDensity.size() != r.size()) {

        throw std::invalid_argument(
            "La malla y las densidades spin deben tener el mismo tamano."
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

    const std::vector<double> hartree =
        calculateHartreePotential(
            r,
            density
        );

    alphaPotential =
        calculateSpinExchangeCorrelationPotential(
            functional,
            r,
            alphaDensity,
            betaDensity,
            0
        );

    betaPotential =
        calculateSpinExchangeCorrelationPotential(
            functional,
            r,
            alphaDensity,
            betaDensity,
            1
        );

    for (std::size_t i = 0;
         i < r.size();
         ++i) {

        const double nuclearPotential =
            -static_cast<double>(Z) /
            r[i];

        alphaPotential[i] +=
            nuclearPotential +
            hartree[i];

        betaPotential[i] +=
            nuclearPotential +
            hartree[i];
    }
}

double calculateDensityDifference(
    const std::vector<double>& r,
    const std::vector<double>& oldDensity,
    const std::vector<double>& newDensity
) {
    if (r.size() != oldDensity.size() ||
        r.size() != newDensity.size()) {

        throw std::invalid_argument(
            "La malla y las densidades deben tener el mismo tamano."
        );
    }

    if (r.size() < 2) {
        throw std::invalid_argument(
            "La malla radial debe contener al menos dos puntos."
        );
    }

    const double dr =
        r[1] - r[0];

    if (dr <= 0.0) {
        throw std::invalid_argument(
            "El paso radial debe ser mayor que cero."
        );
    }

    double differenceNormSquared = 0.0;
    double densityNormSquared = 0.0;

    for (std::size_t i = 0;
         i < r.size();
         ++i) {

        const double weight =
            4.0 *
            DFTConstants::PI *
            r[i] *
            r[i];

        const double difference =
            newDensity[i] -
            oldDensity[i];

        differenceNormSquared +=
            difference *
            difference *
            weight;

        densityNormSquared +=
            newDensity[i] *
            newDensity[i] *
            weight;
    }

    const double differenceNorm =
        std::sqrt(
            differenceNormSquared * dr
        );

    const double densityNorm =
        std::sqrt(
            densityNormSquared * dr
        );

    if (densityNorm <= DFTConstants::EPS) {
        return differenceNorm;
    }

    return differenceNorm / densityNorm;
}

double calculateSpinDensityDifference(
    const std::vector<double>& r,
    const std::vector<double>& oldAlphaDensity,
    const std::vector<double>& oldBetaDensity,
    const std::vector<double>& newAlphaDensity,
    const std::vector<double>& newBetaDensity
) {
    const double alphaDifference =
        calculateDensityDifference(
            r,
            oldAlphaDensity,
            newAlphaDensity
        );

    const double betaDifference =
        calculateDensityDifference(
            r,
            oldBetaDensity,
            newBetaDensity
        );

    return std::max(
        alphaDifference,
        betaDifference
    );
}

void mixDensity(
    std::vector<double>& density,
    const std::vector<double>& output,
    double mixing
) {
    if (density.size() != output.size()) {
        throw std::invalid_argument(
            "Las densidades deben tener el mismo tamano."
        );
    }

    for (std::size_t i = 0;
         i < density.size();
         ++i) {

        density[i] =
            (1.0 - mixing) *
            density[i] +
            mixing *
            output[i];
    }
}

}

SCFResult solveSelfConsistentField(
    const RadialGrid& grid,
    const AtomicConfiguration& configuration,
    const XCFunctional& functional
) {
    const std::vector<double>& r =
        grid.coordinates();

    if (r.size() < 2) {
        throw std::invalid_argument(
            "La malla radial debe contener al menos dos puntos."
        );
    }

    if (configuration.Z <= 0) {
        throw std::invalid_argument(
            "El numero atomico debe ser mayor que cero."
        );
    }

    std::vector<double> alphaDensity;
    std::vector<double> betaDensity;

    buildInitialSpinDensities(
        r,
        configuration,
        alphaDensity,
        betaDensity
    );

    double previousEnergy =
        std::numeric_limits<double>::infinity();

    double previousDensityDifference =
        std::numeric_limits<double>::infinity();

    double mixing =
        DFTConstants::MIXING;

    SCFResult result;

    for (int iteration = 1;
         iteration <= DFTConstants::MAX_SCF_ITERATIONS;
         ++iteration) {

        std::vector<double> alphaPotential;
        std::vector<double> betaPotential;

        buildEffectivePotentials(
            r,
            alphaDensity,
            betaDensity,
            configuration.Z,
            functional,
            alphaPotential,
            betaPotential
        );

        const std::vector<AtomicOrbital> orbitals =
            solveAllSpinOrbitals(
                r,
                alphaPotential,
                betaPotential,
                configuration
            );

        const std::vector<double> outputAlphaDensity =
            calculateSpinDensity(
                r,
                orbitals,
                SpinChannel::Alpha
            );

        const std::vector<double> outputBetaDensity =
            calculateSpinDensity(
                r,
                orbitals,
                SpinChannel::Beta
            );

        std::vector<double> oldDensity(
            r.size(),
            0.0
        );

        std::vector<double> outputDensity(
            r.size(),
            0.0
        );

        for (std::size_t i = 0;
             i < r.size();
             ++i) {

            oldDensity[i] =
                alphaDensity[i] +
                betaDensity[i];

            outputDensity[i] =
                outputAlphaDensity[i] +
                outputBetaDensity[i];
        }

        const std::vector<double> hartreePotential =
            calculateHartreePotential(
                r,
                outputDensity
            );

        const EnergyComponents energy =
            calculateTotalEnergy(
                functional,
                r,
                outputAlphaDensity,
                outputBetaDensity,
                orbitals,
                hartreePotential,
                configuration.Z
            );

        const double densityDifference =
            calculateDensityDifference(
                r,
                oldDensity,
                outputDensity
            );

        const double spinDensityDifference =
            calculateSpinDensityDifference(
                r,
                alphaDensity,
                betaDensity,
                outputAlphaDensity,
                outputBetaDensity
            );

        const double effectiveDensityDifference =
            std::max(
                densityDifference,
                spinDensityDifference
            );

        const double energyDifference =
            std::isfinite(previousEnergy)
                ? std::abs(
                    energy.total -
                    previousEnergy
                )
                : std::numeric_limits<double>::infinity();

        const double residual =
            calculateMaximumKSResidual(
                r,
                alphaPotential,
                betaPotential,
                orbitals
            );

        result.orbitals =
            orbitals;

        result.alphaDensity =
            outputAlphaDensity;

        result.betaDensity =
            outputBetaDensity;

        result.density =
            outputDensity;

        result.alphaEffectivePotential =
            alphaPotential;

        result.betaEffectivePotential =
            betaPotential;

        result.effectivePotential =
            alphaPotential;

        result.energy =
            energy;

        result.densityDifference =
            effectiveDensityDifference;

        result.energyDifference =
            energyDifference;

        result.maxKSResidual =
            residual;

        result.iterations =
            iteration;

        if (effectiveDensityDifference <
                DFTConstants::DENSITY_TOL &&
            energyDifference <
                DFTConstants::ENERGY_TOL &&
            residual <
                DFTConstants::KS_RESIDUAL_TOL) {

            result.converged = true;
            break;
        }

        if (std::isfinite(previousDensityDifference)) {
            if (effectiveDensityDifference >
                previousDensityDifference *
                OSCILLATION_FACTOR) {

                mixing =
                    std::max(
                        MIN_MIXING,
                        mixing * MIXING_DECREASE
                    );
            }
            else if (effectiveDensityDifference <
                     previousDensityDifference) {

                mixing =
                    std::min(
                        MAX_MIXING,
                        mixing * MIXING_INCREASE
                    );
            }
        }

        mixDensity(
            alphaDensity,
            outputAlphaDensity,
            mixing
        );

        mixDensity(
            betaDensity,
            outputBetaDensity,
            mixing
        );

        previousDensityDifference =
            effectiveDensityDifference;

        previousEnergy =
            energy.total;
    }

    return result;
}