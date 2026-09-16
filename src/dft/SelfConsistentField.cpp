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
#include <vector>

namespace {

std::vector<AtomicOrbital> solveOrbitals(
    const std::vector<double>& r,
    const std::vector<double>& effectivePotential,
    const AtomicConfiguration& configuration
) {
    std::vector<AtomicOrbital> orbitals;
    std::map<int, std::size_t> orbitalIndices;

    for (const ElectronicState& state : configuration.states) {
        const TridiagonalMatrix hamiltonian =
            buildKohnShamHamiltonian(
                r,
                effectivePotential,
                state.l
            );

        const std::size_t stateIndex =
            orbitalIndices[state.l]++;

        orbitals.push_back(
            solveOrbital(
                hamiltonian,
                r,
                state.n,
                state.l,
                state.electrons,
                stateIndex
            )
        );
    }

    return orbitals;
}

double calculateMaximumKSResidual(
    const std::vector<double>& r,
    const std::vector<double>& effectivePotential,
    const std::vector<AtomicOrbital>& orbitals
) {
    if (r.size() < 2) {
        throw std::invalid_argument(
            "La malla radial debe contener al menos dos puntos."
        );
    }

    const double dr = r[1] - r[0];

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

        const TridiagonalMatrix hamiltonian =
            buildKohnShamHamiltonian(
                r,
                effectivePotential,
                orbital.l
            );

        double residualNorm = 0.0;

        for (std::size_t i = 0; i < r.size(); ++i) {
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
                residual * residual;
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

std::vector<double> buildInitialDensity(
    const std::vector<double>& r,
    const AtomicConfiguration& configuration
) {
    std::vector<AtomicOrbital> orbitals;

    for (const ElectronicState& state : configuration.states) {
        AtomicOrbital orbital;
        orbital.n = state.n;
        orbital.l = state.l;
        orbital.electrons = state.electrons;
        orbital.u.resize(r.size());

        const double scale =
            std::max(
                0.1,
                static_cast<double>(state.n) /
                static_cast<double>(configuration.Z)
            );

        for (std::size_t i = 0; i < r.size(); ++i) {
            const double x =
                r[i] / scale;

            orbital.u[i] =
                std::pow(
                    r[i],
                    state.l + 1
                ) *
                std::exp(-x);
        }

        normalizeVector(
            orbital.u,
            r[1] - r[0]
        );

        orbitals.push_back(
            std::move(orbital)
        );
    }

    return calculateElectronDensity(
        r,
        orbitals
    );
}

std::vector<double> buildEffectivePotential(
    const std::vector<double>& r,
    const std::vector<double>& density,
    int Z
) {
    const std::vector<double> hartree =
        calculateHartreePotential(
            r,
            density
        );

    const std::vector<double> exchangeCorrelation =
        calculateExchangeCorrelationPotential(
            density
        );

    std::vector<double> potential(r.size());

    for (std::size_t i = 0; i < r.size(); ++i) {
        potential[i] =
            -static_cast<double>(Z) / r[i] +
            hartree[i] +
            exchangeCorrelation[i];
    }

    return potential;
}

}

SCFResult solveSelfConsistentField(
    const RadialGrid& grid,
    const AtomicConfiguration& configuration
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

    std::vector<double> density =
        buildInitialDensity(
            r,
            configuration
        );

    double previousEnergy =
        std::numeric_limits<double>::infinity();

    SCFResult result;

    for (int iteration = 1;
         iteration <= DFTConstants::MAX_SCF_ITERATIONS;
         ++iteration) {

        const std::vector<double> effectivePotential =
            buildEffectivePotential(
                r,
                density,
                configuration.Z
            );

        const std::vector<AtomicOrbital> orbitals =
            solveOrbitals(
                r,
                effectivePotential,
                configuration
            );

        const std::vector<double> outputDensity =
            calculateElectronDensity(
                r,
                orbitals
            );

        const std::vector<double> hartreePotential =
            calculateHartreePotential(
                r,
                outputDensity
            );

        const EnergyComponents energy =
            calculateTotalEnergy(
                r,
                outputDensity,
                orbitals,
                hartreePotential,
                configuration.Z
            );

        const double densityDifference =
            maxAbsoluteDifference(
                density,
                outputDensity
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
                effectivePotential,
                orbitals
            );

        result.orbitals =
            orbitals;

        result.density =
            outputDensity;

        result.effectivePotential =
            effectivePotential;

        result.energy =
            energy;

        result.densityDifference =
            densityDifference;

        result.energyDifference =
            energyDifference;

        result.maxKSResidual =
            residual;

        result.iterations =
            iteration;

        if (densityDifference <
                DFTConstants::DENSITY_TOL &&
            energyDifference <
                DFTConstants::ENERGY_TOL &&
            residual <
                DFTConstants::KS_RESIDUAL_TOL) {

            result.converged = true;
            break;
        }

        std::vector<double> mixedDensity(
            density.size()
        );

        for (std::size_t i = 0;
             i < density.size();
             ++i) {

            mixedDensity[i] =
                (1.0 - DFTConstants::MIXING) *
                density[i] +
                DFTConstants::MIXING *
                outputDensity[i];
        }

        density =
            std::move(mixedDensity);

        previousEnergy =
            energy.total;
    }

    return result;
}