#include "SelfConsistentFieldMolecule.h"

#include "DFTConstants.h"
#include "EigenvalueSolverMolecule.h"
#include "ElectronDensityMolecule.h"
#include "ExchangeCorrelationMolecule.h"
#include "HartreePotentialMolecule.h"
#include "KohnShamHamiltonianMolecule.h"
#include "NuclearPotential.h"
#include "TotalEnergyMolecule.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

constexpr double MIN_MIXING = 0.10;
constexpr double MAX_MIXING = 0.50;
constexpr double MIXING_INCREASE = 1.10;
constexpr double MIXING_DECREASE = 0.50;
constexpr double OSCILLATION_FACTOR = 1.05;

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

double calculateMolecularDensityDifference(
    const CartesianGrid& grid,
    const std::vector<double>& oldDensity,
    const std::vector<double>& newDensity
) {
    if (oldDensity.size() != grid.getSize() ||
        newDensity.size() != grid.getSize()) {

        throw std::invalid_argument(
            "La malla cartesiana y las densidades deben tener el mismo tamano."
        );
    }

    const double volumeElement =
        grid.getDx() *
        grid.getDy() *
        grid.getDz();

    double differenceNormSquared = 0.0;
    double densityNormSquared = 0.0;

    for (std::size_t i = 0;
         i < grid.getSize();
         ++i) {

        const double difference =
            newDensity[i] -
            oldDensity[i];

        differenceNormSquared +=
            difference *
            difference;

        densityNormSquared +=
            newDensity[i] *
            newDensity[i];
    }

    const double differenceNorm =
        std::sqrt(
            differenceNormSquared *
            volumeElement
        );

    const double densityNorm =
        std::sqrt(
            densityNormSquared *
            volumeElement
        );

    if (densityNorm <= DFTConstants::EPS) {
        return differenceNorm;
    }

    return differenceNorm / densityNorm;
}

double calculateMolecularSpinDensityDifference(
    const CartesianGrid& grid,
    const std::vector<double>& oldAlphaDensity,
    const std::vector<double>& oldBetaDensity,
    const std::vector<double>& newAlphaDensity,
    const std::vector<double>& newBetaDensity
) {
    const double alphaDifference =
        calculateMolecularDensityDifference(
            grid,
            oldAlphaDensity,
            newAlphaDensity
        );

    const double betaDifference =
        calculateMolecularDensityDifference(
            grid,
            oldBetaDensity,
            newBetaDensity
        );

    return std::max(
        alphaDifference,
        betaDifference
    );
}

double calculateMaximumMolecularKSResidual(
    const CartesianGrid& grid,
    const std::vector<double>& alphaPotential,
    const std::vector<double>& betaPotential,
    const std::vector<MolecularOrbital>& orbitals
) {
    if (alphaPotential.size() != grid.getSize() ||
        betaPotential.size() != grid.getSize()) {

        throw std::invalid_argument(
            "Los potenciales moleculares y la malla deben tener el mismo tamano."
        );
    }

    const double volumeElement =
        grid.getDx() *
        grid.getDy() *
        grid.getDz();

    double maximumResidual = 0.0;

    for (const MolecularOrbital& orbital : orbitals) {
        if (orbital.psi.size() != grid.getSize()) {
            throw std::invalid_argument(
                "El orbital molecular y la malla deben tener el mismo tamano."
            );
        }

        const std::vector<double>& effectivePotential =
            orbital.spin == SpinChannel::Alpha
                ? alphaPotential
                : betaPotential;

        const std::vector<double> hPsi =
            applyMolecularKohnShamHamiltonian(
                grid,
                effectivePotential,
                orbital.psi
            );

        double residualNormSquared = 0.0;

        for (std::size_t i = 0;
             i < grid.getSize();
             ++i) {

            const double residual =
                hPsi[i] -
                orbital.eigenvalue *
                orbital.psi[i];

            residualNormSquared +=
                residual *
                residual;
        }

        const double residualNorm =
            std::sqrt(
                residualNormSquared *
                volumeElement
            );

        maximumResidual =
            std::max(
                maximumResidual,
                residualNorm
            );
    }

    return maximumResidual;
}

std::vector<int> buildSpinOccupations(
    int electronCount
) {
    if (electronCount < 0) {
        throw std::invalid_argument(
            "El numero de electrones no puede ser negativo."
        );
    }

    std::vector<int> occupations;

    for (int i = 0;
         i < electronCount;
         ++i) {

        occupations.push_back(1);
    }

    return occupations;
}

void buildInitialMolecularDensities(
    const CartesianGrid& grid,
    const Molecule& molecule,
    int alphaElectrons,
    int betaElectrons,
    std::vector<double>& alphaDensity,
    std::vector<double>& betaDensity
) {
    const std::vector<double> nuclearPotential =
        NuclearPotential::calculate(
            molecule,
            grid
        );

    const std::vector<int> alphaOccupations =
        buildSpinOccupations(
            alphaElectrons
        );

    const std::vector<int> betaOccupations =
        buildSpinOccupations(
            betaElectrons
        );

    const std::vector<MolecularOrbital> alphaOrbitals =
        solveMolecularOrbitals(
            grid,
            nuclearPotential,
            alphaOccupations.size(),
            alphaOccupations,
            SpinChannel::Alpha
        );

    const std::vector<MolecularOrbital> betaOrbitals =
        solveMolecularOrbitals(
            grid,
            nuclearPotential,
            betaOccupations.size(),
            betaOccupations,
            SpinChannel::Beta
        );

    alphaDensity =
        calculateMolecularSpinDensity(
            alphaOrbitals,
            SpinChannel::Alpha
        );

    betaDensity =
        calculateMolecularSpinDensity(
            betaOrbitals,
            SpinChannel::Beta
        );
}

}

MolecularResult solveMolecularSelfConsistentField(
    const CartesianGrid& grid,
    const Molecule& molecule,
    int charge,
    const XCFunctional& functional
) {
    if (grid.getNx() < 2 ||
        grid.getNy() < 2 ||
        grid.getNz() < 2) {

        throw std::invalid_argument(
            "La malla cartesiana debe contener al menos dos puntos por dimension."
        );
    }

    if (molecule.getNucleusCount() == 0) {
        throw std::invalid_argument(
            "La molecula debe contener al menos un nucleo."
        );
    }

    const int nuclearCharge =
        [&molecule]() {
            int total = 0;

            for (const Molecule::Nucleus& nucleus :
                 molecule.getNuclei()) {

                total += nucleus.atomicNumber;
            }

            return total;
        }();

    const int electronCount =
        nuclearCharge -
        charge;

    if (electronCount < 0) {
        throw std::invalid_argument(
            "La carga molecular produce un numero negativo de electrones."
        );
    }

    if (electronCount == 0) {
        throw std::invalid_argument(
            "El SCF molecular requiere al menos un electron."
        );
    }

    const int alphaElectrons =
        (electronCount + 1) / 2;

    const int betaElectrons =
        electronCount / 2;

    const std::vector<double> nuclearPotential =
        NuclearPotential::calculate(
            molecule,
            grid
        );

    std::vector<double> alphaDensity;
    std::vector<double> betaDensity;

    buildInitialMolecularDensities(
        grid,
        molecule,
        alphaElectrons,
        betaElectrons,
        alphaDensity,
        betaDensity
    );

    if (alphaDensity.size() != grid.getSize() ||
        betaDensity.size() != grid.getSize()) {

        throw std::runtime_error(
            "Las densidades moleculares iniciales tienen un tamano incorrecto."
        );
    }

    const std::vector<int> alphaOccupations =
        buildSpinOccupations(
            alphaElectrons
        );

    const std::vector<int> betaOccupations =
        buildSpinOccupations(
            betaElectrons
        );

    double previousEnergy =
        std::numeric_limits<double>::infinity();

    double previousDensityDifference =
        std::numeric_limits<double>::infinity();

    double mixing =
        DFTConstants::MIXING;

    MolecularResult molecularResult;

    molecularResult.electrons =
        electronCount;

    molecularResult.charge =
        charge;

    SCFResult& result =
        molecularResult.scf;

    for (int iteration = 1;
         iteration <= DFTConstants::MAX_SCF_ITERATIONS;
         ++iteration) {

        std::vector<double> density(
            grid.getSize(),
            0.0
        );

        for (std::size_t i = 0;
             i < grid.getSize();
             ++i) {

            density[i] =
                alphaDensity[i] +
                betaDensity[i];
        }

        const std::vector<double> hartreePotential =
            calculateHartreePotential(
                grid,
                density
            );

        const std::vector<double> alphaXCPotential =
            calculateMolecularSpinExchangeCorrelationPotential(
                functional,
                grid,
                alphaDensity,
                betaDensity,
                0
            );

        const std::vector<double> betaXCPotential =
            calculateMolecularSpinExchangeCorrelationPotential(
                functional,
                grid,
                alphaDensity,
                betaDensity,
                1
            );

        std::vector<double> alphaPotential(
            grid.getSize(),
            0.0
        );

        std::vector<double> betaPotential(
            grid.getSize(),
            0.0
        );

        for (std::size_t i = 0;
             i < grid.getSize();
             ++i) {

            alphaPotential[i] =
                nuclearPotential[i] +
                hartreePotential[i] +
                alphaXCPotential[i];

            betaPotential[i] =
                nuclearPotential[i] +
                hartreePotential[i] +
                betaXCPotential[i];
        }

        const std::vector<MolecularOrbital> alphaOrbitals =
            solveMolecularOrbitals(
                grid,
                alphaPotential,
                alphaOccupations.size(),
                alphaOccupations,
                SpinChannel::Alpha
            );

        const std::vector<MolecularOrbital> betaOrbitals =
            solveMolecularOrbitals(
                grid,
                betaPotential,
                betaOccupations.size(),
                betaOccupations,
                SpinChannel::Beta
            );

        std::vector<MolecularOrbital> orbitals;

        orbitals.reserve(
            alphaOrbitals.size() +
            betaOrbitals.size()
        );

        orbitals.insert(
            orbitals.end(),
            alphaOrbitals.begin(),
            alphaOrbitals.end()
        );

        orbitals.insert(
            orbitals.end(),
            betaOrbitals.begin(),
            betaOrbitals.end()
        );

        const std::vector<double> outputAlphaDensity =
            calculateMolecularSpinDensity(
                alphaOrbitals,
                SpinChannel::Alpha
            );

        const std::vector<double> outputBetaDensity =
            calculateMolecularSpinDensity(
                betaOrbitals,
                SpinChannel::Beta
            );

        std::vector<double> oldDensity(
            grid.getSize(),
            0.0
        );

        std::vector<double> outputDensity(
            grid.getSize(),
            0.0
        );

        for (std::size_t i = 0;
             i < grid.getSize();
             ++i) {

            oldDensity[i] =
                alphaDensity[i] +
                betaDensity[i];

            outputDensity[i] =
                outputAlphaDensity[i] +
                outputBetaDensity[i];
        }

        const std::vector<double> outputHartreePotential =
            calculateHartreePotential(
                grid,
                outputDensity
            );

        const EnergyComponents energy =
            calculateMolecularTotalEnergy(
                functional,
                molecule,
                grid,
                outputAlphaDensity,
                outputBetaDensity,
                orbitals,
                outputHartreePotential,
                nuclearPotential
            );

        const double densityDifference =
            calculateMolecularDensityDifference(
                grid,
                oldDensity,
                outputDensity
            );

        const double spinDensityDifference =
            calculateMolecularSpinDensityDifference(
                grid,
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
            calculateMaximumMolecularKSResidual(
                grid,
                alphaPotential,
                betaPotential,
                orbitals
            );

        result.orbitals.clear();

        result.molecularOrbitals =
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

    return molecularResult;
}