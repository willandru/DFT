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
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

constexpr double MIN_MIXING = 0.10;
constexpr double MAX_MIXING = 0.50;
constexpr double MIXING_INCREASE = 1.10;
constexpr double MIXING_DECREASE = 0.50;
constexpr double OSCILLATION_FACTOR = 1.05;

struct MolecularSCFTiming
{
    double initialization = 0.0;

    double density = 0.0;
    double hartree = 0.0;

    double exchangeCorrelationAlpha = 0.0;
    double exchangeCorrelationBeta = 0.0;

    double potentialConstruction = 0.0;

    double orbitalAlpha = 0.0;
    double orbitalBeta = 0.0;

    double outputDensity = 0.0;
    double outputHartree = 0.0;

    double totalEnergy = 0.0;

    double densityDifference = 0.0;
    double ksResidual = 0.0;
    double mixing = 0.0;

    double totalIterations = 0.0;
};

struct MolecularSCFIterationTiming
{
    double density = 0.0;
    double hartree = 0.0;

    double exchangeCorrelationAlpha = 0.0;
    double exchangeCorrelationBeta = 0.0;

    double potentialConstruction = 0.0;

    double orbitalAlpha = 0.0;
    double orbitalBeta = 0.0;

    double outputDensity = 0.0;
    double outputHartree = 0.0;

    double totalEnergy = 0.0;

    double densityDifference = 0.0;
    double ksResidual = 0.0;
    double mixing = 0.0;

    double total = 0.0;
};

double elapsedSeconds(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end
)
{
    return std::chrono::duration<double>(
        end - start
    ).count();
}

void mixDensity(
    std::vector<double>& density,
    const std::vector<double>& output,
    double mixing
)
{
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
)
{
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
)
{
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
)
{
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
)
{
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

std::vector<MolecularOrbital> convertOrbitalsToSpin(
    const std::vector<MolecularOrbital>& orbitals,
    SpinChannel spin
)
{
    std::vector<MolecularOrbital> converted =
        orbitals;

    for (MolecularOrbital& orbital : converted) {
        orbital.spin = spin;
    }

    return converted;
}

/*
 * Ejecuta el solver molecular sin permitir que sus diagnosticos
 * internos lleguen a la salida principal.
 *
 * El solver conserva exactamente su calculo y sus resultados.
 */
std::vector<MolecularOrbital> solveMolecularOrbitalsSilently(
    const CartesianGrid& grid,
    const std::vector<double>& potential,
    std::size_t orbitalCount,
    const std::vector<int>& occupations,
    SpinChannel spin
)
{
    std::ostringstream suppressedOutput;

    std::streambuf* originalBuffer =
        std::cout.rdbuf(
            suppressedOutput.rdbuf()
        );

    try {

        const std::vector<MolecularOrbital> orbitals =
            solveMolecularOrbitals(
                grid,
                potential,
                orbitalCount,
                occupations,
                spin
            );

        std::cout.rdbuf(originalBuffer);

        return orbitals;
    }
    catch (...) {

        std::cout.rdbuf(originalBuffer);

        throw;
    }
}

void printSCFHeader()
{
    std::cout
        << "\n"
        << "==============================================================================================================\n"
        << "SCF MOLECULAR\n"
        << "==============================================================================================================\n"
        << "Iter | E (Ha)         | dE         | dRho      | KS        | Mix   | E Alpha      | E Beta       | tA(ms) | tB(ms) | tSCF(s)\n"
        << "-----|----------------|------------|-----------|-----------|-------|--------------|--------------|--------|--------|--------\n";
}

void printSCFIteration(
    int iteration,
    double energy,
    double energyDifference,
    double densityDifference,
    double residual,
    double mixing,
    double iterationTime,
    const std::vector<MolecularOrbital>& alphaOrbitals,
    const std::vector<MolecularOrbital>& betaOrbitals,
    double alphaTime,
    double betaTime
)
{
    std::cout
        << std::right
        << std::setw(4)
        << iteration
        << " | ";

    std::cout
        << std::scientific
        << std::setprecision(10)
        << std::setw(14)
        << energy
        << " | ";

    if (std::isfinite(energyDifference)) {
        std::cout
            << std::scientific
            << std::setprecision(4)
            << std::setw(10)
            << energyDifference;
    }
    else {
        std::cout
            << std::setw(10)
            << "inf";
    }

    std::cout
        << " | "
        << std::scientific
        << std::setprecision(4)
        << std::setw(9)
        << densityDifference
        << " | "
        << std::scientific
        << std::setprecision(4)
        << std::setw(9)
        << residual
        << " | "
        << std::fixed
        << std::setprecision(3)
        << std::setw(5)
        << mixing
        << " | ";

    if (!alphaOrbitals.empty()) {
        std::cout
            << std::scientific
            << std::setprecision(6)
            << std::setw(12)
            << alphaOrbitals.front().eigenvalue;
    }
    else {
        std::cout
            << std::setw(12)
            << "N/A";
    }

    std::cout
        << " | ";

    if (!betaOrbitals.empty()) {
        std::cout
            << std::scientific
            << std::setprecision(6)
            << std::setw(12)
            << betaOrbitals.front().eigenvalue;
    }
    else {
        std::cout
            << std::setw(12)
            << "N/A";
    }

    std::cout
        << " | "
        << std::fixed
        << std::setprecision(3)
        << std::setw(6)
        << alphaTime * 1000.0
        << " | "
        << std::setw(6)
        << betaTime * 1000.0
        << " | "
        << std::setw(7)
        << iterationTime
        << "\n";
}

void buildInitialMolecularDensities(
    const CartesianGrid& grid,
    const Molecule& molecule,
    int alphaElectrons,
    int betaElectrons,
    std::vector<double>& alphaDensity,
    std::vector<double>& betaDensity
)
{
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
        solveMolecularOrbitalsSilently(
            grid,
            nuclearPotential,
            alphaOccupations.size(),
            alphaOccupations,
            SpinChannel::Alpha
        );

    std::vector<MolecularOrbital> betaOrbitals;

    if (alphaElectrons == betaElectrons) {
        betaOrbitals =
            convertOrbitalsToSpin(
                alphaOrbitals,
                SpinChannel::Beta
            );
    }
    else {
        betaOrbitals =
            solveMolecularOrbitalsSilently(
                grid,
                nuclearPotential,
                betaOccupations.size(),
                betaOccupations,
                SpinChannel::Beta
            );
    }

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

void printTimeLine(
    const char* label,
    double seconds
)
{
    std::cout
        << std::left
        << std::setw(30)
        << label
        << std::right
        << std::fixed
        << std::setprecision(3)
        << std::setw(12)
        << seconds * 1000.0
        << " ms\n";
}

}

MolecularResult solveMolecularSelfConsistentField(
    const CartesianGrid& grid,
    const Molecule& molecule,
    int charge,
    const XCFunctional& functional
)
{
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

    const bool closedShell =
        alphaElectrons == betaElectrons;

    MolecularSCFTiming timing;

    const auto scfStart =
        std::chrono::steady_clock::now();

    const auto initializationStart =
        std::chrono::steady_clock::now();

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

    const auto initializationEnd =
        std::chrono::steady_clock::now();

    timing.initialization =
        elapsedSeconds(
            initializationStart,
            initializationEnd
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

    printSCFHeader();

    for (int iteration = 1;
         iteration <= DFTConstants::MAX_SCF_ITERATIONS;
         ++iteration) {

        const auto iterationStart =
            std::chrono::steady_clock::now();

        MolecularSCFIterationTiming iterationTiming;

        auto start =
            std::chrono::steady_clock::now();

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

        auto end =
            std::chrono::steady_clock::now();

        iterationTiming.density =
            elapsedSeconds(start, end);

        timing.density +=
            iterationTiming.density;

        start =
            std::chrono::steady_clock::now();

        const std::vector<double> hartreePotential =
            calculateHartreePotential(
                grid,
                density
            );

        end =
            std::chrono::steady_clock::now();

        iterationTiming.hartree =
            elapsedSeconds(start, end);

        timing.hartree +=
            iterationTiming.hartree;

        start =
            std::chrono::steady_clock::now();

        const std::vector<double> alphaXCPotential =
            calculateMolecularSpinExchangeCorrelationPotential(
                functional,
                grid,
                alphaDensity,
                betaDensity,
                0
            );

        end =
            std::chrono::steady_clock::now();

        iterationTiming.exchangeCorrelationAlpha =
            elapsedSeconds(start, end);

        timing.exchangeCorrelationAlpha +=
            iterationTiming.exchangeCorrelationAlpha;

        std::vector<double> betaXCPotential;

        if (closedShell) {

            betaXCPotential =
                alphaXCPotential;

            iterationTiming.exchangeCorrelationBeta =
                0.0;
        }
        else {

            start =
                std::chrono::steady_clock::now();

            betaXCPotential =
                calculateMolecularSpinExchangeCorrelationPotential(
                    functional,
                    grid,
                    alphaDensity,
                    betaDensity,
                    1
                );

            end =
                std::chrono::steady_clock::now();

            iterationTiming.exchangeCorrelationBeta =
                elapsedSeconds(start, end);
        }

        timing.exchangeCorrelationBeta +=
            iterationTiming.exchangeCorrelationBeta;

        start =
            std::chrono::steady_clock::now();

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

        end =
            std::chrono::steady_clock::now();

        iterationTiming.potentialConstruction =
            elapsedSeconds(start, end);

        timing.potentialConstruction +=
            iterationTiming.potentialConstruction;

        start =
            std::chrono::steady_clock::now();

        const std::vector<MolecularOrbital> alphaOrbitals =
            solveMolecularOrbitalsSilently(
                grid,
                alphaPotential,
                alphaOccupations.size(),
                alphaOccupations,
                SpinChannel::Alpha
            );

        end =
            std::chrono::steady_clock::now();

        iterationTiming.orbitalAlpha =
            elapsedSeconds(start, end);

        timing.orbitalAlpha +=
            iterationTiming.orbitalAlpha;

        std::vector<MolecularOrbital> betaOrbitals;

        if (closedShell) {

            betaOrbitals =
                convertOrbitalsToSpin(
                    alphaOrbitals,
                    SpinChannel::Beta
                );

            iterationTiming.orbitalBeta =
                0.0;
        }
        else {

            start =
                std::chrono::steady_clock::now();

            betaOrbitals =
                solveMolecularOrbitalsSilently(
                    grid,
                    betaPotential,
                    betaOccupations.size(),
                    betaOccupations,
                    SpinChannel::Beta
                );

            end =
                std::chrono::steady_clock::now();

            iterationTiming.orbitalBeta =
                elapsedSeconds(start, end);
        }

        timing.orbitalBeta +=
            iterationTiming.orbitalBeta;

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

        start =
            std::chrono::steady_clock::now();

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

        end =
            std::chrono::steady_clock::now();

        iterationTiming.outputDensity =
            elapsedSeconds(start, end);

        timing.outputDensity +=
            iterationTiming.outputDensity;

        start =
            std::chrono::steady_clock::now();

        const std::vector<double> outputHartreePotential =
            calculateHartreePotential(
                grid,
                outputDensity
            );

        end =
            std::chrono::steady_clock::now();

        iterationTiming.outputHartree =
            elapsedSeconds(start, end);

        timing.outputHartree +=
            iterationTiming.outputHartree;

        start =
            std::chrono::steady_clock::now();

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

        end =
            std::chrono::steady_clock::now();

        iterationTiming.totalEnergy =
            elapsedSeconds(start, end);

        timing.totalEnergy +=
            iterationTiming.totalEnergy;

        start =
            std::chrono::steady_clock::now();

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

        end =
            std::chrono::steady_clock::now();

        iterationTiming.densityDifference =
            elapsedSeconds(start, end);

        timing.densityDifference +=
            iterationTiming.densityDifference;

        start =
            std::chrono::steady_clock::now();

        const double residual =
            calculateMaximumMolecularKSResidual(
                grid,
                alphaPotential,
                betaPotential,
                orbitals
            );

        end =
            std::chrono::steady_clock::now();

        iterationTiming.ksResidual =
            elapsedSeconds(start, end);

        timing.ksResidual +=
            iterationTiming.ksResidual;

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

        const bool converged =
            effectiveDensityDifference <
                DFTConstants::DENSITY_TOL &&
            energyDifference <
                DFTConstants::ENERGY_TOL &&
            residual <
                DFTConstants::KS_RESIDUAL_TOL;

        if (converged) {

            result.converged = true;

            const auto iterationEnd =
                std::chrono::steady_clock::now();

            iterationTiming.total =
                elapsedSeconds(
                    iterationStart,
                    iterationEnd
                );

            timing.totalIterations +=
                iterationTiming.total;

            printSCFIteration(
                iteration,
                energy.total,
                energyDifference,
                effectiveDensityDifference,
                residual,
                mixing,
                iterationTiming.total,
                alphaOrbitals,
                betaOrbitals,
                iterationTiming.orbitalAlpha,
                iterationTiming.orbitalBeta
            );

            break;
        }

        start =
            std::chrono::steady_clock::now();

        if (std::isfinite(previousDensityDifference)) {

            if (effectiveDensityDifference >
                previousDensityDifference *
                OSCILLATION_FACTOR) {

                mixing =
                    std::max(
                        MIN_MIXING,
                        mixing *
                        MIXING_DECREASE
                    );
            }
            else if (effectiveDensityDifference <
                     previousDensityDifference) {

                mixing =
                    std::min(
                        MAX_MIXING,
                        mixing *
                        MIXING_INCREASE
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

        end =
            std::chrono::steady_clock::now();

        iterationTiming.mixing =
            elapsedSeconds(start, end);

        timing.mixing +=
            iterationTiming.mixing;

        previousDensityDifference =
            effectiveDensityDifference;

        previousEnergy =
            energy.total;

        const auto iterationEnd =
            std::chrono::steady_clock::now();

        iterationTiming.total =
            elapsedSeconds(
                iterationStart,
                iterationEnd
            );

        timing.totalIterations +=
            iterationTiming.total;

        printSCFIteration(
            iteration,
            energy.total,
            energyDifference,
            effectiveDensityDifference,
            residual,
            mixing,
            iterationTiming.total,
            alphaOrbitals,
            betaOrbitals,
            iterationTiming.orbitalAlpha,
            iterationTiming.orbitalBeta
        );
    }

    const auto scfEnd =
        std::chrono::steady_clock::now();

    const double scfTotalTime =
        elapsedSeconds(
            scfStart,
            scfEnd
        );

    const double measuredTime =
        timing.initialization +
        timing.totalIterations;

    std::cout
        << "\n"
        << "============================================================\n"
        << "SCF MOLECULAR\n"
        << "============================================================\n"
        << "Iteraciones       : "
        << result.iterations
        << "\n"
        << "Convergencia      : "
        << (result.converged ? "SI" : "NO")
        << "\n"
        << "Energia final     : "
        << std::scientific
        << std::setprecision(12)
        << result.energy.total
        << " Ha\n"
        << "Tiempo SCF        : "
        << std::fixed
        << std::setprecision(6)
        << scfTotalTime
        << " s\n"
        << "============================================================\n"
        << "PROFILING\n"
        << "============================================================\n";

    printTimeLine(
        "Inicializacion",
        timing.initialization
    );

    printTimeLine(
        "Densidad",
        timing.density
    );

    printTimeLine(
        "Hartree",
        timing.hartree
    );

    printTimeLine(
        "XC Alpha",
        timing.exchangeCorrelationAlpha
    );

    printTimeLine(
        "XC Beta",
        timing.exchangeCorrelationBeta
    );

    printTimeLine(
        "Construccion potencial",
        timing.potentialConstruction
    );

    printTimeLine(
        "Orbitales Alpha",
        timing.orbitalAlpha
    );

    printTimeLine(
        "Orbitales Beta",
        timing.orbitalBeta
    );

    printTimeLine(
        "Densidad de salida",
        timing.outputDensity
    );

    printTimeLine(
        "Hartree salida",
        timing.outputHartree
    );

    printTimeLine(
        "Energia total",
        timing.totalEnergy
    );

    printTimeLine(
        "Diferencias de densidad",
        timing.densityDifference
    );

    printTimeLine(
        "Residuo Kohn-Sham",
        timing.ksResidual
    );

    printTimeLine(
        "Mixing",
        timing.mixing
    );

    std::cout
        << "------------------------------------------------------------\n";

    printTimeLine(
        "Tiempo SCF medido",
        measuredTime
    );

    printTimeLine(
        "Tiempo SCF real",
        scfTotalTime
    );

    const double unaccountedTime =
        std::max(
            0.0,
            scfTotalTime -
            measuredTime
        );

    printTimeLine(
        "Tiempo no clasificado",
        unaccountedTime
    );

    std::cout
        << "============================================================\n";

    return molecularResult;
}