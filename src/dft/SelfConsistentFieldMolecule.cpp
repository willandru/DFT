#include "SelfConsistentFieldMolecule.h"

#include "DFTConstants.h"
#include "ElectronDensityMolecule.h"
#include "ExchangeCorrelationMolecule.h"
#include "HartreePotentialMolecule.h"
#include "MolecularSCFHelpers.h"
#include "MolecularSCFIteration.h"
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
#include <string>
#include <utility>
#include <vector>

namespace
{

constexpr double MIN_MIXING = 0.10;
constexpr double MAX_MIXING = 0.50;
constexpr double MIXING_INCREASE = 1.10;
constexpr double MIXING_DECREASE = 0.75;
constexpr double OSCILLATION_FACTOR = 1.05;

template <typename T>
void printField(
    const char* label,
    const T& value
)
{
    std::cout
        << std::left
        << std::setw(42)
        << label
        << std::right
        << value
        << '\n';
}

}

MolecularResult solveMolecularSelfConsistentField(
    const CartesianGrid& grid,
    const Molecule& molecule,
    int charge,
    int multiplicity,
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

    if (multiplicity < 1) {
        throw std::invalid_argument(
            "La multiplicidad electronica debe ser al menos 1."
        );
    }

    const int nuclearCharge =
        [&molecule]() {
            int total = 0;

            for (const Molecule::Nucleus& nucleus :
                 molecule.getNuclei())
                total += nucleus.atomicNumber;

            return total;
        }();

    const int electronCount =
        nuclearCharge - charge;

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

    const int spinDifference =
        multiplicity - 1;

    if (spinDifference > electronCount) {
        throw std::invalid_argument(
            "La multiplicidad electronica es incompatible con el numero de electrones."
        );
    }

    if ((electronCount - spinDifference) % 2 != 0) {
        throw std::invalid_argument(
            "La multiplicidad electronica es incompatible con la paridad del numero de electrones."
        );
    }

    const int alphaElectrons =
        (electronCount + spinDifference) / 2;

    const int betaElectrons =
        (electronCount - spinDifference) / 2;

    const bool closedShell =
        multiplicity == 1;

    MolecularSCFTiming timing;
    MolecularSCFConvergenceAnalysis convergenceAnalysis;

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

    std::vector<MolecularOrbital> previousAlphaOrbitals;
    std::vector<MolecularOrbital> previousBetaOrbitals;

    buildInitialMolecularDensities(
        grid,
        molecule,
        alphaElectrons,
        betaElectrons,
        closedShell,
        alphaDensity,
        betaDensity,
        previousAlphaOrbitals,
        previousBetaOrbitals
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
        buildSpinOccupations(alphaElectrons);

    const std::vector<int> betaOccupations =
        buildSpinOccupations(betaElectrons);

    double previousEnergy =
        std::numeric_limits<double>::infinity();

    double previousDensityDifference =
        std::numeric_limits<double>::infinity();

    double mixing =
        DFTConstants::MIXING;

    MolecularResult molecularResult;

    molecularResult.electrons = electronCount;
    molecularResult.charge = charge;

    SCFResult& result =
        molecularResult.scf;

    printSCFHeader();

    for (int iteration = 1;
         iteration <= DFTConstants::MAX_SCF_ITERATIONS;
         ++iteration) {

        const auto iterationStart =
            std::chrono::steady_clock::now();

        MolecularSCFIterationTiming iterationTiming;

        const double iterationMixing =
            mixing;

        auto start =
            std::chrono::steady_clock::now();

        const MolecularSCFIterationResult iterationResult =
            executeMolecularSCFIteration(
                grid,
                molecule,
                functional,
                nuclearPotential,
                alphaDensity,
                betaDensity,
                alphaElectrons,
                betaElectrons,
                closedShell,
                previousAlphaOrbitals,
                previousBetaOrbitals
            );

        auto end =
            std::chrono::steady_clock::now();

        const double iterationTotalTime =
            elapsedSeconds(
                start,
                end
            );

        iterationTiming.total =
            iterationTotalTime;

        timing.totalIterations +=
            iterationTotalTime;

        const std::vector<double>& density =
            iterationResult.density;

        const std::vector<double>& hartreePotential =
            iterationResult.hartreePotential;

        const std::vector<double>& alphaXCPotential =
            iterationResult.alphaXCPotential;

        const std::vector<double>& betaXCPotential =
            iterationResult.betaXCPotential;

        const std::vector<double>& alphaPotential =
            iterationResult.alphaPotential;

        const std::vector<double>& betaPotential =
            iterationResult.betaPotential;

        const std::vector<MolecularOrbital>& alphaOrbitals =
            iterationResult.alphaOrbitals;

        const std::vector<MolecularOrbital>& betaOrbitals =
            iterationResult.betaOrbitals;

        const std::vector<MolecularOrbital>& orbitals =
            iterationResult.orbitals;

        const std::vector<double>& outputAlphaDensity =
            iterationResult.outputAlphaDensity;

        const std::vector<double>& outputBetaDensity =
            iterationResult.outputBetaDensity;

        const std::vector<double>& outputDensity =
            iterationResult.outputDensity;

        const std::vector<double>& outputHartreePotential =
            iterationResult.outputHartreePotential;

        const EnergyComponents& energy =
            iterationResult.energy;

        const double effectiveDensityDifference =
            iterationResult.effectiveDensityDifference;

        const double residual =
            iterationResult.residual;

        const double energyDifference =
            std::isfinite(previousEnergy)
            ? std::abs(
                energy.total -
                previousEnergy
            )
            : std::numeric_limits<double>::infinity();

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

        const bool energyConverged =
            energyDifference <
            DFTConstants::ENERGY_TOL;

        const bool densityConverged =
            effectiveDensityDifference <
            DFTConstants::DENSITY_TOL;

        const bool ksConverged =
            residual <
            DFTConstants::KS_RESIDUAL_TOL;

        if (energyConverged)
            convergenceAnalysis.lastEnergyCriterionIteration =
                iteration;

        if (densityConverged)
            convergenceAnalysis.lastDensityCriterionIteration =
                iteration;

        if (ksConverged)
            convergenceAnalysis.lastKSCriterionIteration =
                iteration;

        convergenceAnalysis.finalEnergyDifference =
            energyDifference;

        convergenceAnalysis.finalDensityDifference =
            effectiveDensityDifference;

        convergenceAnalysis.finalKSResidual =
            residual;

        convergenceAnalysis.finalMixing =
            iterationMixing;

        convergenceAnalysis.minimumEnergyDifference =
            std::min(
                convergenceAnalysis.minimumEnergyDifference,
                energyDifference
            );

        convergenceAnalysis.minimumDensityDifference =
            std::min(
                convergenceAnalysis.minimumDensityDifference,
                effectiveDensityDifference
            );

        convergenceAnalysis.minimumKSResidual =
            std::min(
                convergenceAnalysis.minimumKSResidual,
                residual
            );

        const bool converged =
            energyConverged &&
            densityConverged &&
            ksConverged;

        if (!converged) {
            previousAlphaOrbitals =
                alphaOrbitals;

            if (!closedShell)
                previousBetaOrbitals =
                    betaOrbitals;
        }

        const auto iterationEnd =
            std::chrono::steady_clock::now();

        iterationTiming.total =
            elapsedSeconds(
                iterationStart,
                iterationEnd
            );

        timing.totalIterations =
            timing.totalIterations -
            iterationTotalTime +
            iterationTiming.total;

        printSCFIteration(
            iteration,
            energy.total,
            energyDifference,
            effectiveDensityDifference,
            residual,
            iterationMixing,
            iterationTiming.total,
            alphaOrbitals,
            betaOrbitals,
            iterationTiming.orbitalAlpha,
            iterationTiming.orbitalBeta
        );

        if (converged) {
            result.converged = true;
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
            else if (
                effectiveDensityDifference <
                previousDensityDifference
            ) {

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
            iterationMixing
        );

        mixDensity(
            betaDensity,
            outputBetaDensity,
            iterationMixing
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
    }

    const auto scfEnd =
        std::chrono::steady_clock::now();

    const double scfTotalTime =
        elapsedSeconds(
            scfStart,
            scfEnd
        );

    std::cout
        << "\n"
        << "============================================================\n"
        << "SCF MOLECULAR\n"
        << "============================================================\n";

    printField(
        "Iteraciones",
        result.iterations
    );

    printField(
        "Convergencia",
        result.converged ? "SI" : "NO"
    );

    std::cout
        << std::scientific
        << std::setprecision(12);

    printField(
        "Energia final",
        result.energy.total
    );

    printField(
        "Tiempo SCF",
        [&]() {
            std::ostringstream value;

            value
                << std::fixed
                << std::setprecision(6)
                << scfTotalTime
                << " s";

            return value.str();
        }()
    );

    printProfiling(
        timing,
        scfTotalTime
    );

    printConvergenceAnalysis(
        convergenceAnalysis
    );

    return molecularResult;
}