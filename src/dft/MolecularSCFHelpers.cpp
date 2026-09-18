#include "MolecularSCFHelpers.h"

#include "DFTConstants.h"
#include "EigenvalueSolverMolecule.h"
#include "ElectronDensityMolecule.h"
#include "HartreePotentialMolecule.h"
#include "KohnShamHamiltonianMolecule.h"
#include "NuclearPotential.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

double elapsedSeconds(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end
)
{
    return std::chrono::duration<double>(end - start).count();
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

    for (std::size_t i = 0; i < density.size(); ++i) {
        density[i] =
            (1.0 - mixing) * density[i] +
            mixing * output[i];
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

    for (std::size_t i = 0; i < grid.getSize(); ++i) {
        const double difference =
            newDensity[i] - oldDensity[i];

        differenceNormSquared +=
            difference * difference;

        densityNormSquared +=
            newDensity[i] * newDensity[i];
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

    if (densityNorm <= DFTConstants::EPS)
        return differenceNorm;

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
    return std::max(
        calculateMolecularDensityDifference(
            grid,
            oldAlphaDensity,
            newAlphaDensity
        ),
        calculateMolecularDensityDifference(
            grid,
            oldBetaDensity,
            newBetaDensity
        )
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

        const auto& effectivePotential =
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

        for (std::size_t i = 0; i < grid.getSize(); ++i) {
            const double residual =
                hPsi[i] -
                orbital.eigenvalue * orbital.psi[i];

            residualNormSquared +=
                residual * residual;
        }

        maximumResidual =
            std::max(
                maximumResidual,
                std::sqrt(
                    residualNormSquared *
                    volumeElement
                )
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

    return std::vector<int>(
        static_cast<std::size_t>(electronCount),
        1
    );
}

std::vector<MolecularOrbital> convertOrbitalsToSpin(
    const std::vector<MolecularOrbital>& orbitals,
    SpinChannel spin
)
{
    auto converted = orbitals;

    for (auto& orbital : converted)
        orbital.spin = spin;

    return converted;
}

std::vector<MolecularOrbital> solveMolecularOrbitalsSilently(
    const CartesianGrid& grid,
    const std::vector<double>& potential,
    std::size_t orbitalCount,
    const std::vector<int>& occupations,
    SpinChannel spin,
    const std::vector<MolecularOrbital>& initialOrbitals
)
{
    std::ostringstream suppressedOutput;

    std::streambuf* originalBuffer =
        std::cout.rdbuf(
            suppressedOutput.rdbuf()
        );

    try {
        auto orbitals =
            solveMolecularOrbitals(
                grid,
                potential,
                orbitalCount,
                occupations,
                spin,
                initialOrbitals
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
    constexpr const char* header =
        "\n"
        "========================================================================================================================\n"
        "SCF MOLECULAR\n"
        "========================================================================================================================\n"
        "Iter | E (Ha)         | dE         | dRho      | KS        | Mix   | E Alpha      | E Beta       | tA(ms) | tB(ms) | tSCF(s)\n"
        "-----|----------------|------------|-----------|-----------|-------|--------------|--------------|--------|--------|---------\n";

    std::cout << header;
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
    auto printNumber =
        [](double value, int width, int precision) {
            std::cout
                << std::scientific
                << std::setprecision(precision)
                << std::setw(width)
                << value;
        };

    auto printOrbital =
        [](const std::vector<MolecularOrbital>& orbitals) {
            if (orbitals.empty()) {
                std::cout << std::setw(12) << "N/A";
            }
            else {
                std::cout
                    << std::scientific
                    << std::setprecision(6)
                    << std::setw(12)
                    << orbitals.front().eigenvalue;
            }
        };

    std::cout
        << std::right
        << std::setw(4)
        << iteration
        << " | ";

    printNumber(energy, 14, 10);

    std::cout << " | ";

    if (std::isfinite(energyDifference))
        printNumber(energyDifference, 10, 4);
    else
        std::cout << std::setw(10) << "inf";

    std::cout << " | ";

    printNumber(densityDifference, 9, 4);

    std::cout << " | ";

    printNumber(residual, 9, 4);

    std::cout
        << " | "
        << std::fixed
        << std::setprecision(3)
        << std::setw(5)
        << mixing
        << " | ";

    printOrbital(alphaOrbitals);

    std::cout << " | ";

    printOrbital(betaOrbitals);

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
        << std::setprecision(6)
        << std::setw(8)
        << iterationTime
        << '\n';
}

namespace {

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

void printTimeLine(
    const char* label,
    double seconds
)
{
    printField(
        label,
        [&]() {
            std::ostringstream value;

            value
                << std::fixed
                << std::setprecision(3)
                << seconds * 1000.0
                << " ms";

            return value.str();
        }()
    );
}

}

void printProfiling(
    const MolecularSCFTiming& timing,
    double scfTotalTime
)
{
    const double measuredTime =
        timing.initialization +
        timing.totalIterations;

    const double unaccountedTime =
        std::max(
            0.0,
            scfTotalTime - measuredTime
        );

    std::cout
        << "\n"
        << "============================================================\n"
        << "PROFILING\n"
        << "============================================================\n";

    const std::pair<const char*, double> profiling[] = {
        {"Inicializacion", timing.initialization},
        {"Densidad", timing.density},
        {"Hartree", timing.hartree},
        {"XC Alpha", timing.exchangeCorrelationAlpha},
        {"XC Beta", timing.exchangeCorrelationBeta},
        {"Construccion potencial", timing.potentialConstruction},
        {"Orbitales Alpha", timing.orbitalAlpha},
        {"Orbitales Beta", timing.orbitalBeta},
        {"Densidad de salida", timing.outputDensity},
        {"Hartree salida", timing.outputHartree},
        {"Energia total", timing.totalEnergy},
        {"Diferencias de densidad", timing.densityDifference},
        {"Residuo Kohn-Sham", timing.ksResidual},
        {"Mixing", timing.mixing},
        {"Tiempo SCF medido", measuredTime},
        {"Tiempo SCF real", scfTotalTime},
        {"Tiempo no clasificado", unaccountedTime}
    };

    for (const auto& [label, seconds] : profiling)
        printTimeLine(label, seconds);

    std::cout
        << "------------------------------------------------------------\n";
}

void printConvergenceAnalysis(
    const MolecularSCFConvergenceAnalysis& analysis
)
{
    std::cout
        << "\n"
        << "============================================================\n"
        << "SCF CONVERGENCE ANALYSIS\n"
        << "============================================================\n";

    std::cout
        << std::scientific
        << std::setprecision(6);

    const std::pair<const char*, double> tolerances[] = {
        {"DENSITY_TOL", DFTConstants::DENSITY_TOL},
        {"ENERGY_TOL", DFTConstants::ENERGY_TOL},
        {"KS_RESIDUAL_TOL", DFTConstants::KS_RESIDUAL_TOL}
    };

    for (const auto& [label, value] : tolerances)
        printField(label, value);

    std::cout
        << "------------------------------------------------------------\n";

    const std::pair<const char*, int> criteria[] = {
        {
            "Ultima iteracion con dE < ENERGY_TOL",
            analysis.lastEnergyCriterionIteration
        },
        {
            "Ultima iteracion con dRho < DENSITY_TOL",
            analysis.lastDensityCriterionIteration
        },
        {
            "Ultima iteracion con KS < KS_RESIDUAL_TOL",
            analysis.lastKSCriterionIteration
        }
    };

    for (const auto& [label, value] : criteria)
        printField(label, value);

    std::cout
        << "------------------------------------------------------------\n";

    const std::pair<const char*, double> minimums[] = {
        {"Minimo dE observado", analysis.minimumEnergyDifference},
        {"Minimo dRho observado", analysis.minimumDensityDifference},
        {"Minimo KS observado", analysis.minimumKSResidual}
    };

    for (const auto& [label, value] : minimums)
        printField(label, value);

    std::cout
        << "------------------------------------------------------------\n";

    const std::pair<const char*, double> finalValues[] = {
        {"Valor final dE", analysis.finalEnergyDifference},
        {"Valor final dRho", analysis.finalDensityDifference},
        {"Valor final KS", analysis.finalKSResidual},
        {"Mix final", analysis.finalMixing}
    };

    for (const auto& [label, value] : finalValues)
        printField(label, value);

    std::cout
        << "============================================================\n";
}

void buildInitialMolecularDensities(
    const CartesianGrid& grid,
    const Molecule& molecule,
    int alphaElectrons,
    int betaElectrons,
    bool closedShell,
    std::vector<double>& alphaDensity,
    std::vector<double>& betaDensity,
    std::vector<MolecularOrbital>& initialAlphaOrbitals,
    std::vector<MolecularOrbital>& initialBetaOrbitals
)
{
    const std::vector<double> nuclearPotential =
        NuclearPotential::calculate(
            molecule,
            grid
        );

    const std::vector<int> alphaOccupations =
        buildSpinOccupations(alphaElectrons);

    const std::vector<int> betaOccupations =
        buildSpinOccupations(betaElectrons);

    initialAlphaOrbitals =
        solveMolecularOrbitalsSilently(
            grid,
            nuclearPotential,
            alphaOccupations.size(),
            alphaOccupations,
            SpinChannel::Alpha,
            {}
        );

    if (closedShell) {
        initialBetaOrbitals =
            convertOrbitalsToSpin(
                initialAlphaOrbitals,
                SpinChannel::Beta
            );
    }
    else {
        initialBetaOrbitals =
            solveMolecularOrbitalsSilently(
                grid,
                nuclearPotential,
                betaOccupations.size(),
                betaOccupations,
                SpinChannel::Beta,
                {}
            );
    }

    alphaDensity =
        calculateMolecularSpinDensity(
            initialAlphaOrbitals,
            SpinChannel::Alpha
        );

    betaDensity =
        calculateMolecularSpinDensity(
            initialBetaOrbitals,
            SpinChannel::Beta
        );
}