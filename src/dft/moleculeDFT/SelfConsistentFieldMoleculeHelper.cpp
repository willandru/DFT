#include "SelfConsistentFieldMoleculeHelper.h"

#include "DFTConstants.h"
#include "EigenvalueSolverMolecule.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace MolecularSCFHelper
{

namespace
{

constexpr std::size_t timingIndex(
    TimingStage stage
)
{
    return static_cast<std::size_t>(stage);
}

double elapsedSeconds(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end
)
{
    return std::chrono::duration<double>(
        end - start
    ).count();
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

void Timer::start()
{
    scfStart_ =
        std::chrono::steady_clock::now();

    scfRunning_ = true;
}

double Timer::stop()
{
    if (!scfRunning_)
    {
        return 0.0;
    }

    const auto end =
        std::chrono::steady_clock::now();

    scfTotalTime_ =
        elapsedSeconds(
            scfStart_,
            end
        );

    scfRunning_ = false;

    return scfTotalTime_;
}

double Timer::elapsed() const
{
    return scfTotalTime_;
}

void Timer::begin(
    TimingStage stage
)
{
    currentStage_ =
        stage;

    stageTimeStart_ =
        std::chrono::steady_clock::now();

    stageRunning_ = true;
}

void Timer::end(
    TimingStage stage
)
{
    if (!stageRunning_)
    {
        return;
    }

    if (stage != currentStage_)
    {
        throw std::logic_error(
            "El estado de timing no coincide con el estado iniciado."
        );
    }

    const auto end =
        std::chrono::steady_clock::now();

    times_[timingIndex(stage)] +=
        elapsedSeconds(
            stageTimeStart_,
            end
        );

    stageRunning_ = false;
}

double Timer::get(
    TimingStage stage
) const
{
    return times_[timingIndex(stage)];
}

double Timer::getTotalIterations() const
{
    return get(
        TimingStage::TotalIterations
    );
}

double Timer::getMeasuredTime() const
{
    return
        get(TimingStage::Initialization) +
        get(TimingStage::Density) +
        get(TimingStage::Hartree) +
        get(TimingStage::ExchangeCorrelationAlpha) +
        get(TimingStage::ExchangeCorrelationBeta) +
        get(TimingStage::PotentialConstruction) +
        get(TimingStage::OrbitalAlpha) +
        get(TimingStage::OrbitalBeta) +
        get(TimingStage::OutputDensity) +
        get(TimingStage::OutputHartree) +
        get(TimingStage::TotalEnergy) +
        get(TimingStage::DensityDifference) +
        get(TimingStage::KSResidual) +
        get(TimingStage::Mixing);
}

double Timer::getSCFTotalTime() const
{
    return scfTotalTime_;
}

double Timer::getUnaccountedTime() const
{
    return std::max(
        0.0,
        getSCFTotalTime() -
        getMeasuredTime()
    );
}

std::vector<MolecularOrbital>
solveMolecularOrbitalsSilently(
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

    try
    {
        const std::vector<MolecularOrbital> orbitals =
            solveMolecularOrbitals(
                grid,
                potential,
                orbitalCount,
                occupations,
                spin,
                initialOrbitals
            );

        std::cout.rdbuf(
            originalBuffer
        );

        return orbitals;
    }
    catch (...)
    {
        std::cout.rdbuf(
            originalBuffer
        );

        throw;
    }
}

void printSCFHeader()
{
    std::cout
        << "\n"
        << "========================================================================================================================\n"
        << "SCF MOLECULAR\n"
        << "========================================================================================================================\n"
        << "Iter | E (Ha)         | dE         | dRho      | KS        | Mix   | E Alpha      | E Beta       | tA(ms) | tB(ms) | tSCF(s)\n"
        << "-----|----------------|------------|-----------|-----------|-------|--------------|--------------|--------|--------|---------\n";
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

    if (std::isfinite(energyDifference))
    {
        std::cout
            << std::scientific
            << std::setprecision(4)
            << std::setw(10)
            << energyDifference;
    }
    else
    {
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

    if (!alphaOrbitals.empty())
    {
        std::cout
            << std::scientific
            << std::setprecision(6)
            << std::setw(12)
            << alphaOrbitals.front().eigenvalue;
    }
    else
    {
        std::cout
            << std::setw(12)
            << "N/A";
    }

    std::cout
        << " | ";

    if (!betaOrbitals.empty())
    {
        std::cout
            << std::scientific
            << std::setprecision(6)
            << std::setw(12)
            << betaOrbitals.front().eigenvalue;
    }
    else
    {
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
        << std::fixed
        << std::setprecision(6)
        << std::setw(8)
        << iterationTime
        << "\n";
}

void printSCFSummary(
    int iterations,
    bool converged,
    double finalEnergy
)
{
    std::cout
        << "\n"
        << "============================================================\n"
        << "SCF MOLECULAR\n"
        << "============================================================\n"
        << "Iteraciones       : "
        << iterations
        << "\n"
        << "Convergencia      : "
        << (converged ? "SI" : "NO")
        << "\n"
        << "Energia final     : "
        << std::scientific
        << std::setprecision(12)
        << finalEnergy
        << " Ha\n"
        << "Tiempo SCF        : ";
}

void printProfiling(
    const Timer& timer
)
{
    std::cout
        << std::fixed
        << std::setprecision(6)
        << timer.getSCFTotalTime()
        << " s\n"
        << "============================================================\n"
        << "PROFILING\n"
        << "============================================================\n";

    printTimeLine(
        "Inicializacion",
        timer.get(
            TimingStage::Initialization
        )
    );

    printTimeLine(
        "Densidad",
        timer.get(
            TimingStage::Density
        )
    );

    printTimeLine(
        "Hartree",
        timer.get(
            TimingStage::Hartree
        )
    );

    printTimeLine(
        "XC Alpha",
        timer.get(
            TimingStage::ExchangeCorrelationAlpha
        )
    );

    printTimeLine(
        "XC Beta",
        timer.get(
            TimingStage::ExchangeCorrelationBeta
        )
    );

    printTimeLine(
        "Construccion potencial",
        timer.get(
            TimingStage::PotentialConstruction
        )
    );

    printTimeLine(
        "Orbitales Alpha",
        timer.get(
            TimingStage::OrbitalAlpha
        )
    );

    printTimeLine(
        "Orbitales Beta",
        timer.get(
            TimingStage::OrbitalBeta
        )
    );

    printTimeLine(
        "Densidad de salida",
        timer.get(
            TimingStage::OutputDensity
        )
    );

    printTimeLine(
        "Hartree salida",
        timer.get(
            TimingStage::OutputHartree
        )
    );

    printTimeLine(
        "Energia total",
        timer.get(
            TimingStage::TotalEnergy
        )
    );

    printTimeLine(
        "Diferencias de densidad",
        timer.get(
            TimingStage::DensityDifference
        )
    );

    printTimeLine(
        "Residuo Kohn-Sham",
        timer.get(
            TimingStage::KSResidual
        )
    );

    printTimeLine(
        "Mixing",
        timer.get(
            TimingStage::Mixing
        )
    );

    std::cout
        << "------------------------------------------------------------\n";

    printTimeLine(
        "Tiempo SCF medido",
        timer.getMeasuredTime()
    );

    printTimeLine(
        "Tiempo SCF real",
        timer.getSCFTotalTime()
    );

    printTimeLine(
        "Tiempo no clasificado",
        timer.getUnaccountedTime()
    );
}

void printConvergenceAnalysis(
    const ConvergenceAnalysis& analysis
)
{
    std::cout
        << "\n"
        << "========================================================================================================================\n"
        << "SCF CONVERGENCE ANALYSIS\n"
        << "========================================================================================================================\n";

    std::cout
        << std::scientific
        << std::setprecision(6);

    std::cout
        << "DENSITY_TOL       : "
        << DFTConstants::DENSITY_TOL
        << "\n";

    std::cout
        << "ENERGY_TOL        : "
        << DFTConstants::ENERGY_TOL
        << "\n";

    std::cout
        << "KS_RESIDUAL_TOL   : "
        << DFTConstants::KS_RESIDUAL_TOL
        << "\n";

    std::cout
        << "------------------------------------------------------------------------------------------------------------------------\n";

    std::cout
        << "Ultima iteracion con dE   < ENERGY_TOL      : "
        << analysis.lastEnergyCriterionIteration
        << "\n";

    std::cout
        << "Ultima iteracion con dRho < DENSITY_TOL     : "
        << analysis.lastDensityCriterionIteration
        << "\n";

    std::cout
        << "Ultima iteracion con KS   < KS_RESIDUAL_TOL : "
        << analysis.lastKSCriterionIteration
        << "\n";

    std::cout
        << "------------------------------------------------------------------------------------------------------------------------\n";

    std::cout
        << "Minimo dE   observado     : "
        << analysis.minimumEnergyDifference
        << "\n";

    std::cout
        << "Minimo dRho observado     : "
        << analysis.minimumDensityDifference
        << "\n";

    std::cout
        << "Minimo KS observado       : "
        << analysis.minimumKSResidual
        << "\n";

    std::cout
        << "------------------------------------------------------------------------------------------------------------------------\n";

    std::cout
        << "Valor final dE            : "
        << analysis.finalEnergyDifference
        << "\n";

    std::cout
        << "Valor final dRho          : "
        << analysis.finalDensityDifference
        << "\n";

    std::cout
        << "Valor final KS            : "
        << analysis.finalKSResidual
        << "\n";

    std::cout
        << "Mix final                 : "
        << std::fixed
        << std::setprecision(6)
        << analysis.finalMixing
        << "\n";

    std::cout
        << "========================================================================================================================\n";
}

void printSCFEndLine()
{
    std::cout
        << "============================================================\n";
}

}