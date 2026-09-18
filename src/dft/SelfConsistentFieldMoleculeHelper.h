#pragma once

#include "CartesianGrid.h"
#include "DFTData.h"

#include <chrono>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace MolecularSCFHelper {

enum class TimingStage
{
    Initialization,

    Density,
    Hartree,

    ExchangeCorrelationAlpha,
    ExchangeCorrelationBeta,

    PotentialConstruction,

    OrbitalAlpha,
    OrbitalBeta,

    OutputDensity,
    OutputHartree,

    TotalEnergy,

    DensityDifference,
    KSResidual,

    Mixing,

    TotalIterations
};

class Timer
{
public:

    void start();

    double stop();

    double elapsed() const;

    void begin(TimingStage stage);

    void end(TimingStage stage);

    double get(TimingStage stage) const;

    double getTotalIterations() const;

    double getMeasuredTime() const;

    double getSCFTotalTime() const;

    double getUnaccountedTime() const;

private:

    std::chrono::steady_clock::time_point scfStart_;

    std::chrono::steady_clock::time_point stageStart_;

    std::chrono::steady_clock::time_point iterationStart_;

    std::chrono::steady_clock::time_point stageTimeStart_;

    bool scfRunning_ = false;

    bool stageRunning_ = false;

    bool iterationRunning_ = false;

    TimingStage currentStage_ =
        TimingStage::Initialization;

    double scfTotalTime_ = 0.0;

    double times_[15] = {};
};

struct ConvergenceAnalysis
{
    int lastEnergyCriterionIteration = 0;

    int lastDensityCriterionIteration = 0;

    int lastKSCriterionIteration = 0;

    double finalEnergyDifference =
        std::numeric_limits<double>::infinity();

    double finalDensityDifference =
        std::numeric_limits<double>::infinity();

    double finalKSResidual =
        std::numeric_limits<double>::infinity();

    double finalMixing = 0.0;

    double minimumEnergyDifference =
        std::numeric_limits<double>::infinity();

    double minimumDensityDifference =
        std::numeric_limits<double>::infinity();

    double minimumKSResidual =
        std::numeric_limits<double>::infinity();
};

std::vector<MolecularOrbital> solveMolecularOrbitalsSilently(
    const CartesianGrid& grid,
    const std::vector<double>& potential,
    std::size_t orbitalCount,
    const std::vector<int>& occupations,
    SpinChannel spin,
    const std::vector<MolecularOrbital>& initialOrbitals
);

void printSCFHeader();

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
);

void printSCFSummary(
    int iterations,
    bool converged,
    double finalEnergy
);

void printProfiling(
    const Timer& timer
);

void printConvergenceAnalysis(
    const ConvergenceAnalysis& analysis
);

void printSCFEndLine();

}