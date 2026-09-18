#pragma once

#include "CartesianGrid.h"
#include "DFTData.h"
#include "Molecule.h"

#include <chrono>
#include <vector>

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

struct MolecularSCFConvergenceAnalysis
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

double elapsedSeconds(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end
);

void mixDensity(
    std::vector<double>& density,
    const std::vector<double>& output,
    double mixing
);

double calculateMolecularDensityDifference(
    const CartesianGrid& grid,
    const std::vector<double>& oldDensity,
    const std::vector<double>& newDensity
);

double calculateMolecularSpinDensityDifference(
    const CartesianGrid& grid,
    const std::vector<double>& oldAlphaDensity,
    const std::vector<double>& oldBetaDensity,
    const std::vector<double>& newAlphaDensity,
    const std::vector<double>& newBetaDensity
);

double calculateMaximumMolecularKSResidual(
    const CartesianGrid& grid,
    const std::vector<double>& alphaPotential,
    const std::vector<double>& betaPotential,
    const std::vector<MolecularOrbital>& orbitals
);

std::vector<int> buildSpinOccupations(
    int electronCount
);

std::vector<MolecularOrbital> convertOrbitalsToSpin(
    const std::vector<MolecularOrbital>& orbitals,
    SpinChannel spin
);

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

void printProfiling(
    const MolecularSCFTiming& timing,
    double scfTotalTime
);

void printConvergenceAnalysis(
    const MolecularSCFConvergenceAnalysis& analysis
);

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
);