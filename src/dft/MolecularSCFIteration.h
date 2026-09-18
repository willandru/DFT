#pragma once

#include "CartesianGrid.h"
#include "DFTData.h"
#include "Molecule.h"
#include "XCFunctional.h"

#include <vector>

struct MolecularSCFIterationResult
{
    std::vector<double> density;

    std::vector<double> hartreePotential;

    std::vector<double> alphaXCPotential;
    std::vector<double> betaXCPotential;

    std::vector<double> alphaPotential;
    std::vector<double> betaPotential;

    std::vector<MolecularOrbital> alphaOrbitals;
    std::vector<MolecularOrbital> betaOrbitals;
    std::vector<MolecularOrbital> orbitals;

    std::vector<double> outputAlphaDensity;
    std::vector<double> outputBetaDensity;
    std::vector<double> outputDensity;

    std::vector<double> outputHartreePotential;

    EnergyComponents energy;

    double densityDifference = 0.0;
    double spinDensityDifference = 0.0;
    double effectiveDensityDifference = 0.0;

    double residual = 0.0;
};

MolecularSCFIterationResult executeMolecularSCFIteration(
    const CartesianGrid& grid,
    const Molecule& molecule,
    const XCFunctional& functional,
    const std::vector<double>& nuclearPotential,
    const std::vector<double>& alphaDensity,
    const std::vector<double>& betaDensity,
    int alphaElectrons,
    int betaElectrons,
    bool closedShell,
    const std::vector<MolecularOrbital>& previousAlphaOrbitals,
    const std::vector<MolecularOrbital>& previousBetaOrbitals
);