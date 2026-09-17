#pragma once

#include "CartesianGrid.h"
#include "DFTData.h"
#include "Molecule.h"
#include "XCFunctional.h"

#include <vector>

double calculateKineticEnergy(
    const std::vector<double>& r,
    const std::vector<AtomicOrbital>& orbitals
);

double calculateExternalEnergy(
    const std::vector<double>& r,
    const std::vector<double>& density,
    int Z
);

EnergyComponents calculateTotalEnergy(
    const XCFunctional& functional,
    const std::vector<double>& r,
    const std::vector<double>& alphaDensity,
    const std::vector<double>& betaDensity,
    const std::vector<AtomicOrbital>& orbitals,
    const std::vector<double>& hartreePotential,
    int Z
);

double calculateMolecularKineticEnergy(
    const CartesianGrid& grid,
    const std::vector<MolecularOrbital>& orbitals
);

double calculateMolecularExternalEnergy(
    const CartesianGrid& grid,
    const std::vector<double>& density,
    const std::vector<double>& nuclearPotential
);

EnergyComponents calculateMolecularTotalEnergy(
    const XCFunctional& functional,
    const Molecule& molecule,
    const CartesianGrid& grid,
    const std::vector<double>& alphaDensity,
    const std::vector<double>& betaDensity,
    const std::vector<MolecularOrbital>& orbitals,
    const std::vector<double>& hartreePotential,
    const std::vector<double>& nuclearPotential
);