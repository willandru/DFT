#pragma once

#include "CartesianGrid.h"
#include "DFTData.h"
#include "Molecule.h"
#include "XCFunctional.h"

#include <vector>

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