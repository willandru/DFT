#pragma once

#include "CartesianGrid.h"
#include "DFTData.h"
#include "Molecule.h"
#include "XCFunctional.h"

MolecularResult solveMolecularSelfConsistentField(
    const CartesianGrid& grid,
    const Molecule& molecule,
    int charge,
    int multiplicity,
    const XCFunctional& functional
);