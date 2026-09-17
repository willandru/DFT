#pragma once

#include "CartesianGrid.h"
#include "DFTData.h"
#include "Molecule.h"
#include "RadialGrid.h"
#include "XCFunctional.h"

SCFResult solveSelfConsistentField(
    const RadialGrid& grid,
    const AtomicConfiguration& configuration,
    const XCFunctional& functional
);

MolecularResult solveMolecularSelfConsistentField(
    const CartesianGrid& grid,
    const Molecule& molecule,
    int charge,
    const XCFunctional& functional
);