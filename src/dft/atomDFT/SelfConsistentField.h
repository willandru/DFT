#pragma once

#include "DFTData.h"
#include "RadialGrid.h"
#include "XCFunctional.h"

SCFResult solveSelfConsistentField(
    const RadialGrid& grid,
    const AtomicConfiguration& configuration,
    const XCFunctional& functional
);