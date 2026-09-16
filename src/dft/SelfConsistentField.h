#pragma once

#include "DFTData.h"
#include "RadialGrid.h"

SCFResult solveSelfConsistentField(
    const RadialGrid& grid,
    const AtomicConfiguration& configuration
);