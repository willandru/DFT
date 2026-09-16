#pragma once

#include "DFTData.h"
#include "RadialGrid.h"

AtomicResult solveAtom(
    const RadialGrid& grid,
    const AtomicConfiguration& configuration
);