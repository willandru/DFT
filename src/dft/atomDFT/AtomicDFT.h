#pragma once

#include "DFTData.h"
#include "RadialGrid.h"
#include "XCFunctional.h"

AtomicResult solveAtom(
    const RadialGrid& grid,
    const AtomicConfiguration& configuration,
    const XCFunctional& functional
);