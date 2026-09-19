#pragma once

#include "DFTData.h"

#include <vector>

bool testCoulombHydrogen(
    const std::vector<double>& r,
    const std::vector<double>& potential
);

bool testSCFResults(
    const std::vector<AtomicResult>& results
);

bool testElectronNumbers(
    const std::vector<AtomicResult>& results
);

bool testNumericalElectronNumbers(
    const std::vector<AtomicResult>& results
);

bool testOrbitalNorms(
    const std::vector<AtomicResult>& results,
    double tolerance
);

bool testKohnShamExpectationValues(
    const std::vector<AtomicResult>& results,
    double tolerance
);

bool testEnergyDecomposition(
    const std::vector<AtomicResult>& results,
    double tolerance
);
