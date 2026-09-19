#pragma once

#include "DFTData.h"

#include <cstddef>
#include <vector>

std::size_t countEigenvaluesBelow(
    const TridiagonalMatrix& matrix,
    double energy
);

double findEigenvalue(
    const TridiagonalMatrix& matrix,
    std::size_t index,
    double lowerBound,
    double upperBound
);

std::vector<double> solveEigenvector(
    const TridiagonalMatrix& matrix,
    double eigenvalue,
    std::size_t maxIterations = 1000
);

AtomicOrbital solveOrbital(
    const TridiagonalMatrix& matrix,
    const std::vector<double>& r,
    int n,
    int l,
    int electrons,
    std::size_t orbitalIndex
);