#pragma once

#include "DFTData.h"

#include <vector>

double dotProduct(
    const std::vector<double>& a,
    const std::vector<double>& b
);

double vectorNorm(
    const std::vector<double>& v
);

void normalizeVector(
    std::vector<double>& v,
    double dr
);

double maxAbsoluteDifference(
    const std::vector<double>& a,
    const std::vector<double>& b
);

std::vector<double> solveTridiagonal(
    const TridiagonalMatrix& matrix,
    const std::vector<double>& rhs
);