#pragma once

#include <cstddef>
#include <vector>

double davidsonDot(
    const std::vector<double>& a,
    const std::vector<double>& b,
    double dV
);

double davidsonNormSquared(
    const std::vector<double>& v,
    double dV
);

bool davidsonValidNorm(
    const std::vector<double>& v,
    double dV,
    double toleranceSquared = 1.0e-20
);

void davidsonNormalize(
    std::vector<double>& v,
    double dV,
    double toleranceSquared = 1.0e-20
);

void davidsonOrthogonalize(
    std::vector<double>& v,
    const std::vector<std::vector<double>>& basis,
    double dV
);

double davidsonRayleigh(
    const std::vector<double>& psi,
    const std::vector<double>& hPsi,
    double dV
);

double davidsonResidualNorm(
    const std::vector<double>& psi,
    const std::vector<double>& hPsi,
    double eigenvalue,
    double dV
);

struct DavidsonEigenpair {
    double value;
    std::vector<double> vector;
};

DavidsonEigenpair davidsonDiagonalize(
    std::vector<std::vector<double>> matrix
);

std::vector<double> davidsonCombine(
    const std::vector<std::vector<double>>& basis,
    const std::vector<double>& coefficients
);