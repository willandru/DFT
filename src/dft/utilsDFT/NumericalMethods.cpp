#include "NumericalMethods.h"

#include "DFTConstants.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

double dotProduct(
    const std::vector<double>& a,
    const std::vector<double>& b
) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Los vectores deben tener el mismo tamano."
        );
    }

    double result = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }

    return result;
}

double vectorNorm(
    const std::vector<double>& v
) {
    return std::sqrt(dotProduct(v, v));
}

void normalizeVector(
    std::vector<double>& v,
    double dr
) {
    if (v.empty()) {
        throw std::invalid_argument(
            "No se puede normalizar un vector vacio."
        );
    }

    if (dr <= 0.0) {
        throw std::invalid_argument(
            "El paso radial debe ser mayor que cero."
        );
    }

    double norm = std::sqrt(dr * dotProduct(v, v));

    if (norm < DFTConstants::EPS) {
        throw std::runtime_error(
            "No se puede normalizar un vector con norma cero."
        );
    }

    for (double& value : v) {
        value /= norm;
    }
}

double maxAbsoluteDifference(
    const std::vector<double>& a,
    const std::vector<double>& b
) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Los vectores deben tener el mismo tamano."
        );
    }

    double maximum = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i) {
        maximum = std::max(
            maximum,
            std::abs(a[i] - b[i])
        );
    }

    return maximum;
}

std::vector<double> solveTridiagonal(
    const TridiagonalMatrix& matrix,
    const std::vector<double>& rhs
) {
    const std::size_t n = matrix.diagonal.size();

    if (n == 0 ||
        matrix.lower.size() != n - 1 ||
        matrix.upper.size() != n - 1 ||
        rhs.size() != n) {
        throw std::invalid_argument(
            "Dimensiones invalidas para la matriz tridiagonal."
        );
    }

    std::vector<double> upper = matrix.upper;
    std::vector<double> diagonal = matrix.diagonal;
    std::vector<double> solution = rhs;

    for (std::size_t i = 1; i < n; ++i) {
        if (std::abs(diagonal[i - 1]) < DFTConstants::EPS) {
            throw std::runtime_error(
                "Pivote numericamente singular en el solver tridiagonal."
            );
        }

        const double factor =
            matrix.lower[i - 1] / diagonal[i - 1];

        diagonal[i] -= factor * upper[i - 1];
        solution[i] -= factor * solution[i - 1];
    }

    if (std::abs(diagonal[n - 1]) < DFTConstants::EPS) {
        throw std::runtime_error(
            "Pivote final numericamente singular en el solver tridiagonal."
        );
    }

    solution[n - 1] /= diagonal[n - 1];

    for (std::size_t i = n - 1; i-- > 0;) {
        if (std::abs(diagonal[i]) < DFTConstants::EPS) {
            throw std::runtime_error(
                "Pivote numericamente singular en el solver tridiagonal."
            );
        }

        solution[i] =
            (solution[i] - upper[i] * solution[i + 1]) /
            diagonal[i];
    }

    return solution;
}