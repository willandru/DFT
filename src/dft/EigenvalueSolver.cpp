#include "EigenvalueSolver.h"

#include "DFTConstants.h"
#include "NumericalMethods.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

double sturmPivot(
    const TridiagonalMatrix& matrix,
    std::size_t i,
    double energy,
    double previous
) {
    if (std::abs(previous) < DFTConstants::EPS) {
        previous =
            std::copysign(
                DFTConstants::EPS,
                previous == 0.0 ? 1.0 : previous
            );
    }

    return
        matrix.diagonal[i] -
        energy -
        matrix.lower[i - 1] *
        matrix.upper[i - 1] /
        previous;
}

}

std::size_t countEigenvaluesBelow(
    const TridiagonalMatrix& matrix,
    double energy
) {
    const std::size_t n =
        matrix.diagonal.size();

    if (n == 0 ||
        matrix.lower.size() != n - 1 ||
        matrix.upper.size() != n - 1) {
        throw std::invalid_argument(
            "Matriz tridiagonal invalida."
        );
    }

    std::size_t count = 0;

    double previous =
        matrix.diagonal[0] - energy;

    if (previous < 0.0) {
        ++count;
    }

    if (std::abs(previous) < DFTConstants::EPS) {
        previous =
            std::copysign(
                DFTConstants::EPS,
                previous == 0.0 ? 1.0 : previous
            );
    }

    for (std::size_t i = 1; i < n; ++i) {
        const double pivot =
            sturmPivot(
                matrix,
                i,
                energy,
                previous
            );

        if (pivot < 0.0) {
            ++count;
        }

        previous =
            std::abs(pivot) < DFTConstants::EPS
                ? std::copysign(
                    DFTConstants::EPS,
                    pivot == 0.0 ? 1.0 : pivot
                )
                : pivot;
    }

    return count;
}

double findEigenvalue(
    const TridiagonalMatrix& matrix,
    std::size_t index,
    double lowerBound,
    double upperBound
) {
    const std::size_t n =
        matrix.diagonal.size();

    if (n == 0 ||
        matrix.lower.size() != n - 1 ||
        matrix.upper.size() != n - 1) {
        throw std::invalid_argument(
            "Matriz tridiagonal invalida."
        );
    }

    if (index >= n) {
        throw std::invalid_argument(
            "Indice de autovalor fuera del rango."
        );
    }

    if (lowerBound >= upperBound) {
        throw std::invalid_argument(
            "Intervalo invalido para la busqueda del autovalor."
        );
    }

    double low =
        lowerBound;

    double high =
        upperBound;

    std::size_t lowCount =
        countEigenvaluesBelow(
            matrix,
            low
        );

    std::size_t highCount =
        countEigenvaluesBelow(
            matrix,
            high
        );

    double width =
        high - low;

    for (int expansion = 0;
         expansion < 100;
         ++expansion) {

        if (lowCount <= index &&
            index < highCount) {
            break;
        }

        width *= 2.0;

        if (lowCount > index) {
            low -= width;

            lowCount =
                countEigenvaluesBelow(
                    matrix,
                    low
                );
        }

        if (highCount <= index) {
            high += width;

            highCount =
                countEigenvaluesBelow(
                    matrix,
                    high
                );
        }
    }

    if (lowCount > index ||
        index >= highCount) {
        throw std::runtime_error(
            "No se pudo acotar el autovalor solicitado."
        );
    }

    for (int iteration = 0;
         iteration < 300;
         ++iteration) {

        const double mid =
            0.5 *
            (low + high);

        const std::size_t count =
            countEigenvaluesBelow(
                matrix,
                mid
            );

        if (count <= index) {
            low = mid;
        } else {
            high = mid;
        }

        const double tolerance =
            1.0e-13 *
            std::max(
                1.0,
                std::max(
                    std::abs(low),
                    std::abs(high)
                )
            );

        if (high - low <= tolerance) {
            break;
        }
    }

    return
        0.5 *
        (low + high);
}

std::vector<double> solveEigenvector(
    const TridiagonalMatrix& matrix,
    double eigenvalue,
    std::size_t maxIterations
) {
    const std::size_t n =
        matrix.diagonal.size();

    if (n == 0 ||
        matrix.lower.size() != n - 1 ||
        matrix.upper.size() != n - 1) {
        throw std::invalid_argument(
            "Matriz tridiagonal invalida."
        );
    }

    std::vector<double> vector(n);

    for (std::size_t i = 0; i < n; ++i) {
        vector[i] =
            std::sin(
                static_cast<double>(i + 1)
            );
    }

    double norm =
        vectorNorm(vector);

    if (norm < DFTConstants::EPS) {
        throw std::runtime_error(
            "Vector inicial numericamente nulo."
        );
    }

    for (double& value : vector) {
        value /= norm;
    }

    const double shift =
        eigenvalue +
        1.0e-10 *
        std::max(
            1.0,
            std::abs(eigenvalue)
        );

    TridiagonalMatrix shifted =
        matrix;

    for (double& value :
         shifted.diagonal) {
        value -= shift;
    }

    for (std::size_t iteration = 0;
         iteration < maxIterations;
         ++iteration) {

        std::vector<double> next =
            solveTridiagonal(
                shifted,
                vector
            );

        const double nextNorm =
            vectorNorm(next);

        if (nextNorm < DFTConstants::EPS) {
            throw std::runtime_error(
                "Autovector numericamente nulo."
            );
        }

        for (double& value : next) {
            value /= nextNorm;
        }

        if (dotProduct(vector, next) < 0.0) {
            for (double& value : next) {
                value = -value;
            }
        }

        const double difference =
            maxAbsoluteDifference(
                vector,
                next
            );

        vector =
            std::move(next);

        if (difference < 1.0e-12) {
            break;
        }
    }

    return vector;
}

AtomicOrbital solveOrbital(
    const TridiagonalMatrix& matrix,
    const std::vector<double>& r,
    int n,
    int l,
    int electrons,
    std::size_t orbitalIndex
) {
    if (matrix.diagonal.size() != r.size()) {
        throw std::invalid_argument(
            "La matriz y la malla radial deben tener el mismo tamano."
        );
    }

    if (r.size() < 2) {
        throw std::invalid_argument(
            "La malla radial debe contener al menos dos puntos."
        );
    }

    if (orbitalIndex >= matrix.diagonal.size()) {
        throw std::invalid_argument(
            "Indice de orbital fuera del rango."
        );
    }

    const double minimum =
        *std::min_element(
            matrix.diagonal.begin(),
            matrix.diagonal.end()
        );

    const double maximum =
        *std::max_element(
            matrix.diagonal.begin(),
            matrix.diagonal.end()
        );

    const double lowerBound =
        minimum - 10.0;

    const double upperBound =
        maximum + 10.0;

    const double eigenvalue =
        findEigenvalue(
            matrix,
            orbitalIndex,
            lowerBound,
            upperBound
        );

    std::vector<double> u =
        solveEigenvector(
            matrix,
            eigenvalue
        );

    normalizeVector(
        u,
        r[1] - r[0]
    );

    AtomicOrbital orbital;

    orbital.n =
        n;

    orbital.l =
        l;

    orbital.electrons =
        electrons;

    orbital.eigenvalue =
        eigenvalue;

    orbital.u =
        std::move(u);

    return orbital;
}