#include "DavidsonMath.h"

#include "DFTConstants.h"
#include "NumericalMethods.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

constexpr double VECTOR_TOLERANCE_SQUARED = 1.0e-20;
constexpr double DENSE_EIGENVALUE_TOLERANCE = 1.0e-13;
constexpr std::size_t DENSE_EIGENVALUE_MAX_ITERATIONS = 10000;

}


double davidsonDot(
    const std::vector<double>& a,
    const std::vector<double>& b,
    double dV
) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Los vectores deben tener el mismo tamano."
        );
    }

    return dV * dotProduct(a, b);
}


double davidsonNormSquared(
    const std::vector<double>& v,
    double dV
) {
    return davidsonDot(v, v, dV);
}


bool davidsonValidNorm(
    const std::vector<double>& v,
    double dV,
    double toleranceSquared
) {
    const double n2 =
        davidsonNormSquared(v, dV);

    return std::isfinite(n2) &&
           n2 > toleranceSquared;
}


void davidsonNormalize(
    std::vector<double>& v,
    double dV,
    double toleranceSquared
) {
    const double n2 =
        davidsonNormSquared(v, dV);

    if (!std::isfinite(n2) ||
        n2 <= toleranceSquared) {

        throw std::runtime_error(
            "No se puede normalizar el vector molecular."
        );
    }

    const double n =
        std::sqrt(n2);

    for (double& value : v) {
        value /= n;
    }
}


void davidsonOrthogonalize(
    std::vector<double>& v,
    const std::vector<std::vector<double>>& basis,
    double dV
) {
    for (int pass = 0; pass < 2; ++pass) {

        for (const auto& b : basis) {

            const double projection =
                davidsonDot(b, v, dV);

            for (std::size_t i = 0;
                 i < v.size();
                 ++i) {

                v[i] -=
                    projection * b[i];
            }
        }
    }
}


double davidsonRayleigh(
    const std::vector<double>& psi,
    const std::vector<double>& hPsi,
    double dV
) {
    const double denominator =
        davidsonNormSquared(psi, dV);

    if (!std::isfinite(denominator) ||
        denominator <= VECTOR_TOLERANCE_SQUARED) {

        throw std::runtime_error(
            "Norma invalida en el cociente de Rayleigh."
        );
    }

    return davidsonDot(psi, hPsi, dV) /
           denominator;
}


double davidsonResidualNorm(
    const std::vector<double>& psi,
    const std::vector<double>& hPsi,
    double eigenvalue,
    double dV
) {
    std::vector<double> residual(
        psi.size(),
        0.0
    );

    for (std::size_t i = 0;
         i < psi.size();
         ++i) {

        residual[i] =
            hPsi[i] -
            eigenvalue * psi[i];
    }

    const double n2 =
        davidsonNormSquared(residual, dV);

    if (!std::isfinite(n2)) {
        return std::numeric_limits<double>::infinity();
    }

    return std::sqrt(n2);
}


DavidsonEigenpair davidsonDiagonalize(
    std::vector<std::vector<double>> matrix
) {
    const std::size_t n =
        matrix.size();

    if (n == 0) {
        throw std::invalid_argument(
            "La matriz no puede estar vacia."
        );
    }

    std::vector<std::vector<double>> vectors(
        n,
        std::vector<double>(n, 0.0)
    );

    for (std::size_t i = 0;
         i < n;
         ++i) {

        vectors[i][i] = 1.0;
    }

    for (std::size_t iteration = 0;
         iteration < DENSE_EIGENVALUE_MAX_ITERATIONS;
         ++iteration) {

        double maximum = 0.0;
        std::size_t p = 0;
        std::size_t q = 0;

        for (std::size_t i = 0;
             i < n;
             ++i) {

            for (std::size_t j = i + 1;
                 j < n;
                 ++j) {

                const double value =
                    std::abs(matrix[i][j]);

                if (value > maximum) {
                    maximum = value;
                    p = i;
                    q = j;
                }
            }
        }

        if (maximum <=
            DENSE_EIGENVALUE_TOLERANCE) {

            break;
        }

        const double app =
            matrix[p][p];

        const double aqq =
            matrix[q][q];

        const double apq =
            matrix[p][q];

        if (apq == 0.0) {
            continue;
        }

        const double tau =
            (aqq - app) /
            (2.0 * apq);

        const double t =
            std::copysign(
                1.0 /
                (
                    std::abs(tau) +
                    std::sqrt(
                        1.0 +
                        tau * tau
                    )
                ),
                tau
            );

        const double c =
            1.0 /
            std::sqrt(
                1.0 + t * t
            );

        const double s =
            t * c;

        matrix[p][p] =
            app - t * apq;

        matrix[q][q] =
            aqq + t * apq;

        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;

        for (std::size_t k = 0;
             k < n;
             ++k) {

            if (k == p || k == q) {
                continue;
            }

            const double mkp =
                matrix[k][p];

            const double mkq =
                matrix[k][q];

            matrix[k][p] =
                c * mkp -
                s * mkq;

            matrix[p][k] =
                matrix[k][p];

            matrix[k][q] =
                s * mkp +
                c * mkq;

            matrix[q][k] =
                matrix[k][q];
        }

        for (std::size_t k = 0;
             k < n;
             ++k) {

            const double vkp =
                vectors[k][p];

            const double vkq =
                vectors[k][q];

            vectors[k][p] =
                c * vkp -
                s * vkq;

            vectors[k][q] =
                s * vkp +
                c * vkq;
        }
    }

    std::size_t index = 0;

    for (std::size_t i = 1;
         i < n;
         ++i) {

        if (matrix[i][i] <
            matrix[index][index]) {

            index = i;
        }
    }

    std::vector<double> eigenvector(n);

    for (std::size_t i = 0;
         i < n;
         ++i) {

        eigenvector[i] =
            vectors[i][index];
    }

    double norm = 0.0;

    for (double value : eigenvector) {
        norm += value * value;
    }

    norm =
        std::sqrt(norm);

    if (!std::isfinite(norm) ||
        norm <= DFTConstants::EPS) {

        throw std::runtime_error(
            "Autovector denso invalido."
        );
    }

    for (double& value : eigenvector) {
        value /= norm;
    }

    return {
        matrix[index][index],
        std::move(eigenvector)
    };
}


std::vector<double> davidsonCombine(
    const std::vector<std::vector<double>>& basis,
    const std::vector<double>& coefficients
) {
    if (basis.empty()) {
        throw std::invalid_argument(
            "La base no puede estar vacia."
        );
    }

    if (basis.size() !=
        coefficients.size()) {

        throw std::invalid_argument(
            "La base y los coeficientes no coinciden."
        );
    }

    std::vector<double> result(
        basis.front().size(),
        0.0
    );

    for (std::size_t j = 0;
         j < basis.size();
         ++j) {

        for (std::size_t i = 0;
             i < result.size();
             ++i) {

            result[i] +=
                coefficients[j] *
                basis[j][i];
        }
    }

    return result;
}