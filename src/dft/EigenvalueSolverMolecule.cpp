#include "EigenvalueSolverMolecule.h"

#include "DFTConstants.h"
#include "KohnShamHamiltonianMolecule.h"
#include "NumericalMethods.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double MO_RESIDUAL_TOLERANCE = 1.0e-7;

constexpr double SUBSPACE_ORTHOGONALITY_TOLERANCE =
    1.0e-12;

constexpr double SUBSPACE_ORTHOGONALITY_TOLERANCE_SQUARED =
    SUBSPACE_ORTHOGONALITY_TOLERANCE *
    SUBSPACE_ORTHOGONALITY_TOLERANCE;

constexpr double VECTOR_RECOVERY_TOLERANCE_SQUARED =
    1.0e-20;

/*
 * Un subespacio mas pequeno reduce el coste de la diagonalizacion
 * proyectada y, mas importante, limita la cantidad de vectores
 * grandes que deben combinarse en cada iteracion.
 *
 * El metodo Davidson utiliza el residuo preacondicionado para
 * generar nuevas direcciones, por lo que no necesita crecer hasta
 * un subespacio tan grande como el metodo anterior.
 */
constexpr std::size_t MAX_SUBSPACE_DIMENSION = 16;

constexpr double DENSE_EIGENVALUE_TOLERANCE = 1.0e-13;

constexpr std::size_t DENSE_EIGENVALUE_MAX_ITERATIONS = 10000;

/*
 * Si el denominador del precondicionador es demasiado pequeno,
 * se evita una division numericamente inestable.
 */
constexpr double DAVIDSON_DENOMINATOR_TOLERANCE = 1.0e-10;


/*
 * ================================================================
 * VECTOR UTILITIES
 * ================================================================
 */

double molecularDotProduct(
    const std::vector<double>& a,
    const std::vector<double>& b,
    double volumeElement
) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Los vectores moleculares deben tener el mismo tamano."
        );
    }

    if (volumeElement <= 0.0) {
        throw std::invalid_argument(
            "El elemento de volumen debe ser mayor que cero."
        );
    }

    return volumeElement * dotProduct(a, b);
}


void printVectorDiagnostic(
    const std::string& label,
    const std::vector<double>& vector,
    double volumeElement
) {
    double squaredNorm = 0.0;
    double maximumAbsoluteValue = 0.0;
    std::size_t nonFiniteValues = 0;

    for (double value : vector) {
        if (!std::isfinite(value)) {
            ++nonFiniteValues;
            continue;
        }

        squaredNorm += value * value;

        maximumAbsoluteValue =
            std::max(
                maximumAbsoluteValue,
                std::abs(value)
            );
    }

    squaredNorm *= volumeElement;

    std::cout
        << "    [DIAGNOSTICO] "
        << label
        << "\n"
        << "      Dimension:       "
        << vector.size()
        << "\n"
        << "      dV:              "
        << volumeElement
        << "\n"
        << "      Norma^2:         "
        << squaredNorm
        << "\n"
        << "      Norma:           "
        << std::sqrt(
               std::max(
                   0.0,
                   squaredNorm
               )
           )
        << "\n"
        << "      Max |valor|:     "
        << maximumAbsoluteValue
        << "\n"
        << "      No finitos:      "
        << nonFiniteValues
        << "\n";
}


double molecularNorm(
    const std::vector<double>& vector,
    double volumeElement,
    const std::string& context = ""
) {
    const double squaredNorm =
        molecularDotProduct(
            vector,
            vector,
            volumeElement
        );

    if (!std::isfinite(squaredNorm) ||
        !(squaredNorm >
          VECTOR_RECOVERY_TOLERANCE_SQUARED)) {

        std::cout
            << "\n"
            << "    >>> ERROR DE NORMA MOLECULAR <<<\n"
            << "      Contexto:         "
            << context
            << "\n"
            << "      Umbral norma^2:   "
            << VECTOR_RECOVERY_TOLERANCE_SQUARED
            << "\n"
            << "      Norma^2:          "
            << squaredNorm
            << "\n"
            << "      dV:               "
            << volumeElement
            << "\n";

        printVectorDiagnostic(
            "Vector que no pudo normalizarse",
            vector,
            volumeElement
        );

        throw std::runtime_error(
            "La funcion de onda molecular tiene norma "
            "numericamente nula."
        );
    }

    return std::sqrt(squaredNorm);
}


bool hasMolecularNorm(
    const std::vector<double>& vector,
    double volumeElement,
    double toleranceSquared
) {
    const double squaredNorm =
        molecularDotProduct(
            vector,
            vector,
            volumeElement
        );

    return std::isfinite(squaredNorm) &&
           squaredNorm > toleranceSquared;
}


void normalizeMolecularVector(
    std::vector<double>& vector,
    double volumeElement,
    const std::string& context = ""
) {
    const double norm =
        molecularNorm(
            vector,
            volumeElement,
            context
        );

    for (double& value : vector) {
        value /= norm;
    }
}


/*
 * ================================================================
 * ORTHOGONALIZATION
 * ================================================================
 */

void orthogonalizeAgainstBasis(
    std::vector<double>& vector,
    const std::vector<std::vector<double>>& basis,
    double volumeElement
) {
    for (int pass = 0; pass < 2; ++pass) {

        for (const std::vector<double>& basisVector : basis) {

            if (basisVector.size() != vector.size()) {
                throw std::invalid_argument(
                    "Los vectores de la base deben tener "
                    "el mismo tamano."
                );
            }

            const double projection =
                molecularDotProduct(
                    basisVector,
                    vector,
                    volumeElement
                );

            for (std::size_t i = 0;
                 i < vector.size();
                 ++i) {

                vector[i] -=
                    projection *
                    basisVector[i];
            }
        }
    }
}


void orthogonalizeAgainstPreviousOrbitals(
    std::vector<double>& vector,
    const std::vector<MolecularOrbital>& previousOrbitals,
    double volumeElement
) {
    std::vector<double> orbitalNormSquared;

    orbitalNormSquared.reserve(
        previousOrbitals.size()
    );

    for (const MolecularOrbital& orbital :
         previousOrbitals) {

        if (orbital.psi.size() != vector.size()) {
            throw std::invalid_argument(
                "Los orbitales moleculares deben estar "
                "definidos sobre la misma malla."
            );
        }

        const double normSquared =
            molecularDotProduct(
                orbital.psi,
                orbital.psi,
                volumeElement
            );

        if (!std::isfinite(normSquared) ||
            !(normSquared >
              VECTOR_RECOVERY_TOLERANCE_SQUARED)) {

            throw std::runtime_error(
                "Un orbital molecular previo no tiene "
                "una norma numericamente valida."
            );
        }

        orbitalNormSquared.push_back(
            normSquared
        );
    }

    for (int pass = 0; pass < 2; ++pass) {

        for (std::size_t orbitalIndex = 0;
             orbitalIndex < previousOrbitals.size();
             ++orbitalIndex) {

            const MolecularOrbital& orbital =
                previousOrbitals[orbitalIndex];

            const double projectionNumerator =
                molecularDotProduct(
                    orbital.psi,
                    vector,
                    volumeElement
                );

            const double projection =
                projectionNumerator /
                orbitalNormSquared[orbitalIndex];

            for (std::size_t i = 0;
                 i < vector.size();
                 ++i) {

                vector[i] -=
                    projection *
                    orbital.psi[i];
            }
        }
    }
}


/*
 * ================================================================
 * RAYLEIGH / RESIDUAL
 * ================================================================
 */

double rayleighQuotient(
    const std::vector<double>& psi,
    const std::vector<double>& hPsi,
    double volumeElement
) {
    const double denominator =
        molecularDotProduct(
            psi,
            psi,
            volumeElement
        );

    if (!std::isfinite(denominator) ||
        !(denominator >
          VECTOR_RECOVERY_TOLERANCE_SQUARED)) {

        throw std::runtime_error(
            "No se puede calcular el cociente de Rayleigh "
            "porque la norma^2 del orbital no es valida."
        );
    }

    const double numerator =
        molecularDotProduct(
            psi,
            hPsi,
            volumeElement
        );

    return numerator / denominator;
}


double molecularResidual(
    const std::vector<double>& psi,
    const std::vector<double>& hPsi,
    double eigenvalue,
    double volumeElement
) {
    if (psi.size() != hPsi.size()) {
        throw std::invalid_argument(
            "La funcion de onda y Hpsi deben tener "
            "el mismo tamano."
        );
    }

    double residualSquared = 0.0;

    for (std::size_t i = 0;
         i < psi.size();
         ++i) {

        const double residual =
            hPsi[i] -
            eigenvalue *
            psi[i];

        residualSquared +=
            residual *
            residual;
    }

    return std::sqrt(
        residualSquared *
        volumeElement
    );
}


/*
 * ================================================================
 * INITIAL MOLECULAR VECTORS
 * ================================================================
 */

std::vector<double> buildInitialMolecularVector(
    const CartesianGrid& grid,
    std::size_t orbitalIndex
) {
    const std::size_t dimension =
        grid.getSize();

    std::vector<double> vector(
        dimension,
        0.0
    );

    const double centerX =
        0.5 *
        (
            grid.getXMin() +
            grid.getXMax()
        );

    const double centerY =
        0.5 *
        (
            grid.getYMin() +
            grid.getYMax()
        );

    const double centerZ =
        0.5 *
        (
            grid.getZMin() +
            grid.getZMax()
        );

    const double sigma =
        0.8 +
        0.18 *
        static_cast<double>(
            orbitalIndex
        );

    const double inverseTwoSigmaSquared =
        1.0 /
        (
            2.0 *
            sigma *
            sigma
        );

    const std::size_t mode =
        orbitalIndex % 9;

    const std::size_t nx =
        grid.getNx();

    const std::size_t ny =
        grid.getNy();

    const std::size_t nz =
        grid.getNz();

    for (std::size_t k = 0;
         k < nz;
         ++k) {

        const double z =
            grid.getZ(k) -
            centerZ;

        const std::size_t zOffset =
            k * nx * ny;

        for (std::size_t j = 0;
             j < ny;
             ++j) {

            const double y =
                grid.getY(j) -
                centerY;

            const std::size_t rowOffset =
                zOffset +
                j * nx;

            for (std::size_t i = 0;
                 i < nx;
                 ++i) {

                const double x =
                    grid.getX(i) -
                    centerX;

                const double radiusSquared =
                    x * x +
                    y * y +
                    z * z;

                const double gaussian =
                    std::exp(
                        -radiusSquared *
                        inverseTwoSigmaSquared
                    );

                double polynomial;

                switch (mode) {

                case 0:
                    polynomial = 1.0;
                    break;

                case 1:
                    polynomial = x;
                    break;

                case 2:
                    polynomial = y;
                    break;

                case 3:
                    polynomial = z;
                    break;

                case 4:
                    polynomial = x * y;
                    break;

                case 5:
                    polynomial = x * z;
                    break;

                case 6:
                    polynomial = y * z;
                    break;

                case 7:
                    polynomial =
                        x * x -
                        y * y;
                    break;

                case 8:
                    polynomial =
                        2.0 * z * z -
                        x * x -
                        y * y;
                    break;

                default:
                    polynomial = 1.0;
                    break;
                }

                vector[rowOffset + i] =
                    polynomial *
                    gaussian;
            }
        }
    }

    return vector;
}


/*
 * ================================================================
 * DENSE SYMMETRIC EIGENSOLVER
 * ================================================================
 */

struct DenseSymmetricEigenpair {
    double eigenvalue;
    std::vector<double> eigenvector;
};


DenseSymmetricEigenpair diagonalizeDenseSymmetricMatrix(
    std::vector<std::vector<double>> matrix
) {
    const std::size_t n =
        matrix.size();

    if (n == 0) {
        throw std::invalid_argument(
            "La matriz densa no puede estar vacia."
        );
    }

    for (const std::vector<double>& row : matrix) {

        if (row.size() != n) {
            throw std::invalid_argument(
                "La matriz densa debe ser cuadrada."
            );
        }
    }

    std::vector<std::vector<double>> eigenvectors(
        n,
        std::vector<double>(
            n,
            0.0
        )
    );

    for (std::size_t i = 0;
         i < n;
         ++i) {

        eigenvectors[i][i] = 1.0;
    }

    for (std::size_t iteration = 0;
         iteration < DENSE_EIGENVALUE_MAX_ITERATIONS;
         ++iteration) {

        double maximumOffDiagonal = 0.0;

        std::size_t p = 0;
        std::size_t q = 0;

        for (std::size_t i = 0;
             i < n;
             ++i) {

            for (std::size_t j = i + 1;
                 j < n;
                 ++j) {

                const double magnitude =
                    std::abs(
                        matrix[i][j]
                    );

                if (magnitude >
                    maximumOffDiagonal) {

                    maximumOffDiagonal =
                        magnitude;

                    p = i;
                    q = j;
                }
            }
        }

        double diagonalScale = 1.0;

        for (std::size_t i = 0;
             i < n;
             ++i) {

            diagonalScale =
                std::max(
                    diagonalScale,
                    std::abs(
                        matrix[i][i]
                    )
                );
        }

        if (maximumOffDiagonal <=
            DENSE_EIGENVALUE_TOLERANCE *
            diagonalScale) {

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
                1.0 +
                t * t
            );

        const double s =
            t *
            c;

        matrix[p][p] =
            app -
            t * apq;

        matrix[q][q] =
            aqq +
            t * apq;

        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;

        for (std::size_t k = 0;
             k < n;
             ++k) {

            if (k == p || k == q) {
                continue;
            }

            const double akp =
                matrix[k][p];

            const double akq =
                matrix[k][q];

            const double newKp =
                c * akp -
                s * akq;

            const double newKq =
                s * akp +
                c * akq;

            matrix[k][p] = newKp;
            matrix[p][k] = newKp;

            matrix[k][q] = newKq;
            matrix[q][k] = newKq;
        }

        for (std::size_t k = 0;
             k < n;
             ++k) {

            const double vkp =
                eigenvectors[k][p];

            const double vkq =
                eigenvectors[k][q];

            eigenvectors[k][p] =
                c * vkp -
                s * vkq;

            eigenvectors[k][q] =
                s * vkp +
                c * vkq;
        }
    }

    std::size_t minimumIndex = 0;

    for (std::size_t i = 1;
         i < n;
         ++i) {

        if (matrix[i][i] <
            matrix[minimumIndex][minimumIndex]) {

            minimumIndex = i;
        }
    }

    std::vector<double> eigenvector(n);

    for (std::size_t i = 0;
         i < n;
         ++i) {

        eigenvector[i] =
            eigenvectors[i][minimumIndex];
    }

    const double norm =
        std::sqrt(
            dotProduct(
                eigenvector,
                eigenvector
            )
        );

    if (!(norm > DFTConstants::EPS) ||
        !std::isfinite(norm)) {

        throw std::runtime_error(
            "El autovector denso es numericamente nulo."
        );
    }

    for (double& value :
         eigenvector) {

        value /= norm;
    }

    return DenseSymmetricEigenpair{
        matrix[minimumIndex][minimumIndex],
        std::move(eigenvector)
    };
}


/*
 * ================================================================
 * SUBSPACE OPERATIONS
 * ================================================================
 */

std::vector<double> buildSubspaceVector(
    const std::vector<std::vector<double>>& basis,
    const std::vector<double>& coefficients
) {
    if (basis.empty()) {
        throw std::invalid_argument(
            "La base del subespacio no puede estar vacia."
        );
    }

    if (basis.size() != coefficients.size()) {
        throw std::invalid_argument(
            "La base y los coeficientes deben "
            "tener el mismo tamano."
        );
    }

    std::vector<double> vector(
        basis.front().size(),
        0.0
    );

    for (std::size_t j = 0;
         j < basis.size();
         ++j) {

        if (basis[j].size() != vector.size()) {
            throw std::invalid_argument(
                "Los vectores del subespacio deben "
                "tener el mismo tamano."
            );
        }

        const double coefficient =
            coefficients[j];

        if (coefficient == 0.0) {
            continue;
        }

        for (std::size_t i = 0;
             i < vector.size();
             ++i) {

            vector[i] +=
                coefficient *
                basis[j][i];
        }
    }

    return vector;
}


std::vector<double> buildSubspaceHamiltonianVector(
    const std::vector<std::vector<double>>& hBasis,
    const std::vector<double>& coefficients
) {
    if (hBasis.empty()) {
        throw std::invalid_argument(
            "H(base) no puede estar vacio."
        );
    }

    if (hBasis.size() != coefficients.size()) {
        throw std::invalid_argument(
            "H(base) y los coeficientes deben "
            "tener el mismo tamano."
        );
    }

    std::vector<double> hVector(
        hBasis.front().size(),
        0.0
    );

    for (std::size_t j = 0;
         j < hBasis.size();
         ++j) {

        if (hBasis[j].size() != hVector.size()) {
            throw std::invalid_argument(
                "Los vectores H(base) deben "
                "tener el mismo tamano."
            );
        }

        const double coefficient =
            coefficients[j];

        if (coefficient == 0.0) {
            continue;
        }

        for (std::size_t i = 0;
             i < hVector.size();
             ++i) {

            hVector[i] +=
                coefficient *
                hBasis[j][i];
        }
    }

    return hVector;
}


void appendProjectedHamiltonianColumn(
    std::vector<std::vector<double>>& projectedHamiltonian,
    const std::vector<std::vector<double>>& basis,
    const std::vector<std::vector<double>>& hBasis,
    std::size_t newIndex,
    double volumeElement
) {
    const std::size_t dimension =
        basis.size();

    if (dimension == 0 ||
        dimension != hBasis.size() ||
        newIndex >= dimension) {

        throw std::invalid_argument(
            "Dimensiones invalidas al actualizar "
            "el Hamiltoniano proyectado."
        );
    }

    projectedHamiltonian.resize(
        dimension
    );

    for (std::vector<double>& row :
         projectedHamiltonian) {

        row.resize(
            dimension,
            0.0
        );
    }

    for (std::size_t i = 0;
         i <= newIndex;
         ++i) {

        const double value =
            molecularDotProduct(
                basis[i],
                hBasis[newIndex],
                volumeElement
            );

        projectedHamiltonian[i][newIndex] =
            value;

        projectedHamiltonian[newIndex][i] =
            value;
    }
}


/*
 * ================================================================
 * DAVIDSON PRECONDITIONER
 * ================================================================
 *
 * Para el Hamiltoniano:
 *
 *     H = -1/2 nabla^2 + Veff
 *
 * y el Laplaciano central de siete puntos:
 *
 *     diag(-1/2 nabla^2)
 *
 * es:
 *
 *     1/dx^2 + 1/dy^2 + 1/dz^2
 *
 * Por tanto:
 *
 *     diag(H) =
 *         1/dx^2 +
 *         1/dy^2 +
 *         1/dz^2 +
 *         Veff
 *
 * El vector Davidson se obtiene como:
 *
 *     t_i = r_i / (theta - H_ii)
 *
 * con proteccion cuando el denominador se aproxima a cero.
 *
 * No se aplica H nuevamente para construir esta direccion.
 *
 * ================================================================
 */

std::vector<double> buildHamiltonianDiagonal(
    const CartesianGrid& grid,
    const std::vector<double>& effectivePotential
) {
    if (effectivePotential.size() != grid.getSize()) {
        throw std::invalid_argument(
            "El potencial efectivo y la malla deben "
            "tener la misma dimension."
        );
    }

    const double dx =
        grid.getDx();

    const double dy =
        grid.getDy();

    const double dz =
        grid.getDz();

    if (dx <= 0.0 ||
        dy <= 0.0 ||
        dz <= 0.0) {

        throw std::invalid_argument(
            "Los espaciamientos de la malla deben "
            "ser positivos."
        );
    }

    const double kineticDiagonal =
        1.0 / (dx * dx) +
        1.0 / (dy * dy) +
        1.0 / (dz * dz);

    std::vector<double> diagonal(
        effectivePotential.size()
    );

    for (std::size_t i = 0;
         i < effectivePotential.size();
         ++i) {

        diagonal[i] =
            kineticDiagonal +
            effectivePotential[i];
    }

    return diagonal;
}


bool buildDavidsonCorrection(
    const std::vector<double>& residual,
    const std::vector<double>& hamiltonianDiagonal,
    double eigenvalue,
    double volumeElement,
    std::vector<double>& correction
) {
    if (residual.size() !=
        hamiltonianDiagonal.size()) {

        throw std::invalid_argument(
            "El residuo y la diagonal del Hamiltoniano "
            "deben tener el mismo tamano."
        );
    }

    correction.resize(
        residual.size()
    );

    bool usable = true;

    for (std::size_t i = 0;
         i < residual.size();
         ++i) {

        const double denominator =
            eigenvalue -
            hamiltonianDiagonal[i];

        if (!std::isfinite(denominator) ||
            std::abs(denominator) <
            DAVIDSON_DENOMINATOR_TOLERANCE) {

            correction[i] =
                residual[i];

            continue;
        }

        correction[i] =
            residual[i] /
            denominator;
    }

    orthogonalizeAgainstBasis(
        correction,
        std::vector<std::vector<double>>{},
        volumeElement
    );

    /*
     * La llamada anterior no modifica el vector porque la base
     * esta vacia. Se mantiene el flujo de validacion centralizado.
     */

    for (double value : correction) {

        if (!std::isfinite(value)) {
            usable = false;
            break;
        }
    }

    if (!usable) {
        return false;
    }

    const double normSquared =
        molecularDotProduct(
            correction,
            correction,
            volumeElement
        );

    return std::isfinite(normSquared) &&
           normSquared >
           SUBSPACE_ORTHOGONALITY_TOLERANCE_SQUARED;
}


/*
 * ================================================================
 * INDEPENDENT INITIAL VECTOR
 * ================================================================
 */

bool buildIndependentInitialVector(
    const CartesianGrid& grid,
    std::size_t orbitalIndex,
    const std::vector<MolecularOrbital>& previousOrbitals,
    double volumeElement,
    std::vector<double>& result
) {
    for (std::size_t seed = orbitalIndex;
         seed < orbitalIndex + 64;
         ++seed) {

        std::vector<double> candidate =
            buildInitialMolecularVector(
                grid,
                seed
            );

        orthogonalizeAgainstPreviousOrbitals(
            candidate,
            previousOrbitals,
            volumeElement
        );

        if (!hasMolecularNorm(
                candidate,
                volumeElement,
                VECTOR_RECOVERY_TOLERANCE_SQUARED)) {

            continue;
        }

        normalizeMolecularVector(
            candidate,
            volumeElement,
            "semilla inicial"
        );

        result =
            std::move(candidate);

        return true;
    }

    return false;
}


bool buildInitialVectorFromPreviousOrbital(
    const CartesianGrid& grid,
    const MolecularOrbital& initialGuess,
    const std::vector<MolecularOrbital>& previousOrbitals,
    double volumeElement,
    std::vector<double>& result
) {
    if (initialGuess.psi.size() !=
        grid.getSize()) {

        throw std::invalid_argument(
            "El orbital inicial y la malla cartesiana "
            "deben tener la misma dimension."
        );
    }

    result =
        initialGuess.psi;

    for (double value : result) {

        if (!std::isfinite(value)) {
            return false;
        }
    }

    orthogonalizeAgainstPreviousOrbitals(
        result,
        previousOrbitals,
        volumeElement
    );

    if (!hasMolecularNorm(
            result,
            volumeElement,
            VECTOR_RECOVERY_TOLERANCE_SQUARED)) {

        return false;
    }

    normalizeMolecularVector(
        result,
        volumeElement,
        "orbital SCF anterior"
    );

    return true;
}


bool buildInitialVector(
    const CartesianGrid& grid,
    std::size_t orbitalIndex,
    const std::vector<MolecularOrbital>& previousOrbitals,
    double volumeElement,
    const MolecularOrbital* initialGuess,
    std::vector<double>& result
) {
    if (initialGuess != nullptr) {

        if (buildInitialVectorFromPreviousOrbital(
                grid,
                *initialGuess,
                previousOrbitals,
                volumeElement,
                result)) {

            return true;
        }
    }

    return buildIndependentInitialVector(
        grid,
        orbitalIndex,
        previousOrbitals,
        volumeElement,
        result
    );
}

} // namespace


/*
 * ================================================================
 * SINGLE MOLECULAR ORBITAL
 * ================================================================
 */

MolecularOrbital solveMolecularOrbital(
    const CartesianGrid& grid,
    const std::vector<double>& effectivePotential,
    std::size_t orbitalIndex,
    SpinChannel spin,
    int electrons,
    const std::vector<MolecularOrbital>& previousOrbitals,
    const MolecularOrbital* initialGuess,
    std::size_t maxIterations
) {
    if (effectivePotential.size() !=
        grid.getSize()) {

        throw std::invalid_argument(
            "El potencial efectivo y la malla cartesiana "
            "deben tener el mismo tamano."
        );
    }

    if (electrons < 0 ||
        electrons > 2) {

        throw std::invalid_argument(
            "La ocupacion de un orbital molecular debe "
            "estar entre cero y dos."
        );
    }

    if (maxIterations == 0) {

        throw std::invalid_argument(
            "El numero maximo de iteraciones debe ser "
            "mayor que cero."
        );
    }

    const double volumeElement =
        grid.getDx() *
        grid.getDy() *
        grid.getDz();

    if (volumeElement <= 0.0) {

        throw std::invalid_argument(
            "El elemento de volumen debe ser mayor que cero."
        );
    }

    const std::size_t dimension =
        grid.getSize();

    if (dimension == 0) {

        throw std::invalid_argument(
            "La malla cartesiana no puede estar vacia."
        );
    }

    const std::string spinName =
        spin == SpinChannel::Alpha
            ? "Alpha"
            : "Beta";

    std::cout
        << "\n"
        << "  ----------------------------------------\n"
        << "  Orbital molecular "
        << orbitalIndex
        << " | Spin "
        << spinName
        << " | e="
        << electrons
        << "\n"
        << "  ----------------------------------------\n"
        << "    Orbitales previos: "
        << previousOrbitals.size()
        << "\n"
        << "    Dimension malla:   "
        << dimension
        << "\n"
        << "    dV:                "
        << volumeElement
        << "\n";


    /*
     * ------------------------------------------------------------
     * DAVIDSON PRECONDITIONER
     * ------------------------------------------------------------
     */

    const std::vector<double>
        hamiltonianDiagonal =
            buildHamiltonianDiagonal(
                grid,
                effectivePotential
            );


    /*
     * ------------------------------------------------------------
     * INITIAL VECTOR
     * ------------------------------------------------------------
     */

    std::vector<double> initialVector;

    if (!buildInitialVector(
            grid,
            orbitalIndex,
            previousOrbitals,
            volumeElement,
            initialGuess,
            initialVector)) {

        throw std::runtime_error(
            "No se pudo construir un vector inicial "
            "ortogonal a los orbitales anteriores."
        );
    }


    /*
     * ------------------------------------------------------------
     * SUBSPACE
     * ------------------------------------------------------------
     */

    std::vector<std::vector<double>> basis;
    std::vector<std::vector<double>> hBasis;

    basis.reserve(
        MAX_SUBSPACE_DIMENSION
    );

    hBasis.reserve(
        MAX_SUBSPACE_DIMENSION
    );

    basis.push_back(
        std::move(initialVector)
    );

    std::vector<std::vector<double>>
        projectedHamiltonian;

    projectedHamiltonian.reserve(
        MAX_SUBSPACE_DIMENSION
    );

    std::size_t hApplications = 0;

    double finalEigenvalue =
        std::numeric_limits<double>::infinity();

    double finalResidual =
        std::numeric_limits<double>::infinity();

    std::vector<double> finalVector;


    /*
     * ============================================================
     * DAVIDSON ITERATION
     * ============================================================
     */

    while (hApplications < maxIterations) {

        /*
         * --------------------------------------------------------
         * APPLY H TO NEW BASIS VECTOR
         * --------------------------------------------------------
         */

        if (hBasis.size() < basis.size()) {

            const std::size_t index =
                hBasis.size();

            std::vector<double> hVector =
                applyMolecularKohnShamHamiltonian(
                    grid,
                    effectivePotential,
                    basis[index]
                );

            if (hVector.size() != dimension) {

                throw std::runtime_error(
                    "El Hamiltoniano molecular devolvio "
                    "una dimension incorrecta."
                );
            }

            hBasis.push_back(
                std::move(hVector)
            );

            ++hApplications;

            appendProjectedHamiltonianColumn(
                projectedHamiltonian,
                basis,
                hBasis,
                index,
                volumeElement
            );
        }


        /*
         * --------------------------------------------------------
         * PROJECTED EIGENPROBLEM
         * --------------------------------------------------------
         */

        const DenseSymmetricEigenpair
            projectedEigenpair =
                diagonalizeDenseSymmetricMatrix(
                    projectedHamiltonian
                );


        /*
         * --------------------------------------------------------
         * RITZ VECTOR
         * --------------------------------------------------------
         */

        std::vector<double> ritzVector =
            buildSubspaceVector(
                basis,
                projectedEigenpair.eigenvector
            );

        orthogonalizeAgainstPreviousOrbitals(
            ritzVector,
            previousOrbitals,
            volumeElement
        );

        const double ritzNormSquared =
            molecularDotProduct(
                ritzVector,
                ritzVector,
                volumeElement
            );


        /*
         * --------------------------------------------------------
         * RITZ COLLAPSE / RECOVERY
         * --------------------------------------------------------
         */

        if (!std::isfinite(ritzNormSquared) ||
            !(ritzNormSquared >
              VECTOR_RECOVERY_TOLERANCE_SQUARED)) {

            if (basis.size() >=
                MAX_SUBSPACE_DIMENSION) {

                throw std::runtime_error(
                    "El vector de Ritz colapso en un "
                    "subespacio completo."
                );
            }

            std::vector<double> recoveryVector;

            if (!buildIndependentInitialVector(
                    grid,
                    orbitalIndex + basis.size(),
                    previousOrbitals,
                    volumeElement,
                    recoveryVector)) {

                throw std::runtime_error(
                    "No se pudo construir un vector de "
                    "recuperacion para el subespacio."
                );
            }

            orthogonalizeAgainstBasis(
                recoveryVector,
                basis,
                volumeElement
            );

            if (!hasMolecularNorm(
                    recoveryVector,
                    volumeElement,
                    VECTOR_RECOVERY_TOLERANCE_SQUARED)) {

                throw std::runtime_error(
                    "El vector de recuperacion es "
                    "numericamente dependiente."
                );
            }

            normalizeMolecularVector(
                recoveryVector,
                volumeElement,
                "vector de recuperacion"
            );

            basis.push_back(
                std::move(recoveryVector)
            );

            continue;
        }


        /*
         * --------------------------------------------------------
         * NORMALIZE RITZ
         * --------------------------------------------------------
         */

        normalizeMolecularVector(
            ritzVector,
            volumeElement,
            "Ritz orbital " +
            std::to_string(
                orbitalIndex
            )
        );


        /*
         * --------------------------------------------------------
         * H(RITZ)
         * --------------------------------------------------------
         *
         * Como el vector Ritz pertenece al subespacio actual:
         *
         *     H Ritz = sum_j c_j H basis_j
         *
         * No se vuelve a aplicar H al Ritz.
         *
         * --------------------------------------------------------
         */

        std::vector<double> hRitz =
            buildSubspaceHamiltonianVector(
                hBasis,
                projectedEigenpair.eigenvector
            );


        /*
         * --------------------------------------------------------
         * RAYLEIGH QUOTIENT
         * --------------------------------------------------------
         */

        finalEigenvalue =
            rayleighQuotient(
                ritzVector,
                hRitz,
                volumeElement
            );


        /*
         * --------------------------------------------------------
         * RESIDUAL
         * --------------------------------------------------------
         */

        finalResidual =
            molecularResidual(
                ritzVector,
                hRitz,
                finalEigenvalue,
                volumeElement
            );

        finalVector =
            ritzVector;


        /*
         * --------------------------------------------------------
         * CONVERGENCE
         * --------------------------------------------------------
         */

        if (finalResidual <
            MO_RESIDUAL_TOLERANCE) {

            std::cout
                << "    CONVERGIO\n"
                << "      Orbital:          "
                << orbitalIndex
                << "\n"
                << "      Spin:             "
                << spinName
                << "\n"
                << "      H aplicaciones:   "
                << hApplications
                << "\n"
                << "      Subespacio:       "
                << basis.size()
                << "\n"
                << "      Autovalor:        "
                << finalEigenvalue
                << "\n"
                << "      Residuo:          "
                << finalResidual
                << "\n";

            break;
        }


        /*
         * --------------------------------------------------------
         * RESIDUAL
         * --------------------------------------------------------
         */

        std::vector<double> residual(
            dimension,
            0.0
        );

        for (std::size_t i = 0;
             i < dimension;
             ++i) {

            residual[i] =
                hRitz[i] -
                finalEigenvalue *
                ritzVector[i];
        }


        /*
         * --------------------------------------------------------
         * ORTHOGONALIZE RESIDUAL
         * --------------------------------------------------------
         */

        orthogonalizeAgainstPreviousOrbitals(
            residual,
            previousOrbitals,
            volumeElement
        );

        orthogonalizeAgainstBasis(
            residual,
            basis,
            volumeElement
        );


        const double residualNormSquared =
            molecularDotProduct(
                residual,
                residual,
                volumeElement
            );

        if (!std::isfinite(residualNormSquared) ||
            !(residualNormSquared >
              SUBSPACE_ORTHOGONALITY_TOLERANCE_SQUARED)) {

            throw std::runtime_error(
                "El residuo molecular se volvio "
                "numericamente dependiente del subespacio."
            );
        }


        /*
         * --------------------------------------------------------
         * DAVIDSON CORRECTION
         * --------------------------------------------------------
         *
         * En lugar de introducir directamente:
         *
         *     r
         *
         * introducimos:
         *
         *     D^{-1} r
         *
         * donde:
         *
         *     D_ii = epsilon - H_ii
         *
         * Esto concentra la nueva direccion en las componentes
         * que pueden corregir mas eficazmente el autovector.
         * --------------------------------------------------------
         */

        std::vector<double> correction;

        if (!buildDavidsonCorrection(
                residual,
                hamiltonianDiagonal,
                finalEigenvalue,
                volumeElement,
                correction)) {

            correction =
                std::move(residual);
        }


        /*
         * --------------------------------------------------------
         * ORTHOGONALIZE CORRECTION
         * --------------------------------------------------------
         */

        orthogonalizeAgainstPreviousOrbitals(
            correction,
            previousOrbitals,
            volumeElement
        );

        orthogonalizeAgainstBasis(
            correction,
            basis,
            volumeElement
        );


        const double correctionNormSquared =
            molecularDotProduct(
                correction,
                correction,
                volumeElement
            );

        if (!std::isfinite(correctionNormSquared) ||
            !(correctionNormSquared >
              SUBSPACE_ORTHOGONALITY_TOLERANCE_SQUARED)) {

            /*
             * El precondicionamiento puede perder efectividad
             * cuando la diagonal presenta un polo cercano.
             *
             * En ese caso se recupera el residuo original.
             */

            correction =
                std::move(residual);

            orthogonalizeAgainstPreviousOrbitals(
                correction,
                previousOrbitals,
                volumeElement
            );

            orthogonalizeAgainstBasis(
                correction,
                basis,
                volumeElement
            );
        }


        if (!hasMolecularNorm(
                correction,
                volumeElement,
                SUBSPACE_ORTHOGONALITY_TOLERANCE_SQUARED)) {

            throw std::runtime_error(
                "La correccion Davidson es numericamente "
                "dependiente del subespacio."
            );
        }

        normalizeMolecularVector(
            correction,
            volumeElement,
            "correccion Davidson"
        );


        /*
         * --------------------------------------------------------
         * EXTEND SUBSPACE
         * --------------------------------------------------------
         */

        if (basis.size() <
            MAX_SUBSPACE_DIMENSION) {

            basis.push_back(
                std::move(correction)
            );

            continue;
        }


        /*
         * ========================================================
         * DAVIDSON RESTART
         * ========================================================
         *
         * Se conserva el mejor Ritz y la direccion de correccion.
         *
         * La correccion NO necesita una segunda aplicacion de H
         * antes de ser almacenada: H(correction) se calcula una
         * sola vez aqui.
         * ========================================================
         */

        if (hApplications >= maxIterations) {
            break;
        }

        std::vector<double> hCorrection =
            applyMolecularKohnShamHamiltonian(
                grid,
                effectivePotential,
                correction
            );

        ++hApplications;

        if (hCorrection.size() != dimension) {

            throw std::runtime_error(
                "El Hamiltoniano molecular devolvio "
                "una dimension incorrecta para la "
                "correccion Davidson."
            );
        }


        /*
         * --------------------------------------------------------
         * RESTART
         * --------------------------------------------------------
         */

        basis.clear();
        hBasis.clear();
        projectedHamiltonian.clear();

        basis.reserve(
            MAX_SUBSPACE_DIMENSION
        );

        hBasis.reserve(
            MAX_SUBSPACE_DIMENSION
        );

        basis.push_back(
            std::move(ritzVector)
        );

        hBasis.push_back(
            std::move(hRitz)
        );

        basis.push_back(
            std::move(correction)
        );

        hBasis.push_back(
            std::move(hCorrection)
        );


        projectedHamiltonian.assign(
            2,
            std::vector<double>(
                2,
                0.0
            )
        );

        projectedHamiltonian[0][0] =
            molecularDotProduct(
                basis[0],
                hBasis[0],
                volumeElement
            );

        projectedHamiltonian[0][1] =
            molecularDotProduct(
                basis[0],
                hBasis[1],
                volumeElement
            );

        projectedHamiltonian[1][0] =
            projectedHamiltonian[0][1];

        projectedHamiltonian[1][1] =
            molecularDotProduct(
                basis[1],
                hBasis[1],
                volumeElement
            );
    }


    /*
     * ============================================================
     * FINAL VALIDATION
     * ============================================================
     */

    if (finalVector.empty() ||
        !(finalResidual <
          MO_RESIDUAL_TOLERANCE)) {

        throw std::runtime_error(
            "El orbital molecular no alcanzo la convergencia.\n"
            "Canal: " +
            std::string(
                spin == SpinChannel::Alpha
                    ? "Alpha"
                    : "Beta"
            ) +
            "\n"
            "Indice orbital: " +
            std::to_string(
                orbitalIndex
            ) +
            "\n"
            "Aplicaciones de H utilizadas: " +
            std::to_string(
                hApplications
            ) +
            "\n"
            "Dimension actual del subespacio: " +
            std::to_string(
                basis.size()
            ) +
            "\n"
            "Autovalor: " +
            std::to_string(
                finalEigenvalue
            ) +
            "\n"
            "Residuo final: " +
            std::to_string(
                finalResidual
            )
        );
    }


    /*
     * ============================================================
     * RESULT
     * ============================================================
     */

    MolecularOrbital orbital;

    orbital.spin =
        spin;

    orbital.electrons =
        electrons;

    orbital.eigenvalue =
        finalEigenvalue;

    orbital.psi =
        std::move(finalVector);

    return orbital;
}


/*
 * ================================================================
 * MULTIPLE MOLECULAR ORBITALS
 * ================================================================
 */

std::vector<MolecularOrbital> solveMolecularOrbitals(
    const CartesianGrid& grid,
    const std::vector<double>& effectivePotential,
    std::size_t numberOfOrbitals,
    const std::vector<int>& occupations,
    SpinChannel spin,
    const std::vector<MolecularOrbital>& initialOrbitals,
    std::size_t maxIterations
) {
    if (numberOfOrbitals == 0) {

        throw std::invalid_argument(
            "El numero de orbitales moleculares debe "
            "ser mayor que cero."
        );
    }

    if (occupations.size() !=
        numberOfOrbitals) {

        throw std::invalid_argument(
            "El numero de ocupaciones debe coincidir "
            "con el numero de orbitales."
        );
    }

    if (!initialOrbitals.empty() &&
        initialOrbitals.size() != numberOfOrbitals) {

        throw std::invalid_argument(
            "El numero de orbitales iniciales debe "
            "coincidir con el numero de orbitales solicitados."
        );
    }

    std::vector<MolecularOrbital> orbitals;

    orbitals.reserve(
        numberOfOrbitals
    );

    for (std::size_t i = 0;
         i < numberOfOrbitals;
         ++i) {

        if (occupations[i] < 0 ||
            occupations[i] > 2) {

            throw std::invalid_argument(
                "Las ocupaciones moleculares deben estar "
                "entre cero y dos."
            );
        }

        const MolecularOrbital* initialGuess =
            initialOrbitals.empty()
                ? nullptr
                : &initialOrbitals[i];

        MolecularOrbital orbital =
            solveMolecularOrbital(
                grid,
                effectivePotential,
                i,
                spin,
                occupations[i],
                orbitals,
                initialGuess,
                maxIterations
            );

        orbitals.push_back(
            std::move(orbital)
        );
    }

    return orbitals;
}