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

constexpr std::size_t MAX_SUBSPACE_DIMENSION = 24;

constexpr double DENSE_EIGENVALUE_TOLERANCE = 1.0e-13;

constexpr std::size_t DENSE_EIGENVALUE_MAX_ITERATIONS = 10000;


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

    const double sigmaSquared =
        sigma *
        sigma;

    const std::size_t mode =
        orbitalIndex % 9;

    for (std::size_t k = 0;
         k < grid.getNz();
         ++k) {

        const double z =
            grid.getZ(k) -
            centerZ;

        for (std::size_t j = 0;
             j < grid.getNy();
             ++j) {

            const double y =
                grid.getY(j) -
                centerY;

            for (std::size_t i = 0;
                 i < grid.getNx();
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
                        -radiusSquared /
                        (2.0 * sigmaSquared)
                    );

                double polynomial = 1.0;

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
                    polynomial =
                        x * y;
                    break;

                case 5:
                    polynomial =
                        x * z;
                    break;

                case 6:
                    polynomial =
                        y * z;
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

                const std::size_t index =
                    grid.getIndex(
                        i,
                        j,
                        k
                    );

                vector[index] =
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

        double maximumOffDiagonal =
            0.0;

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

        double diagonalScale =
            1.0;

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

        matrix[p][q] =
            0.0;

        matrix[q][p] =
            0.0;

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

            matrix[k][p] =
                newKp;

            matrix[p][k] =
                newKp;

            matrix[k][q] =
                newKq;

            matrix[q][k] =
                newKq;
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

    std::size_t minimumIndex =
        0;

    for (std::size_t i = 1;
         i < n;
         ++i) {

        if (matrix[i][i] <
            matrix[minimumIndex][minimumIndex]) {

            minimumIndex =
                i;
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

    if (basis.size() !=
        coefficients.size()) {

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

        if (basis[j].size() !=
            vector.size()) {

            throw std::invalid_argument(
                "Los vectores del subespacio deben "
                "tener el mismo tamano."
            );
        }

        for (std::size_t i = 0;
             i < vector.size();
             ++i) {

            vector[i] +=
                coefficients[j] *
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

    if (hBasis.size() !=
        coefficients.size()) {

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

        if (hBasis[j].size() !=
            hVector.size()) {

            throw std::invalid_argument(
                "Los vectores H(base) deben "
                "tener el mismo tamano."
            );
        }

        for (std::size_t i = 0;
             i < hVector.size();
             ++i) {

            hVector[i] +=
                coefficients[j] *
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

        if (basis[i].size() !=
            hBasis[newIndex].size()) {

            throw std::invalid_argument(
                "Los vectores de la base y H(base) "
                "deben tener el mismo tamano."
            );
        }

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
     * INITIAL VECTOR
     * ------------------------------------------------------------
     */

    std::vector<double> initialVector;

    if (!buildIndependentInitialVector(
            grid,
            orbitalIndex,
            previousOrbitals,
            volumeElement,
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

    std::size_t hApplications =
        0;

    double finalEigenvalue =
        std::numeric_limits<double>::infinity();

    double finalResidual =
        std::numeric_limits<double>::infinity();

    std::vector<double> finalVector;


    /*
     * ------------------------------------------------------------
     * ITERATIVE SUBSPACE SOLVER
     * ------------------------------------------------------------
     */

    while (hApplications < maxIterations) {

        /*
         * Apply H only to a newly added basis vector.
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
         * PROJECTED HAMILTONIAN
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

        const std::size_t currentSubspaceDimension =
            basis.size();

        (void)currentSubspaceDimension;

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
         * RESIDUAL VECTOR
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
         * El residuo se hace ortogonal a los orbitales ya
         * determinados y a la base actual.
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


        /*
         * --------------------------------------------------------
         * RESIDUAL STAGNATION
         * --------------------------------------------------------
         */

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
         * NORMALIZE RESIDUAL
         * --------------------------------------------------------
         */

        normalizeMolecularVector(
            residual,
            volumeElement,
            "residuo molecular"
        );


        /*
         * --------------------------------------------------------
         * EXTEND SUBSPACE
         * --------------------------------------------------------
         */

        if (basis.size() <
            MAX_SUBSPACE_DIMENSION) {

            basis.push_back(
                std::move(residual)
            );

            continue;
        }


        /*
         * --------------------------------------------------------
         * RESIDUAL-BASED RESTART
         * --------------------------------------------------------
         *
         * Se conserva:
         *
         *   1. el Ritz actual
         *   2. el residuo normalizado
         *
         * El residuo YA esta ortogonalizado antes de aplicar H.
         * Por tanto no debe modificarse despues de calcular H(r).
         *
         * Esto mantiene la identidad matematica:
         *
         *       hResidual = H * residual
         *
         * --------------------------------------------------------
         */

        if (hApplications >= maxIterations) {
            break;
        }

        std::vector<double> hResidual =
            applyMolecularKohnShamHamiltonian(
                grid,
                effectivePotential,
                residual
            );

        ++hApplications;

        if (hResidual.size() != dimension) {

            throw std::runtime_error(
                "El Hamiltoniano molecular devolvio "
                "una dimension incorrecta para el residuo."
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

        basis.push_back(
            std::move(ritzVector)
        );

        hBasis.push_back(
            std::move(hRitz)
        );

        basis.push_back(
            std::move(residual)
        );

        hBasis.push_back(
            std::move(hResidual)
        );


        /*
         * Construccion exacta del Hamiltoniano proyectado 2x2.
         *
         * La base es:
         *
         *       { Ritz, Residuo }
         *
         * y ambas funciones estan normalizadas y ortogonales.
         */

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
     * ------------------------------------------------------------
     * FINAL VALIDATION
     * ------------------------------------------------------------
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
     * ------------------------------------------------------------
     * RESULT
     * ------------------------------------------------------------
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

        MolecularOrbital orbital =
            solveMolecularOrbital(
                grid,
                effectivePotential,
                i,
                spin,
                occupations[i],
                orbitals,
                maxIterations
            );

        orbitals.push_back(
            std::move(orbital)
        );
    }

    return orbitals;
}