#include "EigenvalueSolver.h"

#include "DFTConstants.h"
#include "KohnShamHamiltonian.h"
#include "NumericalMethods.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double MO_RESIDUAL_TOLERANCE = 1.0e-7;
constexpr double LANCZOS_ORTHOGONALITY_TOLERANCE = 1.0e-12;
constexpr double RADIAL_EIGENVALUE_TOLERANCE = 1.0e-13;

constexpr std::size_t MIN_LANCZOS_DIMENSION = 24;
constexpr std::size_t MAX_LANCZOS_DIMENSION = 80;

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

double molecularNorm(
    const std::vector<double>& vector,
    double volumeElement
) {
    const double squaredNorm =
        molecularDotProduct(
            vector,
            vector,
            volumeElement
        );

    if (!(squaredNorm > DFTConstants::EPS) ||
        !std::isfinite(squaredNorm)) {

        throw std::runtime_error(
            "La funcion de onda molecular tiene norma "
            "numericamente nula."
        );
    }

    return std::sqrt(squaredNorm);
}

void normalizeMolecularVector(
    std::vector<double>& vector,
    double volumeElement
) {
    const double norm =
        molecularNorm(
            vector,
            volumeElement
        );

    for (double& value : vector) {
        value /= norm;
    }
}

void orthogonalizeAgainstBasis(
    std::vector<double>& vector,
    const std::vector<std::vector<double>>& basis,
    double volumeElement
) {
    /*
        Modified Gram-Schmidt.

        Se realiza dos veces para reducir la perdida de
        ortogonalidad numerica cuando la base crece.
    */
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
    for (int pass = 0; pass < 2; ++pass) {
        for (const MolecularOrbital& orbital :
             previousOrbitals) {

            if (orbital.psi.size() != vector.size()) {
                throw std::invalid_argument(
                    "Los orbitales moleculares deben estar "
                    "definidos sobre la misma malla."
                );
            }

            const double projection =
                molecularDotProduct(
                    orbital.psi,
                    vector,
                    volumeElement
                );

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

void projectHamiltonianVector(
    std::vector<double>& hVector,
    const std::vector<MolecularOrbital>& previousOrbitals,
    double volumeElement
) {
    /*
        H proyectado sobre el complemento ortogonal de los
        orbitales previamente obtenidos:

            P H psi

        con

            P = I - sum |phi_i><phi_i|

        Esto es necesario para que el problema de Lanczos
        corresponda al espacio permitido para el nuevo orbital.
    */
    for (const MolecularOrbital& orbital :
         previousOrbitals) {

        if (orbital.psi.size() != hVector.size()) {
            throw std::invalid_argument(
                "Los orbitales previos y Hpsi deben tener "
                "la misma dimension."
            );
        }

        const double projection =
            molecularDotProduct(
                orbital.psi,
                hVector,
                volumeElement
            );

        for (std::size_t i = 0;
             i < hVector.size();
             ++i) {

            hVector[i] -=
                projection *
                orbital.psi[i];
        }
    }
}

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

    if (!(denominator > DFTConstants::EPS)) {
        throw std::runtime_error(
            "No se puede calcular el cociente de Rayleigh."
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
            eigenvalue * psi[i];

        residualSquared +=
            residual * residual;
    }

    return std::sqrt(
        residualSquared *
        volumeElement
    );
}

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

    /*
        Variamos el ancho para generar estados iniciales
        linealmente independientes.
    */
    const double sigma =
        0.8 +
        0.18 *
        static_cast<double>(
            orbitalIndex
        );

    const double sigmaSquared =
        sigma * sigma;

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

TridiagonalMatrix buildLanczosMatrix(
    const std::vector<double>& alpha,
    const std::vector<double>& beta
) {
    if (alpha.empty()) {
        throw std::invalid_argument(
            "La base de Lanczos no puede estar vacia."
        );
    }

    if (beta.size() + 1 != alpha.size()) {
        throw std::invalid_argument(
            "Los coeficientes de Lanczos tienen tamanos "
            "incompatibles."
        );
    }

    TridiagonalMatrix matrix;

    matrix.diagonal =
        alpha;

    matrix.lower =
        beta;

    matrix.upper =
        beta;

    return matrix;
}

std::vector<double> buildRitzVector(
    const std::vector<std::vector<double>>& basis,
    const std::vector<double>& coefficients,
    double volumeElement
) {
    if (basis.empty()) {
        throw std::invalid_argument(
            "La base de Lanczos no puede estar vacia."
        );
    }

    if (basis.size() != coefficients.size()) {
        throw std::invalid_argument(
            "La base y los coeficientes de Ritz deben "
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
                "Los vectores de Lanczos deben tener "
                "el mismo tamano."
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

    normalizeMolecularVector(
        vector,
        volumeElement
    );

    return vector;
}

} // namespace

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
        matrix.diagonal[0] -
        energy;

    if (previous < 0.0) {
        ++count;
    }

    if (std::abs(previous) <
        DFTConstants::EPS) {

        previous =
            std::copysign(
                DFTConstants::EPS,
                previous == 0.0
                    ? 1.0
                    : previous
            );
    }

    for (std::size_t i = 1;
         i < n;
         ++i) {

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
            std::abs(pivot) <
                DFTConstants::EPS
                ? std::copysign(
                    DFTConstants::EPS,
                    pivot == 0.0
                        ? 1.0
                        : pivot
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
            "Intervalo invalido para la busqueda."
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
        }
        else {
            high = mid;
        }

        const double tolerance =
            RADIAL_EIGENVALUE_TOLERANCE *
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

    for (std::size_t i = 0;
         i < n;
         ++i) {

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
            "La matriz y la malla radial deben "
            "tener el mismo tamano."
        );
    }

    if (r.size() < 2) {
        throw std::invalid_argument(
            "La malla radial debe contener al menos "
            "dos puntos."
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

    const double eigenvalue =
        findEigenvalue(
            matrix,
            orbitalIndex,
            minimum - 10.0,
            maximum + 10.0
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

    /*
        Generamos varios vectores iniciales y elegimos uno
        ortogonal a los orbitales anteriores.
    */
    std::vector<double> initialVector;

    bool initialFound = false;

    for (std::size_t seed = orbitalIndex;
         seed < orbitalIndex + 16;
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

        const double normSquared =
            molecularDotProduct(
                candidate,
                candidate,
                volumeElement
            );

        if (normSquared >
            LANCZOS_ORTHOGONALITY_TOLERANCE) {

            normalizeMolecularVector(
                candidate,
                volumeElement
            );

            initialVector =
                std::move(candidate);

            initialFound =
                true;

            break;
        }
    }

    if (!initialFound) {
        throw std::runtime_error(
            "No se pudo construir un vector inicial "
            "ortogonal a los orbitales anteriores."
        );
    }

    /*
        Dimension fija y moderada.

        No usamos maxIterations como dimension del
        subespacio. maxIterations es un limite de trabajo,
        no una razon para construir miles de vectores.
    */
    const std::size_t lanczosDimension =
        std::min(
            dimension,
            std::max(
                MIN_LANCZOS_DIMENSION,
                std::min(
                    MAX_LANCZOS_DIMENSION,
                    maxIterations
                )
            )
        );

    std::vector<std::vector<double>> basis;

    basis.reserve(
        lanczosDimension
    );

    std::vector<double> alpha;

    alpha.reserve(
        lanczosDimension
    );

    std::vector<double> beta;

    beta.reserve(
        lanczosDimension > 0
            ? lanczosDimension - 1
            : 0
    );

    std::vector<double> previousVector(
        dimension,
        0.0
    );

    std::vector<double> currentVector =
        std::move(initialVector);

    double finalEigenvalue =
        std::numeric_limits<double>::infinity();

    double finalResidual =
        std::numeric_limits<double>::infinity();

    std::vector<double> finalVector;

    bool converged =
        false;

    for (std::size_t iteration = 0;
         iteration < lanczosDimension;
         ++iteration) {

        /*
            q_m pertenece al complemento ortogonal de los
            orbitales previos y de los vectores anteriores
            de Lanczos.
        */
        orthogonalizeAgainstPreviousOrbitals(
            currentVector,
            previousOrbitals,
            volumeElement
        );

        orthogonalizeAgainstBasis(
            currentVector,
            basis,
            volumeElement
        );

        const double currentNormSquared =
            molecularDotProduct(
                currentVector,
                currentVector,
                volumeElement
            );

        if (currentNormSquared <=
            LANCZOS_ORTHOGONALITY_TOLERANCE) {

            break;
        }

        normalizeMolecularVector(
            currentVector,
            volumeElement
        );

        basis.push_back(
            currentVector
        );

        /*
            Aplicamos H una sola vez.
        */
        std::vector<double> hCurrent =
            applyMolecularKohnShamHamiltonian(
                grid,
                effectivePotential,
                currentVector
            );

        /*
            H debe proyectarse sobre el complemento
            ortogonal a los orbitales previamente obtenidos.
        */
        projectHamiltonianVector(
            hCurrent,
            previousOrbitals,
            volumeElement
        );

        const double diagonalElement =
            molecularDotProduct(
                currentVector,
                hCurrent,
                volumeElement
            );

        alpha.push_back(
            diagonalElement
        );

        /*
            Problema reducido de Lanczos.
        */
        const TridiagonalMatrix lanczosMatrix =
            buildLanczosMatrix(
                alpha,
                beta
            );

        const double minimum =
            *std::min_element(
                lanczosMatrix.diagonal.begin(),
                lanczosMatrix.diagonal.end()
            );

        const double maximum =
            *std::max_element(
                lanczosMatrix.diagonal.begin(),
                lanczosMatrix.diagonal.end()
            );

        const double ritzEigenvalue =
            findEigenvalue(
                lanczosMatrix,
                0,
                minimum - 10.0,
                maximum + 10.0
            );

        const std::vector<double> ritzCoefficients =
            solveEigenvector(
                lanczosMatrix,
                ritzEigenvalue,
                200
            );

        std::vector<double> ritzVector =
            buildRitzVector(
                basis,
                ritzCoefficients,
                volumeElement
            );

        /*
            La reconstruccion de Ritz puede perder una cantidad
            muy pequena de ortogonalidad por redondeo.
        */
        orthogonalizeAgainstPreviousOrbitals(
            ritzVector,
            previousOrbitals,
            volumeElement
        );

        normalizeMolecularVector(
            ritzVector,
            volumeElement
        );

        std::vector<double> hRitz =
            applyMolecularKohnShamHamiltonian(
                grid,
                effectivePotential,
                ritzVector
            );

        /*
            Residuo fisico del problema original:

                H psi = epsilon psi
        */
        finalEigenvalue =
            rayleighQuotient(
                ritzVector,
                hRitz,
                volumeElement
            );

        finalResidual =
            molecularResidual(
                ritzVector,
                hRitz,
                finalEigenvalue,
                volumeElement
            );

        finalVector =
            ritzVector;

        if (finalResidual <
            MO_RESIDUAL_TOLERANCE) {

            converged =
                true;

            break;
        }

        /*
            Ya no podemos generar otro vector.
        */
        if (iteration + 1 >=
            lanczosDimension) {

            break;
        }

        /*
            Recurrencia de Lanczos:

                w =
                    H q_m
                    - alpha_m q_m
                    - beta_(m-1) q_(m-1)
        */
        std::vector<double> nextVector =
            hCurrent;

        for (std::size_t i = 0;
             i < dimension;
             ++i) {

            nextVector[i] -=
                diagonalElement *
                currentVector[i];

            if (iteration > 0) {
                nextVector[i] -=
                    beta.back() *
                    previousVector[i];
            }
        }

        /*
            En el problema proyectado también debemos
            mantener el siguiente vector fuera de los
            orbitales anteriores.
        */
        orthogonalizeAgainstPreviousOrbitals(
            nextVector,
            previousOrbitals,
            volumeElement
        );

        /*
            Reortogonalizacion completa contra la base
            de Lanczos.
        */
        orthogonalizeAgainstBasis(
            nextVector,
            basis,
            volumeElement
        );

        const double nextNormSquared =
            molecularDotProduct(
                nextVector,
                nextVector,
                volumeElement
            );

        if (nextNormSquared <=
            LANCZOS_ORTHOGONALITY_TOLERANCE) {

            break;
        }

        const double nextNorm =
            std::sqrt(
                nextNormSquared
            );

        beta.push_back(
            nextNorm
        );

        previousVector =
            currentVector;

        for (double& value :
             nextVector) {

            value /=
                nextNorm;
        }

        currentVector =
            std::move(nextVector);
    }

    if (!converged) {
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
            "Dimension de Lanczos: " +
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