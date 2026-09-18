#include "BlockDavidsonMolecule.h"

#include "KohnShamHamiltonianMolecule.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

constexpr double RESIDUAL_TOLERANCE = 1.0e-7;
constexpr double LINEAR_DEPENDENCE_TOLERANCE = 1.0e-13;
constexpr double JACOBI_TOLERANCE = 1.0e-12;

constexpr std::size_t EXTRA_SUBSPACE_VECTORS = 8;

struct RitzState
{
    double eigenvalue = 0.0;

    std::vector<double> psi;
    std::vector<double> hPsi;
    std::vector<double> residual;

    double residualNorm = 0.0;
};

double getVolumeElement(
    const CartesianGrid& grid
)
{
    const double dx = grid.getDx();
    const double dy = grid.getDy();
    const double dz = grid.getDz();

    if (dx <= 0.0 ||
        dy <= 0.0 ||
        dz <= 0.0)
    {
        throw std::invalid_argument(
            "Los pasos de la malla cartesiana deben ser mayores que cero."
        );
    }

    return dx * dy * dz;
}

void validateVectorSize(
    const std::vector<double>& vector,
    std::size_t expectedSize,
    const char* message
)
{
    if (vector.size() != expectedSize)
    {
        throw std::invalid_argument(message);
    }
}

void validateBasisConsistency(
    const std::vector<std::vector<double>>& basis,
    const std::vector<std::vector<double>>& hBasis,
    std::size_t gridSize
)
{
    if (basis.size() != hBasis.size())
    {
        throw std::runtime_error(
            "La base y la base transformada por el Hamiltoniano "
            "tienen diferente numero de vectores."
        );
    }

    for (std::size_t i = 0;
         i < basis.size();
         ++i)
    {
        if (basis[i].size() != gridSize)
        {
            throw std::runtime_error(
                "Un vector de la base no tiene el tamano de la malla."
            );
        }

        if (hBasis[i].size() != gridSize)
        {
            throw std::runtime_error(
                "Un vector de H aplicado a la base no tiene el tamano de la malla."
            );
        }
    }
}

double dot(
    const std::vector<double>& a,
    const std::vector<double>& b,
    std::size_t gridSize,
    double volumeElement
)
{
    validateVectorSize(
        a,
        gridSize,
        "El primer vector no tiene el mismo tamano que la malla."
    );

    validateVectorSize(
        b,
        gridSize,
        "El segundo vector no tiene el mismo tamano que la malla."
    );

    double result = 0.0;

    for (std::size_t i = 0;
         i < gridSize;
         ++i)
    {
        result +=
            a[i] *
            b[i];
    }

    return result *
           volumeElement;
}

double norm(
    const std::vector<double>& vector,
    std::size_t gridSize,
    double volumeElement
)
{
    const double value =
        dot(
            vector,
            vector,
            gridSize,
            volumeElement
        );

    return std::sqrt(
        std::max(
            0.0,
            value
        )
    );
}

void scale(
    std::vector<double>& vector,
    double factor,
    std::size_t gridSize
)
{
    validateVectorSize(
        vector,
        gridSize,
        "El vector no tiene el mismo tamano que la malla."
    );

    for (std::size_t i = 0;
         i < gridSize;
         ++i)
    {
        vector[i] *= factor;
    }
}

void axpy(
    std::vector<double>& destination,
    const std::vector<double>& source,
    double factor,
    std::size_t gridSize
)
{
    validateVectorSize(
        destination,
        gridSize,
        "El vector destino no tiene el mismo tamano que la malla."
    );

    validateVectorSize(
        source,
        gridSize,
        "El vector fuente no tiene el mismo tamano que la malla."
    );

    for (std::size_t i = 0;
         i < gridSize;
         ++i)
    {
        destination[i] +=
            factor *
            source[i];
    }
}

bool normalize(
    std::vector<double>& vector,
    std::size_t gridSize,
    double volumeElement
)
{
    const double vectorNorm =
        norm(
            vector,
            gridSize,
            volumeElement
        );

    if (!std::isfinite(vectorNorm) ||
        vectorNorm <= LINEAR_DEPENDENCE_TOLERANCE)
    {
        return false;
    }

    scale(
        vector,
        1.0 / vectorNorm,
        gridSize
    );

    return true;
}

void orthogonalizeOnce(
    std::vector<double>& vector,
    const std::vector<std::vector<double>>& basis,
    std::size_t gridSize,
    double volumeElement
)
{
    for (const std::vector<double>& basisVector :
         basis)
    {
        const double projection =
            dot(
                basisVector,
                vector,
                gridSize,
                volumeElement
            );

        axpy(
            vector,
            basisVector,
            -projection,
            gridSize
        );
    }
}

bool orthogonalizeAndNormalize(
    std::vector<double>& vector,
    const std::vector<std::vector<double>>& basis,
    std::size_t gridSize,
    double volumeElement
)
{
    orthogonalizeOnce(
        vector,
        basis,
        gridSize,
        volumeElement
    );

    /*
     * Second Gram-Schmidt pass for numerical stability.
     */
    orthogonalizeOnce(
        vector,
        basis,
        gridSize,
        volumeElement
    );

    return normalize(
        vector,
        gridSize,
        volumeElement
    );
}

std::vector<double> linearCombination(
    const std::vector<std::vector<double>>& basis,
    const std::vector<double>& coefficients,
    std::size_t gridSize
)
{
    if (basis.size() !=
        coefficients.size())
    {
        throw std::invalid_argument(
            "La base y los coeficientes tienen diferentes dimensiones."
        );
    }

    std::vector<double> result(
        gridSize,
        0.0
    );

    for (std::size_t j = 0;
         j < basis.size();
         ++j)
    {
        validateVectorSize(
            basis[j],
            gridSize,
            "Un vector de la base no tiene el mismo tamano que la malla."
        );

        for (std::size_t i = 0;
             i < gridSize;
             ++i)
        {
            result[i] +=
                coefficients[j] *
                basis[j][i];
        }
    }

    return result;
}

void jacobiDiagonalize(
    const std::vector<std::vector<double>>& input,
    std::vector<double>& eigenvalues,
    std::vector<std::vector<double>>& eigenvectors
)
{
    const std::size_t dimension =
        input.size();

    if (dimension == 0)
    {
        eigenvalues.clear();
        eigenvectors.clear();
        return;
    }

    for (const auto& row : input)
    {
        if (row.size() != dimension)
        {
            throw std::invalid_argument(
                "La matriz proyectada debe ser cuadrada."
            );
        }
    }

    std::vector<std::vector<double>> matrix =
        input;

    eigenvectors.assign(
        dimension,
        std::vector<double>(
            dimension,
            0.0
        )
    );

    for (std::size_t i = 0;
         i < dimension;
         ++i)
    {
        eigenvectors[i][i] = 1.0;
    }

    const std::size_t maximumSweeps =
        std::max(
            std::size_t(50),
            10 *
            dimension *
            dimension
        );

    for (std::size_t sweep = 0;
         sweep < maximumSweeps;
         ++sweep)
    {
        double maximumOffDiagonal =
            0.0;

        std::size_t p = 0;
        std::size_t q = 0;

        for (std::size_t i = 0;
             i < dimension;
             ++i)
        {
            for (std::size_t j = i + 1;
                 j < dimension;
                 ++j)
            {
                const double value =
                    std::abs(
                        matrix[i][j]
                    );

                if (value >
                    maximumOffDiagonal)
                {
                    maximumOffDiagonal =
                        value;

                    p = i;
                    q = j;
                }
            }
        }

        if (maximumOffDiagonal <=
            JACOBI_TOLERANCE)
        {
            break;
        }

        const double app =
            matrix[p][p];

        const double aqq =
            matrix[q][q];

        const double apq =
            matrix[p][q];

        if (std::abs(apq) <=
            JACOBI_TOLERANCE)
        {
            continue;
        }

        const double tau =
            (aqq - app) /
            (2.0 * apq);

        const double sign =
            tau >= 0.0
                ? 1.0
                : -1.0;

        const double t =
            sign /
            (
                std::abs(tau) +
                std::sqrt(
                    1.0 +
                    tau * tau
                )
            );

        const double c =
            1.0 /
            std::sqrt(
                1.0 +
                t * t
            );

        const double s =
            t * c;

        for (std::size_t k = 0;
             k < dimension;
             ++k)
        {
            if (k == p ||
                k == q)
            {
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

        matrix[p][p] =
            c * c * app -
            2.0 * s * c * apq +
            s * s * aqq;

        matrix[q][q] =
            s * s * app +
            2.0 * s * c * apq +
            c * c * aqq;

        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;

        for (std::size_t k = 0;
             k < dimension;
             ++k)
        {
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

    eigenvalues.resize(
        dimension
    );

    for (std::size_t i = 0;
         i < dimension;
         ++i)
    {
        eigenvalues[i] =
            matrix[i][i];
    }

    std::vector<std::size_t> order(
        dimension
    );

    for (std::size_t i = 0;
         i < dimension;
         ++i)
    {
        order[i] = i;
    }

    std::sort(
        order.begin(),
        order.end(),
        [&](std::size_t a,
            std::size_t b)
        {
            return eigenvalues[a] <
                   eigenvalues[b];
        }
    );

    std::vector<double> sortedEigenvalues(
        dimension
    );

    std::vector<std::vector<double>> sortedEigenvectors(
        dimension,
        std::vector<double>(
            dimension,
            0.0
        )
    );

    for (std::size_t column = 0;
         column < dimension;
         ++column)
    {
        const std::size_t original =
            order[column];

        sortedEigenvalues[column] =
            eigenvalues[original];

        for (std::size_t row = 0;
             row < dimension;
             ++row)
        {
            sortedEigenvectors[row][column] =
                eigenvectors[row][original];
        }
    }

    eigenvalues =
        std::move(
            sortedEigenvalues
        );

    eigenvectors =
        std::move(
            sortedEigenvectors
        );
}

std::vector<std::vector<double>> applyHamiltonianToBasis(
    const CartesianGrid& grid,
    const std::vector<double>& effectivePotential,
    const std::vector<std::vector<double>>& basis,
    std::size_t gridSize
)
{
    std::vector<std::vector<double>> hBasis;

    hBasis.reserve(
        basis.size()
    );

    for (const std::vector<double>& vector :
         basis)
    {
        validateVectorSize(
            vector,
            gridSize,
            "Un vector de la base no tiene el mismo tamano que la malla."
        );

        std::vector<double> hVector =
            applyMolecularKohnShamHamiltonian(
                grid,
                effectivePotential,
                vector
            );

        validateVectorSize(
            hVector,
            gridSize,
            "El Hamiltoniano produjo un vector con tamano incorrecto."
        );

        hBasis.push_back(
            std::move(hVector)
        );
    }

    validateBasisConsistency(
        basis,
        hBasis,
        gridSize
    );

    return hBasis;
}

std::vector<std::vector<double>> buildProjectedMatrix(
    const std::vector<std::vector<double>>& basis,
    const std::vector<std::vector<double>>& hBasis,
    std::size_t gridSize,
    double volumeElement
)
{
    validateBasisConsistency(
        basis,
        hBasis,
        gridSize
    );

    const std::size_t dimension =
        basis.size();

    std::vector<std::vector<double>> matrix(
        dimension,
        std::vector<double>(
            dimension,
            0.0
        )
    );

    for (std::size_t i = 0;
         i < dimension;
         ++i)
    {
        for (std::size_t j = i;
             j < dimension;
             ++j)
        {
            const double value =
                dot(
                    basis[i],
                    hBasis[j],
                    gridSize,
                    volumeElement
                );

            matrix[i][j] =
                value;

            matrix[j][i] =
                value;
        }
    }

    return matrix;
}

std::vector<RitzState> calculateRitzStates(
    const std::vector<std::vector<double>>& basis,
    const std::vector<std::vector<double>>& hBasis,
    std::size_t numberOfOrbitals,
    std::size_t gridSize,
    double volumeElement
)
{
    validateBasisConsistency(
        basis,
        hBasis,
        gridSize
    );

    if (basis.empty())
    {
        throw std::runtime_error(
            "El subespacio Block-Davidson esta vacio."
        );
    }

    const std::vector<std::vector<double>> projectedMatrix =
        buildProjectedMatrix(
            basis,
            hBasis,
            gridSize,
            volumeElement
        );

    std::vector<double> eigenvalues;

    std::vector<std::vector<double>> eigenvectors;

    jacobiDiagonalize(
        projectedMatrix,
        eigenvalues,
        eigenvectors
    );

    const std::size_t stateCount =
        std::min(
            numberOfOrbitals,
            basis.size()
        );

    std::vector<RitzState> states;

    states.reserve(
        stateCount
    );

    for (std::size_t state = 0;
         state < stateCount;
         ++state)
    {
        std::vector<double> coefficients(
            basis.size(),
            0.0
        );

        for (std::size_t j = 0;
             j < basis.size();
             ++j)
        {
            coefficients[j] =
                eigenvectors[j][state];
        }

        std::vector<double> psi =
            linearCombination(
                basis,
                coefficients,
                gridSize
            );

        if (!normalize(
                psi,
                gridSize,
                volumeElement))
        {
            throw std::runtime_error(
                "No fue posible normalizar un estado de Ritz."
            );
        }

        std::vector<double> hPsi =
            linearCombination(
                hBasis,
                coefficients,
                gridSize
            );

        const double eigenvalue =
            eigenvalues[state];

        std::vector<double> residual =
            hPsi;

        axpy(
            residual,
            psi,
            -eigenvalue,
            gridSize
        );

        const double residualNorm =
            norm(
                residual,
                gridSize,
                volumeElement
            );

        RitzState ritzState;

        ritzState.eigenvalue =
            eigenvalue;

        ritzState.psi =
            std::move(psi);

        ritzState.hPsi =
            std::move(hPsi);

        ritzState.residual =
            std::move(residual);

        ritzState.residualNorm =
            residualNorm;

        states.push_back(
            std::move(ritzState)
        );
    }

    return states;
}

std::vector<double> buildPreconditionedResidual(
    const RitzState& state,
    const std::vector<double>& effectivePotential,
    const CartesianGrid& grid
)
{
    const std::size_t gridSize =
        grid.getSize();

    validateVectorSize(
        state.residual,
        gridSize,
        "El residuo no tiene el mismo tamano que la malla."
    );

    validateVectorSize(
        effectivePotential,
        gridSize,
        "El potencial efectivo no tiene el mismo tamano que la malla."
    );

    const double dx =
        grid.getDx();

    const double dy =
        grid.getDy();

    const double dz =
        grid.getDz();

    const double diagonalKinetic =
        1.0 / (dx * dx) +
        1.0 / (dy * dy) +
        1.0 / (dz * dz);

    std::vector<double> correction(
        gridSize,
        0.0
    );

    for (std::size_t i = 0;
         i < gridSize;
         ++i)
    {
        const double diagonal =
            diagonalKinetic +
            effectivePotential[i];

        double denominator =
            state.eigenvalue -
            diagonal;

        const double denominatorScale =
            std::max(
                {
                    1.0,
                    std::abs(
                        state.eigenvalue
                    ),
                    std::abs(
                        diagonal
                    )
                }
            );

        const double denominatorFloor =
            1.0e-8 *
            denominatorScale;

        if (std::abs(denominator) <
            denominatorFloor)
        {
            denominator =
                denominator >= 0.0
                    ? denominatorFloor
                    : -denominatorFloor;
        }

        correction[i] =
            state.residual[i] /
            denominator;
    }

    return correction;
}

bool addIndependentVector(
    std::vector<std::vector<double>>& target,
    std::vector<double> vector,
    const std::vector<std::vector<double>>& against,
    std::size_t gridSize,
    double volumeElement
)
{
    validateVectorSize(
        vector,
        gridSize,
        "El vector candidato no tiene el mismo tamano que la malla."
    );

    orthogonalizeOnce(
        vector,
        against,
        gridSize,
        volumeElement
    );

    /*
     * Second pass against the same accepted set.
     */
    orthogonalizeOnce(
        vector,
        against,
        gridSize,
        volumeElement
    );

    if (!normalize(
            vector,
            gridSize,
            volumeElement))
    {
        return false;
    }

    target.push_back(
        std::move(vector)
    );

    return true;
}

bool allConverged(
    const std::vector<RitzState>& states,
    std::size_t numberOfOrbitals
)
{
    if (states.size() <
        numberOfOrbitals)
    {
        return false;
    }

    for (std::size_t i = 0;
         i < numberOfOrbitals;
         ++i)
    {
        if (!std::isfinite(
                states[i].residualNorm))
        {
            return false;
        }

        if (states[i].residualNorm >
            RESIDUAL_TOLERANCE)
        {
            return false;
        }
    }

    return true;
}

void rebuildSubspace(
    const CartesianGrid& grid,
    const std::vector<double>& effectivePotential,
    std::vector<std::vector<double>>& basis,
    std::vector<std::vector<double>>& hBasis,
    std::size_t gridSize
)
{
    /*
     * This function deliberately reconstructs hBasis from basis.
     *
     * basis and hBasis are therefore never maintained independently
     * across a restart.
     */
    hBasis =
        applyHamiltonianToBasis(
            grid,
            effectivePotential,
            basis,
            gridSize
        );

    validateBasisConsistency(
        basis,
        hBasis,
        gridSize
    );
}

} // namespace

std::vector<MolecularOrbital> solveMolecularOrbitalsBlockDavidson(
    const CartesianGrid& grid,
    const std::vector<double>& effectivePotential,
    std::size_t numberOfOrbitals,
    const std::vector<int>& occupations,
    SpinChannel spin,
    const std::vector<MolecularOrbital>& initialOrbitals,
    std::size_t maxIterations
)
{
    const std::size_t gridSize =
        grid.getSize();

    if (gridSize == 0)
    {
        throw std::invalid_argument(
            "La malla cartesiana no puede estar vacia."
        );
    }

    if (effectivePotential.size() !=
        gridSize)
    {
        throw std::invalid_argument(
            "El potencial efectivo y la malla deben tener el mismo tamano."
        );
    }

    if (numberOfOrbitals == 0)
    {
        return {};
    }

    if (occupations.size() <
        numberOfOrbitals)
    {
        throw std::invalid_argument(
            "La lista de ocupaciones no contiene suficientes orbitales."
        );
    }

    if (maxIterations == 0)
    {
        throw std::invalid_argument(
            "El numero maximo de iteraciones debe ser mayor que cero."
        );
    }

    const double volumeElement =
        getVolumeElement(
            grid
        );

    /*
     * ---------------------------------------------------------------
     * Initial subspace
     * ---------------------------------------------------------------
     */
    std::vector<std::vector<double>> basis;

    basis.reserve(
        numberOfOrbitals +
        EXTRA_SUBSPACE_VECTORS
    );

    /*
     * Previous SCF orbitals are preferred.
     */
    for (const MolecularOrbital& orbital :
         initialOrbitals)
    {
        if (basis.size() >=
            numberOfOrbitals)
        {
            break;
        }

        if (orbital.spin !=
            spin)
        {
            continue;
        }

        if (orbital.psi.size() !=
            gridSize)
        {
            continue;
        }

        std::vector<double> vector =
            orbital.psi;

        if (orthogonalizeAndNormalize(
                vector,
                basis,
                gridSize,
                volumeElement))
        {
            basis.push_back(
                std::move(vector)
            );
        }
    }

    /*
     * Deterministic initial functions for missing states.
     */
    std::size_t seedIndex = 0;

    while (basis.size() <
           numberOfOrbitals)
    {
        std::vector<double> seed(
            gridSize,
            0.0
        );

        const double pi =
            3.1415926535897932384626433832795;

        const double frequency =
            1.0 +
            static_cast<double>(
                seedIndex % 32
            );

        const double phase =
            0.271 *
            static_cast<double>(
                seedIndex
            );

        if (gridSize == 1)
        {
            seed[0] = 1.0;
        }
        else
        {
            const double denominator =
                static_cast<double>(
                    gridSize - 1
                );

            for (std::size_t i = 0;
                 i < gridSize;
                 ++i)
            {
                const double x =
                    static_cast<double>(i) /
                    denominator;

                const double envelope =
                    std::sin(
                        pi * x
                    );

                seed[i] =
                    envelope *
                    (
                        std::sin(
                            pi *
                            frequency *
                            x +
                            phase
                        ) +
                        0.25 *
                        std::cos(
                            2.0 *
                            pi *
                            x *
                            (
                                1.0 +
                                0.125 *
                                static_cast<double>(
                                    seedIndex % 8
                                )
                            )
                        )
                    );
            }
        }

        ++seedIndex;

        if (!orthogonalizeAndNormalize(
                seed,
                basis,
                gridSize,
                volumeElement))
        {
            if (seedIndex >
                100000)
            {
                throw std::runtime_error(
                    "No fue posible construir un subespacio inicial "
                    "linealmente independiente."
                );
            }

            continue;
        }

        basis.push_back(
            std::move(seed)
        );
    }

    /*
     * hBasis is created exclusively from basis.
     */
    std::vector<std::vector<double>> hBasis;

    rebuildSubspace(
        grid,
        effectivePotential,
        basis,
        hBasis,
        gridSize
    );

    const std::size_t maximumSubspaceDimension =
        std::min(
            gridSize,
            numberOfOrbitals +
            EXTRA_SUBSPACE_VECTORS
        );

    std::vector<RitzState> states;

    bool converged =
        false;

    /*
     * ---------------------------------------------------------------
     * Block-Davidson iterations
     * ---------------------------------------------------------------
     */
    for (std::size_t iteration = 0;
         iteration < maxIterations;
         ++iteration)
    {
        validateBasisConsistency(
            basis,
            hBasis,
            gridSize
        );

        states =
            calculateRitzStates(
                basis,
                hBasis,
                numberOfOrbitals,
                gridSize,
                volumeElement
            );

        if (allConverged(
                states,
                numberOfOrbitals))
        {
            converged =
                true;

            break;
        }

        /*
         * -----------------------------------------------------------
         * Build correction block.
         * -----------------------------------------------------------
         */
        std::vector<std::vector<double>> correctionBlock;

        correctionBlock.reserve(
            numberOfOrbitals
        );

        for (std::size_t state = 0;
             state < numberOfOrbitals;
             ++state)
        {
            if (states[state].residualNorm <=
                RESIDUAL_TOLERANCE)
            {
                continue;
            }

            std::vector<double> correction =
                buildPreconditionedResidual(
                    states[state],
                    effectivePotential,
                    grid
                );

            std::vector<std::vector<double>> against =
                basis;

            for (const auto& accepted :
                 correctionBlock)
            {
                against.push_back(
                    accepted
                );
            }

            if (!addIndependentVector(
                    correctionBlock,
                    std::move(correction),
                    against,
                    gridSize,
                    volumeElement))
            {
                /*
                 * The preconditioner produced a dependent direction.
                 * Use the physical residual itself.
                 */
                std::vector<double> residual =
                    states[state].residual;

                against =
                    basis;

                for (const auto& accepted :
                     correctionBlock)
                {
                    against.push_back(
                        accepted
                    );
                }

                addIndependentVector(
                    correctionBlock,
                    std::move(residual),
                    against,
                    gridSize,
                    volumeElement
                );
            }
        }

        if (correctionBlock.empty())
        {
            break;
        }

        /*
         * -----------------------------------------------------------
         * If there is enough room, expand normally.
         * -----------------------------------------------------------
         */
        const std::size_t availableSpace =
            maximumSubspaceDimension >
                    basis.size()
                ? maximumSubspaceDimension -
                  basis.size()
                : 0;

        if (availableSpace > 0)
        {
            const std::size_t numberToAdd =
                std::min(
                    availableSpace,
                    correctionBlock.size()
                );

            for (std::size_t i = 0;
                 i < numberToAdd;
                 ++i)
            {
                basis.push_back(
                    std::move(
                        correctionBlock[i]
                    )
                );
            }

            rebuildSubspace(
                grid,
                effectivePotential,
                basis,
                hBasis,
                gridSize
            );

            continue;
        }

        /*
         * -----------------------------------------------------------
         * Restart.
         *
         * Keep:
         *
         *   - requested Ritz vectors;
         *   - independent correction directions.
         *
         * IMPORTANT:
         * hBasis is NOT copied or partially modified here.
         * It is rebuilt completely from the new basis.
         * -----------------------------------------------------------
         */
        std::vector<std::vector<double>> restartedBasis;

        restartedBasis.reserve(
            maximumSubspaceDimension
        );

        for (std::size_t state = 0;
             state < numberOfOrbitals;
             ++state)
        {
            std::vector<double> vector =
                states[state].psi;

            if (!orthogonalizeAndNormalize(
                    vector,
                    restartedBasis,
                    gridSize,
                    volumeElement))
            {
                continue;
            }

            restartedBasis.push_back(
                std::move(vector)
            );
        }

        for (const std::vector<double>& correction :
             correctionBlock)
        {
            if (restartedBasis.size() >=
                maximumSubspaceDimension)
            {
                break;
            }

            std::vector<double> vector =
                correction;

            if (!orthogonalizeAndNormalize(
                    vector,
                    restartedBasis,
                    gridSize,
                    volumeElement))
            {
                continue;
            }

            restartedBasis.push_back(
                std::move(vector)
            );
        }

        /*
         * The restart must preserve at least the requested number of
         * linearly independent states.
         */
        if (restartedBasis.size() <
            numberOfOrbitals)
        {
            throw std::runtime_error(
                "El restart de Block-Davidson no pudo conservar "
                "un subespacio suficiente."
            );
        }

        basis =
            std::move(
                restartedBasis
            );

        /*
         * This is the critical consistency operation:
         *
         * basis and hBasis are reconstructed together.
         */
        rebuildSubspace(
            grid,
            effectivePotential,
            basis,
            hBasis,
            gridSize
        );
    }

    /*
     * ---------------------------------------------------------------
     * Final Ritz extraction.
     * ---------------------------------------------------------------
     */
    validateBasisConsistency(
        basis,
        hBasis,
        gridSize
    );

    states =
        calculateRitzStates(
            basis,
            hBasis,
            numberOfOrbitals,
            gridSize,
            volumeElement
        );

    if (!allConverged(
            states,
            numberOfOrbitals))
    {
        throw std::runtime_error(
            "Block-Davidson no convergio dentro del numero maximo de iteraciones."
        );
    }

    /*
     * ---------------------------------------------------------------
     * Final physical orthonormalization.
     * ---------------------------------------------------------------
     */
    std::vector<MolecularOrbital> orbitals;

    orbitals.reserve(
        numberOfOrbitals
    );

    std::vector<std::vector<double>> finalBasis;

    finalBasis.reserve(
        numberOfOrbitals
    );

    for (std::size_t state = 0;
         state < numberOfOrbitals;
         ++state)
    {
        std::vector<double> psi =
            states[state].psi;

        if (!orthogonalizeAndNormalize(
                psi,
                finalBasis,
                gridSize,
                volumeElement))
        {
            throw std::runtime_error(
                "No fue posible ortonormalizar "
                "los orbitales moleculares finales."
            );
        }

        finalBasis.push_back(
            psi
        );

        const std::vector<double> hPsi =
            applyMolecularKohnShamHamiltonian(
                grid,
                effectivePotential,
                psi
            );

        validateVectorSize(
            hPsi,
            gridSize,
            "El Hamiltoniano produjo un orbital final "
            "con tamano incorrecto."
        );

        const double eigenvalue =
            dot(
                psi,
                hPsi,
                gridSize,
                volumeElement
            );

        if (!std::isfinite(
                eigenvalue))
        {
            throw std::runtime_error(
                "El autovalor molecular obtenido no es finito."
            );
        }

        /*
         * Verify the final physical residual after the final
         * orthonormalization.
         */
        std::vector<double> residual =
            hPsi;

        axpy(
            residual,
            psi,
            -eigenvalue,
            gridSize
        );

        const double residualNorm =
            norm(
                residual,
                gridSize,
                volumeElement
            );

        if (!std::isfinite(
                residualNorm) ||
            residualNorm >
                RESIDUAL_TOLERANCE)
        {
            throw std::runtime_error(
                "El orbital molecular final no satisface "
                "el criterio de residuo de Block-Davidson."
            );
        }

        MolecularOrbital orbital;

        orbital.spin =
            spin;

        orbital.electrons =
            occupations[state];

        orbital.eigenvalue =
            eigenvalue;

        orbital.psi =
            std::move(psi);

        orbitals.push_back(
            std::move(orbital)
        );
    }

    if (!converged)
    {
        throw std::runtime_error(
            "Block-Davidson finalizo sin una solucion convergida."
        );
    }

    return orbitals;
}