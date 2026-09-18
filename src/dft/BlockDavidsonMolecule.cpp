#include "BlockDavidsonMolecule.h"

#include "KohnShamHamiltonianMolecule.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

constexpr double RESIDUAL_TOLERANCE = 1.0e-7;
constexpr double ORTHOGONALIZATION_TOLERANCE = 1.0e-12;
constexpr double DENOMINATOR_TOLERANCE = 1.0e-10;
constexpr double JACOBI_TOLERANCE = 1.0e-13;

constexpr std::size_t JACOBI_MAX_ITERATIONS = 10000;

constexpr std::size_t SUBSPACE_MULTIPLIER = 4;
constexpr std::size_t SUBSPACE_EXTRA = 8;

double dot(
    const std::vector<double>& a,
    const std::vector<double>& b
)
{
    if (a.size() != b.size())
    {
        throw std::invalid_argument(
            "Block-Davidson: dimensiones incompatibles en producto escalar."
        );
    }

    double result = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        result += a[i] * b[i];
    }

    return result;
}

double norm(
    const std::vector<double>& vector
)
{
    return std::sqrt(dot(vector, vector));
}

bool normalize(
    std::vector<double>& vector
)
{
    const double vectorNorm = norm(vector);

    if (!std::isfinite(vectorNorm) ||
        vectorNorm <= ORTHOGONALIZATION_TOLERANCE)
    {
        return false;
    }

    for (double& value : vector)
    {
        value /= vectorNorm;
    }

    return true;
}

void orthogonalize(
    std::vector<double>& vector,
    const std::vector<std::vector<double>>& basis
)
{
    /*
     * Dos pasadas de Gram-Schmidt modifican la estabilidad numerica
     * cuando el subespacio contiene vectores casi linealmente
     * dependientes.
     */
    for (int pass = 0; pass < 2; ++pass)
    {
        for (const auto& basisVector : basis)
        {
            const double projection =
                dot(vector, basisVector);

            for (std::size_t i = 0;
                 i < vector.size();
                 ++i)
            {
                vector[i] -=
                    projection * basisVector[i];
            }
        }
    }
}

bool makeIndependent(
    std::vector<double>& vector,
    const std::vector<std::vector<double>>& basis
)
{
    orthogonalize(
        vector,
        basis
    );

    return normalize(vector);
}

std::vector<double> linearCombination(
    const std::vector<std::vector<double>>& basis,
    const std::vector<double>& coefficients
)
{
    if (basis.size() != coefficients.size())
    {
        throw std::invalid_argument(
            "Block-Davidson: dimensiones incompatibles en combinacion lineal."
        );
    }

    if (basis.empty())
    {
        return {};
    }

    std::vector<double> result(
        basis.front().size(),
        0.0
    );

    for (std::size_t i = 0;
         i < basis.size();
         ++i)
    {
        const double coefficient =
            coefficients[i];

        for (std::size_t j = 0;
             j < result.size();
             ++j)
        {
            result[j] +=
                coefficient * basis[i][j];
        }
    }

    return result;
}

std::vector<double> subtractScaled(
    const std::vector<double>& a,
    const std::vector<double>& b,
    double scale
)
{
    if (a.size() != b.size())
    {
        throw std::invalid_argument(
            "Block-Davidson: dimensiones incompatibles en resta."
        );
    }

    std::vector<double> result(a.size());

    for (std::size_t i = 0;
         i < a.size();
         ++i)
    {
        result[i] =
            a[i] -
            scale * b[i];
    }

    return result;
}

struct SymmetricEigenResult
{
    std::vector<double> eigenvalues;
    std::vector<std::vector<double>> eigenvectors;
};

SymmetricEigenResult diagonalizeSymmetric(
    const std::vector<std::vector<double>>& input
)
{
    const std::size_t n = input.size();

    if (n == 0)
    {
        return {};
    }

    for (const auto& row : input)
    {
        if (row.size() != n)
        {
            throw std::invalid_argument(
                "Block-Davidson: matriz proyectada no cuadrada."
            );
        }
    }

    std::vector<std::vector<double>> matrix =
        input;

    std::vector<std::vector<double>> eigenvectors(
        n,
        std::vector<double>(n, 0.0)
    );

    for (std::size_t i = 0; i < n; ++i)
    {
        eigenvectors[i][i] = 1.0;
    }

    for (std::size_t iteration = 0;
         iteration < JACOBI_MAX_ITERATIONS;
         ++iteration)
    {
        std::size_t p = 0;
        std::size_t q = 0;

        double maximumOffDiagonal = 0.0;

        for (std::size_t i = 0;
             i < n;
             ++i)
        {
            for (std::size_t j = i + 1;
                 j < n;
                 ++j)
            {
                const double magnitude =
                    std::abs(matrix[i][j]);

                if (magnitude > maximumOffDiagonal)
                {
                    maximumOffDiagonal = magnitude;
                    p = i;
                    q = j;
                }
            }
        }

        if (maximumOffDiagonal <
            JACOBI_TOLERANCE)
        {
            break;
        }

        const double app = matrix[p][p];
        const double aqq = matrix[q][q];
        const double apq = matrix[p][q];

        const double theta =
            0.5 *
            std::atan2(
                2.0 * apq,
                aqq - app
            );

        const double c = std::cos(theta);
        const double s = std::sin(theta);

        for (std::size_t k = 0;
             k < n;
             ++k)
        {
            if (k == p || k == q)
            {
                continue;
            }

            const double akp =
                matrix[k][p];

            const double akq =
                matrix[k][q];

            matrix[k][p] =
                c * akp -
                s * akq;

            matrix[p][k] =
                matrix[k][p];

            matrix[k][q] =
                s * akp +
                c * akq;

            matrix[q][k] =
                matrix[k][q];
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
             k < n;
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

    std::vector<std::size_t> ordering(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        ordering[i] = i;
    }

    std::sort(
        ordering.begin(),
        ordering.end(),
        [&](std::size_t a,
            std::size_t b)
        {
            return matrix[a][a] <
                   matrix[b][b];
        }
    );

    SymmetricEigenResult result;

    result.eigenvalues.resize(n);

    result.eigenvectors.assign(
        n,
        std::vector<double>(
            n,
            0.0
        )
    );

    for (std::size_t newIndex = 0;
         newIndex < n;
         ++newIndex)
    {
        const std::size_t oldIndex =
            ordering[newIndex];

        result.eigenvalues[newIndex] =
            matrix[oldIndex][oldIndex];

        for (std::size_t row = 0;
             row < n;
             ++row)
        {
            result.eigenvectors[row][newIndex] =
                eigenvectors[row][oldIndex];
        }
    }

    return result;
}

std::vector<std::vector<double>> buildProjectedHamiltonian(
    const std::vector<std::vector<double>>& basis,
    const std::vector<std::vector<double>>& hBasis
)
{
    const std::size_t dimension =
        basis.size();

    if (hBasis.size() != dimension)
    {
        throw std::invalid_argument(
            "Block-Davidson: basis y H*basis tienen dimensiones distintas."
        );
    }

    std::vector<std::vector<double>> projected(
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
                    hBasis[j]
                );

            projected[i][j] = value;
            projected[j][i] = value;
        }
    }

    return projected;
}

std::vector<double> buildHamiltonianDiagonal(
    const CartesianGrid& grid,
    const std::vector<double>& effectivePotential
)
{
    const std::size_t gridSize =
        grid.getSize();

    if (effectivePotential.size() != gridSize)
    {
        throw std::invalid_argument(
            "Block-Davidson: potencial efectivo incompatible con la malla."
        );
    }

    const double dx = grid.getDx();
    const double dy = grid.getDy();
    const double dz = grid.getDz();

    if (dx <= 0.0 ||
        dy <= 0.0 ||
        dz <= 0.0)
    {
        throw std::invalid_argument(
            "Block-Davidson: los espaciamientos de la malla deben ser positivos."
        );
    }

    /*
     * Para el operador cinetico
     *
     *     -1/2 nabla^2
     *
     * discretizado mediante diferencias finitas centrales, la
     * contribucion diagonal es
     *
     *     1/dx^2 + 1/dy^2 + 1/dz^2.
     *
     * La diagonal del Hamiltoniano es entonces
     *
     *     H_ii =
     *         1/dx^2 +
     *         1/dy^2 +
     *         1/dz^2 +
     *         V_eff(i).
     */
    const double kineticDiagonal =
        1.0 / (dx * dx) +
        1.0 / (dy * dy) +
        1.0 / (dz * dz);

    std::vector<double> diagonal(
        gridSize
    );

    for (std::size_t i = 0;
         i < gridSize;
         ++i)
    {
        diagonal[i] =
            kineticDiagonal +
            effectivePotential[i];
    }

    return diagonal;
}

std::vector<double> buildSeed(
    const CartesianGrid& grid,
    std::size_t seedIndex
)
{
    const std::size_t nx =
        grid.getNx();

    const std::size_t ny =
        grid.getNy();

    const std::size_t nz =
        grid.getNz();

    const double xmin =
        grid.getXMin();

    const double xmax =
        grid.getXMax();

    const double ymin =
        grid.getYMin();

    const double ymax =
        grid.getYMax();

    const double zmin =
        grid.getZMin();

    const double zmax =
        grid.getZMax();

    const double xCenter =
        0.5 * (xmin + xmax);

    const double yCenter =
        0.5 * (ymin + ymax);

    const double zCenter =
        0.5 * (zmin + zmax);

    const double xScale =
        std::max(
            0.5 * std::abs(xmax - xmin),
            1.0
        );

    const double yScale =
        std::max(
            0.5 * std::abs(ymax - ymin),
            1.0
        );

    const double zScale =
        std::max(
            0.5 * std::abs(zmax - zmin),
            1.0
        );

    const std::size_t family =
        seedIndex % 12;

    const std::size_t frequency =
        seedIndex / 12 + 1;

    std::vector<double> seed(
        grid.getSize(),
        0.0
    );

    for (std::size_t i = 0;
         i < nx;
         ++i)
    {
        const double x =
            (grid.getX(i) - xCenter) /
            xScale;

        const double sx =
            (grid.getX(i) - xmin) /
            std::max(
                xmax - xmin,
                1.0
            );

        for (std::size_t j = 0;
             j < ny;
             ++j)
        {
            const double y =
                (grid.getY(j) - yCenter) /
                yScale;

            const double sy =
                (grid.getY(j) - ymin) /
                std::max(
                    ymax - ymin,
                    1.0
                );

            for (std::size_t k = 0;
                 k < nz;
                 ++k)
            {
                const double z =
                    (grid.getZ(k) - zCenter) /
                    zScale;

                const double sz =
                    (grid.getZ(k) - zmin) /
                    std::max(
                        zmax - zmin,
                        1.0
                    );

                double value = 0.0;

                if (seedIndex < 12)
                {
                    const double gaussian =
                        std::exp(
                            -0.5 *
                            (x * x +
                             y * y +
                             z * z)
                        );

                    switch (family)
                    {
                        case 0:
                            value = gaussian;
                            break;

                        case 1:
                            value = x * gaussian;
                            break;

                        case 2:
                            value = y * gaussian;
                            break;

                        case 3:
                            value = z * gaussian;
                            break;

                        case 4:
                            value = x * y * gaussian;
                            break;

                        case 5:
                            value = x * z * gaussian;
                            break;

                        case 6:
                            value = y * z * gaussian;
                            break;

                        case 7:
                            value =
                                (x * x -
                                 y * y) *
                                gaussian;
                            break;

                        case 8:
                            value =
                                (2.0 * z * z -
                                 x * x -
                                 y * y) *
                                gaussian;
                            break;

                        case 9:
                            value =
                                (x * x +
                                 y * y +
                                 z * z) *
                                gaussian;
                            break;

                        case 10:
                            value =
                                x * x * x *
                                gaussian;
                            break;

                        case 11:
                            value =
                                y * y * y *
                                gaussian;
                            break;
                    }
                }
                else
                {
                    const double pi =
                        3.14159265358979323846;

                    const double frequencyValue =
                        static_cast<double>(
                            frequency
                        );

                    switch (family % 6)
                    {
                        case 0:
                            value =
                                std::sin(
                                    frequencyValue *
                                    pi * sx
                                ) *
                                std::sin(
                                    frequencyValue *
                                    pi * sy
                                ) *
                                std::sin(
                                    frequencyValue *
                                    pi * sz
                                );
                            break;

                        case 1:
                            value =
                                std::cos(
                                    frequencyValue *
                                    pi * sx
                                ) *
                                std::sin(
                                    frequencyValue *
                                    pi * sy
                                ) *
                                std::sin(
                                    frequencyValue *
                                    pi * sz
                                );
                            break;

                        case 2:
                            value =
                                std::sin(
                                    frequencyValue *
                                    pi * sx
                                ) *
                                std::cos(
                                    frequencyValue *
                                    pi * sy
                                ) *
                                std::sin(
                                    frequencyValue *
                                    pi * sz
                                );
                            break;

                        case 3:
                            value =
                                std::sin(
                                    frequencyValue *
                                    pi * sx
                                ) *
                                std::sin(
                                    frequencyValue *
                                    pi * sy
                                ) *
                                std::cos(
                                    frequencyValue *
                                    pi * sz
                                );
                            break;

                        case 4:
                            value =
                                std::cos(
                                    frequencyValue *
                                    pi * sx
                                ) *
                                std::cos(
                                    frequencyValue *
                                    pi * sy
                                ) *
                                std::sin(
                                    frequencyValue *
                                    pi * sz
                                );
                            break;

                        case 5:
                            value =
                                std::sin(
                                    frequencyValue *
                                    pi * sx
                                ) *
                                std::cos(
                                    frequencyValue *
                                    pi * sy
                                ) *
                                std::cos(
                                    frequencyValue *
                                    pi * sz
                                );
                            break;
                    }
                }

                seed[
                    grid.getIndex(i, j, k)
                ] = value;
            }
        }
    }

    return seed;
}

bool addBasisVector(
    std::vector<double> vector,
    std::vector<std::vector<double>>& basis
)
{
    if (!makeIndependent(
            vector,
            basis))
    {
        return false;
    }

    basis.push_back(
        std::move(vector)
    );

    return true;
}

struct RitzBlock
{
    std::vector<double> eigenvalues;
    std::vector<std::vector<double>> orbitals;
    std::vector<std::vector<double>> hOrbitals;
    std::vector<std::vector<double>> residuals;
    std::vector<double> residualNorms;
};

RitzBlock calculateRitzBlock(
    const std::vector<std::vector<double>>& basis,
    const std::vector<std::vector<double>>& hBasis,
    std::size_t numberOfOrbitals
)
{
    const auto projected =
        buildProjectedHamiltonian(
            basis,
            hBasis
        );

    const SymmetricEigenResult eigensystem =
        diagonalizeSymmetric(
            projected
        );

    if (eigensystem.eigenvalues.size() <
        numberOfOrbitals)
    {
        throw std::runtime_error(
            "Block-Davidson: el subespacio no contiene suficientes estados."
        );
    }

    RitzBlock block;

    block.eigenvalues.resize(
        numberOfOrbitals
    );

    block.orbitals.reserve(
        numberOfOrbitals
    );

    block.hOrbitals.reserve(
        numberOfOrbitals
    );

    block.residuals.reserve(
        numberOfOrbitals
    );

    block.residualNorms.reserve(
        numberOfOrbitals
    );

    for (std::size_t orbital = 0;
         orbital < numberOfOrbitals;
         ++orbital)
    {
        block.eigenvalues[orbital] =
            eigensystem.eigenvalues[orbital];

        std::vector<double> coefficients(
            basis.size(),
            0.0
        );

        for (std::size_t i = 0;
             i < basis.size();
             ++i)
        {
            coefficients[i] =
                eigensystem.eigenvectors[i][orbital];
        }

        std::vector<double> orbitalVector =
            linearCombination(
                basis,
                coefficients
            );

        std::vector<double> hOrbitalVector =
            linearCombination(
                hBasis,
                coefficients
            );

        if (!normalize(orbitalVector))
        {
            throw std::runtime_error(
                "Block-Davidson: no se pudo normalizar un orbital de Ritz."
            );
        }

        /*
         * Los coeficientes de un problema proyectado con una base
         * ortonormal ya producen un vector normalizado. La normalizacion
         * anterior solo corrige errores numericos acumulados.
         */
        const double orbitalNorm =
            norm(orbitalVector);

        if (orbitalNorm <=
            ORTHOGONALIZATION_TOLERANCE)
        {
            throw std::runtime_error(
                "Block-Davidson: norma invalida para un orbital de Ritz."
            );
        }

        /*
         * La combinacion H*V debe recibir exactamente la misma
         * normalizacion que V.
         */
        for (double& value : hOrbitalVector)
        {
            value /= orbitalNorm;
        }

        const std::vector<double> residual =
            subtractScaled(
                hOrbitalVector,
                orbitalVector,
                block.eigenvalues[orbital]
            );

        block.orbitals.push_back(
            std::move(orbitalVector)
        );

        block.hOrbitals.push_back(
            std::move(hOrbitalVector)
        );

        block.residualNorms.push_back(
            norm(residual)
        );

        block.residuals.push_back(
            residual
        );
    }

    return block;
}

std::vector<double> buildCorrection(
    const std::vector<double>& residual,
    double eigenvalue,
    const std::vector<double>& hamiltonianDiagonal
)
{
    std::vector<double> correction(
        residual.size(),
        0.0
    );

    for (std::size_t i = 0;
         i < residual.size();
         ++i)
    {
        const double denominator =
            eigenvalue -
            hamiltonianDiagonal[i];

        if (std::isfinite(denominator) &&
            std::abs(denominator) >
                DENOMINATOR_TOLERANCE)
        {
            correction[i] =
                residual[i] /
                denominator;
        }
        else
        {
            /*
             * Cuando el precondicionador diagonal presenta una
             * singularidad local, el propio residual continua siendo
             * una direccion valida de expansion.
             */
            correction[i] =
                residual[i];
        }
    }

    return correction;
}

bool converged(
    const std::vector<double>& residualNorms
)
{
    if (residualNorms.empty())
    {
        return false;
    }

    for (const double residual : residualNorms)
    {
        if (!std::isfinite(residual) ||
            residual >= RESIDUAL_TOLERANCE)
        {
            return false;
        }
    }

    return true;
}

std::size_t maximumSubspaceDimension(
    std::size_t numberOfOrbitals,
    std::size_t gridSize
)
{
    const std::size_t requested =
        numberOfOrbitals *
        SUBSPACE_MULTIPLIER +
        SUBSPACE_EXTRA;

    return std::min(
        gridSize,
        std::max(
            requested,
            numberOfOrbitals
        )
    );
}

bool buildInitialBlock(
    const CartesianGrid& grid,
    std::size_t numberOfOrbitals,
    const std::vector<MolecularOrbital>& initialOrbitals,
    std::vector<std::vector<double>>& basis
)
{
    basis.clear();

    basis.reserve(
        numberOfOrbitals
    );

    const std::size_t gridSize =
        grid.getSize();

    /*
     * Primero se aprovechan los orbitales iniciales proporcionados por
     * el SCF. Cada uno se vuelve a proyectar contra el bloque para
     * garantizar independencia lineal.
     */
    for (const MolecularOrbital& orbital :
         initialOrbitals)
    {
        if (basis.size() >=
            numberOfOrbitals)
        {
            break;
        }

        if (orbital.psi.size() !=
            gridSize)
        {
            continue;
        }

        std::vector<double> vector =
            orbital.psi;

        addBasisVector(
            std::move(vector),
            basis
        );
    }

    /*
     * Si los orbitales iniciales no son suficientes o contienen
     * vectores dependientes, se generan semillas deterministas.
     */
    const std::size_t maximumSeeds =
        std::max<std::size_t>(
            128,
            numberOfOrbitals * 32
        );

    for (std::size_t seedIndex = 0;
         basis.size() < numberOfOrbitals &&
         seedIndex < maximumSeeds;
         ++seedIndex)
    {
        std::vector<double> seed =
            buildSeed(
                grid,
                seedIndex
            );

        addBasisVector(
            std::move(seed),
            basis
        );
    }

    return basis.size() ==
           numberOfOrbitals;
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
            "Block-Davidson: la malla no puede estar vacia."
        );
    }

    if (effectivePotential.size() !=
        gridSize)
    {
        throw std::invalid_argument(
            "Block-Davidson: el potencial efectivo no coincide con la malla."
        );
    }

    if (numberOfOrbitals == 0)
    {
        return {};
    }

    if (numberOfOrbitals > gridSize)
    {
        throw std::invalid_argument(
            "Block-Davidson: el numero de orbitales excede el numero de puntos de la malla."
        );
    }

    if (occupations.size() !=
        numberOfOrbitals)
    {
        throw std::invalid_argument(
            "Block-Davidson: las ocupaciones no coinciden con el numero de orbitales."
        );
    }

    for (const int electrons :
         occupations)
    {
        if (electrons < 0 ||
            electrons > 2)
        {
            throw std::invalid_argument(
                "Block-Davidson: las ocupaciones deben estar entre 0 y 2."
            );
        }
    }

    if (maxIterations == 0)
    {
        throw std::invalid_argument(
            "Block-Davidson: maxIterations debe ser mayor que cero."
        );
    }

    /*
     * Diagonal aproximada utilizada exclusivamente por el
     * precondicionador Davidson.
     *
     * No se utiliza para reemplazar al Hamiltoniano real.
     */
    const std::vector<double> hamiltonianDiagonal =
        buildHamiltonianDiagonal(
            grid,
            effectivePotential
        );

    std::vector<std::vector<double>> basis;

    if (!buildInitialBlock(
            grid,
            numberOfOrbitals,
            initialOrbitals,
            basis))
    {
        throw std::runtime_error(
            "Block-Davidson: no fue posible construir un bloque inicial independiente."
        );
    }

    const std::size_t maximumSubspace =
        maximumSubspaceDimension(
            numberOfOrbitals,
            gridSize
        );

    std::vector<std::vector<double>> hBasis;

    hBasis.reserve(
        maximumSubspace
    );

    /*
     * Aplicamos H una sola vez a cada vector del bloque inicial.
     * Estas aplicaciones quedan almacenadas y se reutilizan durante
     * toda la diagonalizacion del subespacio.
     */
    for (const auto& vector : basis)
    {
        std::vector<double> hVector =
            applyMolecularKohnShamHamiltonian(
                grid,
                effectivePotential,
                vector
            );

        if (hVector.size() !=
            gridSize)
        {
            throw std::runtime_error(
                "Block-Davidson: el Hamiltoniano devolvio una dimension incorrecta."
            );
        }

        hBasis.push_back(
            std::move(hVector)
        );
    }

    RitzBlock lastBlock;

    for (std::size_t iteration = 0;
         iteration < maxIterations;
         ++iteration)
    {
        RitzBlock block =
            calculateRitzBlock(
                basis,
                hBasis,
                numberOfOrbitals
            );

        lastBlock = block;

        /*
         * Todos los orbitales se consideran simultaneamente.
         *
         * Esto es la diferencia estructural fundamental respecto al
         * Davidson secuencial: no existe un orbital anterior congelado
         * como restriccion para resolver el siguiente.
         */
        if (converged(
                block.residualNorms))
        {
            std::vector<MolecularOrbital> result;

            result.reserve(
                numberOfOrbitals
            );

            for (std::size_t orbital = 0;
                 orbital < numberOfOrbitals;
                 ++orbital)
            {
                MolecularOrbital molecularOrbital;

                molecularOrbital.spin =
                    spin;

                molecularOrbital.electrons =
                    occupations[orbital];

                molecularOrbital.eigenvalue =
                    block.eigenvalues[orbital];

                molecularOrbital.psi =
                    block.orbitals[orbital];

                result.push_back(
                    std::move(molecularOrbital)
                );
            }

            return result;
        }

        /*
         * Las correcciones de todos los estados no convergidos forman
         * un nuevo bloque.
         */
        std::vector<std::vector<double>> corrections;

        corrections.reserve(
            numberOfOrbitals
        );

        for (std::size_t orbital = 0;
             orbital < numberOfOrbitals;
             ++orbital)
        {
            if (block.residualNorms[orbital] <
                RESIDUAL_TOLERANCE)
            {
                continue;
            }

            std::vector<double> correction =
                buildCorrection(
                    block.residuals[orbital],
                    block.eigenvalues[orbital],
                    hamiltonianDiagonal
                );

            /*
             * Las correcciones deben estar fuera del subespacio actual.
             */
            orthogonalize(
                correction,
                basis
            );

            /*
             * Las correcciones del mismo bloque tambien deben ser
             * mutuamente independientes.
             */
            orthogonalize(
                correction,
                corrections
            );

            if (!normalize(correction))
            {
                /*
                 * Si el precondicionador produjo una direccion
                 * degenerada, se utiliza directamente el residual.
                 */
                correction =
                    block.residuals[orbital];

                orthogonalize(
                    correction,
                    basis
                );

                orthogonalize(
                    correction,
                    corrections
                );

                if (!normalize(correction))
                {
                    continue;
                }
            }

            corrections.push_back(
                std::move(correction)
            );
        }

        /*
         * Expansion normal del subespacio.
         */
        bool expanded = false;

        for (auto& correction :
             corrections)
        {
            if (basis.size() >=
                maximumSubspace)
            {
                break;
            }

            if (!addBasisVector(
                    std::move(correction),
                    basis))
            {
                continue;
            }

            std::vector<double> hCorrection =
                applyMolecularKohnShamHamiltonian(
                    grid,
                    effectivePotential,
                    basis.back()
                );

            if (hCorrection.size() !=
                gridSize)
            {
                throw std::runtime_error(
                    "Block-Davidson: dimension incorrecta al aplicar H a una correccion."
                );
            }

            hBasis.push_back(
                std::move(hCorrection)
            );

            expanded = true;
        }

        if (expanded)
        {
            continue;
        }

        /*
         * Si no hay espacio suficiente para las correcciones, se hace
         * un restart del subespacio.
         *
         * Se conservan simultaneamente todos los Ritz vectors actuales.
         */
        basis.clear();
        hBasis.clear();

        basis.reserve(
            maximumSubspace
        );

        hBasis.reserve(
            maximumSubspace
        );

        for (std::size_t orbital = 0;
             orbital < numberOfOrbitals;
             ++orbital)
        {
            std::vector<double> vector =
                block.orbitals[orbital];

            if (!addBasisVector(
                    std::move(vector),
                    basis))
            {
                continue;
            }

            std::vector<double> hVector =
                applyMolecularKohnShamHamiltonian(
                    grid,
                    effectivePotential,
                    basis.back()
                );

            if (hVector.size() !=
                gridSize)
            {
                throw std::runtime_error(
                    "Block-Davidson: dimension incorrecta durante restart."
                );
            }

            hBasis.push_back(
                std::move(hVector)
            );
        }

        /*
         * Despues del restart se intenta incorporar inmediatamente
         * las correcciones que aun sean independientes.
         */
        for (std::size_t orbital = 0;
             orbital < numberOfOrbitals;
             ++orbital)
        {
            if (block.residualNorms[orbital] <
                RESIDUAL_TOLERANCE)
            {
                continue;
            }

            if (basis.size() >=
                maximumSubspace)
            {
                break;
            }

            std::vector<double> correction =
                buildCorrection(
                    block.residuals[orbital],
                    block.eigenvalues[orbital],
                    hamiltonianDiagonal
                );

            if (!addBasisVector(
                    std::move(correction),
                    basis))
            {
                continue;
            }

            std::vector<double> hCorrection =
                applyMolecularKohnShamHamiltonian(
                    grid,
                    effectivePotential,
                    basis.back()
                );

            if (hCorrection.size() !=
                gridSize)
            {
                throw std::runtime_error(
                    "Block-Davidson: dimension incorrecta al agregar correccion despues de restart."
                );
            }

            hBasis.push_back(
                std::move(hCorrection)
            );
        }
    }

    double maximumResidual = 0.0;

    for (const double residual :
         lastBlock.residualNorms)
    {
        maximumResidual =
            std::max(
                maximumResidual,
                residual
            );
    }

    throw std::runtime_error(
        "El solver Block-Davidson molecular no convergio. "
        "Maximo residual = " +
        std::to_string(maximumResidual) +
        ", iteraciones = " +
        std::to_string(maxIterations)
    );
}