#include "SCFSolver.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace
{
    constexpr double JACOBI_TOLERANCE = 1.0e-12;

    constexpr int JACOBI_MAX_ITERATIONS = 10000;

    constexpr double MIN_OVERLAP_EIGENVALUE = 1.0e-10;

    constexpr double DENSITY_DAMPING = 0.50;
}

SCFSolver::SCFSolver()
    : molecularSystem(nullptr),
      kohnSham(nullptr),
      maxIterations(100),
      energyTolerance(1.0e-8),
      densityTolerance(1.0e-6)
{
}

void SCFSolver::initialize(
    const MolecularSystem& system,
    KohnSham& ks
)
{
    if (!system.isValid())
    {
        throw std::invalid_argument(
            "SCFSolver: invalid molecular system."
        );
    }

    if (!ks.isInitialized())
    {
        throw std::invalid_argument(
            "SCFSolver: Kohn-Sham module has not been initialized."
        );
    }

    const auto& overlap =
        ks.getOverlapMatrix();

    validateSymmetricMatrix(
        overlap,
        "overlap"
    );

    molecularSystem = &system;
    kohnSham = &ks;
}

SCFSolver::Result SCFSolver::solve()
{
    if (!isInitialized())
    {
        throw std::runtime_error(
            "SCFSolver: module has not been initialized."
        );
    }

    const auto& overlap =
        kohnSham->getOverlapMatrix();

    const auto& coreHamiltonian =
        kohnSham->getCoreHamiltonian();

    validateSymmetricMatrix(
        overlap,
        "overlap"
    );

    validateSymmetricMatrix(
        coreHamiltonian,
        "core Hamiltonian"
    );

    const std::size_t dimension =
        overlap.size();

    if (dimension == 0)
    {
        throw std::runtime_error(
            "SCFSolver: basis dimension is zero."
        );
    }

    const int alphaElectrons =
        molecularSystem->getAlphaElectronCount();

    const int betaElectrons =
        molecularSystem->getBetaElectronCount();

    if (alphaElectrons < 0 ||
        betaElectrons < 0)
    {
        throw std::runtime_error(
            "SCFSolver: invalid electron occupation."
        );
    }

    if (
        alphaElectrons >
            static_cast<int>(dimension) ||
        betaElectrons >
            static_cast<int>(dimension)
    )
    {
        throw std::runtime_error(
            "SCFSolver: basis set does not contain enough "
            "spatial orbitals for the electron configuration."
        );
    }

    const Matrix orthogonalizer =
        symmetricOrthogonalizer(
            overlap
        );

    Matrix density =
        createMatrix(dimension);

    Matrix previousDensity =
        density;

    Matrix coefficients =
        createMatrix(dimension);

    std::vector<double> orbitalEnergies(
        dimension,
        0.0
    );

    Result result;

    result.nuclearRepulsionEnergy =
        kohnSham->getFockMatrix().empty()
            ? 0.0
            : 0.0;

    /*
        The nuclear repulsion is independent of
        the electronic SCF iterations and comes
        directly from the IntegralEngine.
    */

    /*
        KohnSham does not expose the IntegralEngine
        directly, but the nuclear repulsion is
        needed for the final total energy.

        It is reconstructed here from the molecular
        geometry in atomic units.
    */

    for (std::size_t A = 0;
         A < molecularSystem->getAtomCount();
         ++A)
    {
        const auto& atomA =
            molecularSystem->getAtom(A);

        for (std::size_t B = A + 1;
             B < molecularSystem->getAtomCount();
             ++B)
        {
            const auto& atomB =
                molecularSystem->getAtom(B);

            const double dx =
                atomA.x - atomB.x;

            const double dy =
                atomA.y - atomB.y;

            const double dz =
                atomA.z - atomB.z;

            const double distance =
                std::sqrt(
                    dx * dx +
                    dy * dy +
                    dz * dz
                );

            if (distance <= 1.0e-12)
            {
                throw std::runtime_error(
                    "SCFSolver: two nuclei occupy "
                    "the same position."
                );
            }

            result.nuclearRepulsionEnergy +=
                static_cast<double>(
                    atomA.atomicNumber
                ) *
                static_cast<double>(
                    atomB.atomicNumber
                ) /
                distance;
        }
    }

    double previousEnergy =
        std::numeric_limits<double>::infinity();

    for (int iteration = 1;
         iteration <= maxIterations;
         ++iteration)
    {
        kohnSham->buildFockMatrix(
            density
        );

        const auto& fock =
            kohnSham->getFockMatrix();

        validateSymmetricMatrix(
            fock,
            "Fock"
        );

        const Matrix transformedFock =
            transformHamiltonian(
                fock,
                orthogonalizer
            );

        Matrix transformedCoefficients;

        diagonalizeSymmetric(
            transformedFock,
            orbitalEnergies,
            transformedCoefficients
        );

        coefficients =
            transformCoefficients(
                orthogonalizer,
                transformedCoefficients
            );

        Matrix newDensity =
            buildDensityMatrix(
                coefficients,
                alphaElectrons,
                betaElectrons
            );

        Matrix mixedDensity =
            add(
                scale(
                    newDensity,
                    DENSITY_DAMPING
                ),
                scale(
                    density,
                    1.0 - DENSITY_DAMPING
                )
            );

        const double electronicEnergy =
            calculateElectronicEnergy(
                mixedDensity,
                coreHamiltonian,
                fock
            );

        const double totalEnergy =
            electronicEnergy +
            result.nuclearRepulsionEnergy;

        const double densityChange =
            matrixDifference(
                mixedDensity,
                density
            );

        double energyChange =
            std::numeric_limits<double>::infinity();

        if (std::isfinite(previousEnergy))
        {
            energyChange =
                std::abs(
                    totalEnergy -
                    previousEnergy
                );
        }

        result.iterations =
            iteration;

        result.electronicEnergy =
            electronicEnergy;

        result.totalEnergy =
            totalEnergy;

        result.energyChange =
            energyChange;

        result.densityChange =
            densityChange;

        previousDensity =
            density;

        density =
            std::move(mixedDensity);

        if (
            energyChange < energyTolerance &&
            densityChange < densityTolerance
        )
        {
            result.converged = true;

            break;
        }

        previousEnergy =
            totalEnergy;
    }

    /*
        Rebuild the final Fock matrix from the
        final density and obtain the final orbitals.
    */

    kohnSham->buildFockMatrix(
        density
    );

    const auto& finalFock =
        kohnSham->getFockMatrix();

    const Matrix finalTransformedFock =
        transformHamiltonian(
            finalFock,
            orthogonalizer
        );

    Matrix finalTransformedCoefficients;

    diagonalizeSymmetric(
        finalTransformedFock,
        orbitalEnergies,
        finalTransformedCoefficients
    );

    coefficients =
        transformCoefficients(
            orthogonalizer,
            finalTransformedCoefficients
        );

    result.electronicEnergy =
        calculateElectronicEnergy(
            density,
            coreHamiltonian,
            finalFock
        );

    result.totalEnergy =
        result.electronicEnergy +
        result.nuclearRepulsionEnergy;

    result.orbitalEnergies =
        orbitalEnergies;

    result.coefficients =
        coefficients;

    result.densityMatrix =
        density;

    result.densityChange =
        matrixDifference(
            density,
            previousDensity
        );

    return result;
}

void SCFSolver::setMaxIterations(
    int value
)
{
    if (value <= 0)
    {
        throw std::invalid_argument(
            "SCFSolver: maximum iterations must be positive."
        );
    }

    maxIterations = value;
}

void SCFSolver::setEnergyTolerance(
    double tolerance
)
{
    if (!std::isfinite(tolerance) ||
        tolerance <= 0.0)
    {
        throw std::invalid_argument(
            "SCFSolver: energy tolerance must be positive."
        );
    }

    energyTolerance =
        tolerance;
}

void SCFSolver::setDensityTolerance(
    double tolerance
)
{
    if (!std::isfinite(tolerance) ||
        tolerance <= 0.0)
    {
        throw std::invalid_argument(
            "SCFSolver: density tolerance must be positive."
        );
    }

    densityTolerance =
        tolerance;
}

int SCFSolver::getMaxIterations() const
{
    return maxIterations;
}

double SCFSolver::getEnergyTolerance() const
{
    return energyTolerance;
}

double SCFSolver::getDensityTolerance() const
{
    return densityTolerance;
}

bool SCFSolver::isInitialized() const
{
    return
        molecularSystem != nullptr &&
        kohnSham != nullptr;
}

SCFSolver::Matrix
SCFSolver::createMatrix(
    std::size_t dimension
)
{
    return Matrix(
        dimension,
        std::vector<double>(
            dimension,
            0.0
        )
    );
}

SCFSolver::Matrix
SCFSolver::identityMatrix(
    std::size_t dimension
)
{
    Matrix result =
        createMatrix(dimension);

    for (std::size_t i = 0;
         i < dimension;
         ++i)
    {
        result[i][i] = 1.0;
    }

    return result;
}

SCFSolver::Matrix
SCFSolver::transpose(
    const Matrix& matrix
)
{
    if (matrix.empty())
    {
        return {};
    }

    const std::size_t rows =
        matrix.size();

    const std::size_t columns =
        matrix.front().size();

    Matrix result(
        columns,
        std::vector<double>(
            rows,
            0.0
        )
    );

    for (std::size_t i = 0;
         i < rows;
         ++i)
    {
        if (matrix[i].size() != columns)
        {
            throw std::invalid_argument(
                "SCFSolver: invalid matrix."
            );
        }

        for (std::size_t j = 0;
             j < columns;
             ++j)
        {
            result[j][i] =
                matrix[i][j];
        }
    }

    return result;
}

SCFSolver::Matrix
SCFSolver::multiply(
    const Matrix& a,
    const Matrix& b
)
{
    if (a.empty() ||
        b.empty())
    {
        return {};
    }

    const std::size_t aRows =
        a.size();

    const std::size_t aColumns =
        a.front().size();

    const std::size_t bRows =
        b.size();

    const std::size_t bColumns =
        b.front().size();

    if (aColumns != bRows)
    {
        throw std::invalid_argument(
            "SCFSolver: incompatible matrix dimensions."
        );
    }

    for (const auto& row : a)
    {
        if (row.size() != aColumns)
        {
            throw std::invalid_argument(
                "SCFSolver: invalid matrix A."
            );
        }
    }

    for (const auto& row : b)
    {
        if (row.size() != bColumns)
        {
            throw std::invalid_argument(
                "SCFSolver: invalid matrix B."
            );
        }
    }

    Matrix result(
        aRows,
        std::vector<double>(
            bColumns,
            0.0
        )
    );

    for (std::size_t i = 0;
         i < aRows;
         ++i)
    {
        for (std::size_t k = 0;
             k < aColumns;
             ++k)
        {
            const double value =
                a[i][k];

            if (std::abs(value) < 1.0e-15)
            {
                continue;
            }

            for (std::size_t j = 0;
                 j < bColumns;
                 ++j)
            {
                result[i][j] +=
                    value *
                    b[k][j];
            }
        }
    }

    return result;
}

SCFSolver::Matrix
SCFSolver::scale(
    const Matrix& matrix,
    double factor
)
{
    Matrix result =
        matrix;

    for (auto& row : result)
    {
        for (double& value : row)
        {
            value *= factor;
        }
    }

    return result;
}

SCFSolver::Matrix
SCFSolver::add(
    const Matrix& a,
    const Matrix& b
)
{
    if (a.size() != b.size())
    {
        throw std::invalid_argument(
            "SCFSolver: incompatible matrix dimensions."
        );
    }

    Matrix result =
        a;

    for (std::size_t i = 0;
         i < a.size();
         ++i)
    {
        if (a[i].size() != b[i].size())
        {
            throw std::invalid_argument(
                "SCFSolver: incompatible matrix dimensions."
            );
        }

        for (std::size_t j = 0;
             j < a[i].size();
             ++j)
        {
            result[i][j] +=
                b[i][j];
        }
    }

    return result;
}

SCFSolver::Matrix
SCFSolver::subtract(
    const Matrix& a,
    const Matrix& b
)
{
    if (a.size() != b.size())
    {
        throw std::invalid_argument(
            "SCFSolver: incompatible matrix dimensions."
        );
    }

    Matrix result =
        a;

    for (std::size_t i = 0;
         i < a.size();
         ++i)
    {
        if (a[i].size() != b[i].size())
        {
            throw std::invalid_argument(
                "SCFSolver: incompatible matrix dimensions."
            );
        }

        for (std::size_t j = 0;
             j < a[i].size();
             ++j)
        {
            result[i][j] -=
                b[i][j];
        }
    }

    return result;
}

double SCFSolver::matrixDifference(
    const Matrix& a,
    const Matrix& b
)
{
    if (a.size() != b.size())
    {
        throw std::invalid_argument(
            "SCFSolver: incompatible matrix dimensions."
        );
    }

    double maximum =
        0.0;

    for (std::size_t i = 0;
         i < a.size();
         ++i)
    {
        if (a[i].size() != b[i].size())
        {
            throw std::invalid_argument(
                "SCFSolver: incompatible matrix dimensions."
            );
        }

        for (std::size_t j = 0;
             j < a[i].size();
             ++j)
        {
            maximum =
                std::max(
                    maximum,
                    std::abs(
                        a[i][j] -
                        b[i][j]
                    )
                );
        }
    }

    return maximum;
}

SCFSolver::Matrix
SCFSolver::symmetricOrthogonalizer(
    const Matrix& overlap
)
{
    validateSymmetricMatrix(
        overlap,
        "overlap"
    );

    const std::size_t dimension =
        overlap.size();

    std::vector<double> eigenvalues;

    Matrix eigenvectors;

    diagonalizeSymmetric(
        overlap,
        eigenvalues,
        eigenvectors
    );

    Matrix result =
        createMatrix(dimension);

    for (std::size_t i = 0;
         i < dimension;
         ++i)
    {
        if (
            eigenvalues[i] <=
            MIN_OVERLAP_EIGENVALUE
        )
        {
            throw std::runtime_error(
                "SCFSolver: overlap matrix is singular "
                "or linearly dependent."
            );
        }

        const double inverseSqrt =
            1.0 /
            std::sqrt(
                eigenvalues[i]
            );

        for (std::size_t mu = 0;
             mu < dimension;
             ++mu)
        {
            for (std::size_t nu = 0;
                 nu < dimension;
                 ++nu)
            {
                result[mu][nu] +=
                    eigenvectors[mu][i] *
                    inverseSqrt *
                    eigenvectors[nu][i];
            }
        }
    }

    return result;
}

void SCFSolver::diagonalizeSymmetric(
    Matrix matrix,
    std::vector<double>& eigenvalues,
    Matrix& eigenvectors
)
{
    validateSymmetricMatrix(
        matrix,
        "symmetric matrix"
    );

    const std::size_t dimension =
        matrix.size();

    eigenvectors =
        identityMatrix(
            dimension
        );

    for (int iteration = 0;
         iteration < JACOBI_MAX_ITERATIONS;
         ++iteration)
    {
        std::size_t p = 0;
        std::size_t q = 0;

        double largestOffDiagonal =
            0.0;

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
                    largestOffDiagonal)
                {
                    largestOffDiagonal =
                        value;

                    p = i;
                    q = j;
                }
            }
        }

        if (
            largestOffDiagonal <
            JACOBI_TOLERANCE
        )
        {
            break;
        }

        const double app =
            matrix[p][p];

        const double aqq =
            matrix[q][q];

        const double apq =
            matrix[p][q];

        const double angle =
            0.5 *
            std::atan2(
                2.0 * apq,
                aqq - app
            );

        const double cosine =
            std::cos(angle);

        const double sine =
            std::sin(angle);

        for (std::size_t k = 0;
             k < dimension;
             ++k)
        {
            if (k == p ||
                k == q)
            {
                continue;
            }

            const double mkp =
                matrix[k][p];

            const double mkq =
                matrix[k][q];

            matrix[k][p] =
                cosine * mkp -
                sine * mkq;

            matrix[p][k] =
                matrix[k][p];

            matrix[k][q] =
                sine * mkp +
                cosine * mkq;

            matrix[q][k] =
                matrix[k][q];
        }

        matrix[p][p] =
            cosine * cosine * app -
            2.0 *
            sine *
            cosine *
            apq +
            sine * sine * aqq;

        matrix[q][q] =
            sine * sine * app +
            2.0 *
            sine *
            cosine *
            apq +
            cosine * cosine * aqq;

        matrix[p][q] =
            0.0;

        matrix[q][p] =
            0.0;

        for (std::size_t k = 0;
             k < dimension;
             ++k)
        {
            const double vkp =
                eigenvectors[k][p];

            const double vkq =
                eigenvectors[k][q];

            eigenvectors[k][p] =
                cosine * vkp -
                sine * vkq;

            eigenvectors[k][q] =
                sine * vkp +
                cosine * vkq;
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
        [&eigenvalues](
            std::size_t a,
            std::size_t b
        )
        {
            return
                eigenvalues[a] <
                eigenvalues[b];
        }
    );

    std::vector<double> sortedEigenvalues(
        dimension
    );

    Matrix sortedEigenvectors =
        createMatrix(
            dimension
        );

    for (std::size_t i = 0;
         i < dimension;
         ++i)
    {
        sortedEigenvalues[i] =
            eigenvalues[order[i]];

        for (std::size_t j = 0;
             j < dimension;
             ++j)
        {
            sortedEigenvectors[j][i] =
                eigenvectors[j][order[i]];
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

SCFSolver::Matrix
SCFSolver::transformHamiltonian(
    const Matrix& hamiltonian,
    const Matrix& orthogonalizer
)
{
    return multiply(
        transpose(
            orthogonalizer
        ),
        multiply(
            hamiltonian,
            orthogonalizer
        )
    );
}

SCFSolver::Matrix
SCFSolver::transformCoefficients(
    const Matrix& orthogonalizer,
    const Matrix& eigenvectors
)
{
    return multiply(
        orthogonalizer,
        eigenvectors
    );
}

SCFSolver::Matrix
SCFSolver::buildDensityMatrix(
    const Matrix& coefficients,
    int alphaElectrons,
    int betaElectrons
)
{
    const std::size_t dimension =
        coefficients.size();

    Matrix density =
        createMatrix(
            dimension
        );

    if (
        alphaElectrons < 0 ||
        betaElectrons < 0
    )
    {
        throw std::invalid_argument(
            "SCFSolver: electron counts cannot be negative."
        );
    }

    if (
        alphaElectrons >
            static_cast<int>(dimension) ||
        betaElectrons >
            static_cast<int>(dimension)
    )
    {
        throw std::invalid_argument(
            "SCFSolver: too many electrons for available orbitals."
        );
    }

    /*
        A spatial orbital can contain:
            2 electrons -> alpha + beta
            1 electron  -> alpha or beta

        The occupation is therefore obtained directly
        from the number of alpha and beta electrons.
    */

    for (std::size_t orbital = 0;
         orbital < dimension;
         ++orbital)
    {
        double occupation =
            0.0;

        if (
            orbital <
            static_cast<std::size_t>(
                alphaElectrons
            )
        )
        {
            occupation += 1.0;
        }

        if (
            orbital <
            static_cast<std::size_t>(
                betaElectrons
            )
        )
        {
            occupation += 1.0;
        }

        if (occupation == 0.0)
        {
            continue;
        }

        for (std::size_t mu = 0;
             mu < dimension;
             ++mu)
        {
            for (std::size_t nu = 0;
                 nu < dimension;
                 ++nu)
            {
                density[mu][nu] +=
                    occupation *
                    coefficients[mu][orbital] *
                    coefficients[nu][orbital];
            }
        }
    }

    return density;
}

double SCFSolver::calculateElectronicEnergy(
    const Matrix& density,
    const Matrix& coreHamiltonian,
    const Matrix& fockMatrix
)
{
    if (
        density.size() !=
        coreHamiltonian.size() ||
        density.size() !=
        fockMatrix.size()
    )
    {
        throw std::invalid_argument(
            "SCFSolver: incompatible energy matrix dimensions."
        );
    }

    const std::size_t dimension =
        density.size();

    double energy =
        0.0;

    for (std::size_t mu = 0;
         mu < dimension;
         ++mu)
    {
        if (
            density[mu].size() != dimension ||
            coreHamiltonian[mu].size() != dimension ||
            fockMatrix[mu].size() != dimension
        )
        {
            throw std::invalid_argument(
                "SCFSolver: invalid energy matrix."
            );
        }

        for (std::size_t nu = 0;
             nu < dimension;
             ++nu)
        {
            energy +=
                0.5 *
                density[mu][nu] *
                (
                    coreHamiltonian[mu][nu] +
                    fockMatrix[mu][nu]
                );
        }
    }

    return energy;
}

void SCFSolver::validateSymmetricMatrix(
    const Matrix& matrix,
    const char* name
)
{
    const std::size_t dimension =
        matrix.size();

    if (dimension == 0)
    {
        throw std::invalid_argument(
            std::string(
                "SCFSolver: "
            ) +
            name +
            " matrix is empty."
        );
    }

    constexpr double SYMMETRY_TOLERANCE =
        1.0e-10;

    for (std::size_t i = 0;
         i < dimension;
         ++i)
    {
        if (matrix[i].size() != dimension)
        {
            throw std::invalid_argument(
                std::string(
                    "SCFSolver: "
                ) +
                name +
                " matrix must be square."
            );
        }

        for (std::size_t j = i + 1;
             j < dimension;
             ++j)
        {
            if (
                std::abs(
                    matrix[i][j] -
                    matrix[j][i]
                ) >
                SYMMETRY_TOLERANCE
            )
            {
                throw std::invalid_argument(
                    std::string(
                        "SCFSolver: "
                    ) +
                    name +
                    " matrix is not symmetric."
                );
            }
        }
    }
}