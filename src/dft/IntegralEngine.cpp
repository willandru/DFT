#include "IntegralEngine.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "PhysicalConstants.h"

namespace
{
    constexpr double PI =
        PhysicalConstants::PI;

    constexpr double EPSILON =
        1.0e-12;
}

IntegralEngine::IntegralEngine()
    : basisSet(nullptr),
      nuclearRepulsionEnergy(0.0),
      gridPointsPerAxis(16),
      gridSpacing(0.75),
      gridMargin(4.0),
      laplacianStep(0.05)
{
}

IntegralEngine::IntegralEngine(
    const BasisSet& basis
)
    : IntegralEngine()
{
    initialize(basis);
}

void IntegralEngine::initialize(
    const BasisSet& basis
)
{
    if (!basis.isInitialized())
    {
        throw std::invalid_argument(
            "IntegralEngine: basis set has not been initialized."
        );
    }

    basisSet = &basis;

    grid.clear();

    basisValues.clear();

    laplacianValues.clear();

    nuclearPotentialValues.clear();

    overlapMatrix.clear();

    kineticMatrix.clear();

    nuclearAttractionMatrix.clear();

    twoElectronIntegrals.dimension = 0;

    twoElectronIntegrals.values.clear();

    nuclearRepulsionEnergy = 0.0;
}

void IntegralEngine::calculate()
{
    if (!isInitialized())
    {
        throw std::runtime_error(
            "IntegralEngine: basis set has not been initialized."
        );
    }

    const std::size_t nBasis =
        basisSet->getFunctionCount();

    if (nBasis == 0)
    {
        throw std::runtime_error(
            "IntegralEngine: basis set contains no basis functions."
        );
    }

    buildGrid();

    evaluateBasisFunctions();

    evaluateLaplacians();

    evaluateNuclearPotential();

    calculateOneElectronIntegrals();

    calculateTwoElectronIntegrals();

    calculateNuclearRepulsion();
}

const IntegralEngine::Matrix&
IntegralEngine::getOverlapMatrix() const
{
    if (!isInitialized())
    {
        throw std::runtime_error(
            "IntegralEngine: basis set has not been initialized."
        );
    }

    return overlapMatrix;
}

const IntegralEngine::Matrix&
IntegralEngine::getKineticMatrix() const
{
    if (!isInitialized())
    {
        throw std::runtime_error(
            "IntegralEngine: basis set has not been initialized."
        );
    }

    return kineticMatrix;
}

const IntegralEngine::Matrix&
IntegralEngine::getNuclearAttractionMatrix() const
{
    if (!isInitialized())
    {
        throw std::runtime_error(
            "IntegralEngine: basis set has not been initialized."
        );
    }

    return nuclearAttractionMatrix;
}

const IntegralEngine::TwoElectronIntegrals&
IntegralEngine::getTwoElectronIntegrals() const
{
    if (!isInitialized())
    {
        throw std::runtime_error(
            "IntegralEngine: basis set has not been initialized."
        );
    }

    return twoElectronIntegrals;
}

double IntegralEngine::getNuclearRepulsionEnergy() const
{
    if (!isInitialized())
    {
        throw std::runtime_error(
            "IntegralEngine: basis set has not been initialized."
        );
    }

    return nuclearRepulsionEnergy;
}

bool IntegralEngine::isInitialized() const
{
    return basisSet != nullptr;
}

double&
IntegralEngine::TwoElectronIntegrals::operator()(
    std::size_t mu,
    std::size_t nu,
    std::size_t lambda,
    std::size_t sigma
)
{
    if (mu >= dimension ||
        nu >= dimension ||
        lambda >= dimension ||
        sigma >= dimension)
    {
        throw std::out_of_range(
            "IntegralEngine: two-electron integral index out of range."
        );
    }

    return values[
        IntegralEngine::tensorIndex(
            dimension,
            mu,
            nu,
            lambda,
            sigma
        )
    ];
}

double
IntegralEngine::TwoElectronIntegrals::operator()(
    std::size_t mu,
    std::size_t nu,
    std::size_t lambda,
    std::size_t sigma
) const
{
    if (mu >= dimension ||
        nu >= dimension ||
        lambda >= dimension ||
        sigma >= dimension)
    {
        throw std::out_of_range(
            "IntegralEngine: two-electron integral index out of range."
        );
    }

    return values[
        IntegralEngine::tensorIndex(
            dimension,
            mu,
            nu,
            lambda,
            sigma
        )
    ];
}

std::size_t IntegralEngine::tensorIndex(
    std::size_t dimension,
    std::size_t mu,
    std::size_t nu,
    std::size_t lambda,
    std::size_t sigma
)
{
    return
        (((mu * dimension + nu)
        * dimension + lambda)
        * dimension + sigma);
}

IntegralEngine::Matrix
IntegralEngine::createMatrix(
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

void IntegralEngine::buildGrid()
{
    grid.clear();

    const auto& atoms =
        basisSet->getMolecularSystem().getAtoms();

    if (atoms.empty())
    {
        throw std::runtime_error(
            "IntegralEngine: molecular system contains no atoms."
        );
    }

    double minX = atoms.front().x;
    double maxX = atoms.front().x;

    double minY = atoms.front().y;
    double maxY = atoms.front().y;

    double minZ = atoms.front().z;
    double maxZ = atoms.front().z;

    for (const auto& atom : atoms)
    {
        minX = std::min(minX, atom.x);
        maxX = std::max(maxX, atom.x);

        minY = std::min(minY, atom.y);
        maxY = std::max(maxY, atom.y);

        minZ = std::min(minZ, atom.z);
        maxZ = std::max(maxZ, atom.z);
    }

    minX -= gridMargin;
    maxX += gridMargin;

    minY -= gridMargin;
    maxY += gridMargin;

    minZ -= gridMargin;
    maxZ += gridMargin;

    const double widthX =
        maxX - minX;

    const double widthY =
        maxY - minY;

    const double widthZ =
        maxZ - minZ;

    const int nx =
        std::max(
            gridPointsPerAxis,
            static_cast<int>(
                std::ceil(
                    widthX / gridSpacing
                )
            ) + 1
        );

    const int ny =
        std::max(
            gridPointsPerAxis,
            static_cast<int>(
                std::ceil(
                    widthY / gridSpacing
                )
            ) + 1
        );

    const int nz =
        std::max(
            gridPointsPerAxis,
            static_cast<int>(
                std::ceil(
                    widthZ / gridSpacing
                )
            ) + 1
        );

    const double dx =
        widthX /
        static_cast<double>(nx - 1);

    const double dy =
        widthY /
        static_cast<double>(ny - 1);

    const double dz =
        widthZ /
        static_cast<double>(nz - 1);

    const double volumeElement =
        dx * dy * dz;

    const std::size_t totalPoints =
        static_cast<std::size_t>(nx) *
        static_cast<std::size_t>(ny) *
        static_cast<std::size_t>(nz);

    grid.reserve(totalPoints);

    for (int ix = 0; ix < nx; ++ix)
    {
        const double x =
            minX +
            static_cast<double>(ix) * dx;

        for (int iy = 0; iy < ny; ++iy)
        {
            const double y =
                minY +
                static_cast<double>(iy) * dy;

            for (int iz = 0; iz < nz; ++iz)
            {
                const double z =
                    minZ +
                    static_cast<double>(iz) * dz;

                grid.push_back(
                    {
                        x,
                        y,
                        z,
                        volumeElement
                    }
                );
            }
        }
    }
}

double IntegralEngine::evaluatePrimitiveGaussian(
    const BasisSet::BasisFunction& function,
    const BasisSet::PrimitiveGaussian& primitive,
    double x,
    double y,
    double z
) const
{
    const auto& atom =
        basisSet->getAtom(
            function.atomIndex
        );

    const double dx =
        x - atom.x;

    const double dy =
        y - atom.y;

    const double dz =
        z - atom.z;

    const int lx =
        function.angularMomentumX;

    const int ly =
        function.angularMomentumY;

    const int lz =
        function.angularMomentumZ;

    const double r2 =
        dx * dx +
        dy * dy +
        dz * dz;

    const double alpha =
        primitive.exponent;

    const double normalization =
        std::pow(
            2.0 * alpha / PI,
            0.75
        );

    const double angularPart =
        std::pow(dx, lx) *
        std::pow(dy, ly) *
        std::pow(dz, lz);

    return
        primitive.coefficient *
        normalization *
        angularPart *
        std::exp(
            -alpha * r2
        );
}

double IntegralEngine::evaluateBasisFunction(
    const BasisSet::BasisFunction& function,
    double x,
    double y,
    double z
) const
{
    double value = 0.0;

    for (const auto& primitive :
         function.primitives)
    {
        value +=
            evaluatePrimitiveGaussian(
                function,
                primitive,
                x,
                y,
                z
            );
    }

    return value;
}

double IntegralEngine::evaluateBasisFunctionLaplacian(
    const BasisSet::BasisFunction& function,
    double x,
    double y,
    double z
) const
{
    const double h =
        laplacianStep;

    const double center =
        evaluateBasisFunction(
            function,
            x,
            y,
            z
        );

    const double xPlus =
        evaluateBasisFunction(
            function,
            x + h,
            y,
            z
        );

    const double xMinus =
        evaluateBasisFunction(
            function,
            x - h,
            y,
            z
        );

    const double yPlus =
        evaluateBasisFunction(
            function,
            x,
            y + h,
            z
        );

    const double yMinus =
        evaluateBasisFunction(
            function,
            x,
            y - h,
            z
        );

    const double zPlus =
        evaluateBasisFunction(
            function,
            x,
            y,
            z + h
        );

    const double zMinus =
        evaluateBasisFunction(
            function,
            x,
            y,
            z - h
        );

    const double secondDerivativeX =
        (
            xPlus -
            2.0 * center +
            xMinus
        ) / (h * h);

    const double secondDerivativeY =
        (
            yPlus -
            2.0 * center +
            yMinus
        ) / (h * h);

    const double secondDerivativeZ =
        (
            zPlus -
            2.0 * center +
            zMinus
        ) / (h * h);

    return
        secondDerivativeX +
        secondDerivativeY +
        secondDerivativeZ;
}

void IntegralEngine::evaluateBasisFunctions()
{
    const std::size_t nBasis =
        basisSet->getFunctionCount();

    const std::size_t nPoints =
        grid.size();

    basisValues.assign(
        nBasis * nPoints,
        0.0
    );

    for (std::size_t mu = 0;
         mu < nBasis;
         ++mu)
    {
        const auto& function =
            basisSet->getFunction(mu);

        for (std::size_t p = 0;
             p < nPoints;
             ++p)
        {
            const auto& point =
                grid[p];

            basisValues[
                mu * nPoints + p
            ] =
                evaluateBasisFunction(
                    function,
                    point.x,
                    point.y,
                    point.z
                );
        }
    }
}

void IntegralEngine::evaluateLaplacians()
{
    const std::size_t nBasis =
        basisSet->getFunctionCount();

    const std::size_t nPoints =
        grid.size();

    laplacianValues.assign(
        nBasis * nPoints,
        0.0
    );

    for (std::size_t mu = 0;
         mu < nBasis;
         ++mu)
    {
        const auto& function =
            basisSet->getFunction(mu);

        for (std::size_t p = 0;
             p < nPoints;
             ++p)
        {
            const auto& point =
                grid[p];

            laplacianValues[
                mu * nPoints + p
            ] =
                evaluateBasisFunctionLaplacian(
                    function,
                    point.x,
                    point.y,
                    point.z
                );
        }
    }
}

double IntegralEngine::evaluateNuclearPotential(
    double x,
    double y,
    double z
) const
{
    double potential = 0.0;

    const auto& nuclei =
        basisSet->getMolecularSystem().getAtoms();

    for (const auto& nucleus : nuclei)
    {
        const double dx =
            x - nucleus.x;

        const double dy =
            y - nucleus.y;

        const double dz =
            z - nucleus.z;

        const double r =
            std::sqrt(
                dx * dx +
                dy * dy +
                dz * dz
            );

        if (r > EPSILON)
        {
            potential -=
                static_cast<double>(
                    nucleus.atomicNumber
                ) / r;
        }
    }

    return potential;
}

void IntegralEngine::evaluateNuclearPotential()
{
    nuclearPotentialValues.assign(
        grid.size(),
        0.0
    );

    for (std::size_t p = 0;
         p < grid.size();
         ++p)
    {
        nuclearPotentialValues[p] =
            evaluateNuclearPotential(
                grid[p].x,
                grid[p].y,
                grid[p].z
            );
    }
}

void IntegralEngine::calculateOneElectronIntegrals()
{
    const std::size_t nBasis =
        basisSet->getFunctionCount();

    const std::size_t nPoints =
        grid.size();

    overlapMatrix =
        createMatrix(nBasis);

    kineticMatrix =
        createMatrix(nBasis);

    nuclearAttractionMatrix =
        createMatrix(nBasis);

    for (std::size_t mu = 0;
         mu < nBasis;
         ++mu)
    {
        for (std::size_t nu = 0;
             nu <= mu;
             ++nu)
        {
            double overlap = 0.0;

            double kinetic = 0.0;

            double attraction = 0.0;

            for (std::size_t p = 0;
                 p < nPoints;
                 ++p)
            {
                const double weight =
                    grid[p].weight;

                const double phiMu =
                    basisValues[
                        mu * nPoints + p
                    ];

                const double phiNu =
                    basisValues[
                        nu * nPoints + p
                    ];

                const double laplacianNu =
                    laplacianValues[
                        nu * nPoints + p
                    ];

                const double nuclearPotential =
                    nuclearPotentialValues[p];

                overlap +=
                    weight *
                    phiMu *
                    phiNu;

                kinetic +=
                    weight *
                    phiMu *
                    (
                        -0.5 *
                        laplacianNu
                    );

                attraction +=
                    weight *
                    phiMu *
                    nuclearPotential *
                    phiNu;
            }

            overlapMatrix[mu][nu] =
                overlap;

            overlapMatrix[nu][mu] =
                overlap;

            kineticMatrix[mu][nu] =
                kinetic;

            kineticMatrix[nu][mu] =
                kinetic;

            nuclearAttractionMatrix[mu][nu] =
                attraction;

            nuclearAttractionMatrix[nu][mu] =
                attraction;
        }
    }
}

void IntegralEngine::calculateTwoElectronIntegrals()
{
    const std::size_t nBasis =
        basisSet->getFunctionCount();

    const std::size_t nPoints =
        grid.size();

    twoElectronIntegrals.dimension =
        nBasis;

    const std::size_t totalSize =
        nBasis *
        nBasis *
        nBasis *
        nBasis;

    twoElectronIntegrals.values.assign(
        totalSize,
        0.0
    );

    std::vector<double> pairValues(
        nBasis *
        nBasis *
        nPoints,
        0.0
    );

    for (std::size_t mu = 0;
         mu < nBasis;
         ++mu)
    {
        for (std::size_t nu = 0;
             nu < nBasis;
             ++nu)
        {
            for (std::size_t p = 0;
                 p < nPoints;
                 ++p)
            {
                pairValues[
                    (mu * nBasis + nu)
                    * nPoints + p
                ] =
                    basisValues[
                        mu * nPoints + p
                    ] *
                    basisValues[
                        nu * nPoints + p
                    ];
            }
        }
    }

    for (std::size_t mu = 0;
         mu < nBasis;
         ++mu)
    {
        for (std::size_t nu = 0;
             nu < nBasis;
             ++nu)
        {
            for (std::size_t lambda = 0;
                 lambda < nBasis;
                 ++lambda)
            {
                for (std::size_t sigma = 0;
                     sigma < nBasis;
                     ++sigma)
                {
                    double integral = 0.0;

                    for (std::size_t p = 0;
                         p < nPoints;
                         ++p)
                    {
                        const double rho1 =
                            pairValues[
                                (mu * nBasis + nu)
                                * nPoints + p
                            ];

                        if (std::abs(rho1) < EPSILON)
                        {
                            continue;
                        }

                        for (std::size_t q = 0;
                             q < nPoints;
                             ++q)
                        {
                            const double rho2 =
                                pairValues[
                                    (lambda * nBasis + sigma)
                                    * nPoints + q
                                ];

                            if (std::abs(rho2) < EPSILON)
                            {
                                continue;
                            }

                            const double dx =
                                grid[p].x -
                                grid[q].x;

                            const double dy =
                                grid[p].y -
                                grid[q].y;

                            const double dz =
                                grid[p].z -
                                grid[q].z;

                            const double distanceSquared =
                                dx * dx +
                                dy * dy +
                                dz * dz;

                            if (distanceSquared <
                                EPSILON * EPSILON)
                            {
                                continue;
                            }

                            const double distance =
                                std::sqrt(
                                    distanceSquared
                                );

                            integral +=
                                grid[p].weight *
                                grid[q].weight *
                                rho1 *
                                rho2 /
                                distance;
                        }
                    }

                    twoElectronIntegrals(
                        mu,
                        nu,
                        lambda,
                        sigma
                    ) = integral;
                }
            }
        }
    }
}

void IntegralEngine::calculateNuclearRepulsion()
{
    const auto& nuclei =
        basisSet->getMolecularSystem().getAtoms();

    nuclearRepulsionEnergy = 0.0;

    for (std::size_t A = 0;
         A < nuclei.size();
         ++A)
    {
        for (std::size_t B = A + 1;
             B < nuclei.size();
             ++B)
        {
            const double dx =
                nuclei[A].x -
                nuclei[B].x;

            const double dy =
                nuclei[A].y -
                nuclei[B].y;

            const double dz =
                nuclei[A].z -
                nuclei[B].z;

            const double distance =
                std::sqrt(
                    dx * dx +
                    dy * dy +
                    dz * dz
                );

            if (distance < EPSILON)
            {
                throw std::runtime_error(
                    "IntegralEngine: two nuclei occupy "
                    "the same position."
                );
            }

            nuclearRepulsionEnergy +=
                static_cast<double>(
                    nuclei[A].atomicNumber
                ) *
                static_cast<double>(
                    nuclei[B].atomicNumber
                ) /
                distance;
        }
    }
}