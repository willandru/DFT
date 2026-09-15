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

    constexpr double INTEGRAL_SYMMETRY_TOLERANCE =
        1.0e-8;
}

IntegralEngine::IntegralEngine()
    : basisSet(nullptr),
      nuclearRepulsionEnergy(0.0)
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

    basisSet =
        &basis;

    overlapMatrix.clear();
    kineticMatrix.clear();
    nuclearAttractionMatrix.clear();

    twoElectronIntegrals.dimension =
        0;

    twoElectronIntegrals.values.clear();

    nuclearRepulsionEnergy =
        0.0;
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

double IntegralEngine::calculateOneDimensionalOverlap(
    int angularMomentumA,
    int angularMomentumB,
    double centerA,
    double centerB,
    double alpha,
    double beta
) const
{
    if (angularMomentumA < 0 ||
        angularMomentumB < 0)
    {
        throw std::invalid_argument(
            "IntegralEngine: angular momentum cannot be negative."
        );
    }

    if (angularMomentumA > 3 ||
        angularMomentumB > 3)
    {
        throw std::runtime_error(
            "IntegralEngine: one-dimensional overlap currently "
            "supports angular momentum up to 3."
        );
    }

    const double gamma =
        alpha + beta;

    const double reducedExponent =
        alpha * beta / gamma;

    const double difference =
        centerA - centerB;

    const double gaussianFactor =
        std::exp(
            -reducedExponent *
            difference *
            difference
        );

    const double prefactor =
        std::sqrt(
            PI / gamma
        ) *
        gaussianFactor;

    const double productCenter =
        (
            alpha * centerA +
            beta * centerB
        ) / gamma;

    const double PA =
        productCenter - centerA;

    const double PB =
        productCenter - centerB;

    double moments[8] = {};

    moments[0] = 1.0;

    for (int n = 1; n < 8; ++n)
    {
        if (n == 1)
        {
            moments[n] =
                0.0;
        }
        else
        {
            moments[n] =
                static_cast<double>(n - 1) /
                (2.0 * gamma) *
                moments[n - 2];
        }
    }

    double polynomial[7] = {};

    for (int i = 0; i <= angularMomentumA; ++i)
    {
        double coefficientA =
            0.0;

        if (angularMomentumA == 0)
        {
            coefficientA =
                1.0;
        }
        else if (angularMomentumA == 1)
        {
            if (i == 0)
            {
                coefficientA =
                    PA;
            }
            else
            {
                coefficientA =
                    1.0;
            }
        }
        else if (angularMomentumA == 2)
        {
            if (i == 0)
            {
                coefficientA =
                    PA * PA;
            }
            else if (i == 1)
            {
                coefficientA =
                    2.0 * PA;
            }
            else
            {
                coefficientA =
                    1.0;
            }
        }
        else
        {
            if (i == 0)
            {
                coefficientA =
                    PA * PA * PA;
            }
            else if (i == 1)
            {
                coefficientA =
                    3.0 * PA * PA;
            }
            else if (i == 2)
            {
                coefficientA =
                    3.0 * PA;
            }
            else
            {
                coefficientA =
                    1.0;
            }
        }

        for (int j = 0; j <= angularMomentumB; ++j)
        {
            double coefficientB =
                0.0;

            if (angularMomentumB == 0)
            {
                coefficientB =
                    1.0;
            }
            else if (angularMomentumB == 1)
            {
                if (j == 0)
                {
                    coefficientB =
                        PB;
                }
                else
                {
                    coefficientB =
                        1.0;
                }
            }
            else if (angularMomentumB == 2)
            {
                if (j == 0)
                {
                    coefficientB =
                        PB * PB;
                }
                else if (j == 1)
                {
                    coefficientB =
                        2.0 * PB;
                }
                else
                {
                    coefficientB =
                        1.0;
                }
            }
            else
            {
                if (j == 0)
                {
                    coefficientB =
                        PB * PB * PB;
                }
                else if (j == 1)
                {
                    coefficientB =
                        3.0 * PB * PB;
                }
                else if (j == 2)
                {
                    coefficientB =
                        3.0 * PB;
                }
                else
                {
                    coefficientB =
                        1.0;
                }
            }

            polynomial[i + j] +=
                coefficientA *
                coefficientB;
        }
    }

    double integral =
        0.0;

    for (int n = 0;
         n <= angularMomentumA + angularMomentumB;
         ++n)
    {
        integral +=
            polynomial[n] *
            moments[n];
    }

    return
        prefactor *
        integral;
}

double IntegralEngine::calculateCartesianOverlap(
    int angularMomentumAX,
    int angularMomentumAY,
    int angularMomentumAZ,
    int angularMomentumBX,
    int angularMomentumBY,
    int angularMomentumBZ,
    double ax,
    double ay,
    double az,
    double bx,
    double by,
    double bz,
    double alpha,
    double beta
) const
{
    return
        calculateOneDimensionalOverlap(
            angularMomentumAX,
            angularMomentumBX,
            ax,
            bx,
            alpha,
            beta
        ) *
        calculateOneDimensionalOverlap(
            angularMomentumAY,
            angularMomentumBY,
            ay,
            by,
            alpha,
            beta
        ) *
        calculateOneDimensionalOverlap(
            angularMomentumAZ,
            angularMomentumBZ,
            az,
            bz,
            alpha,
            beta
        );
}

double IntegralEngine::calculatePrimitiveOverlap(
    const BasisSet::BasisFunction& functionA,
    const BasisSet::PrimitiveGaussian& primitiveA,
    const BasisSet::BasisFunction& functionB,
    const BasisSet::PrimitiveGaussian& primitiveB
) const
{
    const auto& atomA =
        basisSet->getAtom(
            functionA.atomIndex
        );

    const auto& atomB =
        basisSet->getAtom(
            functionB.atomIndex
        );

    const double alpha =
        primitiveA.exponent;

    const double beta =
        primitiveB.exponent;

    const double normalizationA =
        std::pow(
            2.0 * alpha / PI,
            0.75
        );

    const double normalizationB =
        std::pow(
            2.0 * beta / PI,
            0.75
        );

    return
        primitiveA.coefficient *
        primitiveB.coefficient *
        normalizationA *
        normalizationB *
        calculateCartesianOverlap(
            functionA.angularMomentumX,
            functionA.angularMomentumY,
            functionA.angularMomentumZ,
            functionB.angularMomentumX,
            functionB.angularMomentumY,
            functionB.angularMomentumZ,
            atomA.x,
            atomA.y,
            atomA.z,
            atomB.x,
            atomB.y,
            atomB.z,
            alpha,
            beta
        );
}

double IntegralEngine::calculatePrimitiveKinetic(
    const BasisSet::BasisFunction& functionA,
    const BasisSet::PrimitiveGaussian& primitiveA,
    const BasisSet::BasisFunction& functionB,
    const BasisSet::PrimitiveGaussian& primitiveB
) const
{
    const auto& atomA =
        basisSet->getAtom(
            functionA.atomIndex
        );

    const auto& atomB =
        basisSet->getAtom(
            functionB.atomIndex
        );

    const double alpha =
        primitiveA.exponent;

    const double beta =
        primitiveB.exponent;

    const double normalizationA =
        std::pow(
            2.0 * alpha / PI,
            0.75
        );

    const double normalizationB =
        std::pow(
            2.0 * beta / PI,
            0.75
        );

    const double coefficient =
        primitiveA.coefficient *
        primitiveB.coefficient *
        normalizationA *
        normalizationB;

    double kinetic =
        0.0;

    const int angularA[3] =
    {
        functionA.angularMomentumX,
        functionA.angularMomentumY,
        functionA.angularMomentumZ
    };

    const int angularB[3] =
    {
        functionB.angularMomentumX,
        functionB.angularMomentumY,
        functionB.angularMomentumZ
    };

    const double centersA[3] =
    {
        atomA.x,
        atomA.y,
        atomA.z
    };

    const double centersB[3] =
    {
        atomB.x,
        atomB.y,
        atomB.z
    };

    for (int direction = 0;
         direction < 3;
         ++direction)
    {
        const int l =
            angularB[direction];

        const int otherA1 =
            angularA[(direction + 1) % 3];

        const int otherA2 =
            angularA[(direction + 2) % 3];

        const int otherB1 =
            angularB[(direction + 1) % 3];

        const int otherB2 =
            angularB[(direction + 2) % 3];

        const double transverseOverlap =
            calculateOneDimensionalOverlap(
                otherA1,
                otherB1,
                centersA[(direction + 1) % 3],
                centersB[(direction + 1) % 3],
                alpha,
                beta
            ) *
            calculateOneDimensionalOverlap(
                otherA2,
                otherB2,
                centersA[(direction + 2) % 3],
                centersB[(direction + 2) % 3],
                alpha,
                beta
            );

        if (l >= 2)
        {
            const double integralLower =
                calculateOneDimensionalOverlap(
                    angularA[direction],
                    l - 2,
                    centersA[direction],
                    centersB[direction],
                    alpha,
                    beta
                );

            kinetic +=
                -0.5 *
                static_cast<double>(
                    l * (l - 1)
                ) *
                integralLower *
                transverseOverlap;
        }

        const double integralSame =
            calculateOneDimensionalOverlap(
                angularA[direction],
                l,
                centersA[direction],
                centersB[direction],
                alpha,
                beta
            );

        kinetic +=
            beta *
            static_cast<double>(
                2 * l + 1
            ) *
            integralSame *
            transverseOverlap;

        const double integralHigher =
            calculateOneDimensionalOverlap(
                angularA[direction],
                l + 2,
                centersA[direction],
                centersB[direction],
                alpha,
                beta
            );

        kinetic +=
            -2.0 *
            beta *
            beta *
            integralHigher *
            transverseOverlap;
    }

    return
        coefficient *
        kinetic;
}

double IntegralEngine::boysFunctionF0(
    double value
) const
{
    if (value < EPSILON)
    {
        return 1.0;
    }

    return
        0.5 *
        std::sqrt(
            PI / value
        ) *
        std::erf(
            std::sqrt(value)
        );
}

double IntegralEngine::boysFunction(
    int order,
    double value
) const
{
    if (order < 0)
    {
        throw std::invalid_argument(
            "IntegralEngine: Boys function order cannot be negative."
        );
    }

    if (value < EPSILON)
    {
        return
            1.0 /
            static_cast<double>(
                2 * order + 1
            );
    }

    if (order == 0)
    {
        return boysFunctionF0(value);
    }

    double valueN =
        boysFunctionF0(value);

    for (int n = 0;
         n < order;
         ++n)
    {
        valueN =
            (
                (
                    2.0 *
                    value *
                    valueN
                ) +
                std::exp(-value)
            ) /
            static_cast<double>(
                2 * n + 3
            );
    }

    return valueN;
}

IntegralEngine::HermiteCoefficients
IntegralEngine::calculateHermiteCoefficients(
    int angularMomentumA,
    int angularMomentumB,
    double centerA,
    double centerB,
    double alpha,
    double beta
) const
{
    if (angularMomentumA < 0 ||
        angularMomentumB < 0 ||
        angularMomentumA > 1 ||
        angularMomentumB > 1)
    {
        throw std::runtime_error(
            "IntegralEngine: Hermite coefficients currently "
            "support only s and p functions."
        );
    }

    const double gamma =
        alpha + beta;

    const double productCenter =
        (
            alpha * centerA +
            beta * centerB
        ) / gamma;

    const double PA =
        productCenter - centerA;

    const double PB =
        productCenter - centerB;

    HermiteCoefficients result;

    if (angularMomentumA == 0 &&
        angularMomentumB == 0)
    {
        result.values[0] =
            1.0;
    }
    else if (angularMomentumA == 1 &&
             angularMomentumB == 0)
    {
        result.values[0] =
            PA;

        result.values[1] =
            1.0 /
            (2.0 * gamma);
    }
    else if (angularMomentumA == 0 &&
             angularMomentumB == 1)
    {
        result.values[0] =
            PB;

        result.values[1] =
            1.0 /
            (2.0 * gamma);
    }
    else
    {
        result.values[0] =
            PA * PB +
            1.0 /
            (2.0 * gamma);

        result.values[1] =
            (
                PA + PB
            ) /
            (2.0 * gamma);

        result.values[2] =
            1.0 /
            (
                4.0 *
                gamma *
                gamma
            );
    }

    return result;
}

double IntegralEngine::calculateHermiteCoulombIntegral(
    int t,
    int u,
    int v,
    int order,
    double x,
    double y,
    double z,
    double gamma
) const
{
    if (t < 0 ||
        u < 0 ||
        v < 0 ||
        order < 0)
    {
        return 0.0;
    }

    const double T =
        gamma *
        (
            x * x +
            y * y +
            z * z
        );

    if (t == 0 &&
        u == 0 &&
        v == 0)
    {
        return
            std::pow(
                -2.0 * gamma,
                order
            ) *
            boysFunction(
                order,
                T
            );
    }

    if (t > 0)
    {
        double result =
            x *
            calculateHermiteCoulombIntegral(
                t - 1,
                u,
                v,
                order + 1,
                x,
                y,
                z,
                gamma
            );

        if (t > 1)
        {
            result +=
                static_cast<double>(t - 1) *
                calculateHermiteCoulombIntegral(
                    t - 2,
                    u,
                    v,
                    order + 1,
                    x,
                    y,
                    z,
                    gamma
                );
        }

        return result;
    }

    if (u > 0)
    {
        double result =
            y *
            calculateHermiteCoulombIntegral(
                t,
                u - 1,
                v,
                order + 1,
                x,
                y,
                z,
                gamma
            );

        if (u > 1)
        {
            result +=
                static_cast<double>(u - 1) *
                calculateHermiteCoulombIntegral(
                    t,
                    u - 2,
                    v,
                    order + 1,
                    x,
                    y,
                    z,
                    gamma
                );
        }

        return result;
    }

    double result =
        z *
        calculateHermiteCoulombIntegral(
            t,
            u,
            v - 1,
            order + 1,
            x,
            y,
            z,
            gamma
        );

    if (v > 1)
    {
        result +=
            static_cast<double>(v - 1) *
            calculateHermiteCoulombIntegral(
                t,
                u,
                v - 2,
                order + 1,
                x,
                y,
                z,
                gamma
            );
    }

    return result;
}

double IntegralEngine::calculatePrimitiveNuclearAttraction(
    const BasisSet::BasisFunction& functionA,
    const BasisSet::PrimitiveGaussian& primitiveA,
    const BasisSet::BasisFunction& functionB,
    const BasisSet::PrimitiveGaussian& primitiveB,
    const MolecularSystem::Atom& nucleus
) const
{
    if (functionA.angularMomentumX > 1 ||
        functionA.angularMomentumY > 1 ||
        functionA.angularMomentumZ > 1 ||
        functionB.angularMomentumX > 1 ||
        functionB.angularMomentumY > 1 ||
        functionB.angularMomentumZ > 1)
    {
        throw std::runtime_error(
            "IntegralEngine: nuclear attraction currently "
            "supports only s and p Gaussian functions."
        );
    }

    const auto& atomA =
        basisSet->getAtom(
            functionA.atomIndex
        );

    const auto& atomB =
        basisSet->getAtom(
            functionB.atomIndex
        );

    const double alpha =
        primitiveA.exponent;

    const double beta =
        primitiveB.exponent;

    const double gamma =
        alpha + beta;

    const double reducedExponent =
        alpha * beta / gamma;

    const double ABx =
        atomA.x - atomB.x;

    const double ABy =
        atomA.y - atomB.y;

    const double ABz =
        atomA.z - atomB.z;

    const double distanceABSquared =
        ABx * ABx +
        ABy * ABy +
        ABz * ABz;

    const double gaussianDecay =
        std::exp(
            -reducedExponent *
            distanceABSquared
        );

    const double Px =
        (
            alpha * atomA.x +
            beta * atomB.x
        ) / gamma;

    const double Py =
        (
            alpha * atomA.y +
            beta * atomB.y
        ) / gamma;

    const double Pz =
        (
            alpha * atomA.z +
            beta * atomB.z
        ) / gamma;

    const double PCx =
        Px - nucleus.x;

    const double PCy =
        Py - nucleus.y;

    const double PCz =
        Pz - nucleus.z;

    const HermiteCoefficients hx =
        calculateHermiteCoefficients(
            functionA.angularMomentumX,
            functionB.angularMomentumX,
            atomA.x,
            atomB.x,
            alpha,
            beta
        );

    const HermiteCoefficients hy =
        calculateHermiteCoefficients(
            functionA.angularMomentumY,
            functionB.angularMomentumY,
            atomA.y,
            atomB.y,
            alpha,
            beta
        );

    const HermiteCoefficients hz =
        calculateHermiteCoefficients(
            functionA.angularMomentumZ,
            functionB.angularMomentumZ,
            atomA.z,
            atomB.z,
            alpha,
            beta
        );

    double coulombSum =
        0.0;

    for (int t = 0; t <= 2; ++t)
    {
        if (std::abs(hx.values[t]) < EPSILON)
        {
            continue;
        }

        for (int u = 0; u <= 2; ++u)
        {
            if (std::abs(hy.values[u]) < EPSILON)
            {
                continue;
            }

            for (int v = 0; v <= 2; ++v)
            {
                if (std::abs(hz.values[v]) < EPSILON)
                {
                    continue;
                }

                coulombSum +=
                    hx.values[t] *
                    hy.values[u] *
                    hz.values[v] *
                    calculateHermiteCoulombIntegral(
                        t,
                        u,
                        v,
                        0,
                        PCx,
                        PCy,
                        PCz,
                        gamma
                    );
            }
        }
    }

    const double prefactor =
        -static_cast<double>(
            nucleus.atomicNumber
        ) *
        (
            2.0 *
            PI /
            gamma
        );

    const double normalizationA =
        std::pow(
            2.0 * alpha / PI,
            0.75
        );

    const double normalizationB =
        std::pow(
            2.0 * beta / PI,
            0.75
        );

    return
        primitiveA.coefficient *
        primitiveB.coefficient *
        normalizationA *
        normalizationB *
        prefactor *
        gaussianDecay *
        coulombSum;
}

void IntegralEngine::calculateOneElectronIntegrals()
{
    const std::size_t nBasis =
        basisSet->getFunctionCount();

    overlapMatrix =
        createMatrix(
            nBasis
        );

    kineticMatrix =
        createMatrix(
            nBasis
        );

    nuclearAttractionMatrix =
        createMatrix(
            nBasis
        );

    const auto& nuclei =
        basisSet->getMolecularSystem().getAtoms();

    for (std::size_t mu = 0;
         mu < nBasis;
         ++mu)
    {
        const auto& functionA =
            basisSet->getFunction(
                mu
            );

        for (std::size_t nu = 0;
             nu <= mu;
             ++nu)
        {
            const auto& functionB =
                basisSet->getFunction(
                    nu
                );

            double overlap =
                0.0;

            double kinetic =
                0.0;

            double attraction =
                0.0;

            for (const auto& primitiveA :
                 functionA.primitives)
            {
                for (const auto& primitiveB :
                     functionB.primitives)
                {
                    overlap +=
                        calculatePrimitiveOverlap(
                            functionA,
                            primitiveA,
                            functionB,
                            primitiveB
                        );

                    kinetic +=
                        calculatePrimitiveKinetic(
                            functionA,
                            primitiveA,
                            functionB,
                            primitiveB
                        );

                    for (const auto& nucleus :
                         nuclei)
                    {
                        attraction +=
                            calculatePrimitiveNuclearAttraction(
                                functionA,
                                primitiveA,
                                functionB,
                                primitiveB,
                                nucleus
                            );
                    }
                }
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

    const auto calculatePrimitiveERI =
        [this](
            const BasisSet::BasisFunction& functionA,
            const BasisSet::PrimitiveGaussian& primitiveA,
            const BasisSet::BasisFunction& functionB,
            const BasisSet::PrimitiveGaussian& primitiveB,
            const BasisSet::BasisFunction& functionC,
            const BasisSet::PrimitiveGaussian& primitiveC,
            const BasisSet::BasisFunction& functionD,
            const BasisSet::PrimitiveGaussian& primitiveD
        ) -> double
        {
            if (functionA.angularMomentumX > 1 ||
                functionA.angularMomentumY > 1 ||
                functionA.angularMomentumZ > 1 ||
                functionB.angularMomentumX > 1 ||
                functionB.angularMomentumY > 1 ||
                functionB.angularMomentumZ > 1 ||
                functionC.angularMomentumX > 1 ||
                functionC.angularMomentumY > 1 ||
                functionC.angularMomentumZ > 1 ||
                functionD.angularMomentumX > 1 ||
                functionD.angularMomentumY > 1 ||
                functionD.angularMomentumZ > 1)
            {
                throw std::runtime_error(
                    "IntegralEngine: two-electron integrals currently "
                    "support only s and p Gaussian functions."
                );
            }

            const auto& atomA =
                basisSet->getAtom(
                    functionA.atomIndex
                );

            const auto& atomB =
                basisSet->getAtom(
                    functionB.atomIndex
                );

            const auto& atomC =
                basisSet->getAtom(
                    functionC.atomIndex
                );

            const auto& atomD =
                basisSet->getAtom(
                    functionD.atomIndex
                );

            const double alpha =
                primitiveA.exponent;

            const double beta =
                primitiveB.exponent;

            const double gamma =
                primitiveC.exponent;

            const double delta =
                primitiveD.exponent;

            const double zeta =
                alpha + beta;

            const double eta =
                gamma + delta;

            const double reducedAB =
                alpha * beta / zeta;

            const double reducedCD =
                gamma * delta / eta;

            const double ABx =
                atomA.x - atomB.x;

            const double ABy =
                atomA.y - atomB.y;

            const double ABz =
                atomA.z - atomB.z;

            const double CDx =
                atomC.x - atomD.x;

            const double CDy =
                atomC.y - atomD.y;

            const double CDz =
                atomC.z - atomD.z;

            const double distanceABSquared =
                ABx * ABx +
                ABy * ABy +
                ABz * ABz;

            const double distanceCDSquared =
                CDx * CDx +
                CDy * CDy +
                CDz * CDz;

            const double gaussianAB =
                std::exp(
                    -reducedAB *
                    distanceABSquared
                );

            const double gaussianCD =
                std::exp(
                    -reducedCD *
                    distanceCDSquared
                );

            const double Px =
                (
                    alpha * atomA.x +
                    beta * atomB.x
                ) / zeta;

            const double Py =
                (
                    alpha * atomA.y +
                    beta * atomB.y
                ) / zeta;

            const double Pz =
                (
                    alpha * atomA.z +
                    beta * atomB.z
                ) / zeta;

            const double Qx =
                (
                    gamma * atomC.x +
                    delta * atomD.x
                ) / eta;

            const double Qy =
                (
                    gamma * atomC.y +
                    delta * atomD.y
                ) / eta;

            const double Qz =
                (
                    gamma * atomC.z +
                    delta * atomD.z
                ) / eta;

            const double PQx =
                Px - Qx;

            const double PQy =
                Py - Qy;

            const double PQz =
                Pz - Qz;

            const double PQSquared =
                PQx * PQx +
                PQy * PQy +
                PQz * PQz;

            const double rho =
                zeta * eta /
                (zeta + eta);

            const HermiteCoefficients hxAB =
                calculateHermiteCoefficients(
                    functionA.angularMomentumX,
                    functionB.angularMomentumX,
                    atomA.x,
                    atomB.x,
                    alpha,
                    beta
                );

            const HermiteCoefficients hyAB =
                calculateHermiteCoefficients(
                    functionA.angularMomentumY,
                    functionB.angularMomentumY,
                    atomA.y,
                    atomB.y,
                    alpha,
                    beta
                );

            const HermiteCoefficients hzAB =
                calculateHermiteCoefficients(
                    functionA.angularMomentumZ,
                    functionB.angularMomentumZ,
                    atomA.z,
                    atomB.z,
                    alpha,
                    beta
                );

            const HermiteCoefficients hxCD =
                calculateHermiteCoefficients(
                    functionC.angularMomentumX,
                    functionD.angularMomentumX,
                    atomC.x,
                    atomD.x,
                    gamma,
                    delta
                );

            const HermiteCoefficients hyCD =
                calculateHermiteCoefficients(
                    functionC.angularMomentumY,
                    functionD.angularMomentumY,
                    atomC.y,
                    atomD.y,
                    gamma,
                    delta
                );

            const HermiteCoefficients hzCD =
                calculateHermiteCoefficients(
                    functionC.angularMomentumZ,
                    functionD.angularMomentumZ,
                    atomC.z,
                    atomD.z,
                    gamma,
                    delta
                );

            double coulombSum =
                0.0;

            for (int t = 0; t <= 2; ++t)
            {
                if (std::abs(hxAB.values[t]) < EPSILON)
                {
                    continue;
                }

                for (int u = 0; u <= 2; ++u)
                {
                    if (std::abs(hyAB.values[u]) < EPSILON)
                    {
                        continue;
                    }

                    for (int v = 0; v <= 2; ++v)
                    {
                        if (std::abs(hzAB.values[v]) < EPSILON)
                        {
                            continue;
                        }

                        for (int tau = 0; tau <= 2; ++tau)
                        {
                            if (std::abs(hxCD.values[tau]) < EPSILON)
                            {
                                continue;
                            }

                            for (int upsilon = 0; upsilon <= 2; ++upsilon)
                            {
                                if (std::abs(hyCD.values[upsilon]) < EPSILON)
                                {
                                    continue;
                                }

                                for (int phi = 0; phi <= 2; ++phi)
                                {
                                    if (std::abs(hzCD.values[phi]) < EPSILON)
                                    {
                                        continue;
                                    }

                                    const double hermiteIntegral =
                                        calculateHermiteCoulombIntegral(
                                            t + tau,
                                            u + upsilon,
                                            v + phi,
                                            0,
                                            PQx,
                                            PQy,
                                            PQz,
                                            rho
                                        );

                                    coulombSum +=
                                        hxAB.values[t] *
                                        hyAB.values[u] *
                                        hzAB.values[v] *
                                        hxCD.values[tau] *
                                        hyCD.values[upsilon] *
                                        hzCD.values[phi] *
                                        hermiteIntegral;
                                }
                            }
                        }
                    }
                }
            }

            const double prefactor =
                (
                    2.0 *
                    std::pow(PI, 2.5)
                ) /
                (
                    zeta *
                    eta *
                    std::sqrt(
                        zeta + eta
                    )
                );

            const double normalizationA =
                std::pow(
                    2.0 * alpha / PI,
                    0.75
                );

            const double normalizationB =
                std::pow(
                    2.0 * beta / PI,
                    0.75
                );

            const double normalizationC =
                std::pow(
                    2.0 * gamma / PI,
                    0.75
                );

            const double normalizationD =
                std::pow(
                    2.0 * delta / PI,
                    0.75
                );

            return
                primitiveA.coefficient *
                primitiveB.coefficient *
                primitiveC.coefficient *
                primitiveD.coefficient *
                normalizationA *
                normalizationB *
                normalizationC *
                normalizationD *
                gaussianAB *
                gaussianCD *
                prefactor *
                coulombSum;
        };

    for (std::size_t mu = 0;
         mu < nBasis;
         ++mu)
    {
        const auto& functionA =
            basisSet->getFunction(
                mu
            );

        for (std::size_t nu = 0;
             nu < nBasis;
             ++nu)
        {
            const auto& functionB =
                basisSet->getFunction(
                    nu
                );

            for (std::size_t lambda = 0;
                 lambda < nBasis;
                 ++lambda)
            {
                const auto& functionC =
                    basisSet->getFunction(
                        lambda
                    );

                for (std::size_t sigma = 0;
                     sigma < nBasis;
                     ++sigma)
                {
                    const auto& functionD =
                        basisSet->getFunction(
                            sigma
                        );

                    double value =
                        0.0;

                    for (const auto& primitiveA :
                         functionA.primitives)
                    {
                        for (const auto& primitiveB :
                             functionB.primitives)
                        {
                            for (const auto& primitiveC :
                                 functionC.primitives)
                            {
                                for (const auto& primitiveD :
                                     functionD.primitives)
                                {
                                    value +=
                                        calculatePrimitiveERI(
                                            functionA,
                                            primitiveA,
                                            functionB,
                                            primitiveB,
                                            functionC,
                                            primitiveC,
                                            functionD,
                                            primitiveD
                                        );
                                }
                            }
                        }
                    }

                    twoElectronIntegrals(
                        mu,
                        nu,
                        lambda,
                        sigma
                    ) =
                        value;
                }
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
                    const double value =
                        twoElectronIntegrals(
                            mu,
                            nu,
                            lambda,
                            sigma
                        );

                    const double symmetry1 =
                        twoElectronIntegrals(
                            nu,
                            mu,
                            lambda,
                            sigma
                        );

                    const double symmetry2 =
                        twoElectronIntegrals(
                            mu,
                            nu,
                            sigma,
                            lambda
                        );

                    const double symmetry3 =
                        twoElectronIntegrals(
                            lambda,
                            sigma,
                            mu,
                            nu
                        );

                    const double scale =
                        std::max(
                            {
                                1.0,
                                std::abs(value),
                                std::abs(symmetry1),
                                std::abs(symmetry2),
                                std::abs(symmetry3)
                            }
                        );

                    const double tolerance =
                        INTEGRAL_SYMMETRY_TOLERANCE *
                        scale;

                    if (std::abs(value - symmetry1) > tolerance)
                    {
                        throw std::runtime_error(
                            "IntegralEngine: ERI symmetry "
                            "(mu,nu|lambda,sigma) != "
                            "(nu,mu|lambda,sigma)."
                        );
                    }

                    if (std::abs(value - symmetry2) > tolerance)
                    {
                        throw std::runtime_error(
                            "IntegralEngine: ERI symmetry "
                            "(mu,nu|lambda,sigma) != "
                            "(mu,nu|sigma,lambda)."
                        );
                    }

                    if (std::abs(value - symmetry3) > tolerance)
                    {
                        throw std::runtime_error(
                            "IntegralEngine: ERI symmetry "
                            "(mu,nu|lambda,sigma) != "
                            "(lambda,sigma|mu,nu)."
                        );
                    }
                }
            }
        }
    }
}

void IntegralEngine::calculateNuclearRepulsion()
{
    const auto& nuclei =
        basisSet->getMolecularSystem().getAtoms();

    nuclearRepulsionEnergy =
        0.0;

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