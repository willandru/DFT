#pragma once

#include <cstddef>
#include <vector>

#include "BasisSet.h"

class IntegralEngine
{
public:

    using Matrix = std::vector<std::vector<double>>;

    struct TwoElectronIntegrals
    {
        std::size_t dimension = 0;
        std::vector<double> values;

        double& operator()(
            std::size_t mu,
            std::size_t nu,
            std::size_t lambda,
            std::size_t sigma
        );

        double operator()(
            std::size_t mu,
            std::size_t nu,
            std::size_t lambda,
            std::size_t sigma
        ) const;
    };

    IntegralEngine();

    explicit IntegralEngine(
        const BasisSet& basisSet
    );

    void initialize(
        const BasisSet& basisSet
    );

    void calculate();

    const Matrix& getOverlapMatrix() const;

    const Matrix& getKineticMatrix() const;

    const Matrix& getNuclearAttractionMatrix() const;

    const TwoElectronIntegrals&
    getTwoElectronIntegrals() const;

    double getNuclearRepulsionEnergy() const;

    bool isInitialized() const;

private:

    const BasisSet* basisSet;

    Matrix overlapMatrix;

    Matrix kineticMatrix;

    Matrix nuclearAttractionMatrix;

    TwoElectronIntegrals twoElectronIntegrals;

    double nuclearRepulsionEnergy;

    void calculateOneElectronIntegrals();

    void calculateTwoElectronIntegrals();

    void calculateNuclearRepulsion();

    double calculatePrimitiveOverlap(
        const BasisSet::BasisFunction& functionA,
        const BasisSet::PrimitiveGaussian& primitiveA,
        const BasisSet::BasisFunction& functionB,
        const BasisSet::PrimitiveGaussian& primitiveB
    ) const;

    double calculatePrimitiveKinetic(
        const BasisSet::BasisFunction& functionA,
        const BasisSet::PrimitiveGaussian& primitiveA,
        const BasisSet::BasisFunction& functionB,
        const BasisSet::PrimitiveGaussian& primitiveB
    ) const;

    double calculatePrimitiveNuclearAttraction(
        const BasisSet::BasisFunction& functionA,
        const BasisSet::PrimitiveGaussian& primitiveA,
        const BasisSet::BasisFunction& functionB,
        const BasisSet::PrimitiveGaussian& primitiveB,
        const MolecularSystem::Atom& nucleus
    ) const;

    double boysFunctionF0(
        double value
    ) const;

    double boysFunction(
        int order,
        double value
    ) const;

    double calculateOneDimensionalOverlap(
        int angularMomentumA,
        int angularMomentumB,
        double centerA,
        double centerB,
        double alpha,
        double beta
    ) const;

    double calculateCartesianOverlap(
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
    ) const;

    struct HermiteCoefficients
    {
        double values[3] = {0.0, 0.0, 0.0};
    };

    HermiteCoefficients calculateHermiteCoefficients(
        int angularMomentumA,
        int angularMomentumB,
        double centerA,
        double centerB,
        double alpha,
        double beta
    ) const;

    double calculateHermiteCoulombIntegral(
        int t,
        int u,
        int v,
        int order,
        double x,
        double y,
        double z,
        double gamma
    ) const;

    static Matrix createMatrix(
        std::size_t dimension
    );

    static std::size_t tensorIndex(
        std::size_t dimension,
        std::size_t mu,
        std::size_t nu,
        std::size_t lambda,
        std::size_t sigma
    );
};