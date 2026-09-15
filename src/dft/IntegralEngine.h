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

    struct GridPoint
    {
        double x;
        double y;
        double z;
        double weight;
    };

    const BasisSet* basisSet;

    std::vector<GridPoint> grid;

    std::vector<double> basisValues;

    std::vector<double> laplacianValues;

    std::vector<double> nuclearPotentialValues;

    Matrix overlapMatrix;

    Matrix kineticMatrix;

    Matrix nuclearAttractionMatrix;

    TwoElectronIntegrals twoElectronIntegrals;

    double nuclearRepulsionEnergy;

    int gridPointsPerAxis;

    double gridSpacing;

    double gridMargin;

    double laplacianStep;

    void buildGrid();

    void evaluateBasisFunctions();

    void evaluateLaplacians();

    void evaluateNuclearPotential();

    double evaluateBasisFunction(
        const BasisSet::BasisFunction& function,
        double x,
        double y,
        double z
    ) const;

    double evaluatePrimitiveGaussian(
        const BasisSet::BasisFunction& function,
        const BasisSet::PrimitiveGaussian& primitive,
        double x,
        double y,
        double z
    ) const;

    double evaluateBasisFunctionLaplacian(
        const BasisSet::BasisFunction& function,
        double x,
        double y,
        double z
    ) const;

    double evaluateNuclearPotential(
        double x,
        double y,
        double z
    ) const;

    void calculateOneElectronIntegrals();

    void calculateTwoElectronIntegrals();

    void calculateNuclearRepulsion();

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