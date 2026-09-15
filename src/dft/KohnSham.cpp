#include "KohnSham.h"

#include "PhysicalConstants.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace
{
    constexpr double PI =
        PhysicalConstants::PI;

    constexpr double EPSILON =
        1.0e-12;

    constexpr double DENSITY_EPSILON =
        1.0e-14;

    const double CX =
        0.75 *
        std::pow(
            3.0 / PI,
            1.0 / 3.0
        );
}

KohnSham::KohnSham()
    : molecularSystem(nullptr),
      integralEngine(nullptr),
      exchangeCorrelationEnergy(0.0)
{
}

void KohnSham::initialize(
    const MolecularSystem& system,
    const IntegralEngine& integrals
)
{
    if (!system.isValid())
    {
        throw std::invalid_argument(
            "KohnSham: invalid molecular system."
        );
    }

    if (!integrals.isInitialized())
    {
        throw std::invalid_argument(
            "KohnSham: integral engine has not been initialized."
        );
    }

    const auto& overlap =
        integrals.getOverlapMatrix();

    if (overlap.empty())
    {
        throw std::invalid_argument(
            "KohnSham: overlap matrix is empty."
        );
    }

    const std::size_t dimension =
        overlap.size();

    validateMatrix(
        overlap,
        dimension,
        "overlap"
    );

    molecularSystem = &system;
    integralEngine = &integrals;

    overlapMatrix =
        overlap;

    coreHamiltonian =
        createMatrix(dimension);

    coulombMatrix =
        createMatrix(dimension);

    exchangeCorrelationMatrix =
        createMatrix(dimension);

    fockMatrix =
        createMatrix(dimension);

    exchangeCorrelationEnergy =
        0.0;

    buildCoreHamiltonian();
}

void KohnSham::buildFockMatrix(
    const Matrix& densityMatrix
)
{
    if (!isInitialized())
    {
        throw std::runtime_error(
            "KohnSham: module has not been initialized."
        );
    }

    const std::size_t dimension =
        coreHamiltonian.size();

    validateMatrix(
        densityMatrix,
        dimension,
        "density"
    );

    buildCoulombMatrix(
        densityMatrix
    );

    buildExchangeCorrelationMatrix(
        densityMatrix
    );

    for (std::size_t mu = 0;
         mu < dimension;
         ++mu)
    {
        for (std::size_t nu = 0;
             nu < dimension;
             ++nu)
        {
            fockMatrix[mu][nu] =
                coreHamiltonian[mu][nu] +
                coulombMatrix[mu][nu] +
                exchangeCorrelationMatrix[mu][nu];
        }
    }
}

const KohnSham::Matrix&
KohnSham::getOverlapMatrix() const
{
    if (!isInitialized())
    {
        throw std::runtime_error(
            "KohnSham: module has not been initialized."
        );
    }

    return overlapMatrix;
}

const KohnSham::Matrix&
KohnSham::getCoreHamiltonian() const
{
    if (!isInitialized())
    {
        throw std::runtime_error(
            "KohnSham: module has not been initialized."
        );
    }

    return coreHamiltonian;
}

const KohnSham::Matrix&
KohnSham::getCoulombMatrix() const
{
    if (!isInitialized())
    {
        throw std::runtime_error(
            "KohnSham: module has not been initialized."
        );
    }

    return coulombMatrix;
}

const KohnSham::Matrix&
KohnSham::getExchangeCorrelationMatrix() const
{
    if (!isInitialized())
    {
        throw std::runtime_error(
            "KohnSham: module has not been initialized."
        );
    }

    return exchangeCorrelationMatrix;
}

const KohnSham::Matrix&
KohnSham::getFockMatrix() const
{
    if (!isInitialized())
    {
        throw std::runtime_error(
            "KohnSham: module has not been initialized."
        );
    }

    return fockMatrix;
}

double KohnSham::getExchangeCorrelationEnergy() const
{
    if (!isInitialized())
    {
        throw std::runtime_error(
            "KohnSham: module has not been initialized."
        );
    }

    return exchangeCorrelationEnergy;
}

bool KohnSham::isInitialized() const
{
    return
        molecularSystem != nullptr &&
        integralEngine != nullptr;
}

void KohnSham::buildCoreHamiltonian()
{
    const auto& kinetic =
        integralEngine->getKineticMatrix();

    const auto& nuclear =
        integralEngine->getNuclearAttractionMatrix();

    const std::size_t dimension =
        kinetic.size();

    validateMatrix(
        kinetic,
        dimension,
        "kinetic"
    );

    validateMatrix(
        nuclear,
        dimension,
        "nuclear attraction"
    );

    for (std::size_t mu = 0;
         mu < dimension;
         ++mu)
    {
        for (std::size_t nu = 0;
             nu < dimension;
             ++nu)
        {
            coreHamiltonian[mu][nu] =
                kinetic[mu][nu] +
                nuclear[mu][nu];
        }
    }
}

void KohnSham::buildCoulombMatrix(
    const Matrix& densityMatrix
)
{
    const auto& eri =
        integralEngine->getTwoElectronIntegrals();

    const std::size_t dimension =
        densityMatrix.size();

    if (eri.dimension != dimension)
    {
        throw std::runtime_error(
            "KohnSham: two-electron integral dimension "
            "does not match density dimension."
        );
    }

    coulombMatrix =
        createMatrix(dimension);

    for (std::size_t mu = 0;
         mu < dimension;
         ++mu)
    {
        for (std::size_t nu = 0;
             nu < dimension;
             ++nu)
        {
            double value = 0.0;

            for (std::size_t lambda = 0;
                 lambda < dimension;
                 ++lambda)
            {
                for (std::size_t sigma = 0;
                     sigma < dimension;
                     ++sigma)
                {
                    value +=
                        densityMatrix[lambda][sigma] *
                        eri(
                            mu,
                            nu,
                            lambda,
                            sigma
                        );
                }
            }

            coulombMatrix[mu][nu] =
                value;
        }
    }
}

void KohnSham::buildExchangeCorrelationMatrix(
    const Matrix& densityMatrix
)
{
    const std::size_t dimension =
        densityMatrix.size();

    exchangeCorrelationMatrix =
        createMatrix(dimension);

    double electronCount = 0.0;

    for (std::size_t mu = 0;
         mu < dimension;
         ++mu)
    {
        for (std::size_t nu = 0;
             nu < dimension;
             ++nu)
        {
            electronCount +=
                densityMatrix[mu][nu] *
                overlapMatrix[mu][nu];
        }
    }

    electronCount =
        std::max(
            electronCount,
            0.0
        );

    const double density =
        std::max(
            electronCount,
            DENSITY_EPSILON
        );

    const double exchangePotential =
        -std::pow(
            3.0 / PI *
            density,
            1.0 / 3.0
        );

    const double localExchangeEnergy =
        -CX *
        std::pow(
            density,
            4.0 / 3.0
        );

    exchangeCorrelationEnergy =
        localExchangeEnergy;

    for (std::size_t mu = 0;
         mu < dimension;
         ++mu)
    {
        for (std::size_t nu = 0;
             nu < dimension;
             ++nu)
        {
            exchangeCorrelationMatrix[mu][nu] =
                exchangePotential *
                overlapMatrix[mu][nu];
        }
    }
}

KohnSham::Matrix
KohnSham::createMatrix(
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

void KohnSham::validateMatrix(
    const Matrix& matrix,
    std::size_t dimension,
    const char* name
)
{
    if (matrix.size() != dimension)
    {
        throw std::invalid_argument(
            std::string(
                "KohnSham: invalid "
            ) +
            name +
            " matrix dimension."
        );
    }

    for (const auto& row : matrix)
    {
        if (row.size() != dimension)
        {
            throw std::invalid_argument(
                std::string(
                    "KohnSham: invalid "
                ) +
                name +
                " matrix."
            );
        }
    }
}