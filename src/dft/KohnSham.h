#pragma once

#include <cstddef>
#include <vector>

#include "IntegralEngine.h"
#include "MolecularSystem.h"

class KohnSham
{
public:

    using Matrix = std::vector<std::vector<double>>;

    KohnSham();

    void initialize(
        const MolecularSystem& system,
        const IntegralEngine& integrals
    );

    void buildFockMatrix(
        const Matrix& densityMatrix
    );

    const Matrix& getOverlapMatrix() const;
    const Matrix& getCoreHamiltonian() const;
    const Matrix& getCoulombMatrix() const;
    const Matrix& getExchangeCorrelationMatrix() const;
    const Matrix& getFockMatrix() const;

    double getExchangeCorrelationEnergy() const;

    bool isInitialized() const;

private:

    const MolecularSystem* molecularSystem;
    const IntegralEngine* integralEngine;

    Matrix overlapMatrix;
    Matrix coreHamiltonian;
    Matrix coulombMatrix;
    Matrix exchangeCorrelationMatrix;
    Matrix fockMatrix;

    double exchangeCorrelationEnergy;

    void buildCoreHamiltonian();

    void buildCoulombMatrix(
        const Matrix& densityMatrix
    );

    void buildExchangeCorrelationMatrix(
        const Matrix& densityMatrix
    );

    static Matrix createMatrix(
        std::size_t dimension
    );

    static void validateMatrix(
        const Matrix& matrix,
        std::size_t dimension,
        const char* name
    );
};