#pragma once

#include <cstddef>
#include <vector>

#include "KohnSham.h"
#include "MolecularSystem.h"

class SCFSolver
{
public:

    using Matrix = std::vector<std::vector<double>>;

    struct Result
    {
        bool converged = false;

        int iterations = 0;

        double totalEnergy = 0.0;
        double electronicEnergy = 0.0;
        double nuclearRepulsionEnergy = 0.0;

        double energyChange = 0.0;
        double densityChange = 0.0;

        std::vector<double> orbitalEnergies;

        Matrix coefficients;
        Matrix densityMatrix;
    };

    SCFSolver();

    void initialize(
        const MolecularSystem& system,
        KohnSham& kohnSham
    );

    Result solve();

    void setMaxIterations(int maxIterations);

    void setEnergyTolerance(double tolerance);

    void setDensityTolerance(double tolerance);

    int getMaxIterations() const;

    double getEnergyTolerance() const;

    double getDensityTolerance() const;

    bool isInitialized() const;

private:

    const MolecularSystem* molecularSystem;

    KohnSham* kohnSham;

    int maxIterations;

    double energyTolerance;

    double densityTolerance;

    static Matrix createMatrix(
        std::size_t dimension
    );

    static Matrix identityMatrix(
        std::size_t dimension
    );

    static Matrix transpose(
        const Matrix& matrix
    );

    static Matrix multiply(
        const Matrix& a,
        const Matrix& b
    );

    static Matrix scale(
        const Matrix& matrix,
        double factor
    );

    static Matrix add(
        const Matrix& a,
        const Matrix& b
    );

    static Matrix subtract(
        const Matrix& a,
        const Matrix& b
    );

    static double matrixDifference(
        const Matrix& a,
        const Matrix& b
    );

    static Matrix symmetricOrthogonalizer(
        const Matrix& overlap
    );

    static void diagonalizeSymmetric(
        Matrix matrix,
        std::vector<double>& eigenvalues,
        Matrix& eigenvectors
    );

    static Matrix transformHamiltonian(
        const Matrix& hamiltonian,
        const Matrix& orthogonalizer
    );

    static Matrix transformCoefficients(
        const Matrix& orthogonalizer,
        const Matrix& eigenvectors
    );

    static Matrix buildDensityMatrix(
        const Matrix& coefficients,
        int alphaElectrons,
        int betaElectrons
    );

    static double calculateElectronicEnergy(
        const Matrix& density,
        const Matrix& coreHamiltonian,
        const Matrix& fockMatrix
    );

    static void validateSymmetricMatrix(
        const Matrix& matrix,
        const char* name
    );
};