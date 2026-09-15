#pragma once

#include <cstddef>
#include <vector>

class DFTResults
{
public:

    using Matrix = std::vector<std::vector<double>>;

    DFTResults();

    void clear();

    void setConverged(bool converged);

    void setIterations(int iterations);

    void setTotalEnergy(double energy);

    void setElectronicEnergy(double energy);

    void setNuclearRepulsionEnergy(double energy);

    void setEnergyChange(double change);

    void setDensityChange(double change);

    void setOrbitalEnergies(
        const std::vector<double>& energies
    );

    void setCoefficients(
        const Matrix& coefficients
    );

    void setDensityMatrix(
        const Matrix& densityMatrix
    );

    bool isConverged() const;

    int getIterations() const;

    double getTotalEnergy() const;

    double getElectronicEnergy() const;

    double getNuclearRepulsionEnergy() const;

    double getEnergyChange() const;

    double getDensityChange() const;

    std::size_t getOrbitalCount() const;

    double getOrbitalEnergy(
        std::size_t index
    ) const;

    const std::vector<double>&
    getOrbitalEnergies() const;

    const Matrix&
    getCoefficients() const;

    const Matrix&
    getDensityMatrix() const;

    bool isValid() const;

private:

    bool converged;

    int iterations;

    double totalEnergy;

    double electronicEnergy;

    double nuclearRepulsionEnergy;

    double energyChange;

    double densityChange;

    std::vector<double> orbitalEnergies;

    Matrix coefficients;

    Matrix densityMatrix;

    static void validateMatrix(
        const Matrix& matrix,
        const char* name
    );
};