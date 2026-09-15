#include "DFTResults.h"

#include <cmath>
#include <stdexcept>
#include <string>

DFTResults::DFTResults()
    : converged(false),
      iterations(0),
      totalEnergy(0.0),
      electronicEnergy(0.0),
      nuclearRepulsionEnergy(0.0),
      energyChange(0.0),
      densityChange(0.0)
{
}

void DFTResults::clear()
{
    converged = false;

    iterations = 0;

    totalEnergy = 0.0;
    electronicEnergy = 0.0;
    nuclearRepulsionEnergy = 0.0;

    energyChange = 0.0;
    densityChange = 0.0;

    orbitalEnergies.clear();

    coefficients.clear();

    densityMatrix.clear();
}

void DFTResults::setConverged(
    bool value
)
{
    converged = value;
}

void DFTResults::setIterations(
    int value
)
{
    if (value < 0)
    {
        throw std::invalid_argument(
            "DFTResults: iterations cannot be negative."
        );
    }

    iterations = value;
}

void DFTResults::setTotalEnergy(
    double value
)
{
    if (!std::isfinite(value))
    {
        throw std::invalid_argument(
            "DFTResults: total energy must be finite."
        );
    }

    totalEnergy = value;
}

void DFTResults::setElectronicEnergy(
    double value
)
{
    if (!std::isfinite(value))
    {
        throw std::invalid_argument(
            "DFTResults: electronic energy must be finite."
        );
    }

    electronicEnergy = value;
}

void DFTResults::setNuclearRepulsionEnergy(
    double value
)
{
    if (!std::isfinite(value))
    {
        throw std::invalid_argument(
            "DFTResults: nuclear repulsion energy must be finite."
        );
    }

    nuclearRepulsionEnergy = value;
}

void DFTResults::setEnergyChange(
    double value
)
{
    if (!std::isfinite(value) &&
        !std::isinf(value))
    {
        throw std::invalid_argument(
            "DFTResults: energy change must be a valid number."
        );
    }

    energyChange = value;
}

void DFTResults::setDensityChange(
    double value
)
{
    if (!std::isfinite(value))
    {
        throw std::invalid_argument(
            "DFTResults: density change must be finite."
        );
    }

    densityChange = value;
}

void DFTResults::setOrbitalEnergies(
    const std::vector<double>& energies
)
{
    for (double energy : energies)
    {
        if (!std::isfinite(energy))
        {
            throw std::invalid_argument(
                "DFTResults: orbital energies must be finite."
            );
        }
    }

    orbitalEnergies =
        energies;
}

void DFTResults::setCoefficients(
    const Matrix& value
)
{
    validateMatrix(
        value,
        "coefficients"
    );

    coefficients =
        value;
}

void DFTResults::setDensityMatrix(
    const Matrix& value
)
{
    validateMatrix(
        value,
        "density matrix"
    );

    densityMatrix =
        value;
}

bool DFTResults::isConverged() const
{
    return converged;
}

int DFTResults::getIterations() const
{
    return iterations;
}

double DFTResults::getTotalEnergy() const
{
    return totalEnergy;
}

double DFTResults::getElectronicEnergy() const
{
    return electronicEnergy;
}

double DFTResults::getNuclearRepulsionEnergy() const
{
    return nuclearRepulsionEnergy;
}

double DFTResults::getEnergyChange() const
{
    return energyChange;
}

double DFTResults::getDensityChange() const
{
    return densityChange;
}

std::size_t DFTResults::getOrbitalCount() const
{
    return orbitalEnergies.size();
}

double DFTResults::getOrbitalEnergy(
    std::size_t index
) const
{
    if (index >= orbitalEnergies.size())
    {
        throw std::out_of_range(
            "DFTResults: orbital index out of range."
        );
    }

    return orbitalEnergies[index];
}

const std::vector<double>&
DFTResults::getOrbitalEnergies() const
{
    return orbitalEnergies;
}

const DFTResults::Matrix&
DFTResults::getCoefficients() const
{
    return coefficients;
}

const DFTResults::Matrix&
DFTResults::getDensityMatrix() const
{
    return densityMatrix;
}

bool DFTResults::isValid() const
{
    if (iterations < 0)
    {
        return false;
    }

    if (!std::isfinite(totalEnergy) ||
        !std::isfinite(electronicEnergy) ||
        !std::isfinite(nuclearRepulsionEnergy) ||
        !std::isfinite(densityChange))
    {
        return false;
    }

    for (double energy : orbitalEnergies)
    {
        if (!std::isfinite(energy))
        {
            return false;
        }
    }

    if (!coefficients.empty())
    {
        validateMatrix(
            coefficients,
            "coefficients"
        );
    }

    if (!densityMatrix.empty())
    {
        validateMatrix(
            densityMatrix,
            "density matrix"
        );
    }

    return true;
}

void DFTResults::validateMatrix(
    const Matrix& matrix,
    const char* name
)
{
    if (matrix.empty())
    {
        return;
    }

    const std::size_t dimension =
        matrix.front().size();

    for (const auto& row : matrix)
    {
        if (row.size() != dimension)
        {
            throw std::invalid_argument(
                std::string(
                    "DFTResults: invalid "
                ) +
                name +
                " matrix."
            );
        }
    }
}