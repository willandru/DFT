#include "SelfConsistentFieldMoleculeMath.h"

#include "DFTConstants.h"
#include "KohnShamHamiltonianMolecule.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace MolecularSCFMath
{

void mixDensity(
    std::vector<double>& density,
    const std::vector<double>& output,
    double mixing
)
{
    if (density.size() != output.size()) {
        throw std::invalid_argument(
            "Las densidades deben tener el mismo tamano."
        );
    }

    for (std::size_t i = 0;
         i < density.size();
         ++i) {

        density[i] =
            (1.0 - mixing) *
            density[i] +
            mixing *
            output[i];
    }
}

double calculateMolecularDensityDifference(
    const CartesianGrid& grid,
    const std::vector<double>& oldDensity,
    const std::vector<double>& newDensity
)
{
    if (oldDensity.size() != grid.getSize() ||
        newDensity.size() != grid.getSize()) {

        throw std::invalid_argument(
            "La malla cartesiana y las densidades deben tener el mismo tamano."
        );
    }

    const double volumeElement =
        grid.getDx() *
        grid.getDy() *
        grid.getDz();

    double differenceNormSquared = 0.0;
    double densityNormSquared = 0.0;

    for (std::size_t i = 0;
         i < grid.getSize();
         ++i) {

        const double difference =
            newDensity[i] -
            oldDensity[i];

        differenceNormSquared +=
            difference *
            difference;

        densityNormSquared +=
            newDensity[i] *
            newDensity[i];
    }

    const double differenceNorm =
        std::sqrt(
            differenceNormSquared *
            volumeElement
        );

    const double densityNorm =
        std::sqrt(
            densityNormSquared *
            volumeElement
        );

    if (densityNorm <= DFTConstants::EPS) {
        return differenceNorm;
    }

    return differenceNorm / densityNorm;
}

double calculateMolecularSpinDensityDifference(
    const CartesianGrid& grid,
    const std::vector<double>& oldAlphaDensity,
    const std::vector<double>& oldBetaDensity,
    const std::vector<double>& newAlphaDensity,
    const std::vector<double>& newBetaDensity
)
{
    const double alphaDifference =
        calculateMolecularDensityDifference(
            grid,
            oldAlphaDensity,
            newAlphaDensity
        );

    const double betaDifference =
        calculateMolecularDensityDifference(
            grid,
            oldBetaDensity,
            newBetaDensity
        );

    return std::max(
        alphaDifference,
        betaDifference
    );
}

double calculateMaximumMolecularKSResidual(
    const CartesianGrid& grid,
    const std::vector<double>& alphaPotential,
    const std::vector<double>& betaPotential,
    const std::vector<MolecularOrbital>& orbitals
)
{
    if (alphaPotential.size() != grid.getSize() ||
        betaPotential.size() != grid.getSize()) {

        throw std::invalid_argument(
            "Los potenciales moleculares y la malla deben tener el mismo tamano."
        );
    }

    const double volumeElement =
        grid.getDx() *
        grid.getDy() *
        grid.getDz();

    double maximumResidual = 0.0;

    for (const MolecularOrbital& orbital : orbitals) {

        if (orbital.psi.size() != grid.getSize()) {
            throw std::invalid_argument(
                "El orbital molecular y la malla deben tener el mismo tamano."
            );
        }

        const std::vector<double>& effectivePotential =
            orbital.spin == SpinChannel::Alpha
                ? alphaPotential
                : betaPotential;

        const std::vector<double> hPsi =
            applyMolecularKohnShamHamiltonian(
                grid,
                effectivePotential,
                orbital.psi
            );

        double residualNormSquared = 0.0;

        for (std::size_t i = 0;
             i < grid.getSize();
             ++i) {

            const double residual =
                hPsi[i] -
                orbital.eigenvalue *
                orbital.psi[i];

            residualNormSquared +=
                residual *
                residual;
        }

        const double residualNorm =
            std::sqrt(
                residualNormSquared *
                volumeElement
            );

        maximumResidual =
            std::max(
                maximumResidual,
                residualNorm
            );
    }

    return maximumResidual;
}

std::vector<int> buildSpinOccupations(
    int electronCount
)
{
    if (electronCount < 0) {
        throw std::invalid_argument(
            "El numero de electrones no puede ser negativo."
        );
    }

    std::vector<int> occupations;

    occupations.reserve(
        static_cast<std::size_t>(electronCount)
    );

    for (int i = 0;
         i < electronCount;
         ++i) {

        occupations.push_back(1);
    }

    return occupations;
}

std::vector<MolecularOrbital> convertOrbitalsToSpin(
    const std::vector<MolecularOrbital>& orbitals,
    SpinChannel spin
)
{
    std::vector<MolecularOrbital> converted =
        orbitals;

    for (MolecularOrbital& orbital : converted) {
        orbital.spin = spin;
    }

    return converted;
}

}