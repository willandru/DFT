#include "MolecularSCFIteration.h"

#include "ElectronDensityMolecule.h"
#include "ExchangeCorrelationMolecule.h"
#include "HartreePotentialMolecule.h"
#include "MolecularSCFHelpers.h"
#include "TotalEnergyMolecule.h"

#include <algorithm>
#include <stdexcept>

MolecularSCFIterationResult executeMolecularSCFIteration(
    const CartesianGrid& grid,
    const Molecule& molecule,
    const XCFunctional& functional,
    const std::vector<double>& nuclearPotential,
    const std::vector<double>& alphaDensity,
    const std::vector<double>& betaDensity,
    int alphaElectrons,
    int betaElectrons,
    bool closedShell,
    const std::vector<MolecularOrbital>& previousAlphaOrbitals,
    const std::vector<MolecularOrbital>& previousBetaOrbitals
)
{
    if (alphaDensity.size() != grid.getSize() ||
        betaDensity.size() != grid.getSize()) {

        throw std::invalid_argument(
            "Las densidades moleculares y la malla deben tener el mismo tamano."
        );
    }

    if (nuclearPotential.size() != grid.getSize()) {
        throw std::invalid_argument(
            "El potencial nuclear y la malla deben tener el mismo tamano."
        );
    }

    const std::vector<int> alphaOccupations =
        buildSpinOccupations(alphaElectrons);

    const std::vector<int> betaOccupations =
        buildSpinOccupations(betaElectrons);

    MolecularSCFIterationResult iteration;

    iteration.density.resize(
        grid.getSize(),
        0.0
    );

    for (std::size_t i = 0;
         i < grid.getSize();
         ++i) {

        iteration.density[i] =
            alphaDensity[i] +
            betaDensity[i];
    }

    iteration.hartreePotential =
        calculateHartreePotential(
            grid,
            iteration.density
        );

    iteration.alphaXCPotential =
        calculateMolecularSpinExchangeCorrelationPotential(
            functional,
            grid,
            alphaDensity,
            betaDensity,
            0
        );

    if (closedShell) {
        iteration.betaXCPotential =
            iteration.alphaXCPotential;
    }
    else {
        iteration.betaXCPotential =
            calculateMolecularSpinExchangeCorrelationPotential(
                functional,
                grid,
                alphaDensity,
                betaDensity,
                1
            );
    }

    iteration.alphaPotential.resize(
        grid.getSize(),
        0.0
    );

    iteration.betaPotential.resize(
        grid.getSize(),
        0.0
    );

    for (std::size_t i = 0;
         i < grid.getSize();
         ++i) {

        iteration.alphaPotential[i] =
            nuclearPotential[i] +
            iteration.hartreePotential[i] +
            iteration.alphaXCPotential[i];

        iteration.betaPotential[i] =
            nuclearPotential[i] +
            iteration.hartreePotential[i] +
            iteration.betaXCPotential[i];
    }

    iteration.alphaOrbitals =
        solveMolecularOrbitalsSilently(
            grid,
            iteration.alphaPotential,
            alphaOccupations.size(),
            alphaOccupations,
            SpinChannel::Alpha,
            previousAlphaOrbitals
        );

    if (closedShell) {
        iteration.betaOrbitals =
            convertOrbitalsToSpin(
                iteration.alphaOrbitals,
                SpinChannel::Beta
            );
    }
    else {
        iteration.betaOrbitals =
            solveMolecularOrbitalsSilently(
                grid,
                iteration.betaPotential,
                betaOccupations.size(),
                betaOccupations,
                SpinChannel::Beta,
                previousBetaOrbitals
            );
    }

    iteration.orbitals.reserve(
        iteration.alphaOrbitals.size() +
        iteration.betaOrbitals.size()
    );

    iteration.orbitals.insert(
        iteration.orbitals.end(),
        iteration.alphaOrbitals.begin(),
        iteration.alphaOrbitals.end()
    );

    iteration.orbitals.insert(
        iteration.orbitals.end(),
        iteration.betaOrbitals.begin(),
        iteration.betaOrbitals.end()
    );

    iteration.outputAlphaDensity =
        calculateMolecularSpinDensity(
            iteration.alphaOrbitals,
            SpinChannel::Alpha
        );

    iteration.outputBetaDensity =
        calculateMolecularSpinDensity(
            iteration.betaOrbitals,
            SpinChannel::Beta
        );

    iteration.outputDensity.resize(
        grid.getSize(),
        0.0
    );

    for (std::size_t i = 0;
         i < grid.getSize();
         ++i) {

        iteration.outputDensity[i] =
            iteration.outputAlphaDensity[i] +
            iteration.outputBetaDensity[i];
    }

    iteration.outputHartreePotential =
        calculateHartreePotential(
            grid,
            iteration.outputDensity
        );

    iteration.energy =
        calculateMolecularTotalEnergy(
            functional,
            molecule,
            grid,
            iteration.outputAlphaDensity,
            iteration.outputBetaDensity,
            iteration.orbitals,
            iteration.outputHartreePotential,
            nuclearPotential
        );

    std::vector<double> oldDensity(
        grid.getSize(),
        0.0
    );

    for (std::size_t i = 0;
         i < grid.getSize();
         ++i) {

        oldDensity[i] =
            alphaDensity[i] +
            betaDensity[i];
    }

    iteration.densityDifference =
        calculateMolecularDensityDifference(
            grid,
            oldDensity,
            iteration.outputDensity
        );

    iteration.spinDensityDifference =
        calculateMolecularSpinDensityDifference(
            grid,
            alphaDensity,
            betaDensity,
            iteration.outputAlphaDensity,
            iteration.outputBetaDensity
        );

    iteration.effectiveDensityDifference =
        std::max(
            iteration.densityDifference,
            iteration.spinDensityDifference
        );

    iteration.residual =
        calculateMaximumMolecularKSResidual(
            grid,
            iteration.alphaPotential,
            iteration.betaPotential,
            iteration.orbitals
        );

    return iteration;
}