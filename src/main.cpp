#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "BasisSet.h"
#include "DFTResults.h"
#include "IntegralEngine.h"
#include "KohnSham.h"
#include "MolecularSystem.h"
#include "SCFSolver.h"

int main()
{
    try
    {
        std::cout
            << "========================================\n"
            << " DFT - MVP\n"
            << "========================================\n\n";

        // ------------------------------------------------------------
        // 1. MolecularSystem
        // ------------------------------------------------------------

        MolecularSystem molecule(
            {
                {
                    1,
                    0.0,
                    0.0,
                    0.0
                }
            },
            0,
            2
        );

        std::cout
            << "MolecularSystem\n"
            << "----------------------------------------\n"
            << "Formula: "
            << molecule.getFormula()
            << '\n'
            << "Atoms: "
            << molecule.getAtomCount()
            << '\n'
            << "Nuclear charge: "
            << molecule.getTotalNuclearCharge()
            << '\n'
            << "Electrons: "
            << molecule.getElectronCount()
            << '\n'
            << "Alpha electrons: "
            << molecule.getAlphaElectronCount()
            << '\n'
            << "Beta electrons: "
            << molecule.getBetaElectronCount()
            << '\n'
            << "Multiplicity: "
            << molecule.getMultiplicity()
            << "\n\n";

        // ------------------------------------------------------------
        // 2. BasisSet
        // ------------------------------------------------------------

        BasisSet basisSet;

        basisSet.initialize(
            molecule
        );

        basisSet.addFunction(
            0,
            0,
            0,
            0,
            {
                {
                    1.0,
                    1.0
                }
            },
            "1s"
        );

        std::cout
            << "BasisSet\n"
            << "----------------------------------------\n"
            << "Basis functions: "
            << basisSet.getFunctionCount()
            << '\n'
            << "Function 0: "
            << basisSet.getFunction(0).label
            << "\n\n";

        // ------------------------------------------------------------
        // 3. IntegralEngine
        // ------------------------------------------------------------

        IntegralEngine integralEngine;

        integralEngine.initialize(
            basisSet
        );

        std::cout
            << "IntegralEngine\n"
            << "----------------------------------------\n"
            << "Calculating integrals...\n";

        integralEngine.calculate();

        const auto& overlap =
            integralEngine.getOverlapMatrix();

        const auto& kinetic =
            integralEngine.getKineticMatrix();

        const auto& nuclearAttraction =
            integralEngine.getNuclearAttractionMatrix();

        const auto& twoElectron =
            integralEngine.getTwoElectronIntegrals();

        std::cout
            << "Integrals calculated.\n\n";

        std::cout
            << std::setprecision(12)
            << "Overlap S(0,0): "
            << overlap[0][0]
            << '\n'
            << "Kinetic T(0,0): "
            << kinetic[0][0]
            << '\n'
            << "Nuclear attraction V(0,0): "
            << nuclearAttraction[0][0]
            << '\n'
            << "Two-electron (00|00): "
            << twoElectron(0, 0, 0, 0)
            << '\n'
            << "Nuclear repulsion: "
            << integralEngine.getNuclearRepulsionEnergy()
            << " Eh\n\n";

        // ------------------------------------------------------------
        // 4. KohnSham
        // ------------------------------------------------------------

        KohnSham kohnSham;

        kohnSham.initialize(
            molecule,
            integralEngine
        );

        std::cout
            << "KohnSham\n"
            << "----------------------------------------\n"
            << "Kohn-Sham module initialized.\n\n";

        // ------------------------------------------------------------
        // 5. SCFSolver
        // ------------------------------------------------------------

        SCFSolver scfSolver;

        scfSolver.initialize(
            molecule,
            kohnSham
        );

        scfSolver.setMaxIterations(
            100
        );

        scfSolver.setEnergyTolerance(
            1.0e-8
        );

        scfSolver.setDensityTolerance(
            1.0e-6
        );

        std::cout
            << "SCFSolver\n"
            << "----------------------------------------\n"
            << "Starting SCF...\n";

        const SCFSolver::Result scfResult =
            scfSolver.solve();

        std::cout
            << "SCF finished.\n\n";

        // ------------------------------------------------------------
        // 6. DFTResults
        // ------------------------------------------------------------

        DFTResults results;

        results.setConverged(
            scfResult.converged
        );

        results.setIterations(
            scfResult.iterations
        );

        results.setTotalEnergy(
            scfResult.totalEnergy
        );

        results.setElectronicEnergy(
            scfResult.electronicEnergy
        );

        results.setNuclearRepulsionEnergy(
            scfResult.nuclearRepulsionEnergy
        );

        results.setEnergyChange(
            scfResult.energyChange
        );

        results.setDensityChange(
            scfResult.densityChange
        );

        results.setOrbitalEnergies(
            scfResult.orbitalEnergies
        );

        results.setCoefficients(
            scfResult.coefficients
        );

        results.setDensityMatrix(
            scfResult.densityMatrix
        );

        // ------------------------------------------------------------
        // 7. Results
        // ------------------------------------------------------------

        std::cout
            << "DFTResults\n"
            << "========================================\n"
            << "Converged: "
            << (results.isConverged() ? "yes" : "no")
            << '\n'
            << "Iterations: "
            << results.getIterations()
            << '\n'
            << "Electronic energy: "
            << results.getElectronicEnergy()
            << " Eh\n"
            << "Nuclear repulsion: "
            << results.getNuclearRepulsionEnergy()
            << " Eh\n"
            << "Total energy: "
            << results.getTotalEnergy()
            << " Eh\n"
            << "Energy change: "
            << results.getEnergyChange()
            << " Eh\n"
            << "Density change: "
            << results.getDensityChange()
            << "\n\n";

        std::cout
            << "Orbital energies\n"
            << "----------------------------------------\n";

        for (std::size_t i = 0;
             i < results.getOrbitalCount();
             ++i)
        {
            std::cout
                << "Orbital "
                << i
                << ": "
                << results.getOrbitalEnergy(i)
                << " Eh\n";
        }

        std::cout
            << "\nDensity matrix\n"
            << "----------------------------------------\n";

        const auto& density =
            results.getDensityMatrix();

        for (const auto& row : density)
        {
            for (double value : row)
            {
                std::cout
                    << std::setw(16)
                    << value;
            }

            std::cout << '\n';
        }

        std::cout
            << "\n========================================\n"
            << " DFT MVP TEST FINISHED\n"
            << "========================================\n";
    }
    catch (const std::exception& exception)
    {
        std::cerr
            << "\n========================================\n"
            << " DFT ERROR\n"
            << "========================================\n"
            << exception.what()
            << '\n';

        return 1;
    }

    return 0;
}