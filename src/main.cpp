#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "BasisSet.h"
#include "DFTResults.h"
#include "IntegralEngine.h"
#include "KohnSham.h"
#include "MolecularSystem.h"
#include "SCFSolver.h"

namespace
{
    struct TestCase
    {
        std::string name;
        int atomicNumber;
        int charge;
        int multiplicity;
    };

    void printSeparator()
    {
        std::cout
            << "------------------------------------------------------\n";
    }

    void runTest(const TestCase& test)
    {
        std::cout
            << "\n======================================================\n"
            << " " << test.name << " TEST\n"
            << "======================================================\n\n";

        MolecularSystem molecule(
            {
                {
                    test.atomicNumber,
                    0.0,
                    0.0,
                    0.0
                }
            },
            test.charge,
            test.multiplicity
        );

        std::cout
            << "MolecularSystem\n";

        printSeparator();

        std::cout
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

        BasisSet basisSet;

        basisSet.initialize(
            molecule
        );

        /*
            Minimal validation basis.

            The basis depends only on the atomic number.
            The DFT modules remain independent of the
            particular test element.
        */

        if (test.atomicNumber == 1)
        {
            basisSet.addFunction(
                0,
                0,
                0,
                0,
                {
                    {3.42525091, 0.15432897},
                    {0.62391373, 0.53532814},
                    {0.16885540, 0.44463454}
                },
                "1s"
            );
        }
        else if (test.atomicNumber == 2)
        {
            basisSet.addFunction(
                0,
                0,
                0,
                0,
                {
                    {6.36242139, 0.15432897},
                    {1.15892300, 0.53532814},
                    {0.31364979, 0.44463454}
                },
                "1s"
            );
        }
        else if (test.atomicNumber == 3)
        {
            basisSet.addFunction(
                0,
                0,
                0,
                0,
                {
                    {16.1195750, 0.15432897},
                    {2.9362007, 0.53532814},
                    {0.7946505, 0.44463454}
                },
                "1s"
            );

            basisSet.addFunction(
                0,
                0,
                0,
                0,
                {
                    {0.6362897, -0.09996723},
                    {0.1478601, 0.39951283},
                    {0.0480887, 0.70011547}
                },
                "2s"
            );
        }
        else if (test.atomicNumber == 6)
        {
            basisSet.addFunction(
                0,
                0,
                0,
                0,
                {
                    {71.6168370, 0.15432897},
                    {13.0450960, 0.53532814},
                    {3.5305122, 0.44463454}
                },
                "1s"
            );

            basisSet.addFunction(
                0,
                0,
                0,
                0,
                {
                    {2.9412494, -0.09996723},
                    {0.6834831, 0.39951283},
                    {0.2222899, 0.70011547}
                },
                "2s"
            );

            basisSet.addFunction(
                0,
                1,
                0,
                0,
                {
                    {2.9412494, 0.15591627},
                    {0.6834831, 0.60768372},
                    {0.2222899, 0.39195739}
                },
                "2px"
            );

            basisSet.addFunction(
                0,
                0,
                1,
                0,
                {
                    {2.9412494, 0.15591627},
                    {0.6834831, 0.60768372},
                    {0.2222899, 0.39195739}
                },
                "2py"
            );

            basisSet.addFunction(
                0,
                0,
                0,
                1,
                {
                    {2.9412494, 0.15591627},
                    {0.6834831, 0.60768372},
                    {0.2222899, 0.39195739}
                },
                "2pz"
            );
        }
        else
        {
            throw std::runtime_error(
                "Unsupported atomic number in test."
            );
        }

        std::cout
            << "BasisSet\n";

        printSeparator();

        std::cout
            << "Basis functions: "
            << basisSet.getFunctionCount()
            << '\n';

        for (std::size_t i = 0;
             i < basisSet.getFunctionCount();
             ++i)
        {
            const auto& function =
                basisSet.getFunction(i);

            std::cout
                << "Function "
                << i
                << ": "
                << function.label
                << " ("
                << function.angularMomentumX
                << ","
                << function.angularMomentumY
                << ","
                << function.angularMomentumZ
                << ")\n";
        }

        std::cout << '\n';

        IntegralEngine integralEngine;

        integralEngine.initialize(
            basisSet
        );

        std::cout
            << "IntegralEngine\n";

        printSeparator();

        std::cout
            << "Calculating analytic integrals...\n";

        integralEngine.calculate();

        std::cout
            << "Integrals calculated.\n\n";

        KohnSham kohnSham;

        kohnSham.initialize(
            molecule,
            integralEngine
        );

        std::cout
            << "KohnSham\n";

        printSeparator();

        std::cout
            << "Kohn-Sham module initialized.\n\n";

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
            << "SCFSolver\n";

        printSeparator();

        std::cout
            << "Starting SCF...\n";

        const SCFSolver::Result scfResult =
            scfSolver.solve();

        std::cout
            << "SCF finished.\n\n";

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

        std::cout
            << "DFTResults\n";

        printSeparator();

        std::cout
            << std::setprecision(12)
            << "Converged: "
            << (
                results.isConverged()
                    ? "yes"
                    : "no"
            )
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
            << "Orbital energies\n";

        printSeparator();

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
            << "\n";
    }
}

int main()
{
    try
    {
        std::cout
            << "======================================================\n"
            << " DFT - MVP\n"
            << " Atomic Validation Suite\n"
            << "======================================================\n";

        const std::vector<TestCase> tests =
        {
            {"Hydrogen", 1, 0, 2},
            {"Helium",   2, 0, 1},
            {"Lithium",  3, 0, 2},
            {"Carbon",   6, 0, 3}
        };

        for (const TestCase& test : tests)
        {
            runTest(test);
        }

        std::cout
            << "\n======================================================\n"
            << " ALL ATOMIC TESTS FINISHED\n"
            << "======================================================\n";
    }
    catch (const std::exception& exception)
    {
        std::cerr
            << "\n======================================================\n"
            << " DFT ERROR\n"
            << "======================================================\n"
            << exception.what()
            << '\n';

        return 1;
    }

    return 0;
}