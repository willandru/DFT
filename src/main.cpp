#include "AtomicDFT.h"
#include "CartesianGrid.h"
#include "DFTData.h"
#include "Molecule.h"
#include "PBE96.h"
#include "PZ81.h"
#include "RadialGrid.h"
#include "SelfConsistentField.h"
#include "SelfConsistentFieldMolecule.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct FunctionalResults
{
    std::string name;
    std::vector<AtomicResult> results;
};

/*
 * ================================================================
 * COMMON OUTPUT
 * ================================================================
 */

void printEnergyComponents(
    const EnergyComponents& energy)
{
    std::cout << std::fixed << std::setprecision(10);

    std::cout << "  Kinetic:          "
              << energy.kinetic << '\n';

    std::cout << "  External:         "
              << energy.external << '\n';

    std::cout << "  Hartree:          "
              << energy.hartree << '\n';

    std::cout << "  Exchange-corr.:   "
              << energy.exchangeCorrelation << '\n';

    std::cout << "  Nuclear rep.:     "
              << energy.nuclearRepulsion << '\n';

    std::cout << "  Total:            "
              << energy.total << '\n';
}

void printEnergyVerification(
    const EnergyComponents& energy)
{
    const double sum =
        energy.kinetic +
        energy.external +
        energy.hartree +
        energy.exchangeCorrelation +
        energy.nuclearRepulsion;

    const double difference =
        std::abs(
            sum -
            energy.total
        );

    std::cout << "  Energy sum:       "
              << sum << '\n';

    std::cout << "  Verification:     "
              << std::scientific
              << difference << '\n';
}

/*
 * ================================================================
 * ATOMIC DFT
 * ================================================================
 */

void printAtomicSCFDiagnostics(
    const AtomicResult& result)
{
    const SCFResult& scf =
        result.scf;

    std::cout << "  SCF iterations: "
              << scf.iterations << '\n';

    std::cout << "  Converged:       "
              << (scf.converged ? "yes" : "no")
              << '\n';

    std::cout << "  Density diff:    "
              << std::scientific
              << scf.densityDifference
              << '\n';

    std::cout << "  Energy diff:     "
              << std::scientific
              << scf.energyDifference
              << '\n';

    std::cout << "  Max KS residual: "
              << std::scientific
              << scf.maxKSResidual
              << '\n';
}

AtomicConfiguration buildZincConfiguration()
{
    AtomicConfiguration configuration;

    configuration.Z = 30;
    configuration.symbol = "Zn";

    configuration.states = {
        {1, 0, 1, 1},
        {2, 0, 1, 1},
        {2, 1, 3, 3},
        {3, 0, 1, 1},
        {3, 1, 3, 3},
        {4, 0, 1, 0}
    };

    return configuration;
}

AtomicResult runZinc(
    const RadialGrid& grid,
    const XCFunctional& functional)
{
    const AtomicConfiguration configuration =
        buildZincConfiguration();

    return solveAtom(
        grid,
        configuration,
        functional
    );
}

void printAtomicResult(
    const std::string& functionalName,
    const AtomicResult& result)
{
    std::cout
        << "\n========================================\n";

    std::cout
        << "Atomic DFT: Zn\n";

    std::cout
        << "Functional: "
        << functionalName
        << '\n';

    std::cout
        << "========================================\n";

    std::cout
        << "  Z:                "
        << result.Z
        << '\n';

    std::cout
        << "  Symbol:           "
        << result.symbol
        << '\n';

    std::cout
        << "  Electrons:        "
        << result.electrons
        << '\n';

    printAtomicSCFDiagnostics(
        result
    );

    std::cout
        << "\n  Energy components:\n";

    printEnergyComponents(
        result.scf.energy
    );

    std::cout
        << "\n  Energy verification:\n";

    printEnergyVerification(
        result.scf.energy
    );
}

void runAtomicTests()
{
    std::cout << "\n";
    std::cout
        << "########################################\n";

    std::cout
        << "# ATOMIC DFT TEST\n";

    std::cout
        << "########################################\n";

    const RadialGrid radialGrid(
        2000,
        40.0
    );

    PZ81 pz81;
    PBE96 pbe96;

    const AtomicResult zincPZ81 =
        runZinc(
            radialGrid,
            pz81
        );

    const AtomicResult zincPBE96 =
        runZinc(
            radialGrid,
            pbe96
        );

    printAtomicResult(
        "PZ81",
        zincPZ81
    );

    printAtomicResult(
        "PBE96",
        zincPBE96
    );
}

/*
 * ================================================================
 * MOLECULAR DFT
 * ================================================================
 */

void printMolecularGeometry(
    const std::string& name,
    const Molecule& molecule)
{
    std::cout
        << "\n========================================\n";

    std::cout
        << "Molecular DFT: "
        << name
        << '\n';

    std::cout
        << "========================================\n";

    std::cout
        << "Nuclei:    "
        << molecule.getNucleusCount()
        << '\n';

    std::cout
        << "Charge:    "
        << molecule.getCharge()
        << '\n';

    std::cout
        << "Electrons: "
        << molecule.getElectronCount()
        << '\n';

    std::cout
        << "\nGeometry (bohr):\n";

    const auto& nuclei =
        molecule.getNuclei();

    for (std::size_t i = 0;
         i < nuclei.size();
         ++i)
    {
        std::cout
            << "  Nucleus "
            << i
            << ": Z="
            << nuclei[i].atomicNumber
            << "  ("
            << nuclei[i].position[0]
            << ", "
            << nuclei[i].position[1]
            << ", "
            << nuclei[i].position[2]
            << ")\n";
    }
}

void printMolecularResult(
    const std::string& functionalName,
    const MolecularResult& result)
{
    std::cout
        << "\nFunctional: "
        << functionalName
        << '\n';

    std::cout
        << "  Electrons:        "
        << result.electrons
        << '\n';

    std::cout
        << "  SCF iterations:   "
        << result.scf.iterations
        << '\n';

    std::cout
        << "  Converged:        "
        << (result.scf.converged ? "yes" : "no")
        << '\n';

    std::cout
        << "  Density diff:     "
        << std::scientific
        << result.scf.densityDifference
        << '\n';

    std::cout
        << "  Energy diff:      "
        << std::scientific
        << result.scf.energyDifference
        << '\n';

    std::cout
        << "  Max KS residual:  "
        << std::scientific
        << result.scf.maxKSResidual
        << '\n';

    std::cout
        << "\n  Energy components:\n";

    printEnergyComponents(
        result.scf.energy
    );

    std::cout
        << "\n  Energy verification:\n";

    printEnergyVerification(
        result.scf.energy
    );

    std::cout
        << "\n  Molecular orbitals:\n";

    for (std::size_t i = 0;
         i < result.scf.molecularOrbitals.size();
         ++i)
    {
        const MolecularOrbital& orbital =
            result.scf.molecularOrbitals[i];

        std::cout
            << "    MO "
            << i
            << "  electrons="
            << orbital.electrons
            << "  eigenvalue="
            << std::scientific
            << orbital.eigenvalue
            << '\n';
    }
}

void runMolecule(
    const std::string& name,
    const Molecule& molecule,
    const CartesianGrid& grid,
    const XCFunctional& functional,
    const std::string& functionalName)
{
    printMolecularGeometry(
        name,
        molecule
    );

    const MolecularResult result =
        solveMolecularSelfConsistentField(
            grid,
            molecule,
            molecule.getCharge(),
            functional
        );

    printMolecularResult(
        functionalName,
        result
    );
}

void runMolecularTests()
{
    std::cout << "\n\n";
    std::cout
        << "########################################\n";

    std::cout
        << "# MOLECULAR DFT TEST\n";

    std::cout
        << "########################################\n";

    /*
     * Reduced molecular grid for debugging.
     *
     * Current diagnostic:
     * 11 x 11 x 11
     */
    const CartesianGrid molecularGrid(
        11,
        11,
        11,
        -8.1,
        7.9,
        -8.1,
        7.9,
        -8.1,
        7.9
    );

    /*
     * ------------------------------------------------------------
     * H2
     * ------------------------------------------------------------
     */

    Molecule h2(0);

    h2.addNucleus(
        1,
        -0.7,
        0.0,
        0.0
    );

    h2.addNucleus(
        1,
        0.7,
        0.0,
        0.0
    );

    /*
     * ------------------------------------------------------------
     * H2O
     * ------------------------------------------------------------
     */

    Molecule h2o(0);

    h2o.addNucleus(
        8,
        0.0,
        0.0,
        0.0
    );

    h2o.addNucleus(
        1,
        1.43,
        0.0,
        1.107
    );

    h2o.addNucleus(
        1,
        -1.43,
        0.0,
        1.107
    );

    /*
     * ------------------------------------------------------------
     * CO2
     * ------------------------------------------------------------
     */

    Molecule co2(0);

    co2.addNucleus(
        8,
        -2.20,
        0.0,
        0.0
    );

    co2.addNucleus(
        6,
        0.0,
        0.0,
        0.0
    );

    co2.addNucleus(
        8,
        2.20,
        0.0,
        0.0
    );

    /*
     * ------------------------------------------------------------
     * N2
     * ------------------------------------------------------------
     */

    Molecule n2(0);

    n2.addNucleus(
        7,
        -1.04,
        0.0,
        0.0
    );

    n2.addNucleus(
        7,
        1.04,
        0.0,
        0.0
    );

    PZ81 pz81;
    PBE96 pbe96;

    /*
     * ------------------------------------------------------------
     * H2
     * ------------------------------------------------------------
     */

    runMolecule(
        "H2",
        h2,
        molecularGrid,
        pz81,
        "PZ81"
    );

    runMolecule(
        "H2",
        h2,
        molecularGrid,
        pbe96,
        "PBE96"
    );

    /*
     * ------------------------------------------------------------
     * H2O
     * ------------------------------------------------------------
     */

    runMolecule(
        "H2O",
        h2o,
        molecularGrid,
        pz81,
        "PZ81"
    );

    runMolecule(
        "H2O",
        h2o,
        molecularGrid,
        pbe96,
        "PBE96"
    );

    /*
     * ------------------------------------------------------------
     * CO2
     * ------------------------------------------------------------
     */

    runMolecule(
        "CO2",
        co2,
        molecularGrid,
        pz81,
        "PZ81"
    );

    runMolecule(
        "CO2",
        co2,
        molecularGrid,
        pbe96,
        "PBE96"
    );

    /*
     * ------------------------------------------------------------
     * N2
     * ------------------------------------------------------------
     */

    runMolecule(
        "N2",
        n2,
        molecularGrid,
        pz81,
        "PZ81"
    );

    runMolecule(
        "N2",
        n2,
        molecularGrid,
        pbe96,
        "PBE96"
    );
}

/*
 * ================================================================
 * MAIN
 * ================================================================
 */

int main()
{
    try
    {
        std::cout
            << std::setprecision(10);

        /*
         * ========================================================
         * ATOMIC CALCULATIONS
         * ========================================================
         */

        runAtomicTests();

        /*
         * ========================================================
         * MOLECULAR CALCULATIONS
         * ========================================================
         */

        runMolecularTests();

        /*
         * ========================================================
         * COMPLETION
         * ========================================================
         */

        std::cout << "\n";
        std::cout
            << "########################################\n";

        std::cout
            << "# DFT TEST COMPLETED\n";

        std::cout
            << "########################################\n";

        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr
            << "\nERROR: "
            << exception.what()
            << '\n';

        return 1;
    }
}