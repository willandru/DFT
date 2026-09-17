#include "AtomicDFT.h"
#include "CartesianGrid.h"
#include "DFTConstants.h"
#include "ElectronicConfiguration.h"
#include "Molecule.h"
#include "PBE96.h"
#include "PZ81.h"
#include "RadialGrid.h"
#include "SelfConsistentFieldMolecule.h"

#include <iomanip>
#include <iostream>

int main()
{
    try
    {
        RadialGrid grid(
            DFTConstants::GRID_POINTS,
            DFTConstants::RMAX
        );

        PZ81 pz81;
        PBE96 pbe96;

        // ============================================================
        // ATOMOS Z = 1 - 10
        // PZ81 + PBE96
        // ============================================================

        for (int Z = 1; Z <= 10; ++Z)
        {
            const AtomicConfiguration configuration =
                getAtomicConfiguration(Z);

            const AtomicResult pz81Result =
                solveAtom(
                    grid,
                    configuration,
                    pz81
                );

            const AtomicResult pbe96Result =
                solveAtom(
                    grid,
                    configuration,
                    pbe96
                );

            std::cout
                << "Z = "
                << Z
                << " | PZ81  | E = "
                << std::scientific
                << std::setprecision(10)
                << pz81Result.scf.energy.total
                << " Ha\n";

            std::cout
                << "Z = "
                << Z
                << " | PBE96 | E = "
                << std::scientific
                << std::setprecision(10)
                << pbe96Result.scf.energy.total
                << " Ha\n";
        }

        // ============================================================
        // MOLECULA H2 - PZ81
        // ============================================================

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

        CartesianGrid h2Grid(
            7,
            7,
            7,
            -8.1,
            7.9,
            -8.1,
            7.9,
            -8.1,
            7.9
        );

        const MolecularResult h2PZ81 =
            solveMolecularSelfConsistentField(
                h2Grid,
                h2,
                h2.getCharge(),
                pz81
            );

        std::cout
            << "\n"
            << "H2 | PZ81 | E = "
            << std::scientific
            << std::setprecision(10)
            << h2PZ81.scf.energy.total
            << " Ha\n";

        // ============================================================
        // MOLECULA H2O - PZ81
        // ============================================================

        Molecule h2o(0);

        h2o.addNucleus(
            8,
            0.0,
            0.0,
            0.0
        );

        h2o.addNucleus(
            1,
            1.430,
            0.0,
            1.108
        );

        h2o.addNucleus(
            1,
            -1.430,
            0.0,
            1.108
        );

        CartesianGrid h2oGrid(
            9,
            9,
            9,
            -8.1,
            7.9,
            -8.1,
            7.9,
            -8.1,
            7.9
        );

        const MolecularResult h2oPZ81 =
            solveMolecularSelfConsistentField(
                h2oGrid,
                h2o,
                h2o.getCharge(),
                pz81
            );

        std::cout
            << "\n"
            << "H2O | PZ81 | E = "
            << std::scientific
            << std::setprecision(10)
            << h2oPZ81.scf.energy.total
            << " Ha\n";

        // ============================================================
        // MOLECULA CO2 - PZ81
        // ============================================================

        Molecule co2(0);

        co2.addNucleus(
            6,
            0.0,
            0.0,
            0.0
        );

        co2.addNucleus(
            8,
            -2.192,
            0.0,
            0.0
        );

        co2.addNucleus(
            8,
            2.192,
            0.0,
            0.0
        );

        CartesianGrid co2Grid(
            15,
            15,
            15,
            -8.1,
            7.9,
            -8.1,
            7.9,
            -8.1,
            7.9
        );

        const MolecularResult co2PZ81 =
            solveMolecularSelfConsistentField(
                co2Grid,
                co2,
                co2.getCharge(),
                pz81
            );

        std::cout
            << "\n"
            << "CO2 | PZ81 | E = "
            << std::scientific
            << std::setprecision(10)
            << co2PZ81.scf.energy.total
            << " Ha\n";

        // ============================================================
        // MOLECULA N2 - PZ81
        // ============================================================

        Molecule n2(0);

        n2.addNucleus(
            7,
            -1.037,
            0.0,
            0.0
        );

        n2.addNucleus(
            7,
            1.037,
            0.0,
            0.0
        );

        CartesianGrid n2Grid(
            9,
            9,
            9,
            -8.1,
            7.9,
            -8.1,
            7.9,
            -8.1,
            7.9
        );

        const MolecularResult n2PZ81 =
            solveMolecularSelfConsistentField(
                n2Grid,
                n2,
                n2.getCharge(),
                pz81
            );

        std::cout
            << "\n"
            << "N2 | PZ81 | E = "
            << std::scientific
            << std::setprecision(10)
            << n2PZ81.scf.energy.total
            << " Ha\n";

        // ============================================================
        // MOLECULA O2 - PZ81
        // ============================================================

        Molecule o2(0);

        o2.addNucleus(
            8,
            -1.14,
            0.0,
            0.0
        );

        o2.addNucleus(
            8,
            1.14,
            0.0,
            0.0
        );

        CartesianGrid o2Grid(
            9,
            9,
            9,
            -8.1,
            7.9,
            -8.1,
            7.9,
            -8.1,
            7.9
        );

        const MolecularResult o2PZ81 =
            solveMolecularSelfConsistentField(
                o2Grid,
                o2,
                o2.getCharge(),
                pz81
            );

        std::cout
            << "\n"
            << "O2 | PZ81 | E = "
            << std::scientific
            << std::setprecision(10)
            << o2PZ81.scf.energy.total
            << " Ha\n";

        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "ERROR: "
            << error.what()
            << '\n';

        return 1;
    }
    catch (...)
    {
        std::cerr
            << "ERROR: excepcion desconocida.\n";

        return 1;
    }
}