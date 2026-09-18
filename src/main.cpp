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
        // PRIMEROS 15 ATOMOS - PZ81
        // ============================================================

        std::cout
            << "============================================================\n"
            << "PRIMEROS 5 ATOMOS - PZ81\n"
            << "============================================================\n";

        for (int Z = 1; Z <= 5; ++Z)
        {
            const AtomicConfiguration configuration =
                getAtomicConfiguration(Z);

            const AtomicResult result =
                solveAtom(
                    grid,
                    configuration,
                    pz81
                );

            std::cout
                << configuration.symbol
                << " | PZ81 | E = "
                << std::scientific
                << std::setprecision(10)
                << result.scf.energy.total
                << " Ha\n";
        }

        // ============================================================
        // PRIMEROS 15 ATOMOS - PBE96
        // ============================================================

        std::cout
            << "\n"
            << "============================================================\n"
            << "PRIMEROS 5 ATOMOS - PBE96\n"
            << "============================================================\n";

        for (int Z = 1; Z <= 5; ++Z)
        {
            const AtomicConfiguration configuration =
                getAtomicConfiguration(Z);

            const AtomicResult result =
                solveAtom(
                    grid,
                    configuration,
                    pbe96
                );

            std::cout
                << configuration.symbol
                << " | PBE96 | E = "
                << std::scientific
                << std::setprecision(10)
                << result.scf.energy.total
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

        CartesianGrid molecularGrid(
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
                molecularGrid,
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
        //
        // Geometria aproximada:
        //
        // O  = ( 0.000,  0.000, 0.000) bohr
        // H1 = ( 1.430,  0.000, 1.108) bohr
        // H2 = (-1.430,  0.000, 1.108) bohr
        //
        // Carga molecular = 0
        // Electrones = 10
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

        CartesianGrid waterGrid(
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
                waterGrid,
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
        // MOLECULA N2 - PZ81
        //
        // Geometria lineal aproximada:
        //
        // N1 = (-1.037, 0.000, 0.000) bohr
        // N2 = ( 1.037, 0.000, 0.000) bohr
        //
        // Carga molecular = 0
        // Electrones = 14
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
        // MOLECULA CO2 - PZ81
        //
        // Geometria lineal aproximada:
        //
        // O1 = (-2.192, 0.000, 0.000) bohr
        // C  = ( 0.000, 0.000, 0.000) bohr
        // O2 = ( 2.192, 0.000, 0.000) bohr
        //
        // Distancia C-O ~ 1.16 A
        // Angulo O-C-O = 180 grados
        //
        // Carga molecular = 0
        // Electrones = 22
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
        // MOLECULA O2 - PZ81
        //
        // Geometria lineal aproximada:
        //
        // O1 = (-1.141, 0.000, 0.000) bohr
        // O2 = ( 1.141, 0.000, 0.000) bohr
        //
        // Carga molecular = 0
        // Electrones = 16
        // ============================================================

        Molecule o2(0);

        o2.addNucleus(
            8,
            -1.141,
            0.0,
            0.0
        );

        o2.addNucleus(
            8,
            1.141,
            0.0,
            0.0
        );

        CartesianGrid o2Grid(
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