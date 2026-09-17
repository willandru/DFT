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
        // ATOMO DE HIDROGENO - PZ81
        // ============================================================

        const AtomicConfiguration hydrogen =
            getAtomicConfiguration(1);

        const AtomicResult hPZ81 =
            solveAtom(
                grid,
                hydrogen,
                pz81
            );

        std::cout
            << "H | PZ81  | E = "
            << std::scientific
            << std::setprecision(10)
            << hPZ81.scf.energy.total
            << " Ha\n";

        // ============================================================
        // ATOMO DE HIDROGENO - PBE96
        // ============================================================

        const AtomicResult hPBE96 =
            solveAtom(
                grid,
                hydrogen,
                pbe96
            );

        std::cout
            << "H | PBE96 | E = "
            << std::scientific
            << std::setprecision(10)
            << hPBE96.scf.energy.total
            << " Ha\n";

        // ============================================================
        // ATOMO DE OXIGENO - PZ81
        // ============================================================

        const AtomicConfiguration oxygen =
            getAtomicConfiguration(8);

        const AtomicResult oPZ81 =
            solveAtom(
                grid,
                oxygen,
                pz81
            );

        std::cout
            << "O | PZ81  | E = "
            << std::scientific
            << std::setprecision(10)
            << oPZ81.scf.energy.total
            << " Ha\n";

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