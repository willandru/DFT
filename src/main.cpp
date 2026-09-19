#include "CartesianGrid.h"
#include "Molecule.h"
#include "PZ81.h"
#include "SelfConsistentFieldMolecule.h"

#include <iomanip>
#include <iostream>

int main()
{
    try
    {
        PZ81 pz81;

        std::cout
            << "============================================================\n"
            << "MOLECULAS ORGANICAS - PZ81\n"
            << "============================================================\n";

        // ============================================================
        // CH2O - FORMALDEHIDO
        //
        // Geometria aproximadamente trigonal plana:
        //
        //       H
        //       |
        // H - C = O
        //
        // C  = ( 0.000,  0.000,  0.000)
        // O  = ( 0.000,  0.000,  2.300)
        // H1 = ( 1.770,  0.000, -0.590)
        // H2 = (-1.770,  0.000, -0.590)
        //
        // Electrones = 16
        // Carga = 0
        // Multiplicidad = 1
        // ============================================================

        Molecule ch2o(0);

        ch2o.addNucleus(
            6,
            0.000,
            0.000,
            0.000
        );

        ch2o.addNucleus(
            8,
            0.000,
            0.000,
            2.300
        );

        ch2o.addNucleus(
            1,
            1.770,
            0.000,
            -0.590
        );

        ch2o.addNucleus(
            1,
            -1.770,
            0.000,
            -0.590
        );

        CartesianGrid ch2oGrid(
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

        const MolecularResult ch2oPZ81 =
            solveMolecularSelfConsistentField(
                ch2oGrid,
                ch2o,
                ch2o.getCharge(),
                1,
                pz81
            );

        std::cout
            << "\nCH2O | PZ81 | E = "
            << std::scientific
            << std::setprecision(10)
            << ch2oPZ81.scf.energy.total
            << " Ha\n";

        // ============================================================
        // HCN - CIANURO DE HIDROGENO
        //
        // Geometria lineal:
        //
        // H - C = N
        //
        // H  = (-2.000, 0.000, 0.000)
        // C  = ( 0.000, 0.000, 0.000)
        // N  = ( 2.180, 0.000, 0.000)
        //
        // Electrones = 14
        // Carga = 0
        // Multiplicidad = 1
        // ============================================================

        Molecule hcn(0);

        hcn.addNucleus(
            1,
            -2.000,
            0.000,
            0.000
        );

        hcn.addNucleus(
            6,
            0.000,
            0.000,
            0.000
        );

        hcn.addNucleus(
            7,
            2.180,
            0.000,
            0.000
        );

        CartesianGrid hcnGrid(
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

        const MolecularResult hcnPZ81 =
            solveMolecularSelfConsistentField(
                hcnGrid,
                hcn,
                hcn.getCharge(),
                1,
                pz81
            );

        std::cout
            << "\nHCN | PZ81 | E = "
            << std::scientific
            << std::setprecision(10)
            << hcnPZ81.scf.energy.total
            << " Ha\n";

        // ============================================================
        // C2H2 - ACETILENO
        //
        // Geometria lineal:
        //
        // H - C = C - H
        //
        // H1 = (-3.000, 0.000, 0.000)
        // C1 = (-1.150, 0.000, 0.000)
        // C2 = ( 1.150, 0.000, 0.000)
        // H2 = ( 3.000, 0.000, 0.000)
        //
        // Electrones = 14
        // Carga = 0
        // Multiplicidad = 1
        // ============================================================

        Molecule c2h2(0);

        c2h2.addNucleus(
            1,
            -3.000,
            0.000,
            0.000
        );

        c2h2.addNucleus(
            6,
            -1.150,
            0.000,
            0.000
        );

        c2h2.addNucleus(
            6,
            1.150,
            0.000,
            0.000
        );

        c2h2.addNucleus(
            1,
            3.000,
            0.000,
            0.000
        );

        CartesianGrid c2h2Grid(
            17,
            17,
            17,
            -8.1,
            7.9,
            -8.1,
            7.9,
            -8.1,
            7.9
        );

        const MolecularResult c2h2PZ81 =
            solveMolecularSelfConsistentField(
                c2h2Grid,
                c2h2,
                c2h2.getCharge(),
                1,
                pz81
            );

        std::cout
            << "\nC2H2 | PZ81 | E = "
            << std::scientific
            << std::setprecision(10)
            << c2h2PZ81.scf.energy.total
            << " Ha\n";

        // ============================================================
        // C2H4 - ETENO
        //
        // Geometria plana:
        //
        //        H       H
        //         \     /
        //          C = C
        //         /     \
        //        H       H
        //
        // C1 = (-1.270, 0.000, 0.000)
        // C2 = ( 1.270, 0.000, 0.000)
        //
        // H1 = (-2.300,  1.740, 0.000)
        // H2 = (-2.300, -1.740, 0.000)
        // H3 = ( 2.300,  1.740, 0.000)
        // H4 = ( 2.300, -1.740, 0.000)
        //
        // Electrones = 16
        // Carga = 0
        // Multiplicidad = 1
        // ============================================================

        Molecule c2h4(0);

        c2h4.addNucleus(
            6,
            -1.270,
            0.000,
            0.000
        );

        c2h4.addNucleus(
            6,
            1.270,
            0.000,
            0.000
        );

        c2h4.addNucleus(
            1,
            -2.300,
            1.740,
            0.000
        );

        c2h4.addNucleus(
            1,
            -2.300,
            -1.740,
            0.000
        );

        c2h4.addNucleus(
            1,
            2.300,
            1.740,
            0.000
        );

        c2h4.addNucleus(
            1,
            2.300,
            -1.740,
            0.000
        );

        CartesianGrid c2h4Grid(
            17,
            17,
            17,
            -8.1,
            7.9,
            -8.1,
            7.9,
            -8.1,
            7.9
        );

        const MolecularResult c2h4PZ81 =
            solveMolecularSelfConsistentField(
                c2h4Grid,
                c2h4,
                c2h4.getCharge(),
                1,
                pz81
            );

        std::cout
            << "\nC2H4 | PZ81 | E = "
            << std::scientific
            << std::setprecision(10)
            << c2h4PZ81.scf.energy.total
            << " Ha\n";

        // ============================================================
        // C2H6 - ETANO
        //
        // Geometria aproximada:
        //
        //      H   H
        //       \ /
        //        C-C
        //       /   \
        //      H     H
        //       \   /
        //        H H
        //
        // C1 = (-1.260, 0.000, 0.000)
        // C2 = ( 1.260, 0.000, 0.000)
        //
        // H1 = (-2.000,  1.650,  0.000)
        // H2 = (-2.000, -0.825,  1.429)
        // H3 = (-2.000, -0.825, -1.429)
        //
        // H4 = ( 2.000,  1.650,  0.000)
        // H5 = ( 2.000, -0.825,  1.429)
        // H6 = ( 2.000, -0.825, -1.429)
        //
        // Electrones = 18
        // Carga = 0
        // Multiplicidad = 1
        // ============================================================

        Molecule c2h6(0);

        c2h6.addNucleus(
            6,
            -1.260,
            0.000,
            0.000
        );

        c2h6.addNucleus(
            6,
            1.260,
            0.000,
            0.000
        );

        c2h6.addNucleus(
            1,
            -2.000,
            1.650,
            0.000
        );

        c2h6.addNucleus(
            1,
            -2.000,
            -0.825,
            1.429
        );

        c2h6.addNucleus(
            1,
            -2.000,
            -0.825,
            -1.429
        );

        c2h6.addNucleus(
            1,
            2.000,
            1.650,
            0.000
        );

        c2h6.addNucleus(
            1,
            2.000,
            -0.825,
            1.429
        );

        c2h6.addNucleus(
            1,
            2.000,
            -0.825,
            -1.429
        );

        CartesianGrid c2h6Grid(
            19,
            19,
            19,
            -8.1,
            7.9,
            -8.1,
            7.9,
            -8.1,
            7.9
        );

        const MolecularResult c2h6PZ81 =
            solveMolecularSelfConsistentField(
                c2h6Grid,
                c2h6,
                c2h6.getCharge(),
                1,
                pz81
            );

        std::cout
            << "\nC2H6 | PZ81 | E = "
            << std::scientific
            << std::setprecision(10)
            << c2h6PZ81.scf.energy.total
            << " Ha\n";

        // ============================================================
        // CH3OH - METANOL
        //
        // Geometria aproximada:
        //
        //       H
        //       |
        //   H - C - O - H
        //       |
        //       H
        //
        // C  = ( 0.000,  0.000,  0.000)
        // O  = ( 2.400,  0.000,  0.000)
        //
        // H del grupo CH3:
        //
        // H1 = (-0.700,  1.700,  0.000)
        // H2 = (-0.700, -0.850,  1.472)
        // H3 = (-0.700, -0.850, -1.472)
        //
        // H del grupo OH:
        //
        // H4 = ( 3.300,  0.000,  1.400)
        //
        // Electrones = 18
        // Carga = 0
        // Multiplicidad = 1
        // ============================================================

        Molecule ch3oh(0);

        ch3oh.addNucleus(
            6,
            0.000,
            0.000,
            0.000
        );

        ch3oh.addNucleus(
            8,
            2.400,
            0.000,
            0.000
        );

        ch3oh.addNucleus(
            1,
            -0.700,
            1.700,
            0.000
        );

        ch3oh.addNucleus(
            1,
            -0.700,
            -0.850,
            1.472
        );

        ch3oh.addNucleus(
            1,
            -0.700,
            -0.850,
            -1.472
        );

        ch3oh.addNucleus(
            1,
            3.300,
            0.000,
            1.400
        );

        CartesianGrid ch3ohGrid(
            19,
            19,
            19,
            -8.1,
            7.9,
            -8.1,
            7.9,
            -8.1,
            7.9
        );

        const MolecularResult ch3ohPZ81 =
            solveMolecularSelfConsistentField(
                ch3ohGrid,
                ch3oh,
                ch3oh.getCharge(),
                1,
                pz81
            );

        std::cout
            << "\nCH3OH | PZ81 | E = "
            << std::scientific
            << std::setprecision(10)
            << ch3ohPZ81.scf.energy.total
            << " Ha\n";

        std::cout
            << "\n"
            << "============================================================\n"
            << "FIN DEL ANALISIS\n"
            << "============================================================\n";

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