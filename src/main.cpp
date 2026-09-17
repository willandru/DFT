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

int main() {
    try {
        RadialGrid grid(
            DFTConstants::GRID_POINTS,
            DFTConstants::RMAX
        );

        PZ81 pz81;
        PBE96 pbe96;

        const AtomicConfiguration hydrogen =
            getAtomicConfiguration(1);

        const AtomicResult hPZ81 =
            solveAtom(
                grid,
                hydrogen,
                pz81
            );

        const AtomicResult hPBE96 =
            solveAtom(
                grid,
                hydrogen,
                pbe96
            );

        std::cout
            << "H | PZ81  | E = "
            << std::scientific
            << std::setprecision(10)
            << hPZ81.scf.energy.total
            << " Ha\n";

        std::cout
            << "H | PBE96 | E = "
            << std::scientific
            << std::setprecision(10)
            << hPBE96.scf.energy.total
            << " Ha\n";

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

        const MolecularResult h2PZ81 =
            solveMolecularSelfConsistentField(
                molecularGrid,
                h2,
                h2.getCharge(),
                pz81
            );

        std::cout
            << "H2 | PZ81  | E = "
            << std::scientific
            << std::setprecision(10)
            << h2PZ81.scf.energy.total
            << " Ha\n";

        const auto& h2Nuclei = h2.getNuclei();

        for (std::size_t i = 0; i < h2Nuclei.size(); ++i) {
            const auto& nucleus = h2Nuclei[i];

            std::cout
                << "H2 | Nucleo " << i
                << " | Z = " << nucleus.atomicNumber
                << " | Posicion = ("
                << std::fixed
                << std::setprecision(6)
                << nucleus.position[0] << ", "
                << nucleus.position[1] << ", "
                << nucleus.position[2]
                << ")\n";
        }

        Molecule h2o(0);

        h2o.addNucleus(
            8,
            0.0,
            0.0,
            0.0
        );

        h2o.addNucleus(
            1,
            1.432,
            1.107,
            0.0
        );

        h2o.addNucleus(
            1,
            -1.432,
            1.107,
            0.0
        );

        CartesianGrid waterGrid(
            13,
            13,
            13,
            -6.1,
            5.9,
            -6.1,
            5.9,
            -6.1,
            5.9
        );

        const MolecularResult h2oPZ81 =
            solveMolecularSelfConsistentField(
                waterGrid,
                h2o,
                h2o.getCharge(),
                pz81
            );

        std::cout
            << "H2O | PZ81  | E = "
            << std::scientific
            << std::setprecision(10)
            << h2oPZ81.scf.energy.total
            << " Ha\n";

        const auto& h2oNuclei = h2o.getNuclei();

        for (std::size_t i = 0; i < h2oNuclei.size(); ++i) {
            const auto& nucleus = h2oNuclei[i];

            std::cout
                << "H2O | Nucleo " << i
                << " | Z = " << nucleus.atomicNumber
                << " | Posicion = ("
                << std::fixed
                << std::setprecision(6)
                << nucleus.position[0] << ", "
                << nucleus.position[1] << ", "
                << nucleus.position[2]
                << ")\n";
        }

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "ERROR: "
            << error.what()
            << '\n';

        return 1;
    }
    catch (...) {
        std::cerr
            << "ERROR: excepcion desconocida.\n";

        return 1;
    }
}