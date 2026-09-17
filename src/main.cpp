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

        Molecule co2(0);

        co2.addNucleus(
            8,
            -2.2,
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
            2.2,
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

        const MolecularResult co2PZ81 =
            solveMolecularSelfConsistentField(
                molecularGrid,
                co2,
                co2.getCharge(),
                pz81
            );

        std::cout
            << "CO2 | PZ81 | E = "
            << std::scientific
            << std::setprecision(10)
            << co2PZ81.scf.energy.total
            << " Ha\n";

        const auto& co2Nuclei = co2.getNuclei();

        for (std::size_t i = 0; i < co2Nuclei.size(); ++i)
        {
            const auto& nucleus = co2Nuclei[i];

            std::cout
                << "CO2 | Nucleo " << i
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