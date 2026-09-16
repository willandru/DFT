#include "AtomicDFT.h"
#include "DFTConstants.h"
#include "ElectronicConfiguration.h"
#include "PBE96.h"
#include "PZ81.h"
#include "RadialGrid.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

struct FunctionalResults {
    std::string name;
    std::vector<AtomicResult> results;
};

static FunctionalResults runFunctional(
    const std::string& name,
    const XCFunctional& functional,
    RadialGrid& grid,
    const std::vector<int>& atomicNumbers
) {
    FunctionalResults output;

    output.name = name;
    output.results.reserve(atomicNumbers.size());

    std::cout
        << "\n============================================================\n"
        << " DFT ATOMICO | " << name << " | H-Zn\n"
        << "============================================================\n"
        << "Grid: "
        << DFTConstants::GRID_POINTS
        << " puntos | Rmax: "
        << DFTConstants::RMAX
        << " bohr\n\n";

    for (int Z : atomicNumbers) {
        const AtomicConfiguration configuration =
            getAtomicConfiguration(Z);

        std::cout
            << "Calculando "
            << configuration.symbol
            << " (" << Z << ")... "
            << std::flush;

        AtomicResult result =
            solveAtom(
                grid,
                configuration,
                functional
            );

        output.results.push_back(
            std::move(result)
        );

        const AtomicResult& solved =
            output.results.back();

        int alphaElectrons = 0;
        int betaElectrons = 0;

        for (const AtomicOrbital& orbital :
             solved.scf.orbitals) {

            if (orbital.spin == SpinChannel::Alpha) {
                alphaElectrons += orbital.electrons;
            } else {
                betaElectrons += orbital.electrons;
            }
        }

        std::cout
            << (solved.scf.converged ? "OK" : "NO")
            << "  E = "
            << std::scientific
            << std::setprecision(10)
            << solved.scf.energy.total
            << " Ha"
            << "  N = "
            << solved.electrons
            << " ("
            << alphaElectrons
            << "+"
            << betaElectrons
            << ")"
            << "  iter = "
            << solved.scf.iterations
            << '\n';
    }

    return output;
}

static void printSCFDiagnostics(
    const FunctionalResults& functionalResults
) {
    std::cout
        << "\n============================================================\n"
        << " DIAGNOSTICO SCF | "
        << functionalResults.name
        << "\n"
        << "============================================================\n";

    std::cout
        << "Atom   Iter       dRho           dE             KS_res\n"
        << "------------------------------------------------------------\n";

    for (const AtomicResult& result :
         functionalResults.results) {

        std::cout
            << std::left
            << std::setw(7)
            << result.symbol
            << std::setw(11)
            << result.scf.iterations
            << std::scientific
            << std::setprecision(6)
            << std::setw(15)
            << result.scf.densityDifference
            << std::setw(15)
            << result.scf.energyDifference
            << std::setw(15)
            << result.scf.maxKSResidual
            << '\n';
    }
}

static void printEnergyComponents(
    const FunctionalResults& functionalResults
) {
    std::cout
        << "\n============================================================\n"
        << " ENERGIAS POR COMPONENTE | "
        << functionalResults.name
        << "\n"
        << "============================================================\n";

    std::cout
        << std::scientific
        << std::setprecision(10);

    std::cout
        << "Atom        Ts              Eext            EH"
        << "             Exc             Etot\n"
        << "--------------------------------------------------------------------------\n";

    for (const AtomicResult& result :
         functionalResults.results) {

        const EnergyComponents& energy =
            result.scf.energy;

        std::cout
            << std::left
            << std::setw(5)
            << result.symbol
            << std::right
            << std::setw(17)
            << energy.kinetic
            << std::setw(17)
            << energy.external
            << std::setw(17)
            << energy.hartree
            << std::setw(17)
            << energy.exchangeCorrelation
            << std::setw(17)
            << energy.total
            << '\n';
    }
}

static void printEnergyVerification(
    const FunctionalResults& functionalResults
) {
    std::cout
        << "\n============================================================\n"
        << " VERIFICACION DE LA SUMA | "
        << functionalResults.name
        << "\n"
        << "============================================================\n";

    std::cout
        << "Atom        Ts+Eext+EH+Exc        Etot"
        << "                 Diferencia\n"
        << "--------------------------------------------------------------------------\n";

    std::cout
        << std::scientific
        << std::setprecision(10);

    for (const AtomicResult& result :
         functionalResults.results) {

        const EnergyComponents& energy =
            result.scf.energy;

        const double reconstructed =
            energy.kinetic
            + energy.external
            + energy.hartree
            + energy.exchangeCorrelation;

        const double difference =
            reconstructed - energy.total;

        std::cout
            << std::left
            << std::setw(5)
            << result.symbol
            << std::right
            << std::setw(22)
            << reconstructed
            << std::setw(17)
            << energy.total
            << std::setw(20)
            << difference
            << '\n';
    }
}

static void printFunctionalComparison(
    const FunctionalResults& pz81Results,
    const FunctionalResults& pbe96Results,
    const std::vector<int>& atomicNumbers
) {
    std::cout
        << "\n============================================================\n"
        << " COMPARACION PZ81 vs PBE96 | ENERGIA TOTAL\n"
        << "============================================================\n";

    std::cout
        << "Atom        Z"
        << "          E_PZ81 (Ha)"
        << "          E_PBE96 (Ha)"
        << "          PBE-PZ (Ha)"
        << "        |Delta E|\n"
        << "--------------------------------------------------------------------------------\n";

    std::cout
        << std::scientific
        << std::setprecision(10);

    const std::size_t count =
        std::min(
            {
                pz81Results.results.size(),
                pbe96Results.results.size(),
                atomicNumbers.size()
            }
        );

    for (std::size_t i = 0; i < count; ++i) {
        const AtomicResult& pz =
            pz81Results.results[i];

        const AtomicResult& pbe =
            pbe96Results.results[i];

        const double energyPZ =
            pz.scf.energy.total;

        const double energyPBE =
            pbe.scf.energy.total;

        const double difference =
            energyPBE - energyPZ;

        std::cout
            << std::left
            << std::setw(5)
            << pz.symbol
            << std::right
            << std::setw(8)
            << atomicNumbers[i]
            << std::setw(20)
            << energyPZ
            << std::setw(20)
            << energyPBE
            << std::setw(20)
            << difference
            << std::setw(18)
            << std::abs(difference)
            << '\n';
    }
}

int main() {
    try {
        RadialGrid grid(
            DFTConstants::GRID_POINTS,
            DFTConstants::RMAX
        );

        PZ81 pz81;
        PBE96 pbe96;

        const std::vector<int> atomicNumbers = {
            1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30
        };

        /*
         * ========================================================
         * 1. PZ81
         * ========================================================
         */

        FunctionalResults pz81Results =
            runFunctional(
                "PZ81",
                pz81,
                grid,
                atomicNumbers
            );

        printSCFDiagnostics(
            pz81Results
        );

        printEnergyComponents(
            pz81Results
        );

        printEnergyVerification(
            pz81Results
        );

        /*
         * ========================================================
         * 2. PBE96
         * ========================================================
         */

        FunctionalResults pbe96Results =
            runFunctional(
                "PBE96",
                pbe96,
                grid,
                atomicNumbers
            );

        printSCFDiagnostics(
            pbe96Results
        );

        printEnergyComponents(
            pbe96Results
        );

        printEnergyVerification(
            pbe96Results
        );

        /*
         * ========================================================
         * 3. COMPARACION FINAL
         * ========================================================
         */

        printFunctionalComparison(
            pz81Results,
            pbe96Results,
            atomicNumbers
        );

        std::cout
            << "\n============================================================\n"
            << " CALCULO COMPLETADO\n"
            << "============================================================\n\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "\nERROR: "
            << error.what()
            << '\n';

        return 1;
    }
    catch (...) {
        std::cerr
            << "\nERROR: excepcion desconocida.\n";

        return 1;
    }
}