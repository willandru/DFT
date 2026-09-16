#include "AtomicDFT.h"
#include "DFTConstants.h"
#include "ElectronicConfiguration.h"
#include "RadialGrid.h"

#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

int main() {
    try {
        RadialGrid grid(
            DFTConstants::GRID_POINTS,
            DFTConstants::RMAX
        );

        const std::vector<int> atomicNumbers = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        };

        std::vector<AtomicResult> results;
        results.reserve(atomicNumbers.size());

        std::cout
            << "\nDFT atomico LSDA-PZ81 | H-Ne\n"
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
                    configuration
                );

            results.push_back(
                std::move(result)
            );

            const AtomicResult& solved =
                results.back();

            int alphaElectrons = 0;
            int betaElectrons = 0;

            for (const AtomicOrbital& orbital :
                 solved.scf.orbitals) {

                if (orbital.spin ==
                    SpinChannel::Alpha) {

                    alphaElectrons +=
                        orbital.electrons;
                } else {
                    betaElectrons +=
                        orbital.electrons;
                }
            }

            std::cout
                << (solved.scf.converged ? "OK" : "NO")
                << "  E = "
                << std::scientific
                << std::setprecision(8)
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

        std::cout
            << "\nDIAGNOSTICO SCF\n"
            << "Atom   Iter       dRho           dE             KS_res\n"
            << "------------------------------------------------------------\n";

        for (const AtomicResult& result : results) {
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

        std::cout
            << "\nENERGIAS\n";

        for (const AtomicResult& result : results) {
            std::cout
                << std::setw(3)
                << result.symbol
                << "  "
                << std::scientific
                << std::setprecision(10)
                << result.scf.energy.total
                << " Ha\n";
        }

        std::cout << '\n';

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