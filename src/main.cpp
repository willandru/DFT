#include "AtomicDFT.h"
#include "DFTConstants.h"
#include "DFTTests.h"
#include "ElectronicConfiguration.h"
#include "HartreePotential.h"
#include "RadialGrid.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main() {
    try {
        std::cout
            << "============================================================\n"
            << "                 DFT ATOMICO H -> C\n"
            << "                 LDA-PZ81 | radial\n"
            << "============================================================\n\n";

        RadialGrid grid(
            DFTConstants::GRID_POINTS,
            DFTConstants::RMAX
        );

        const std::vector<int> atomicNumbers = {
            1, 2, 3, 4, 5, 6
        };

        std::vector<AtomicResult> results;

        for (int Z : atomicNumbers) {
            const AtomicConfiguration configuration =
                getAtomicConfiguration(Z);

            std::cout
                << "Calculando "
                << configuration.symbol
                << " (Z = "
                << Z
                << ")...\n";

            AtomicResult result =
                solveAtom(
                    grid,
                    configuration
                );

            results.push_back(result);
        }

        std::cout << '\n';

        std::cout
            << "============================================================\n"
            << "                    RESULTADOS H -> C\n"
            << "============================================================\n\n";

        std::cout
            << std::left
            << std::setw(6) << "Atom"
            << std::setw(6) << "Z"
            << std::setw(8) << "Ne"
            << std::setw(12) << "SCF"
            << std::setw(8) << "Iter"
            << std::right
            << std::setw(20) << "E_total"
            << std::setw(20) << "T_s"
            << std::setw(20) << "E_ext"
            << std::setw(20) << "E_H"
            << std::setw(20) << "E_XC"
            << '\n';

        std::cout
            << std::string(140, '-')
            << '\n';

        for (const AtomicResult& result : results) {
            const EnergyComponents& energy =
                result.scf.energy;

            std::cout
                << std::left
                << std::setw(6) << result.symbol
                << std::setw(6) << result.Z
                << std::setw(8) << result.electrons
                << std::setw(12)
                << (result.scf.converged ? "YES" : "NO")
                << std::setw(8) << result.scf.iterations
                << std::right
                << std::scientific
                << std::setprecision(10)
                << std::setw(20) << energy.total
                << std::setw(20) << energy.kinetic
                << std::setw(20) << energy.external
                << std::setw(20) << energy.hartree
                << std::setw(20) << energy.exchangeCorrelation
                << '\n';
        }

        std::cout << '\n';

        std::cout
            << "============================================================\n"
            << "                 ORBITALES KOHN-SHAM\n"
            << "============================================================\n\n";

        for (const AtomicResult& result : results) {
            std::cout
                << result.symbol
                << "  Z = "
                << result.Z
                << "  Ne = "
                << result.electrons
                << '\n';

            for (const AtomicOrbital& orbital :
                 result.scf.orbitals) {

                std::cout
                    << "  n = "
                    << orbital.n
                    << "  l = "
                    << orbital.l
                    << "  electrones = "
                    << orbital.electrons
                    << "  epsilon = "
                    << std::scientific
                    << std::setprecision(12)
                    << orbital.eigenvalue
                    << " Ha\n";
            }

            std::cout << '\n';
        }

        std::cout
            << "============================================================\n"
            << "                    VALIDACION NUMERICA\n"
            << "============================================================\n\n";

        const std::vector<double>& r =
            grid.coordinates();

        std::vector<double> hydrogenDensity(
            r.size(),
            0.0
        );

        for (std::size_t i = 0;
             i < r.size();
             ++i) {

            hydrogenDensity[i] =
                std::exp(-2.0 * r[i]) /
                DFTConstants::PI;
        }

        const std::vector<double> hydrogenPotential =
            calculateHartreePotential(
                r,
                hydrogenDensity
            );

        const bool coulombTest =
            testCoulombHydrogen(
                r,
                hydrogenPotential
            );

        const bool scfTest =
            testSCFResults(results);

        const bool electronNumberTest =
            testElectronNumbers(results);

        const bool numericalElectronNumberTest =
            testNumericalElectronNumbers(results);

        const bool orbitalNormTest =
            testOrbitalNorms(
                results,
                1.0e-8
            );

        const bool energyTest =
            testEnergyDecomposition(
                results,
                1.0e-10
            );

        std::cout
            << std::left
            << std::setw(35)
            << "Solver Coulomb H"
            << (coulombTest ? "PASS" : "FAIL")
            << '\n';

        std::cout
            << std::setw(35)
            << "SCF H-C"
            << (scfTest ? "PASS" : "FAIL")
            << '\n';

        std::cout
            << std::setw(35)
            << "Numero de electrones"
            << (electronNumberTest ? "PASS" : "FAIL")
            << '\n';

        std::cout
            << std::setw(35)
            << "Numero electronico numerico"
            << (numericalElectronNumberTest ? "PASS" : "FAIL")
            << '\n';

        std::cout
            << std::setw(35)
            << "Normas orbitales"
            << (orbitalNormTest ? "PASS" : "FAIL")
            << '\n';

        std::cout
            << std::setw(35)
            << "Descomposicion energetica"
            << (energyTest ? "PASS" : "FAIL")
            << '\n';

        std::cout << '\n';

        std::cout
            << "============================================================\n"
            << "                    REFERENCIAS\n"
            << "============================================================\n\n";

        std::cout
            << "H  energia Coulomb exacta     = "
            << std::scientific
            << std::setprecision(12)
            << DFTConstants::H_EXACT
            << " Ha\n";

        std::cout
            << "H  energia calculada DFT      = "
            << results[0].scf.energy.total
            << " Ha\n";

        std::cout
            << "He energia calculada DFT      = "
            << results[1].scf.energy.total
            << " Ha\n";

        const bool allPassed =
            coulombTest &&
            scfTest &&
            electronNumberTest &&
            numericalElectronNumberTest &&
            orbitalNormTest &&
            energyTest;

        std::cout << '\n';

        std::cout
            << "============================================================\n"
            << (allPassed
                ? "                    VALIDACION PASS\n"
                : "                    VALIDACION FAIL\n")
            << "============================================================\n";

        return allPassed ? 0 : 1;
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