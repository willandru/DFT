#include "DFTTests.h"

#include "DFTConstants.h"
#include "ElectronDensity.h"
#include "EigenvalueSolver.h"
#include "KohnShamHamiltonian.h"
#include "RadialGrid.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

bool testCoulombHydrogen(
    const std::vector<double>& r,
    const std::vector<double>& potential
) {
    if (r.size() < 2 ||
        r.size() != potential.size()) {
        return false;
    }

    for (double value : r) {
        if (value <= 0.0) {
            return false;
        }
    }

    std::vector<double> coulombPotential(
        r.size()
    );

    for (std::size_t i = 0;
         i < r.size();
         ++i) {
        coulombPotential[i] =
            -1.0 / r[i];
    }

    const TridiagonalMatrix hamiltonian =
        buildKohnShamHamiltonian(
            r,
            coulombPotential,
            0
        );

    const AtomicOrbital orbital =
        solveOrbital(
            hamiltonian,
            r,
            1,
            0,
            1,
            0
        );

    const double error =
        std::abs(
            orbital.eigenvalue -
            DFTConstants::H_EXACT
        );

    return error < 1.0e-4;
}

bool testSCFResults(
    const std::vector<AtomicResult>& results
) {
    if (results.empty()) {
        return false;
    }

    for (const AtomicResult& result : results) {
        if (!result.scf.converged) {
            return false;
        }
    }

    return true;
}

bool testElectronNumbers(
    const std::vector<AtomicResult>& results
) {
    if (results.empty()) {
        return false;
    }

    for (const AtomicResult& result : results) {
        if (result.electrons != result.Z) {
            return false;
        }
    }

    return true;
}

bool testNumericalElectronNumbers(
    const std::vector<AtomicResult>& results
) {
    if (results.empty()) {
        return false;
    }

    bool passed = true;

    for (const AtomicResult& result : results) {
        if (result.scf.density.empty() ||
            result.scf.orbitals.empty()) {
            return false;
        }

        std::vector<double> r(
            result.scf.density.size()
        );

        const double dr =
            DFTConstants::RMAX /
            static_cast<double>(
                DFTConstants::GRID_POINTS + 1
            );

        for (std::size_t i = 0;
             i < r.size();
             ++i) {
            r[i] =
                static_cast<double>(i + 1) *
                dr;
        }

        const double electrons =
            integrateElectronDensity(
                r,
                result.scf.density
            );

        const double expected =
            static_cast<double>(
                result.electrons
            );

        const double error =
            std::abs(
                electrons -
                expected
            );

        const double tolerance =
            1.0e-5 *
            std::max(
                1.0,
                expected
            );

        std::cout
            << std::scientific
            << std::setprecision(12)
            << "Numero electronico "
            << std::setw(2)
            << result.symbol
            << ": esperado="
            << expected
            << " calculado="
            << electrons
            << " error="
            << error
            << " tolerancia="
            << tolerance
            << '\n';

        if (error > tolerance) {
            passed = false;
        }
    }

    return passed;
}

bool testOrbitalNorms(
    const std::vector<AtomicResult>& results,
    double tolerance
) {
    if (results.empty()) {
        return false;
    }

    for (const AtomicResult& result : results) {
        for (const AtomicOrbital& orbital :
             result.scf.orbitals) {

            if (orbital.u.empty()) {
                return false;
            }

            double norm = 0.0;

            const double dr =
                DFTConstants::RMAX /
                static_cast<double>(
                    DFTConstants::GRID_POINTS + 1
                );

            for (double value : orbital.u) {
                norm +=
                    value *
                    value;
            }

            norm *= dr;

            if (std::abs(norm - 1.0) > tolerance) {
                return false;
            }
        }
    }

    return true;
}

bool testKohnShamExpectationValues(
    const std::vector<AtomicResult>& results,
    double tolerance
) {
    if (results.empty()) {
        return false;
    }

    bool passed = true;

    const double dr =
        DFTConstants::RMAX /
        static_cast<double>(
            DFTConstants::GRID_POINTS + 1
        );

    for (const AtomicResult& result : results) {
        const std::vector<double>& r =
            result.scf.density.empty()
                ? std::vector<double>()
                : [&]() {
                    std::vector<double> coordinates(
                        result.scf.density.size()
                    );

                    for (std::size_t i = 0;
                         i < coordinates.size();
                         ++i) {
                        coordinates[i] =
                            static_cast<double>(i + 1) *
                            dr;
                    }

                    return coordinates;
                }();

        if (r.size() != result.scf.effectivePotential.size()) {
            return false;
        }

        for (const AtomicOrbital& orbital :
             result.scf.orbitals) {

            if (orbital.u.size() != r.size()) {
                return false;
            }

            const TridiagonalMatrix hamiltonian =
                buildKohnShamHamiltonian(
                    r,
                    result.scf.effectivePotential,
                    orbital.l
                );

            double expectationValue = 0.0;

            for (std::size_t i = 0;
                 i < r.size();
                 ++i) {

                double value =
                    hamiltonian.diagonal[i] *
                    orbital.u[i];

                if (i > 0) {
                    value +=
                        hamiltonian.lower[i - 1] *
                        orbital.u[i - 1];
                }

                if (i + 1 < r.size()) {
                    value +=
                        hamiltonian.upper[i] *
                        orbital.u[i + 1];
                }

                expectationValue +=
                    orbital.u[i] *
                    value;
            }

            expectationValue *= dr;

            const double error =
                std::abs(
                    expectationValue -
                    orbital.eigenvalue
                );

            std::cout
                << std::scientific
                << std::setprecision(12)
                << "Valor esperado KS "
                << std::setw(2)
                << result.symbol
                << " n="
                << orbital.n
                << " l="
                << orbital.l
                << ": epsilon="
                << orbital.eigenvalue
                << " esperado="
                << expectationValue
                << " error="
                << error
                << " tolerancia="
                << tolerance
                << '\n';

            if (error > tolerance) {
                passed = false;
            }
        }
    }

    return passed;
}

bool testEnergyDecomposition(
    const std::vector<AtomicResult>& results,
    double tolerance
) {
    if (results.empty()) {
        return false;
    }

    for (const AtomicResult& result : results) {
        const EnergyComponents& energy =
            result.scf.energy;

        const double reconstructed =
            energy.kinetic +
            energy.external +
            energy.hartree +
            energy.exchangeCorrelation;

        if (std::abs(
                reconstructed -
                energy.total
            ) > tolerance) {
            return false;
        }
    }

    return true;
}
