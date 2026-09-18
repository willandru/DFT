#include "EigenvalueSolverMolecule.h"

#include "BlockDavidsonMolecule.h"
#include "DavidsonMath.h"
#include "DavidsonNumerical.h"
#include "KohnShamHamiltonianMolecule.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double MO_RESIDUAL_TOLERANCE = 1.0e-7;
constexpr double SUBSPACE_TOLERANCE = 1.0e-12;
constexpr std::size_t MAX_SUBSPACE_DIMENSION = 16;

enum class MolecularOrbitalSolver {
    Davidson,
    BlockDavidson
};

constexpr MolecularOrbitalSolver ACTIVE_MOLECULAR_ORBITAL_SOLVER =
    MolecularOrbitalSolver::BlockDavidson;


const char* spinName(
    SpinChannel spin
) {
    return spin == SpinChannel::Alpha
        ? "Alpha"
        : "Beta";
}

}


MolecularOrbital solveMolecularOrbital(
    const CartesianGrid& grid,
    const std::vector<double>& effectivePotential,
    std::size_t orbitalIndex,
    SpinChannel spin,
    int electrons,
    const std::vector<MolecularOrbital>& previousOrbitals,
    const MolecularOrbital* initialGuess,
    std::size_t maxIterations
) {
    if (effectivePotential.size() !=
        grid.getSize()) {

        throw std::invalid_argument(
            "El potencial y la malla no coinciden."
        );
    }

    if (electrons < 0 ||
        electrons > 2) {

        throw std::invalid_argument(
            "Ocupacion orbital invalida."
        );
    }

    if (maxIterations == 0) {

        throw std::invalid_argument(
            "maxIterations debe ser mayor que cero."
        );
    }

    const double dV =
        grid.getDx() *
        grid.getDy() *
        grid.getDz();

    if (dV <= 0.0) {

        throw std::invalid_argument(
            "Elemento de volumen invalido."
        );
    }

    std::cerr
        << "[MO-DEBUG] START"
        << " orbital=" << orbitalIndex
        << " spin=" << spinName(spin)
        << " electrons=" << electrons
        << " dimension=" << grid.getSize()
        << " maxH=" << maxIterations
        << " previous=" << previousOrbitals.size()
        << " initialGuess=" << (
            initialGuess != nullptr
                ? "yes"
                : "no"
        )
        << "\n";


    const std::vector<double> diagonal =
        davidsonBuildHamiltonianDiagonal(
            grid,
            effectivePotential
        );

    std::vector<double> initialVector;

    bool usedInitialGuess = false;

    if (initialGuess != nullptr &&
        davidsonBuildInitialFromPrevious(
            grid,
            *initialGuess,
            previousOrbitals,
            dV,
            initialVector
        )) {

        usedInitialGuess = true;

    } else if (!davidsonBuildIndependentVector(
                   grid,
                   orbitalIndex,
                   previousOrbitals,
                   dV,
                   initialVector
               )) {

        throw std::runtime_error(
            "No se pudo construir el vector inicial."
        );
    }

    std::cerr
        << "[MO-DEBUG] INITIAL"
        << " orbital=" << orbitalIndex
        << " spin=" << spinName(spin)
        << " source=" << (
            usedInitialGuess
                ? "previous"
                : "independent"
        )
        << "\n";


    std::vector<std::vector<double>> basis;
    std::vector<std::vector<double>> hBasis;

    basis.reserve(
        MAX_SUBSPACE_DIMENSION
    );

    hBasis.reserve(
        MAX_SUBSPACE_DIMENSION
    );

    basis.push_back(
        std::move(initialVector)
    );

    std::vector<std::vector<double>>
        projectedHamiltonian;

    std::size_t hApplications = 0;
    std::size_t davidsonIterations = 0;

    double finalEigenvalue =
        std::numeric_limits<double>::infinity();

    double finalResidual =
        std::numeric_limits<double>::infinity();

    std::vector<double> finalVector;


    while (hApplications < maxIterations) {

        ++davidsonIterations;

        std::cerr
            << "[MO-DEBUG] ITER_START"
            << " orbital=" << orbitalIndex
            << " spin=" << spinName(spin)
            << " iter=" << davidsonIterations
            << " H=" << hApplications
            << " basis=" << basis.size()
            << "\n";


        if (hBasis.size() < basis.size()) {

            const std::size_t index =
                hBasis.size();

            hBasis.push_back(
                applyMolecularKohnShamHamiltonian(
                    grid,
                    effectivePotential,
                    basis[index]
                )
            );

            ++hApplications;

            std::cerr
                << "[MO-DEBUG] H_BASIS"
                << " orbital=" << orbitalIndex
                << " spin=" << spinName(spin)
                << " basisIndex=" << index
                << " H=" << hApplications
                << " basis=" << basis.size()
                << "\n";


            const std::size_t size =
                basis.size();

            projectedHamiltonian.resize(size);

            for (auto& row :
                 projectedHamiltonian) {

                row.resize(
                    size,
                    0.0
                );
            }

            for (std::size_t i = 0;
                 i <= index;
                 ++i) {

                const double value =
                    davidsonDot(
                        basis[i],
                        hBasis[index],
                        dV
                    );

                projectedHamiltonian[i][index] =
                    value;

                projectedHamiltonian[index][i] =
                    value;
            }
        }


        const DavidsonEigenpair projected =
            davidsonDiagonalize(
                projectedHamiltonian
            );


        std::vector<double> ritz =
            davidsonCombine(
                basis,
                projected.vector
            );

        if (!davidsonValidNorm(ritz, dV)) {

            throw std::runtime_error(
                "El vector de Ritz es numericamente nulo."
            );
        }

        davidsonNormalize(
            ritz,
            dV
        );


        std::vector<double> hRitz =
            applyMolecularKohnShamHamiltonian(
                grid,
                effectivePotential,
                ritz
            );

        ++hApplications;


        finalEigenvalue =
            davidsonRayleigh(
                ritz,
                hRitz,
                dV
            );

        finalResidual =
            davidsonResidualNorm(
                ritz,
                hRitz,
                finalEigenvalue,
                dV
            );

        finalVector =
            ritz;


        std::cerr
            << "[MO-DEBUG] RITZ"
            << " orbital=" << orbitalIndex
            << " spin=" << spinName(spin)
            << " iter=" << davidsonIterations
            << " basis=" << basis.size()
            << " H=" << hApplications
            << " eigenvalue=" << finalEigenvalue
            << " residual=" << finalResidual
            << "\n";


        if (finalResidual <
            MO_RESIDUAL_TOLERANCE) {

            std::cerr
                << "[MO-DEBUG] CONVERGED"
                << " orbital=" << orbitalIndex
                << " spin=" << spinName(spin)
                << " iter=" << davidsonIterations
                << " basis=" << basis.size()
                << " H=" << hApplications
                << " eigenvalue=" << finalEigenvalue
                << " residual=" << finalResidual
                << "\n";

            std::cout
                << "Orbital "
                << orbitalIndex
                << " "
                << (
                    spin == SpinChannel::Alpha
                        ? "Alpha"
                        : "Beta"
                )
                << " convergido: "
                << finalEigenvalue
                << " | residual = "
                << finalResidual
                << " | H = "
                << hApplications
                << "\n";

            break;
        }


        std::vector<double> residual(
            grid.getSize(),
            0.0
        );

        for (std::size_t i = 0;
             i < grid.getSize();
             ++i) {

            residual[i] =
                hRitz[i] -
                finalEigenvalue *
                ritz[i];
        }

        davidsonOrthogonalizeAgainstOrbitals(
            residual,
            previousOrbitals,
            dV
        );


        const double residualNormSquared =
            davidsonNormSquared(
                residual,
                dV
            );

        const double residualNorm =
            std::isfinite(residualNormSquared) &&
            residualNormSquared >= 0.0
                ? std::sqrt(residualNormSquared)
                : std::numeric_limits<double>::infinity();


        std::cerr
            << "[MO-DEBUG] RESIDUAL"
            << " orbital=" << orbitalIndex
            << " spin=" << spinName(spin)
            << " iter=" << davidsonIterations
            << " physicalResidual=" << finalResidual
            << " orthogonalizedResidual=" << residualNorm
            << " basis=" << basis.size()
            << "\n";


        if (!davidsonValidNorm(
                residual,
                dV,
                SUBSPACE_TOLERANCE *
                SUBSPACE_TOLERANCE
            )) {

            std::cerr
                << "[MO-DEBUG] RESIDUAL_REJECTED"
                << " orbital=" << orbitalIndex
                << " spin=" << spinName(spin)
                << " iter=" << davidsonIterations
                << " residual=" << residualNorm
                << " threshold=" << SUBSPACE_TOLERANCE
                << "\n";

            std::vector<double> independent;

            if (!davidsonBuildIndependentVector(
                    grid,
                    orbitalIndex + basis.size(),
                    previousOrbitals,
                    dV,
                    independent
                )) {

                throw std::runtime_error(
                    "No se pudo construir una nueva "
                    "direccion independiente para Davidson."
                );
            }

            davidsonOrthogonalize(
                independent,
                basis,
                dV
            );

            if (!davidsonValidNorm(
                    independent,
                    dV,
                    SUBSPACE_TOLERANCE *
                    SUBSPACE_TOLERANCE
                )) {

                throw std::runtime_error(
                    "La nueva direccion Davidson "
                    "es numericamente dependiente."
                );
            }

            davidsonNormalize(
                independent,
                dV,
                SUBSPACE_TOLERANCE *
                SUBSPACE_TOLERANCE
            );

            if (basis.size() <
                MAX_SUBSPACE_DIMENSION) {

                std::cerr
                    << "[MO-DEBUG] EXPAND"
                    << " orbital=" << orbitalIndex
                    << " spin=" << spinName(spin)
                    << " iter=" << davidsonIterations
                    << " oldBasis=" << basis.size()
                    << " newBasis=" << basis.size() + 1
                    << " reason=independent"
                    << "\n";

                basis.push_back(
                    std::move(independent)
                );

                continue;
            }

            if (hApplications >=
                maxIterations) {

                std::cerr
                    << "[MO-DEBUG] LIMIT"
                    << " orbital=" << orbitalIndex
                    << " spin=" << spinName(spin)
                    << " iter=" << davidsonIterations
                    << " H=" << hApplications
                    << " basis=" << basis.size()
                    << " reason=independent_restart_unavailable"
                    << "\n";

                break;
            }

            std::vector<double> hIndependent =
                applyMolecularKohnShamHamiltonian(
                    grid,
                    effectivePotential,
                    independent
                );

            ++hApplications;

            std::cerr
                << "[MO-DEBUG] RESTART"
                << " orbital=" << orbitalIndex
                << " spin=" << spinName(spin)
                << " iter=" << davidsonIterations
                << " H=" << hApplications
                << " reason=independent"
                << "\n";

            basis.clear();
            hBasis.clear();
            projectedHamiltonian.clear();

            basis.push_back(
                std::move(ritz)
            );

            basis.push_back(
                std::move(independent)
            );

            hBasis.push_back(
                std::move(hRitz)
            );

            hBasis.push_back(
                std::move(hIndependent)
            );

            projectedHamiltonian.assign(
                2,
                std::vector<double>(
                    2,
                    0.0
                )
            );

            projectedHamiltonian[0][0] =
                davidsonDot(
                    basis[0],
                    hBasis[0],
                    dV
                );

            projectedHamiltonian[0][1] =
                davidsonDot(
                    basis[0],
                    hBasis[1],
                    dV
                );

            projectedHamiltonian[1][0] =
                projectedHamiltonian[0][1];

            projectedHamiltonian[1][1] =
                davidsonDot(
                    basis[1],
                    hBasis[1],
                    dV
                );

            continue;
        }


        std::vector<double> correction =
            davidsonBuildCorrection(
                residual,
                diagonal,
                finalEigenvalue
            );


        const double correctionNormSquared =
            davidsonNormSquared(
                correction,
                dV
            );

        const double correctionNorm =
            std::isfinite(correctionNormSquared) &&
            correctionNormSquared >= 0.0
                ? std::sqrt(correctionNormSquared)
                : std::numeric_limits<double>::infinity();


        std::cerr
            << "[MO-DEBUG] PRECONDITION"
            << " orbital=" << orbitalIndex
            << " spin=" << spinName(spin)
            << " iter=" << davidsonIterations
            << " residual=" << residualNorm
            << " correctionNorm=" << correctionNorm
            << "\n";


        const bool correctionValid =
            davidsonAppendIndependentCorrection(
                correction,
                previousOrbitals,
                basis,
                dV,
                SUBSPACE_TOLERANCE
            );


        std::cerr
            << "[MO-DEBUG] CORRECTION"
            << " orbital=" << orbitalIndex
            << " spin=" << spinName(spin)
            << " iter=" << davidsonIterations
            << " accepted=" << (
                correctionValid
                    ? "yes"
                    : "no"
            )
            << "\n";


        if (!correctionValid) {

            std::cerr
                << "[MO-DEBUG] FALLBACK"
                << " orbital=" << orbitalIndex
                << " spin=" << spinName(spin)
                << " iter=" << davidsonIterations
                << " reason=preconditioned_correction_rejected"
                << "\n";

            correction =
                residual;

            if (!davidsonAppendIndependentCorrection(
                    correction,
                    previousOrbitals,
                    basis,
                    dV,
                    SUBSPACE_TOLERANCE
                )) {

                std::cerr
                    << "[MO-DEBUG] FALLBACK_REJECTED"
                    << " orbital=" << orbitalIndex
                    << " spin=" << spinName(spin)
                    << " iter=" << davidsonIterations
                    << " reason=raw_residual_rejected"
                    << "\n";

                if (!davidsonBuildIndependentVector(
                        grid,
                        orbitalIndex + basis.size(),
                        previousOrbitals,
                        dV,
                        correction
                    )) {

                    throw std::runtime_error(
                        "No se pudo construir una direccion "
                        "de expansion Davidson valida."
                    );
                }

                davidsonOrthogonalize(
                    correction,
                    basis,
                    dV
                );

                if (!davidsonValidNorm(
                        correction,
                        dV,
                        SUBSPACE_TOLERANCE *
                        SUBSPACE_TOLERANCE
                    )) {

                    throw std::runtime_error(
                        "La direccion de expansion Davidson "
                        "es numericamente dependiente."
                    );
                }

                davidsonNormalize(
                    correction,
                    dV,
                    SUBSPACE_TOLERANCE *
                    SUBSPACE_TOLERANCE
                );

                std::cerr
                    << "[MO-DEBUG] FALLBACK"
                    << " orbital=" << orbitalIndex
                    << " spin=" << spinName(spin)
                    << " iter=" << davidsonIterations
                    << " reason=deterministic_independent"
                    << "\n";
            }
        }


        if (basis.size() <
            MAX_SUBSPACE_DIMENSION) {

            std::cerr
                << "[MO-DEBUG] EXPAND"
                << " orbital=" << orbitalIndex
                << " spin=" << spinName(spin)
                << " iter=" << davidsonIterations
                << " oldBasis=" << basis.size()
                << " newBasis=" << basis.size() + 1
                << " reason=correction"
                << "\n";

            basis.push_back(
                std::move(correction)
            );

            continue;
        }


        if (hApplications >=
            maxIterations) {

            std::cerr
                << "[MO-DEBUG] LIMIT"
                << " orbital=" << orbitalIndex
                << " spin=" << spinName(spin)
                << " iter=" << davidsonIterations
                << " H=" << hApplications
                << " basis=" << basis.size()
                << " reason=maximum_H_applications"
                << "\n";

            break;
        }


        std::vector<double> hCorrection =
            applyMolecularKohnShamHamiltonian(
                grid,
                effectivePotential,
                correction
            );

        ++hApplications;


        std::cerr
            << "[MO-DEBUG] RESTART"
            << " orbital=" << orbitalIndex
            << " spin=" << spinName(spin)
            << " iter=" << davidsonIterations
            << " H=" << hApplications
            << " reason=max_subspace"
            << "\n";


        basis.clear();
        hBasis.clear();
        projectedHamiltonian.clear();


        basis.push_back(
            std::move(ritz)
        );

        basis.push_back(
            std::move(correction)
        );


        hBasis.push_back(
            std::move(hRitz)
        );

        hBasis.push_back(
            std::move(hCorrection)
        );


        projectedHamiltonian.assign(
            2,
            std::vector<double>(
                2,
                0.0
            )
        );


        projectedHamiltonian[0][0] =
            davidsonDot(
                basis[0],
                hBasis[0],
                dV
            );

        projectedHamiltonian[0][1] =
            davidsonDot(
                basis[0],
                hBasis[1],
                dV
            );

        projectedHamiltonian[1][0] =
            projectedHamiltonian[0][1];

        projectedHamiltonian[1][1] =
            davidsonDot(
                basis[1],
                hBasis[1],
                dV
            );
    }


    if (finalVector.empty() ||
        finalResidual >=
            MO_RESIDUAL_TOLERANCE) {

        std::cerr
            << "[MO-DEBUG] FAIL"
            << " orbital=" << orbitalIndex
            << " spin=" << spinName(spin)
            << " iterations=" << davidsonIterations
            << " H=" << hApplications
            << " basis=" << basis.size()
            << " eigenvalue=" << finalEigenvalue
            << " residual=" << finalResidual
            << " tolerance=" << MO_RESIDUAL_TOLERANCE
            << "\n";

        throw std::runtime_error(
            "El orbital molecular no convergio. "
            "Orbital = " +
            std::to_string(orbitalIndex) +
            ", residual = " +
            std::to_string(finalResidual)
        );
    }


    MolecularOrbital orbital;

    orbital.spin =
        spin;

    orbital.electrons =
        electrons;

    orbital.eigenvalue =
        finalEigenvalue;

    orbital.psi =
        std::move(finalVector);

    return orbital;
}


std::vector<MolecularOrbital> solveMolecularOrbitals(
    const CartesianGrid& grid,
    const std::vector<double>& effectivePotential,
    std::size_t numberOfOrbitals,
    const std::vector<int>& occupations,
    SpinChannel spin,
    const std::vector<MolecularOrbital>& initialOrbitals,
    std::size_t maxIterations
) {
    if (numberOfOrbitals == 0) {

        throw std::invalid_argument(
            "El numero de orbitales debe ser mayor que cero."
        );
    }

    if (occupations.size() !=
        numberOfOrbitals) {

        throw std::invalid_argument(
            "Las ocupaciones no coinciden con "
            "el numero de orbitales."
        );
    }

    if (!initialOrbitals.empty() &&
        initialOrbitals.size() !=
            numberOfOrbitals) {

        throw std::invalid_argument(
            "Los orbitales iniciales no coinciden "
            "con el numero solicitado."
        );
    }

    for (const int occupation : occupations) {

        if (occupation < 0 ||
            occupation > 2) {

            throw std::invalid_argument(
                "Ocupacion orbital invalida."
            );
        }
    }


    switch (ACTIVE_MOLECULAR_ORBITAL_SOLVER) {

        case MolecularOrbitalSolver::Davidson:
        {
            std::vector<MolecularOrbital> orbitals;

            orbitals.reserve(
                numberOfOrbitals
            );

            for (std::size_t i = 0;
                 i < numberOfOrbitals;
                 ++i) {

                const MolecularOrbital* initialGuess =
                    initialOrbitals.empty()
                        ? nullptr
                        : &initialOrbitals[i];

                orbitals.push_back(
                    solveMolecularOrbital(
                        grid,
                        effectivePotential,
                        i,
                        spin,
                        occupations[i],
                        orbitals,
                        initialGuess,
                        maxIterations
                    )
                );
            }

            return orbitals;
        }


        case MolecularOrbitalSolver::BlockDavidson:
        {
            return solveMolecularOrbitalsBlockDavidson(
                grid,
                effectivePotential,
                numberOfOrbitals,
                occupations,
                spin,
                initialOrbitals,
                maxIterations
            );
        }
    }


    throw std::runtime_error(
        "Solver de orbitales moleculares no reconocido."
    );
}