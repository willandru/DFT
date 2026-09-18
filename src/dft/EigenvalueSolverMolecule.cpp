#include "EigenvalueSolverMolecule.h"

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

    const std::size_t dimension =
        grid.getSize();

    const std::vector<double> diagonal =
        davidsonBuildHamiltonianDiagonal(
            grid,
            effectivePotential
        );

    std::vector<double> initialVector;

    if (initialGuess != nullptr &&
        davidsonBuildInitialFromPrevious(
            grid,
            *initialGuess,
            previousOrbitals,
            dV,
            initialVector
        )) {

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

    double finalEigenvalue =
        std::numeric_limits<double>::infinity();

    double finalResidual =
        std::numeric_limits<double>::infinity();

    std::vector<double> finalVector;


    while (hApplications < maxIterations) {

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


        /*
         * El vector de Ritz pertenece directamente al
         * subespacio de Davidson que acaba de ser
         * diagonalizado.
         *
         * La base ya fue construida ortogonalmente
         * respecto a los orbitales anteriores. Por tanto,
         * NO se vuelve a proyectar el Ritz contra ellos:
         * hacerlo modificaria el autovector asociado al
         * autovalor proyectado.
         */
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


        if (finalResidual <
            MO_RESIDUAL_TOLERANCE) {

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


        /*
         * Residuo Ritz:
         *
         *     r = H v - theta v
         *
         * Para el autovector del problema proyectado,
         * este residuo es ortogonal al subespacio actual
         * en aritmetica exacta. Por ello NO se proyecta
         * directamente contra basis antes de comprobar
         * su norma.
         */
        std::vector<double> residual(
            dimension,
            0.0
        );

        for (std::size_t i = 0;
             i < dimension;
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

        /*
         * La tolerancia de dependencia se aplica al
         * residual fisico antes de cualquier operacion
         * de precondicionamiento.
         */
        if (!davidsonValidNorm(
                residual,
                dV,
                SUBSPACE_TOLERANCE *
                SUBSPACE_TOLERANCE
            )) {

            /*
             * Si la magnitud del residual es pequena
             * respecto al umbral de independencia, no
             * se intenta fabricar una direccion a partir
             * de el. Se solicita una nueva direccion
             * determinista para continuar el subespacio.
             */
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

                basis.push_back(
                    std::move(independent)
                );

                continue;
            }

            if (hApplications >=
                maxIterations) {

                break;
            }

            std::vector<double> hIndependent =
                applyMolecularKohnShamHamiltonian(
                    grid,
                    effectivePotential,
                    independent
                );

            ++hApplications;

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


        /*
         * Precondicionamiento de Davidson.
         */
        std::vector<double> correction =
            davidsonBuildCorrection(
                residual,
                diagonal,
                finalEigenvalue
            );


        /*
         * La correccion se proyecta una sola vez contra
         * los orbitales anteriores y contra el subespacio
         * Davidson actual.
         */
        const bool correctionValid =
            davidsonAppendIndependentCorrection(
                correction,
                previousOrbitals,
                basis,
                dV,
                SUBSPACE_TOLERANCE
            );


        /*
         * Si el precondicionador destruye la direccion,
         * se recupera a partir del residual original.
         *
         * El residual ya esta separado de los orbitales
         * anteriores. Aqui solamente se elimina cualquier
         * componente numerica contra basis.
         */
        if (!correctionValid) {

            correction =
                residual;

            if (!davidsonAppendIndependentCorrection(
                    correction,
                    previousOrbitals,
                    basis,
                    dV,
                    SUBSPACE_TOLERANCE
                )) {

                /*
                 * El residual puede ser pequeno pero aun
                 * no suficientemente pequeno para satisfacer
                 * la convergencia. En ese caso se construye
                 * una direccion determinista independiente.
                 */
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
            }
        }


        if (basis.size() <
            MAX_SUBSPACE_DIMENSION) {

            basis.push_back(
                std::move(correction)
            );

            continue;
        }


        if (hApplications >=
            maxIterations) {

            break;
        }


        /*
         * Reinicio Davidson:
         *
         * Se conserva el Ritz actual y la correccion
         * como una base minima de dos dimensiones.
         */
        std::vector<double> hCorrection =
            applyMolecularKohnShamHamiltonian(
                grid,
                effectivePotential,
                correction
            );

        ++hApplications;


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

    std::vector<MolecularOrbital> orbitals;

    orbitals.reserve(
        numberOfOrbitals
    );

    for (std::size_t i = 0;
         i < numberOfOrbitals;
         ++i) {

        if (occupations[i] < 0 ||
            occupations[i] > 2) {

            throw std::invalid_argument(
                "Ocupacion orbital invalida."
            );
        }

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