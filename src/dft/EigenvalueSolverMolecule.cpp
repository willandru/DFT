#include "EigenvalueSolverMolecule.h"

#include "DavidsonMath.h"
#include "DFTConstants.h"
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
constexpr double VECTOR_TOLERANCE_SQUARED = 1.0e-20;
constexpr std::size_t MAX_SUBSPACE_DIMENSION = 16;
constexpr double DAVIDSON_DENOMINATOR_TOLERANCE = 1.0e-10;


void orthogonalizeAgainstOrbitals(
    std::vector<double>& v,
    const std::vector<MolecularOrbital>& orbitals,
    double dV
) {
    for (int pass = 0; pass < 2; ++pass) {

        for (const auto& orbital : orbitals) {

            const double orbitalNorm =
                davidsonNormSquared(
                    orbital.psi,
                    dV
                );

            if (!std::isfinite(orbitalNorm) ||
                orbitalNorm <=
                    VECTOR_TOLERANCE_SQUARED) {

                throw std::runtime_error(
                    "Un orbital molecular previo no es valido."
                );
            }

            const double projection =
                davidsonDot(
                    orbital.psi,
                    v,
                    dV
                ) / orbitalNorm;

            for (std::size_t i = 0;
                 i < v.size();
                 ++i) {

                v[i] -=
                    projection *
                    orbital.psi[i];
            }
        }
    }
}


std::vector<double> buildInitialVector(
    const CartesianGrid& grid,
    std::size_t orbitalIndex
) {
    const std::size_t nx =
        grid.getNx();

    const std::size_t ny =
        grid.getNy();

    const std::size_t nz =
        grid.getNz();

    std::vector<double> v(
        grid.getSize(),
        0.0
    );

    const double cx =
        0.5 *
        (grid.getXMin() + grid.getXMax());

    const double cy =
        0.5 *
        (grid.getYMin() + grid.getYMax());

    const double cz =
        0.5 *
        (grid.getZMin() + grid.getZMax());

    const double sigma =
        0.8 +
        0.18 *
        static_cast<double>(orbitalIndex);

    const double factor =
        1.0 /
        (2.0 * sigma * sigma);

    const std::size_t mode =
        orbitalIndex % 9;

    for (std::size_t k = 0;
         k < nz;
         ++k) {

        const double z =
            grid.getZ(k) - cz;

        for (std::size_t j = 0;
             j < ny;
             ++j) {

            const double y =
                grid.getY(j) - cy;

            for (std::size_t i = 0;
                 i < nx;
                 ++i) {

                const double x =
                    grid.getX(i) - cx;

                const double r2 =
                    x * x +
                    y * y +
                    z * z;

                const double gaussian =
                    std::exp(-r2 * factor);

                double polynomial = 1.0;

                switch (mode) {

                case 1:
                    polynomial = x;
                    break;

                case 2:
                    polynomial = y;
                    break;

                case 3:
                    polynomial = z;
                    break;

                case 4:
                    polynomial = x * y;
                    break;

                case 5:
                    polynomial = x * z;
                    break;

                case 6:
                    polynomial = y * z;
                    break;

                case 7:
                    polynomial =
                        x * x -
                        y * y;
                    break;

                case 8:
                    polynomial =
                        2.0 * z * z -
                        x * x -
                        y * y;
                    break;

                default:
                    break;
                }

                const std::size_t index =
                    k * nx * ny +
                    j * nx +
                    i;

                v[index] =
                    polynomial *
                    gaussian;
            }
        }
    }

    return v;
}


std::vector<double> buildHamiltonianDiagonal(
    const CartesianGrid& grid,
    const std::vector<double>& potential
) {
    const double dx =
        grid.getDx();

    const double dy =
        grid.getDy();

    const double dz =
        grid.getDz();

    if (dx <= 0.0 ||
        dy <= 0.0 ||
        dz <= 0.0) {

        throw std::invalid_argument(
            "Espaciamiento de malla invalido."
        );
    }

    const double kinetic =
        1.0 / (dx * dx) +
        1.0 / (dy * dy) +
        1.0 / (dz * dz);

    std::vector<double> diagonal(
        potential.size()
    );

    for (std::size_t i = 0;
         i < potential.size();
         ++i) {

        diagonal[i] =
            kinetic +
            potential[i];
    }

    return diagonal;
}


std::vector<double> davidsonCorrection(
    const std::vector<double>& residual,
    const std::vector<double>& diagonal,
    double eigenvalue
) {
    std::vector<double> correction(
        residual.size(),
        0.0
    );

    for (std::size_t i = 0;
         i < residual.size();
         ++i) {

        const double denominator =
            eigenvalue -
            diagonal[i];

        if (!std::isfinite(denominator) ||
            std::abs(denominator) <
                DAVIDSON_DENOMINATOR_TOLERANCE) {

            correction[i] =
                residual[i];

        } else {

            correction[i] =
                residual[i] /
                denominator;
        }
    }

    return correction;
}


bool buildIndependentVector(
    const CartesianGrid& grid,
    std::size_t orbitalIndex,
    const std::vector<MolecularOrbital>& orbitals,
    double dV,
    std::vector<double>& result
) {
    for (std::size_t seed = orbitalIndex;
         seed < orbitalIndex + 64;
         ++seed) {

        result =
            buildInitialVector(
                grid,
                seed
            );

        orthogonalizeAgainstOrbitals(
            result,
            orbitals,
            dV
        );

        if (!davidsonValidNorm(result, dV)) {
            continue;
        }

        davidsonNormalize(
            result,
            dV
        );

        return true;
    }

    return false;
}


bool buildInitialFromPrevious(
    const CartesianGrid& grid,
    const MolecularOrbital& initial,
    const std::vector<MolecularOrbital>& orbitals,
    double dV,
    std::vector<double>& result
) {
    if (initial.psi.size() !=
        grid.getSize()) {

        return false;
    }

    result =
        initial.psi;

    for (double value : result) {

        if (!std::isfinite(value)) {
            return false;
        }
    }

    orthogonalizeAgainstOrbitals(
        result,
        orbitals,
        dV
    );

    if (!davidsonValidNorm(result, dV)) {
        return false;
    }

    davidsonNormalize(
        result,
        dV
    );

    return true;
}


bool appendIndependentCorrection(
    std::vector<double>& correction,
    const std::vector<MolecularOrbital>& previousOrbitals,
    const std::vector<std::vector<double>>& basis,
    double dV
) {
    orthogonalizeAgainstOrbitals(
        correction,
        previousOrbitals,
        dV
    );

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

        return false;
    }

    davidsonNormalize(
        correction,
        dV,
        SUBSPACE_TOLERANCE *
        SUBSPACE_TOLERANCE
    );

    return true;
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

    const std::size_t dimension =
        grid.getSize();

    const std::vector<double> diagonal =
        buildHamiltonianDiagonal(
            grid,
            effectivePotential
        );

    std::vector<double> initialVector;

    if (initialGuess != nullptr &&
        buildInitialFromPrevious(
            grid,
            *initialGuess,
            previousOrbitals,
            dV,
            initialVector
        )) {

    } else if (!buildIndependentVector(
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

        orthogonalizeAgainstOrbitals(
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

            if (!buildIndependentVector(
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
            davidsonCorrection(
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
            appendIndependentCorrection(
                correction,
                previousOrbitals,
                basis,
                dV
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

            if (!appendIndependentCorrection(
                    correction,
                    previousOrbitals,
                    basis,
                    dV
                )) {

                /*
                 * El residual puede ser pequeno pero aun
                 * no suficientemente pequeno para satisfacer
                 * la convergencia. En ese caso se construye
                 * una direccion determinista independiente.
                 */
                if (!buildIndependentVector(
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