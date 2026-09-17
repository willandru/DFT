#include "EigenvalueSolverMolecule.h"

#include "DFTConstants.h"
#include "KohnShamHamiltonianMolecule.h"
#include "NumericalMethods.h"

#include <algorithm>
#include <chrono>
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
constexpr double DENSE_EIGENVALUE_TOLERANCE = 1.0e-13;
constexpr std::size_t DENSE_EIGENVALUE_MAX_ITERATIONS = 10000;
constexpr double DAVIDSON_DENOMINATOR_TOLERANCE = 1.0e-10;

using Clock = std::chrono::steady_clock;

double elapsedMilliseconds(const Clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(
        Clock::now() - start
    ).count();
}

double dot(
    const std::vector<double>& a,
    const std::vector<double>& b,
    double dV
) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Los vectores deben tener el mismo tamano."
        );
    }

    return dV * dotProduct(a, b);
}

double normSquared(
    const std::vector<double>& v,
    double dV
) {
    return dot(v, v, dV);
}

bool validNorm(
    const std::vector<double>& v,
    double dV,
    double toleranceSquared = VECTOR_TOLERANCE_SQUARED
) {
    const double n2 = normSquared(v, dV);

    return std::isfinite(n2) &&
           n2 > toleranceSquared;
}

void normalize(
    std::vector<double>& v,
    double dV,
    double toleranceSquared = VECTOR_TOLERANCE_SQUARED
) {
    const double n2 = normSquared(v, dV);

    if (!std::isfinite(n2) ||
        n2 <= toleranceSquared) {

        throw std::runtime_error(
            "No se puede normalizar el vector molecular."
        );
    }

    const double n = std::sqrt(n2);

    for (double& value : v) {
        value /= n;
    }
}

void orthogonalize(
    std::vector<double>& v,
    const std::vector<std::vector<double>>& basis,
    double dV
) {
    for (int pass = 0; pass < 2; ++pass) {
        for (const auto& b : basis) {
            const double projection = dot(b, v, dV);

            for (std::size_t i = 0; i < v.size(); ++i) {
                v[i] -= projection * b[i];
            }
        }
    }
}

void orthogonalizeAgainstOrbitals(
    std::vector<double>& v,
    const std::vector<MolecularOrbital>& orbitals,
    double dV
) {
    for (int pass = 0; pass < 2; ++pass) {
        for (const auto& orbital : orbitals) {
            const double orbitalNorm =
                normSquared(orbital.psi, dV);

            if (!std::isfinite(orbitalNorm) ||
                orbitalNorm <= VECTOR_TOLERANCE_SQUARED) {

                throw std::runtime_error(
                    "Un orbital molecular previo no es valido."
                );
            }

            const double projection =
                dot(orbital.psi, v, dV) / orbitalNorm;

            for (std::size_t i = 0; i < v.size(); ++i) {
                v[i] -= projection * orbital.psi[i];
            }
        }
    }
}

double rayleigh(
    const std::vector<double>& psi,
    const std::vector<double>& hPsi,
    double dV
) {
    const double denominator = normSquared(psi, dV);

    if (!std::isfinite(denominator) ||
        denominator <= VECTOR_TOLERANCE_SQUARED) {

        throw std::runtime_error(
            "Norma invalida en el cociente de Rayleigh."
        );
    }

    return dot(psi, hPsi, dV) / denominator;
}

double residualNorm(
    const std::vector<double>& psi,
    const std::vector<double>& hPsi,
    double eigenvalue,
    double dV
) {
    std::vector<double> residual(
        psi.size(),
        0.0
    );

    for (std::size_t i = 0; i < psi.size(); ++i) {
        residual[i] =
            hPsi[i] -
            eigenvalue * psi[i];
    }

    const double n2 = normSquared(residual, dV);

    if (!std::isfinite(n2)) {
        return std::numeric_limits<double>::infinity();
    }

    return std::sqrt(n2);
}

std::vector<double> buildInitialVector(
    const CartesianGrid& grid,
    std::size_t orbitalIndex
) {
    const std::size_t nx = grid.getNx();
    const std::size_t ny = grid.getNy();
    const std::size_t nz = grid.getNz();

    std::vector<double> v(
        grid.getSize(),
        0.0
    );

    const double cx =
        0.5 * (grid.getXMin() + grid.getXMax());

    const double cy =
        0.5 * (grid.getYMin() + grid.getYMax());

    const double cz =
        0.5 * (grid.getZMin() + grid.getZMax());

    const double sigma =
        0.8 +
        0.18 * static_cast<double>(orbitalIndex);

    const double factor =
        1.0 / (2.0 * sigma * sigma);

    const std::size_t mode =
        orbitalIndex % 9;

    for (std::size_t k = 0; k < nz; ++k) {
        const double z = grid.getZ(k) - cz;

        for (std::size_t j = 0; j < ny; ++j) {
            const double y = grid.getY(j) - cy;

            for (std::size_t i = 0; i < nx; ++i) {
                const double x = grid.getX(i) - cx;

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

struct Eigenpair {
    double value;
    std::vector<double> vector;
};

Eigenpair diagonalize(
    std::vector<std::vector<double>> matrix
) {
    const std::size_t n = matrix.size();

    if (n == 0) {
        throw std::invalid_argument(
            "La matriz no puede estar vacia."
        );
    }

    std::vector<std::vector<double>> vectors(
        n,
        std::vector<double>(n, 0.0)
    );

    for (std::size_t i = 0; i < n; ++i) {
        vectors[i][i] = 1.0;
    }

    for (std::size_t iteration = 0;
         iteration < DENSE_EIGENVALUE_MAX_ITERATIONS;
         ++iteration) {

        double maximum = 0.0;
        std::size_t p = 0;
        std::size_t q = 0;

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                const double value =
                    std::abs(matrix[i][j]);

                if (value > maximum) {
                    maximum = value;
                    p = i;
                    q = j;
                }
            }
        }

        if (maximum <= DENSE_EIGENVALUE_TOLERANCE) {
            break;
        }

        const double app = matrix[p][p];
        const double aqq = matrix[q][q];
        const double apq = matrix[p][q];

        if (apq == 0.0) {
            continue;
        }

        const double tau =
            (aqq - app) / (2.0 * apq);

        const double t =
            std::copysign(
                1.0 /
                (
                    std::abs(tau) +
                    std::sqrt(1.0 + tau * tau)
                ),
                tau
            );

        const double c =
            1.0 /
            std::sqrt(1.0 + t * t);

        const double s = t * c;

        matrix[p][p] =
            app - t * apq;

        matrix[q][q] =
            aqq + t * apq;

        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;

        for (std::size_t k = 0; k < n; ++k) {
            if (k == p || k == q) {
                continue;
            }

            const double mkp = matrix[k][p];
            const double mkq = matrix[k][q];

            matrix[k][p] =
                c * mkp -
                s * mkq;

            matrix[p][k] =
                matrix[k][p];

            matrix[k][q] =
                s * mkp +
                c * mkq;

            matrix[q][k] =
                matrix[k][q];
        }

        for (std::size_t k = 0; k < n; ++k) {
            const double vkp = vectors[k][p];
            const double vkq = vectors[k][q];

            vectors[k][p] =
                c * vkp -
                s * vkq;

            vectors[k][q] =
                s * vkp +
                c * vkq;
        }
    }

    std::size_t index = 0;

    for (std::size_t i = 1; i < n; ++i) {
        if (matrix[i][i] <
            matrix[index][index]) {

            index = i;
        }
    }

    std::vector<double> eigenvector(n);

    for (std::size_t i = 0; i < n; ++i) {
        eigenvector[i] =
            vectors[i][index];
    }

    double norm = 0.0;

    for (double value : eigenvector) {
        norm += value * value;
    }

    norm = std::sqrt(norm);

    if (!std::isfinite(norm) ||
        norm <= DFTConstants::EPS) {

        throw std::runtime_error(
            "Autovector denso invalido."
        );
    }

    for (double& value : eigenvector) {
        value /= norm;
    }

    return {
        matrix[index][index],
        std::move(eigenvector)
    };
}

std::vector<double> combine(
    const std::vector<std::vector<double>>& basis,
    const std::vector<double>& coefficients
) {
    if (basis.empty()) {
        throw std::invalid_argument(
            "La base no puede estar vacia."
        );
    }

    if (basis.size() != coefficients.size()) {
        throw std::invalid_argument(
            "La base y los coeficientes no coinciden."
        );
    }

    std::vector<double> result(
        basis.front().size(),
        0.0
    );

    for (std::size_t j = 0; j < basis.size(); ++j) {
        for (std::size_t i = 0; i < result.size(); ++i) {
            result[i] +=
                coefficients[j] *
                basis[j][i];
        }
    }

    return result;
}

std::vector<double> buildHamiltonianDiagonal(
    const CartesianGrid& grid,
    const std::vector<double>& potential
) {
    const double dx = grid.getDx();
    const double dy = grid.getDy();
    const double dz = grid.getDz();

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

        if (!validNorm(result, dV)) {
            continue;
        }

        normalize(result, dV);

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
    if (initial.psi.size() != grid.getSize()) {
        return false;
    }

    result = initial.psi;

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

    if (!validNorm(result, dV)) {
        return false;
    }

    normalize(result, dV);

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

    orthogonalize(
        correction,
        basis,
        dV
    );

    if (!validNorm(
            correction,
            dV,
            SUBSPACE_TOLERANCE *
            SUBSPACE_TOLERANCE
        )) {

        return false;
    }

    normalize(
        correction,
        dV,
        SUBSPACE_TOLERANCE *
        SUBSPACE_TOLERANCE
    );

    return true;
}

} // namespace

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
    const Clock::time_point totalStart =
        Clock::now();

    if (effectivePotential.size() != grid.getSize()) {
        throw std::invalid_argument(
            "El potencial y la malla no coinciden."
        );
    }

    if (electrons < 0 || electrons > 2) {
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

    basis.reserve(MAX_SUBSPACE_DIMENSION);
    hBasis.reserve(MAX_SUBSPACE_DIMENSION);

    basis.push_back(
        std::move(initialVector)
    );

    std::vector<std::vector<double>>
        projectedHamiltonian;

    std::size_t hApplications = 0;
    std::size_t davidsonIterations = 0;
    std::size_t davidsonRestarts = 0;
    std::size_t maximumSubspaceDimension = 1;

    double finalEigenvalue =
        std::numeric_limits<double>::infinity();

    double finalResidual =
        std::numeric_limits<double>::infinity();

    std::vector<double> finalVector;

    while (hApplications < maxIterations) {
        ++davidsonIterations;

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

            maximumSubspaceDimension =
                std::max(
                    maximumSubspaceDimension,
                    size
                );

            projectedHamiltonian.resize(size);

            for (auto& row : projectedHamiltonian) {
                row.resize(size, 0.0);
            }

            for (std::size_t i = 0;
                 i <= index;
                 ++i) {

                const double value =
                    dot(
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

        const Eigenpair projected =
            diagonalize(
                projectedHamiltonian
            );

        std::vector<double> ritz =
            combine(
                basis,
                projected.vector
            );

        if (!validNorm(ritz, dV)) {
            throw std::runtime_error(
                "El vector de Ritz es numericamente nulo."
            );
        }

        normalize(ritz, dV);

        std::vector<double> hRitz =
            applyMolecularKohnShamHamiltonian(
                grid,
                effectivePotential,
                ritz
            );

        ++hApplications;

        finalEigenvalue =
            rayleigh(
                ritz,
                hRitz,
                dV
            );

        finalResidual =
            residualNorm(
                ritz,
                hRitz,
                finalEigenvalue,
                dV
            );

        finalVector = ritz;

        if (finalResidual < MO_RESIDUAL_TOLERANCE) {
            const double totalMilliseconds =
                elapsedMilliseconds(totalStart);

            std::cerr
                << "[MO] "
                << "Orbital=" << orbitalIndex
                << " | Spin="
                << (
                    spin == SpinChannel::Alpha
                        ? "Alpha"
                        : "Beta"
                )
                << " | E=" << finalEigenvalue
                << " | residual=" << finalResidual
                << " | H=" << hApplications
                << " | Iter=" << davidsonIterations
                << " | Subspace="
                << maximumSubspaceDimension
                << " | Restarts="
                << davidsonRestarts
                << " | Time="
                << totalMilliseconds
                << " ms\n";

            break;
        }

        std::vector<double> residual(
            dimension,
            0.0
        );

        for (std::size_t i = 0;
             i < dimension;
             ++i) {

            residual[i] =
                hRitz[i] -
                finalEigenvalue * ritz[i];
        }

        orthogonalizeAgainstOrbitals(
            residual,
            previousOrbitals,
            dV
        );

        if (!validNorm(
                residual,
                dV,
                SUBSPACE_TOLERANCE *
                SUBSPACE_TOLERANCE
            )) {

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

            orthogonalize(
                independent,
                basis,
                dV
            );

            if (!validNorm(
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

            normalize(
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

            if (hApplications >= maxIterations) {
                break;
            }

            std::vector<double> hIndependent =
                applyMolecularKohnShamHamiltonian(
                    grid,
                    effectivePotential,
                    independent
                );

            ++hApplications;
            ++davidsonRestarts;

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

            maximumSubspaceDimension =
                std::max(
                    maximumSubspaceDimension,
                    basis.size()
                );

            projectedHamiltonian.assign(
                2,
                std::vector<double>(2, 0.0)
            );

            projectedHamiltonian[0][0] =
                dot(
                    basis[0],
                    hBasis[0],
                    dV
                );

            projectedHamiltonian[0][1] =
                dot(
                    basis[0],
                    hBasis[1],
                    dV
                );

            projectedHamiltonian[1][0] =
                projectedHamiltonian[0][1];

            projectedHamiltonian[1][1] =
                dot(
                    basis[1],
                    hBasis[1],
                    dV
                );

            continue;
        }

        std::vector<double> correction =
            davidsonCorrection(
                residual,
                diagonal,
                finalEigenvalue
            );

        if (!appendIndependentCorrection(
                correction,
                previousOrbitals,
                basis,
                dV
            )) {

            correction = residual;

            if (!appendIndependentCorrection(
                    correction,
                    previousOrbitals,
                    basis,
                    dV
                )) {

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

                orthogonalize(
                    correction,
                    basis,
                    dV
                );

                if (!validNorm(
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

                normalize(
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

            maximumSubspaceDimension =
                std::max(
                    maximumSubspaceDimension,
                    basis.size()
                );

            continue;
        }

        if (hApplications >= maxIterations) {
            break;
        }

        std::vector<double> hCorrection =
            applyMolecularKohnShamHamiltonian(
                grid,
                effectivePotential,
                correction
            );

        ++hApplications;
        ++davidsonRestarts;

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

        maximumSubspaceDimension =
            std::max(
                maximumSubspaceDimension,
                basis.size()
            );

        projectedHamiltonian.assign(
            2,
            std::vector<double>(2, 0.0)
        );

        projectedHamiltonian[0][0] =
            dot(
                basis[0],
                hBasis[0],
                dV
            );

        projectedHamiltonian[0][1] =
            dot(
                basis[0],
                hBasis[1],
                dV
            );

        projectedHamiltonian[1][0] =
            projectedHamiltonian[0][1];

        projectedHamiltonian[1][1] =
            dot(
                basis[1],
                hBasis[1],
                dV
            );
    }

    if (finalVector.empty() ||
        finalResidual >= MO_RESIDUAL_TOLERANCE) {

        const double totalMilliseconds =
            elapsedMilliseconds(totalStart);

        std::cerr
            << "[MO] "
            << "Orbital=" << orbitalIndex
            << " | NO CONVERGIO"
            << " | residual=" << finalResidual
            << " | H=" << hApplications
            << " | Iter=" << davidsonIterations
            << " | Subspace="
            << maximumSubspaceDimension
            << " | Restarts="
            << davidsonRestarts
            << " | Time="
            << totalMilliseconds
            << " ms\n";

        throw std::runtime_error(
            "El orbital molecular no convergio. "
            "Orbital = " +
            std::to_string(orbitalIndex) +
            ", residual = " +
            std::to_string(finalResidual)
        );
    }

    MolecularOrbital orbital;

    orbital.spin = spin;
    orbital.electrons = electrons;
    orbital.eigenvalue = finalEigenvalue;
    orbital.psi = std::move(finalVector);

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

    if (occupations.size() != numberOfOrbitals) {
        throw std::invalid_argument(
            "Las ocupaciones no coinciden con "
            "el numero de orbitales."
        );
    }

    if (!initialOrbitals.empty() &&
        initialOrbitals.size() != numberOfOrbitals) {

        throw std::invalid_argument(
            "Los orbitales iniciales no coinciden "
            "con el numero solicitado."
        );
    }

    std::vector<MolecularOrbital> orbitals;

    orbitals.reserve(numberOfOrbitals);

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