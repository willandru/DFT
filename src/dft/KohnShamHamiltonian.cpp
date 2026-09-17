#include "KohnShamHamiltonian.h"

#include <cmath>
#include <stdexcept>

TridiagonalMatrix buildKohnShamHamiltonian(
    const std::vector<double>& r,
    const std::vector<double>& effectivePotential,
    int l
) {
    if (r.empty()) {
        throw std::invalid_argument(
            "La malla radial no puede estar vacia."
        );
    }

    if (r.size() != effectivePotential.size()) {
        throw std::invalid_argument(
            "La malla radial y el potencial deben tener el mismo tamano."
        );
    }

    if (l < 0) {
        throw std::invalid_argument(
            "El numero cuantico l no puede ser negativo."
        );
    }

    const std::size_t n = r.size();

    if (n < 2) {
        throw std::invalid_argument(
            "La malla radial debe contener al menos dos puntos."
        );
    }

    const double dr =
        r[1] - r[0];

    if (dr <= 0.0) {
        throw std::invalid_argument(
            "El paso radial debe ser mayor que cero."
        );
    }

    TridiagonalMatrix matrix;

    matrix.lower.resize(n - 1);
    matrix.diagonal.resize(n);
    matrix.upper.resize(n - 1);

    const double inverseDrSquared =
        1.0 / (dr * dr);

    const double kineticDiagonal =
        inverseDrSquared;

    const double kineticOffDiagonal =
        -0.5 * inverseDrSquared;

    const double centrifugalCoefficient =
        0.5 *
        static_cast<double>(
            l * (l + 1)
        );

    for (std::size_t i = 0;
         i < n;
         ++i) {

        const double ri = r[i];

        if (ri <= 0.0) {
            throw std::invalid_argument(
                "Todos los puntos radiales deben ser mayores que cero."
            );
        }

        const double centrifugal =
            centrifugalCoefficient /
            (ri * ri);

        matrix.diagonal[i] =
            kineticDiagonal +
            centrifugal +
            effectivePotential[i];
    }

    for (std::size_t i = 0;
         i < n - 1;
         ++i) {

        matrix.lower[i] =
            kineticOffDiagonal;

        matrix.upper[i] =
            kineticOffDiagonal;
    }

    return matrix;
}

std::vector<double> applyMolecularKohnShamHamiltonian(
    const CartesianGrid& grid,
    const std::vector<double>& effectivePotential,
    const std::vector<double>& psi
) {
    if (effectivePotential.size() != grid.getSize()) {
        throw std::invalid_argument(
            "El potencial efectivo y la malla cartesiana "
            "deben tener el mismo tamano."
        );
    }

    if (psi.size() != grid.getSize()) {
        throw std::invalid_argument(
            "La funcion de onda y la malla cartesiana "
            "deben tener el mismo tamano."
        );
    }

    const std::size_t nx = grid.getNx();
    const std::size_t ny = grid.getNy();
    const std::size_t nz = grid.getNz();

    const double dx = grid.getDx();
    const double dy = grid.getDy();
    const double dz = grid.getDz();

    if (dx <= 0.0 || dy <= 0.0 || dz <= 0.0) {
        throw std::invalid_argument(
            "Los pasos de la malla cartesiana deben ser mayores que cero."
        );
    }

    const double inverseDxSquared =
        1.0 / (dx * dx);

    const double inverseDySquared =
        1.0 / (dy * dy);

    const double inverseDzSquared =
        1.0 / (dz * dz);

    std::vector<double> hPsi(
        grid.getSize(),
        0.0
    );

    for (std::size_t k = 0; k < nz; ++k) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {

                const std::size_t index =
                    grid.getIndex(i, j, k);

                /*
                 * Dirichlet boundary condition:
                 *
                 * psi = 0
                 *
                 * outside the computational domain.
                 *
                 * The central-difference stencil is therefore
                 * applied with zero-valued boundary neighbors.
                 */

                const double center =
                    psi[index];

                double laplacian =
                    -2.0 *
                    (
                        inverseDxSquared +
                        inverseDySquared +
                        inverseDzSquared
                    ) *
                    center;

                if (i > 0) {
                    laplacian +=
                        psi[
                            grid.getIndex(
                                i - 1,
                                j,
                                k
                            )
                        ] *
                        inverseDxSquared;
                }

                if (i + 1 < nx) {
                    laplacian +=
                        psi[
                            grid.getIndex(
                                i + 1,
                                j,
                                k
                            )
                        ] *
                        inverseDxSquared;
                }

                if (j > 0) {
                    laplacian +=
                        psi[
                            grid.getIndex(
                                i,
                                j - 1,
                                k
                            )
                        ] *
                        inverseDySquared;
                }

                if (j + 1 < ny) {
                    laplacian +=
                        psi[
                            grid.getIndex(
                                i,
                                j + 1,
                                k
                            )
                        ] *
                        inverseDySquared;
                }

                if (k > 0) {
                    laplacian +=
                        psi[
                            grid.getIndex(
                                i,
                                j,
                                k - 1
                            )
                        ] *
                        inverseDzSquared;
                }

                if (k + 1 < nz) {
                    laplacian +=
                        psi[
                            grid.getIndex(
                                i,
                                j,
                                k + 1
                            )
                        ] *
                        inverseDzSquared;
                }

                const double kinetic =
                    -0.5 * laplacian;

                const double potential =
                    effectivePotential[index] *
                    center;

                hPsi[index] =
                    kinetic +
                    potential;
            }
        }
    }

    return hPsi;
}