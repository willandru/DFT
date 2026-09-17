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