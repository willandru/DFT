#include "ElectronDensity.h"

#include "DFTConstants.h"

#include <cmath>
#include <stdexcept>

std::vector<double> calculateElectronDensity(
    const std::vector<double>& r,
    const std::vector<AtomicOrbital>& orbitals
) {
    if (r.empty()) {
        throw std::invalid_argument(
            "La malla radial no puede estar vacia."
        );
    }

    std::vector<double> density(
        r.size(),
        0.0
    );

    for (const AtomicOrbital& orbital : orbitals) {
        if (orbital.u.size() != r.size()) {
            throw std::invalid_argument(
                "El orbital y la malla radial deben tener el mismo tamano."
            );
        }

        if (orbital.electrons <= 0) {
            continue;
        }

        for (std::size_t i = 0;
             i < r.size();
             ++i) {

            if (r[i] <= 0.0) {
                throw std::invalid_argument(
                    "Los puntos radiales deben ser mayores que cero."
                );
            }

            density[i] +=
                static_cast<double>(orbital.electrons) *
                orbital.u[i] *
                orbital.u[i] /
                (
                    4.0 *
                    DFTConstants::PI *
                    r[i] *
                    r[i]
                );
        }
    }

    return density;
}

double integrateElectronDensity(
    const std::vector<double>& r,
    const std::vector<double>& density
) {
    if (r.size() != density.size()) {
        throw std::invalid_argument(
            "La malla radial y la densidad deben tener el mismo tamano."
        );
    }

    if (r.size() < 2) {
        throw std::invalid_argument(
            "Se necesitan al menos dos puntos radiales."
        );
    }

    const double dr =
        r[1] - r[0];

    if (dr <= 0.0) {
        throw std::invalid_argument(
            "El paso radial debe ser mayor que cero."
        );
    }

    for (double value : r) {
        if (value <= 0.0) {
            throw std::invalid_argument(
                "Los puntos radiales deben ser mayores que cero."
            );
        }
    }

    double electrons = 0.0;

    for (std::size_t i = 0;
         i < r.size();
         ++i) {

        electrons +=
            4.0 *
            DFTConstants::PI *
            r[i] *
            r[i] *
            density[i];
    }

    return electrons * dr;
}