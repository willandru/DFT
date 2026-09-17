#include "HartreePotential.h"

#include "DFTConstants.h"

#include <cmath>
#include <stdexcept>

std::vector<double> calculateHartreePotential(
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

    const std::size_t n = r.size();
    const double dr = r[1] - r[0];

    if (dr <= 0.0) {
        throw std::invalid_argument(
            "El paso radial debe ser mayor que cero."
        );
    }

    for (double radius : r) {
        if (radius <= 0.0) {
            throw std::invalid_argument(
                "Los puntos radiales deben ser mayores que cero."
            );
        }
    }

    std::vector<double> enclosedCharge(n, 0.0);
    std::vector<double> outerIntegral(n, 0.0);
    std::vector<double> potential(n, 0.0);

    for (std::size_t i = 1; i < n; ++i) {
        const double r0 = r[i - 1];
        const double r1 = r[i];

        const double f0 =
            4.0 * DFTConstants::PI *
            r0 * r0 * density[i - 1];

        const double f1 =
            4.0 * DFTConstants::PI *
            r1 * r1 * density[i];

        enclosedCharge[i] =
            enclosedCharge[i - 1] +
            0.5 * (f0 + f1) * dr;
    }

    for (std::size_t i = n - 1; i-- > 0;) {
        const double r0 = r[i];
        const double r1 = r[i + 1];

        const double f0 =
            4.0 * DFTConstants::PI *
            r0 * density[i];

        const double f1 =
            4.0 * DFTConstants::PI *
            r1 * density[i + 1];

        outerIntegral[i] =
            outerIntegral[i + 1] +
            0.5 * (f0 + f1) * dr;
    }

    for (std::size_t i = 0; i < n; ++i) {
        potential[i] =
            enclosedCharge[i] / r[i] +
            outerIntegral[i];
    }

    return potential;
}

double calculateHartreeEnergy(
    const std::vector<double>& r,
    const std::vector<double>& density,
    const std::vector<double>& hartreePotential
) {
    if (r.size() != density.size() ||
        r.size() != hartreePotential.size()) {
        throw std::invalid_argument(
            "La malla, densidad y potencial de Hartree deben tener el mismo tamano."
        );
    }

    if (r.size() < 2) {
        throw std::invalid_argument(
            "Se necesitan al menos dos puntos radiales."
        );
    }

    const double dr = r[1] - r[0];

    if (dr <= 0.0) {
        throw std::invalid_argument(
            "El paso radial debe ser mayor que cero."
        );
    }

    double energy = 0.0;

    for (std::size_t i = 0; i < r.size(); ++i) {
        if (r[i] <= 0.0) {
            throw std::invalid_argument(
                "Los puntos radiales deben ser mayores que cero."
            );
        }

        const double integrand =
            4.0 * DFTConstants::PI *
            r[i] * r[i] *
            density[i] *
            hartreePotential[i];

        if (i == 0 || i == r.size() - 1) {
            energy += 0.5 * integrand;
        } else {
            energy += integrand;
        }
    }

    return 0.5 * energy * dr;
}