#include "ElectronDensity.h"

#include "DFTConstants.h"

#include <cmath>
#include <stdexcept>

std::vector<double> calculateSpinDensity(
    const std::vector<double>& r,
    const std::vector<AtomicOrbital>& orbitals,
    SpinChannel spin
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

        if (orbital.spin != spin || orbital.electrons <= 0) {
            continue;
        }

        for (std::size_t i = 0; i < r.size(); ++i) {
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

std::vector<double> calculateElectronDensity(
    const std::vector<double>& r,
    const std::vector<AtomicOrbital>& orbitals
) {
    const std::vector<double> alphaDensity =
        calculateSpinDensity(
            r,
            orbitals,
            SpinChannel::Alpha
        );

    const std::vector<double> betaDensity =
        calculateSpinDensity(
            r,
            orbitals,
            SpinChannel::Beta
        );

    std::vector<double> density(
        r.size(),
        0.0
    );

    for (std::size_t i = 0; i < r.size(); ++i) {
        density[i] =
            alphaDensity[i] +
            betaDensity[i];
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

    for (std::size_t i = 0; i < r.size(); ++i) {
        electrons +=
            4.0 *
            DFTConstants::PI *
            r[i] *
            r[i] *
            density[i];
    }

    return electrons * dr;
}

std::vector<double> calculateMolecularSpinDensity(
    const std::vector<MolecularOrbital>& orbitals,
    SpinChannel spin
) {
    if (orbitals.empty()) {
        throw std::invalid_argument(
            "La lista de orbitales moleculares no puede estar vacia."
        );
    }

    std::size_t gridSize = 0;
    bool gridSizeInitialized = false;

    for (const MolecularOrbital& orbital : orbitals) {
        if (orbital.spin != spin || orbital.electrons <= 0) {
            continue;
        }

        if (orbital.psi.empty()) {
            throw std::invalid_argument(
                "Un orbital molecular ocupado no puede tener una funcion de onda vacia."
            );
        }

        if (!gridSizeInitialized) {
            gridSize = orbital.psi.size();
            gridSizeInitialized = true;
        }
        else if (orbital.psi.size() != gridSize) {
            throw std::invalid_argument(
                "Todos los orbitales moleculares deben estar definidos "
                "sobre la misma malla."
            );
        }
    }

    if (!gridSizeInitialized) {
        for (const MolecularOrbital& orbital : orbitals) {
            if (!orbital.psi.empty()) {
                gridSize = orbital.psi.size();
                gridSizeInitialized = true;
                break;
            }
        }
    }

    if (!gridSizeInitialized) {
        throw std::invalid_argument(
            "No existen funciones de onda moleculares validas."
        );
    }

    std::vector<double> density(
        gridSize,
        0.0
    );

    for (const MolecularOrbital& orbital : orbitals) {
        if (orbital.spin != spin || orbital.electrons <= 0) {
            continue;
        }

        for (std::size_t i = 0; i < gridSize; ++i) {
            density[i] +=
                static_cast<double>(orbital.electrons) *
                orbital.psi[i] *
                orbital.psi[i];
        }
    }

    return density;
}

std::vector<double> calculateMolecularElectronDensity(
    const std::vector<MolecularOrbital>& orbitals
) {
    const std::vector<double> alphaDensity =
        calculateMolecularSpinDensity(
            orbitals,
            SpinChannel::Alpha
        );

    const std::vector<double> betaDensity =
        calculateMolecularSpinDensity(
            orbitals,
            SpinChannel::Beta
        );

    if (alphaDensity.size() != betaDensity.size()) {
        throw std::runtime_error(
            "Las densidades alpha y beta deben tener el mismo tamano."
        );
    }

    std::vector<double> density(
        alphaDensity.size(),
        0.0
    );

    for (std::size_t i = 0;
         i < density.size();
         ++i) {

        density[i] =
            alphaDensity[i] +
            betaDensity[i];
    }

    return density;
}

double integrateMolecularElectronDensity(
    const std::vector<double>& density,
    double dx,
    double dy,
    double dz
) {
    if (density.empty()) {
        throw std::invalid_argument(
            "La densidad molecular no puede estar vacia."
        );
    }

    if (dx <= 0.0 || dy <= 0.0 || dz <= 0.0) {
        throw std::invalid_argument(
            "Los pasos de la malla cartesiana deben ser mayores que cero."
        );
    }

    double electrons = 0.0;

    for (double value : density) {
        electrons += value;
    }

    return electrons * dx * dy * dz;
}