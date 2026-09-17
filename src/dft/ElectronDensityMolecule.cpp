#include "ElectronDensityMolecule.h"

#include <stdexcept>

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