#include "ElectronicConfiguration.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Subshell {
    int n;
    int l;
    int electrons;
};

std::vector<ElectronicState> buildStates(
    const std::vector<Subshell>& subshells
) {
    std::vector<ElectronicState> states;
    states.reserve(subshells.size());

    for (const Subshell& subshell : subshells) {
        const int capacityPerSpin =
            2 * subshell.l + 1;

        const int alphaElectrons =
            std::min(
                subshell.electrons,
                capacityPerSpin
            );

        const int betaElectrons =
            subshell.electrons -
            alphaElectrons;

        states.push_back({
            subshell.n,
            subshell.l,
            alphaElectrons,
            betaElectrons
        });
    }

    return states;
}

}

AtomicConfiguration getAtomicConfiguration(int Z) {
    if (Z < 1 || Z > 30) {
        throw std::invalid_argument(
            "Configuracion atomica no disponible para Z = " +
            std::to_string(Z)
        );
    }

    static const char* symbols[] = {
        "",
        "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",
        "Ne", "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar",
        "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co",
        "Ni", "Cu", "Zn"
    };

    std::vector<Subshell> subshells;

    subshells.push_back({
        1,
        0,
        std::min(Z, 2)
    });

    if (Z >= 3) {
        subshells.push_back({
            2,
            0,
            std::min(Z - 2, 2)
        });
    }

    if (Z >= 5) {
        subshells.push_back({
            2,
            1,
            std::min(Z - 4, 6)
        });
    }

    if (Z >= 11) {
        subshells.push_back({
            3,
            0,
            std::min(Z - 10, 2)
        });
    }

    if (Z >= 13) {
        subshells.push_back({
            3,
            1,
            std::min(Z - 12, 6)
        });
    }

    if (Z >= 19) {
        int sElectrons = 1;

        if (Z >= 20) {
            sElectrons = 2;
        }

        if (Z == 24 || Z == 29) {
            sElectrons = 1;
        }

        subshells.push_back({
            4,
            0,
            sElectrons
        });
    }

    if (Z >= 21) {
        int dElectrons =
            Z - 20;

        if (Z == 24) {
            dElectrons = 5;
        }
        else if (Z >= 29) {
            dElectrons = 10;
        }

        subshells.push_back({
            3,
            2,
            dElectrons
        });
    }

    return {
        Z,
        symbols[Z],
        buildStates(subshells)
    };
}