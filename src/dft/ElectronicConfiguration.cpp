#include "ElectronicConfiguration.h"

#include <stdexcept>

AtomicConfiguration getAtomicConfiguration(int Z) {
    switch (Z) {
    case 1:
        return {
            1,
            "H",
            {
                {1, 0, 1}
            }
        };

    case 2:
        return {
            2,
            "He",
            {
                {1, 0, 2}
            }
        };

    case 3:
        return {
            3,
            "Li",
            {
                {1, 0, 2},
                {2, 0, 1}
            }
        };

    case 4:
        return {
            4,
            "Be",
            {
                {1, 0, 2},
                {2, 0, 2}
            }
        };

    case 5:
        return {
            5,
            "B",
            {
                {1, 0, 2},
                {2, 0, 2},
                {2, 1, 1}
            }
        };

    case 6:
        return {
            6,
            "C",
            {
                {1, 0, 2},
                {2, 0, 2},
                {2, 1, 2}
            }
        };

    default:
        throw std::invalid_argument(
            "Configuracion atomica no disponible para Z = " +
            std::to_string(Z)
        );
    }
}