#include "AtomicDFT.h"

#include "SelfConsistentField.h"

#include <stdexcept>

AtomicResult solveAtom(
    const RadialGrid& grid,
    const AtomicConfiguration& configuration,
    const XCFunctional& functional
) {
    if (configuration.Z <= 0) {
        throw std::invalid_argument(
            "El numero atomico debe ser mayor que cero."
        );
    }

    AtomicResult result;

    result.Z =
        configuration.Z;

    result.symbol =
        configuration.symbol;

    for (const ElectronicState& state :
         configuration.states) {

        result.electrons +=
            state.alphaElectrons +
            state.betaElectrons;
    }

    result.scf =
        solveSelfConsistentField(
            grid,
            configuration,
            functional
        );

    return result;
}