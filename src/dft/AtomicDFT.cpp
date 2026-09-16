#include "AtomicDFT.h"

#include "ElectronDensity.h"
#include "SelfConsistentField.h"

AtomicResult solveAtom(
    const RadialGrid& grid,
    const AtomicConfiguration& configuration
) {
    AtomicResult result;

    result.Z = configuration.Z;
    result.symbol = configuration.symbol;

    for (const ElectronicState& state : configuration.states) {
        result.electrons += state.electrons;
    }

    result.scf =
        solveSelfConsistentField(
            grid,
            configuration
        );

    return result;
}