#pragma once

#include "XCInput.h"

struct XCResult {
    double energyPerElectron;

    double potentialAlpha;
    double potentialBeta;

    double gradientCoefficientAlpha;
    double gradientCoefficientBeta;
};

class XCFunctional {
public:
    virtual ~XCFunctional() = default;

    virtual XCResult evaluate(
        const XCInput& input
    ) const = 0;
};