#pragma once

struct XCResult {
    double energyPerElectron;
    double potentialAlpha;
    double potentialBeta;
};

class XCFunctional {
public:
    virtual ~XCFunctional() = default;

    virtual XCResult evaluate(
        double alphaDensity,
        double betaDensity
    ) const = 0;
};