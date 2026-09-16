#pragma once

#include "XCFunctional.h"

class PZ81 final : public XCFunctional {
public:
    XCResult evaluate(
        double alphaDensity,
        double betaDensity
    ) const override;
};