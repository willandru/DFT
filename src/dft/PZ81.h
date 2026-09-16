#pragma once

#include "XCFunctional.h"

class PZ81 final : public XCFunctional {
public:
    XCResult evaluate(
        const XCInput& input
    ) const override;
};