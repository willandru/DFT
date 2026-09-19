#pragma once

#include "XCFunctional.h"

class PBE96 final : public XCFunctional {
public:
    XCResult evaluate(
        const XCInput& input
    ) const override;
};