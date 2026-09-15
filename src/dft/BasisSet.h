#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "MolecularSystem.h"

class BasisSet
{
public:

    struct PrimitiveGaussian
    {
        double exponent;
        double coefficient;
    };

    struct BasisFunction
    {
        std::size_t atomIndex;

        int angularMomentumX;
        int angularMomentumY;
        int angularMomentumZ;

        std::vector<PrimitiveGaussian> primitives;

        std::string label;
    };

    BasisSet();

    explicit BasisSet(const MolecularSystem& system);

    void initialize(const MolecularSystem& system);

    void clear();

    void addFunction(
        std::size_t atomIndex,
        int angularMomentumX,
        int angularMomentumY,
        int angularMomentumZ,
        std::vector<PrimitiveGaussian> primitives,
        const std::string& label
    );

    std::size_t getFunctionCount() const;

    const BasisFunction& getFunction(
        std::size_t index
    ) const;

    const std::vector<BasisFunction>& getFunctions() const;

    const MolecularSystem& getMolecularSystem() const;

    const MolecularSystem::Atom& getAtom(
        std::size_t atomIndex
    ) const;

    bool isInitialized() const;

private:

    const MolecularSystem* molecularSystem;

    std::vector<BasisFunction> functions;

    static void validateAngularMomentum(
        int angularMomentumX,
        int angularMomentumY,
        int angularMomentumZ
    );

    static void validatePrimitive(
        const PrimitiveGaussian& primitive
    );
};