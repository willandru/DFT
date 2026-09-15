#include "BasisSet.h"

#include <cmath>
#include <stdexcept>
#include <utility>

BasisSet::BasisSet()
    : molecularSystem(nullptr)
{
}

BasisSet::BasisSet(const MolecularSystem& system)
    : molecularSystem(nullptr)
{
    initialize(system);
}

void BasisSet::initialize(const MolecularSystem& system)
{
    if (!system.isValid())
    {
        throw std::invalid_argument(
            "BasisSet: cannot initialize from an invalid molecular system."
        );
    }

    molecularSystem = &system;
    functions.clear();
}

void BasisSet::clear()
{
    functions.clear();
}

void BasisSet::addFunction(
    std::size_t atomIndex,
    int angularMomentumX,
    int angularMomentumY,
    int angularMomentumZ,
    std::vector<PrimitiveGaussian> primitives,
    const std::string& label
)
{
    if (!isInitialized())
    {
        throw std::runtime_error(
            "BasisSet: molecular system has not been initialized."
        );
    }

    if (atomIndex >= molecularSystem->getAtomCount())
    {
        throw std::out_of_range(
            "BasisSet: atom index out of range."
        );
    }

    validateAngularMomentum(
        angularMomentumX,
        angularMomentumY,
        angularMomentumZ
    );

    if (primitives.empty())
    {
        throw std::invalid_argument(
            "BasisSet: a basis function must contain at least "
            "one primitive Gaussian."
        );
    }

    for (const PrimitiveGaussian& primitive : primitives)
    {
        validatePrimitive(primitive);
    }

    if (label.empty())
    {
        throw std::invalid_argument(
            "BasisSet: basis function label cannot be empty."
        );
    }

    functions.push_back(
        {
            atomIndex,
            angularMomentumX,
            angularMomentumY,
            angularMomentumZ,
            std::move(primitives),
            label
        }
    );
}

std::size_t BasisSet::getFunctionCount() const
{
    return functions.size();
}

const BasisSet::BasisFunction&
BasisSet::getFunction(std::size_t index) const
{
    if (index >= functions.size())
    {
        throw std::out_of_range(
            "BasisSet: basis function index out of range."
        );
    }

    return functions[index];
}

const std::vector<BasisSet::BasisFunction>&
BasisSet::getFunctions() const
{
    return functions;
}

const MolecularSystem&
BasisSet::getMolecularSystem() const
{
    if (!isInitialized())
    {
        throw std::runtime_error(
            "BasisSet: molecular system has not been initialized."
        );
    }

    return *molecularSystem;
}

const MolecularSystem::Atom&
BasisSet::getAtom(std::size_t atomIndex) const
{
    if (!isInitialized())
    {
        throw std::runtime_error(
            "BasisSet: molecular system has not been initialized."
        );
    }

    return molecularSystem->getAtom(atomIndex);
}

bool BasisSet::isInitialized() const
{
    return molecularSystem != nullptr;
}

void BasisSet::validateAngularMomentum(
    int angularMomentumX,
    int angularMomentumY,
    int angularMomentumZ
)
{
    if (angularMomentumX < 0 ||
        angularMomentumY < 0 ||
        angularMomentumZ < 0)
    {
        throw std::invalid_argument(
            "BasisSet: angular momentum components must be non-negative."
        );
    }
}

void BasisSet::validatePrimitive(
    const PrimitiveGaussian& primitive
)
{
    if (!std::isfinite(primitive.exponent) ||
        primitive.exponent <= 0.0)
    {
        throw std::invalid_argument(
            "BasisSet: Gaussian exponent must be finite and positive."
        );
    }

    if (!std::isfinite(primitive.coefficient))
    {
        throw std::invalid_argument(
            "BasisSet: Gaussian coefficient must be finite."
        );
    }
}