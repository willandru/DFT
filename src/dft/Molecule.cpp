#include "Molecule.h"

#include <stdexcept>

Molecule::Molecule(int charge)
    : charge_(charge)
{
}

void Molecule::addNucleus(
    int atomicNumber,
    double x,
    double y,
    double z
)
{
    if (atomicNumber <= 0)
    {
        throw std::invalid_argument(
            "Atomic number must be greater than zero."
        );
    }

    Nucleus nucleus;
    nucleus.atomicNumber = atomicNumber;
    nucleus.position = { x, y, z };

    nuclei_.push_back(nucleus);
}

void Molecule::clear()
{
    nuclei_.clear();
}

void Molecule::setCharge(int charge)
{
    charge_ = charge;
}

int Molecule::getCharge() const
{
    return charge_;
}

int Molecule::getElectronCount() const
{
    int totalNuclearCharge = 0;

    for (const Nucleus& nucleus : nuclei_)
    {
        totalNuclearCharge += nucleus.atomicNumber;
    }

    return totalNuclearCharge - charge_;
}

std::size_t Molecule::getNucleusCount() const
{
    return nuclei_.size();
}

const std::vector<Molecule::Nucleus>& Molecule::getNuclei() const
{
    return nuclei_;
}