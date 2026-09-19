#pragma once

#include <array>
#include <vector>

class Molecule
{
public:
    struct Nucleus
    {
        int atomicNumber;
        std::array<double, 3> position;
    };

    explicit Molecule(int charge = 0);

    void addNucleus(
        int atomicNumber,
        double x,
        double y,
        double z
    );

    void clear();

    void setCharge(int charge);

    int getCharge() const;
    int getElectronCount() const;
    std::size_t getNucleusCount() const;

    const std::vector<Nucleus>& getNuclei() const;

private:
    std::vector<Nucleus> nuclei_;
    int charge_;
};