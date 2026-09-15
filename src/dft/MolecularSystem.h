#pragma once

#include <cstddef>
#include <string>
#include <vector>

class MolecularSystem
{
public:

    struct Atom
    {
        int atomicNumber;
        double x; // Bohr
        double y; // Bohr
        double z; // Bohr
    };

    MolecularSystem();

    MolecularSystem(
        std::vector<Atom> atoms,
        int charge = 0,
        int multiplicity = 1
    );

    void addAtom(
        int atomicNumber,
        double x,
        double y,
        double z
    );

    void clear();

    void setCharge(int charge);
    void setMultiplicity(int multiplicity);

    int getCharge() const;
    int getMultiplicity() const;

    int getElectronCount() const;
    int getAlphaElectronCount() const;
    int getBetaElectronCount() const;

    int getTotalNuclearCharge() const;

    std::size_t getAtomCount() const;

    const Atom& getAtom(std::size_t index) const;
    const std::vector<Atom>& getAtoms() const;

    static const std::string& getElementSymbol(int atomicNumber);

    std::string getFormula() const;

    bool isValid() const;

private:

    std::vector<Atom> atoms;

    int charge;
    int multiplicity;

    static void validateAtomicNumber(int atomicNumber);
    static void validateMultiplicity(int multiplicity);
};