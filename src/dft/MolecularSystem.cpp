#include "MolecularSystem.h"

#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

MolecularSystem::MolecularSystem()
    : charge(0),
      multiplicity(1)
{
}

MolecularSystem::MolecularSystem(
    std::vector<Atom> atoms,
    int charge,
    int multiplicity
)
    : atoms(std::move(atoms)),
      charge(charge),
      multiplicity(multiplicity)
{
    validateMultiplicity(multiplicity);

    for (const Atom& atom : this->atoms)
    {
        validateAtomicNumber(atom.atomicNumber);
    }

    if (!isValid())
    {
        throw std::invalid_argument(
            "MolecularSystem: invalid charge/multiplicity."
        );
    }
}

void MolecularSystem::addAtom(
    int atomicNumber,
    double x,
    double y,
    double z
)
{
    validateAtomicNumber(atomicNumber);

    atoms.push_back({ atomicNumber, x, y, z });

    if (!isValid())
    {
        atoms.pop_back();

        throw std::invalid_argument(
            "MolecularSystem: adding this atom creates "
            "an invalid electronic configuration."
        );
    }
}

void MolecularSystem::clear()
{
    atoms.clear();
    charge = 0;
    multiplicity = 1;
}

void MolecularSystem::setCharge(int newCharge)
{
    const int oldCharge = charge;
    charge = newCharge;

    if (!isValid())
    {
        charge = oldCharge;

        throw std::invalid_argument(
            "MolecularSystem: invalid charge."
        );
    }
}

void MolecularSystem::setMultiplicity(int newMultiplicity)
{
    validateMultiplicity(newMultiplicity);

    const int oldMultiplicity = multiplicity;
    multiplicity = newMultiplicity;

    if (!isValid())
    {
        multiplicity = oldMultiplicity;

        throw std::invalid_argument(
            "MolecularSystem: invalid multiplicity."
        );
    }
}

int MolecularSystem::getCharge() const
{
    return charge;
}

int MolecularSystem::getMultiplicity() const
{
    return multiplicity;
}

int MolecularSystem::getElectronCount() const
{
    return getTotalNuclearCharge() - charge;
}

int MolecularSystem::getAlphaElectronCount() const
{
    const int electrons = getElectronCount();
    const int spinDifference = multiplicity - 1;

    return (electrons + spinDifference) / 2;
}

int MolecularSystem::getBetaElectronCount() const
{
    const int electrons = getElectronCount();
    const int spinDifference = multiplicity - 1;

    return (electrons - spinDifference) / 2;
}

int MolecularSystem::getTotalNuclearCharge() const
{
    int total = 0;

    for (const Atom& atom : atoms)
    {
        total += atom.atomicNumber;
    }

    return total;
}

std::size_t MolecularSystem::getAtomCount() const
{
    return atoms.size();
}

const MolecularSystem::Atom& MolecularSystem::getAtom(
    std::size_t index
) const
{
    if (index >= atoms.size())
    {
        throw std::out_of_range(
            "MolecularSystem: atom index out of range."
        );
    }

    return atoms[index];
}

const std::vector<MolecularSystem::Atom>&
MolecularSystem::getAtoms() const
{
    return atoms;
}

const std::string& MolecularSystem::getElementSymbol(
    int atomicNumber
)
{
    static const std::unordered_map<int, std::string> symbols =
    {
        {1, "H"},   {2, "He"},  {3, "Li"},  {4, "Be"},
        {5, "B"},   {6, "C"},   {7, "N"},   {8, "O"},
        {9, "F"},   {10, "Ne"}, {11, "Na"}, {12, "Mg"},
        {13, "Al"}, {14, "Si"}, {15, "P"},  {16, "S"},
        {17, "Cl"}, {18, "Ar"}, {19, "K"},  {20, "Ca"},
        {21, "Sc"}, {22, "Ti"}, {23, "V"},  {24, "Cr"},
        {25, "Mn"}, {26, "Fe"}, {27, "Co"}, {28, "Ni"},
        {29, "Cu"}, {30, "Zn"}, {31, "Ga"}, {32, "Ge"},
        {33, "As"}, {34, "Se"}, {35, "Br"}, {36, "Kr"},
        {37, "Rb"}, {38, "Sr"}, {39, "Y"},  {40, "Zr"},
        {41, "Nb"}, {42, "Mo"}, {43, "Tc"}, {44, "Ru"},
        {45, "Rh"}, {46, "Pd"}, {47, "Ag"}, {48, "Cd"},
        {49, "In"}, {50, "Sn"}, {51, "Sb"}, {52, "Te"},
        {53, "I"},  {54, "Xe"}, {55, "Cs"}, {56, "Ba"},
        {57, "La"}, {58, "Ce"}, {59, "Pr"}, {60, "Nd"},
        {61, "Pm"}, {62, "Sm"}, {63, "Eu"}, {64, "Gd"},
        {65, "Tb"}, {66, "Dy"}, {67, "Ho"}, {68, "Er"},
        {69, "Tm"}, {70, "Yb"}, {71, "Lu"}, {72, "Hf"},
        {73, "Ta"}, {74, "W"},  {75, "Re"}, {76, "Os"},
        {77, "Ir"}, {78, "Pt"}, {79, "Au"}, {80, "Hg"},
        {81, "Tl"}, {82, "Pb"}, {83, "Bi"}, {84, "Po"},
        {85, "At"}, {86, "Rn"}, {87, "Fr"}, {88, "Ra"},
        {89, "Ac"}, {90, "Th"}, {91, "Pa"}, {92, "U"},
        {93, "Np"}, {94, "Pu"}, {95, "Am"}, {96, "Cm"},
        {97, "Bk"}, {98, "Cf"}, {99, "Es"}, {100, "Fm"},
        {101, "Md"}, {102, "No"}, {103, "Lr"}, {104, "Rf"},
        {105, "Db"}, {106, "Sg"}, {107, "Bh"}, {108, "Hs"},
        {109, "Mt"}, {110, "Ds"}, {111, "Rg"}, {112, "Cn"},
        {113, "Nh"}, {114, "Fl"}, {115, "Mc"}, {116, "Lv"},
        {117, "Ts"}, {118, "Og"}
    };

    const auto it = symbols.find(atomicNumber);

    if (it == symbols.end())
    {
        throw std::invalid_argument(
            "MolecularSystem: invalid atomic number."
        );
    }

    return it->second;
}

std::string MolecularSystem::getFormula() const
{
    std::map<int, int> counts;

    for (const Atom& atom : atoms)
    {
        ++counts[atom.atomicNumber];
    }

    std::string formula;

    for (const auto& [atomicNumber, count] : counts)
    {
        formula += getElementSymbol(atomicNumber);

        if (count > 1)
        {
            formula += std::to_string(count);
        }
    }

    return formula;
}

bool MolecularSystem::isValid() const
{
    const int electrons = getElectronCount();

    if (electrons < 0)
    {
        return false;
    }

    const int spinDifference = multiplicity - 1;

    if (spinDifference > electrons)
    {
        return false;
    }

    if ((electrons % 2) != (spinDifference % 2))
    {
        return false;
    }

    return true;
}

void MolecularSystem::validateAtomicNumber(int atomicNumber)
{
    if (atomicNumber < 1 || atomicNumber > 118)
    {
        throw std::invalid_argument(
            "MolecularSystem: atomic number must be between 1 and 118."
        );
    }
}

void MolecularSystem::validateMultiplicity(int multiplicity)
{
    if (multiplicity < 1)
    {
        throw std::invalid_argument(
            "MolecularSystem: multiplicity must be >= 1."
        );
    }
}