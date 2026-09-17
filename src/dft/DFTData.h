#pragma once

#include <cstddef>
#include <string>
#include <vector>

enum class SpinChannel {
    Alpha,
    Beta
};

struct ElectronicState {
    int n = 0;
    int l = 0;
    int alphaElectrons = 0;
    int betaElectrons = 0;
};

struct AtomicConfiguration {
    int Z = 0;
    std::string symbol;
    std::vector<ElectronicState> states;
};

struct AtomicOrbital {
    int n = 0;
    int l = 0;
    SpinChannel spin = SpinChannel::Alpha;
    int electrons = 0;
    double eigenvalue = 0.0;
    std::vector<double> u;
};

struct MolecularOrbital {
    SpinChannel spin = SpinChannel::Alpha;
    int electrons = 0;
    double eigenvalue = 0.0;
    std::vector<double> psi;
};

struct TridiagonalMatrix {
    std::vector<double> lower;
    std::vector<double> diagonal;
    std::vector<double> upper;
};

struct EnergyComponents {
    double kinetic = 0.0;
    double external = 0.0;
    double hartree = 0.0;
    double exchangeCorrelation = 0.0;
    double nuclearRepulsion = 0.0;
    double total = 0.0;
};

struct SCFResult {
    std::vector<AtomicOrbital> orbitals;
    std::vector<MolecularOrbital> molecularOrbitals;

    std::vector<double> alphaDensity;
    std::vector<double> betaDensity;
    std::vector<double> density;

    std::vector<double> alphaEffectivePotential;
    std::vector<double> betaEffectivePotential;
    std::vector<double> effectivePotential;

    EnergyComponents energy;

    double densityDifference = 0.0;
    double energyDifference = 0.0;
    double maxKSResidual = 0.0;

    int iterations = 0;
    bool converged = false;
};

struct AtomicResult {
    int Z = 0;
    std::string symbol;
    int electrons = 0;

    SCFResult scf;
};

struct MolecularResult {
    int electrons = 0;
    int charge = 0;

    SCFResult scf;
};