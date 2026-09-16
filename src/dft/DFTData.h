#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct AtomicOrbital {
    int n = 0;
    int l = 0;
    int electrons = 0;
    double eigenvalue = 0.0;
    std::vector<double> u;
};

struct ElectronicState {
    int n = 0;
    int l = 0;
    int electrons = 0;
};

struct AtomicConfiguration {
    int Z = 0;
    std::string symbol;
    std::vector<ElectronicState> states;
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
    double total = 0.0;
};

struct SCFResult {
    std::vector<AtomicOrbital> orbitals;
    std::vector<double> density;
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