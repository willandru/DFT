#pragma once

#include <vector>

double exchangeEnergyDensity(double density);

double correlationEnergyPerElectron(double density);

double exchangeCorrelationEnergyPerElectron(double density);

double exchangeCorrelationPotential(double density);

std::vector<double> calculateSpinExchangeCorrelationPotential(
    const std::vector<double>& alphaDensity,
    const std::vector<double>& betaDensity,
    int spin
);

std::vector<double> calculateExchangeCorrelationPotential(
    const std::vector<double>& density
);

double calculateExchangeCorrelationEnergy(
    const std::vector<double>& r,
    const std::vector<double>& density
);

double calculateSpinExchangeCorrelationEnergy(
    const std::vector<double>& r,
    const std::vector<double>& alphaDensity,
    const std::vector<double>& betaDensity
);