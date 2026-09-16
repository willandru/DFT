#pragma once

#include <cstddef>

namespace DFTConstants {

constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double EPS = 1.0e-14;
constexpr double RHO_FLOOR = 1.0e-20;

constexpr std::size_t GRID_POINTS = 2000;
constexpr double RMAX = 30.0;

constexpr double MIXING = 0.25;
constexpr int MAX_SCF_ITERATIONS = 250;

constexpr double DENSITY_TOL = 1.0e-9;
constexpr double ENERGY_TOL = 1.0e-11;
constexpr double KS_RESIDUAL_TOL = 1.0e-7;

constexpr double H_EXACT = -0.5;

constexpr double ANGULAR_MOMENTUM_EPS = 1.0e-14;

constexpr double EXCHANGE_COEFFICIENT =
    0.7385587663820223;

}