#include "RadialGrid.h"

#include <stdexcept>

RadialGrid::RadialGrid(std::size_t points, double rMax)
    : rMax_(rMax),
      dr_(rMax / static_cast<double>(points + 1)) {
    if (points == 0) {
        throw std::invalid_argument("El numero de puntos debe ser mayor que cero.");
    }

    if (rMax <= 0.0) {
        throw std::invalid_argument("RMAX debe ser mayor que cero.");
    }

    r_.resize(points);

    for (std::size_t i = 0; i < points; ++i) {
        r_[i] = static_cast<double>(i + 1) * dr_;
    }
}

std::size_t RadialGrid::size() const {
    return r_.size();
}

double RadialGrid::rMax() const {
    return rMax_;
}

double RadialGrid::spacing() const {
    return dr_;
}

const std::vector<double>& RadialGrid::coordinates() const {
    return r_;
}