#pragma once

#include <cstddef>
#include <vector>

class RadialGrid {
public:
    RadialGrid(std::size_t points, double rMax);

    std::size_t size() const;
    double rMax() const;
    double spacing() const;
    const std::vector<double>& coordinates() const;

private:
    std::vector<double> r_;
    double rMax_;
    double dr_;
};