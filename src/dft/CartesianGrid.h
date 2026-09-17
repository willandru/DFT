#pragma once

#include <cstddef>

class CartesianGrid
{
public:
    CartesianGrid(
        std::size_t nx,
        std::size_t ny,
        std::size_t nz,
        double xmin,
        double xmax,
        double ymin,
        double ymax,
        double zmin,
        double zmax
    );

    std::size_t getNx() const;
    std::size_t getNy() const;
    std::size_t getNz() const;

    std::size_t getSize() const;

    double getXMin() const;
    double getXMax() const;
    double getYMin() const;
    double getYMax() const;
    double getZMin() const;
    double getZMax() const;

    double getDx() const;
    double getDy() const;
    double getDz() const;

    double getX(std::size_t i) const;
    double getY(std::size_t j) const;
    double getZ(std::size_t k) const;

    std::size_t getIndex(
        std::size_t i,
        std::size_t j,
        std::size_t k
    ) const;

private:
    std::size_t nx_;
    std::size_t ny_;
    std::size_t nz_;

    double xmin_;
    double xmax_;
    double ymin_;
    double ymax_;
    double zmin_;
    double zmax_;

    double dx_;
    double dy_;
    double dz_;
};