#include "CartesianGrid.h"

#include <stdexcept>

CartesianGrid::CartesianGrid(
    std::size_t nx,
    std::size_t ny,
    std::size_t nz,
    double xmin,
    double xmax,
    double ymin,
    double ymax,
    double zmin,
    double zmax
)
    : nx_(nx),
      ny_(ny),
      nz_(nz),
      xmin_(xmin),
      xmax_(xmax),
      ymin_(ymin),
      ymax_(ymax),
      zmin_(zmin),
      zmax_(zmax),
      dx_(0.0),
      dy_(0.0),
      dz_(0.0)
{
    if (nx_ < 2 || ny_ < 2 || nz_ < 2)
    {
        throw std::invalid_argument(
            "CartesianGrid requires at least two points per dimension."
        );
    }

    if (xmax_ <= xmin_ ||
        ymax_ <= ymin_ ||
        zmax_ <= zmin_)
    {
        throw std::invalid_argument(
            "CartesianGrid requires valid spatial bounds."
        );
    }

    dx_ = (xmax_ - xmin_) / static_cast<double>(nx_ - 1);
    dy_ = (ymax_ - ymin_) / static_cast<double>(ny_ - 1);
    dz_ = (zmax_ - zmin_) / static_cast<double>(nz_ - 1);
}

std::size_t CartesianGrid::getNx() const
{
    return nx_;
}

std::size_t CartesianGrid::getNy() const
{
    return ny_;
}

std::size_t CartesianGrid::getNz() const
{
    return nz_;
}

std::size_t CartesianGrid::getSize() const
{
    return nx_ * ny_ * nz_;
}

double CartesianGrid::getXMin() const
{
    return xmin_;
}

double CartesianGrid::getXMax() const
{
    return xmax_;
}

double CartesianGrid::getYMin() const
{
    return ymin_;
}

double CartesianGrid::getYMax() const
{
    return ymax_;
}

double CartesianGrid::getZMin() const
{
    return zmin_;
}

double CartesianGrid::getZMax() const
{
    return zmax_;
}

double CartesianGrid::getDx() const
{
    return dx_;
}

double CartesianGrid::getDy() const
{
    return dy_;
}

double CartesianGrid::getDz() const
{
    return dz_;
}

double CartesianGrid::getX(std::size_t i) const
{
    if (i >= nx_)
    {
        throw std::out_of_range("X grid index out of range.");
    }

    return xmin_ + static_cast<double>(i) * dx_;
}

double CartesianGrid::getY(std::size_t j) const
{
    if (j >= ny_)
    {
        throw std::out_of_range("Y grid index out of range.");
    }

    return ymin_ + static_cast<double>(j) * dy_;
}

double CartesianGrid::getZ(std::size_t k) const
{
    if (k >= nz_)
    {
        throw std::out_of_range("Z grid index out of range.");
    }

    return zmin_ + static_cast<double>(k) * dz_;
}

std::size_t CartesianGrid::getIndex(
    std::size_t i,
    std::size_t j,
    std::size_t k
) const
{
    if (i >= nx_ || j >= ny_ || k >= nz_)
    {
        throw std::out_of_range(
            "Cartesian grid index out of range."
        );
    }

    return (k * ny_ + j) * nx_ + i;
}