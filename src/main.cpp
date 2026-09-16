#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// ================================================================
// Constants
// ================================================================

constexpr double PI =
    3.1415926535897932384626433832795;

constexpr double MIN_DENSITY =
    1.0e-14;

constexpr double MIN_PIVOT =
    1.0e-14;

constexpr double EIGENVALUE_TOLERANCE =
    1.0e-12;

constexpr double ORBITAL_TOLERANCE =
    1.0e-11;

// ================================================================
// LSDA / PW92 constants
//
// PW92 parameterization for the correlation energy of the
// homogeneous electron gas.
//
// Unpolarized: zeta = 0
// Fully polarized: zeta = 1
// ================================================================

struct PW92Parameters
{
    double A;
    double alpha1;
    double beta1;
    double beta2;
    double beta3;
    double beta4;
};

constexpr PW92Parameters PW92_UNPOLARIZED =
{
    0.0310907,
    0.21370,
    7.5957,
    3.5876,
    1.6382,
    0.49294
};

constexpr PW92Parameters PW92_POLARIZED =
{
    0.01554535,
    0.20548,
    14.1189,
    6.1977,
    3.3662,
    0.62517
};

// ================================================================
// Radial grid
//
// u(r) = r R(r)
//
// u(0) = 0
// u(Rmax) = 0
// ================================================================

double radiusAt(
    int i,
    double dr)
{
    return static_cast<double>(i + 1) * dr;
}

// ================================================================
// Results
// ================================================================

struct OrbitalResult
{
    std::vector<double> orbital;
    double eigenvalue;
};

struct SpinOrbitals
{
    std::vector<double> up1s;
    std::vector<double> down1s;

    std::vector<double> up2s;
    std::vector<double> down2s;

    double up1sEnergy = 0.0;
    double down1sEnergy = 0.0;

    double up2sEnergy = 0.0;
    double down2sEnergy = 0.0;
};

struct SpinDensities
{
    std::vector<double> up;
    std::vector<double> down;
};

struct LSDAPotential
{
    std::vector<double> hartree;
    std::vector<double> xcUp;
    std::vector<double> xcDown;

    std::vector<double> effectiveUp;
    std::vector<double> effectiveDown;
};

struct LSDATotalEnergy
{
    double kinetic = 0.0;
    double external = 0.0;
    double hartree = 0.0;
    double exchange = 0.0;
    double correlation = 0.0;
    double total = 0.0;
};

struct AtomicSCFResult
{
    std::string element;

    double Z = 0.0;

    double electronsUp = 0.0;
    double electronsDown = 0.0;

    bool converged = false;
    int iterations = 0;

    double densityError = 0.0;
    double energyChange =
        std::numeric_limits<double>::infinity();

    double spinMoment = 0.0;

    SpinOrbitals orbitals;
    SpinDensities density;

    LSDAPotential potential;
    LSDATotalEnergy energy;
};

// ================================================================
// Tridiagonal solver
// ================================================================

std::vector<double> solveTridiagonal(
    const std::vector<double>& lower,
    const std::vector<double>& diagonal,
    const std::vector<double>& upper,
    const std::vector<double>& rhs)
{
    const int n =
        static_cast<int>(diagonal.size());

    if (n == 0)
        throw std::runtime_error(
            "Tridiagonal system has zero size.");

    if (static_cast<int>(lower.size()) != n - 1 ||
        static_cast<int>(upper.size()) != n - 1 ||
        static_cast<int>(rhs.size()) != n)
    {
        throw std::runtime_error(
            "Invalid tridiagonal system dimensions.");
    }

    std::vector<double> cPrime(
        n - 1,
        0.0);

    std::vector<double> dPrime(
        n,
        0.0);

    std::vector<double> solution(
        n,
        0.0);

    double pivot =
        diagonal[0];

    if (std::abs(pivot) < MIN_PIVOT)
        throw std::runtime_error(
            "Numerically singular tridiagonal system.");

    if (n > 1)
        cPrime[0] =
            upper[0] / pivot;

    dPrime[0] =
        rhs[0] / pivot;

    for (int i = 1;
         i < n;
         ++i)
    {
        pivot =
            diagonal[i] -
            lower[i - 1] *
            cPrime[i - 1];

        if (std::abs(pivot) < MIN_PIVOT)
            throw std::runtime_error(
                "Numerically singular tridiagonal system.");

        if (i < n - 1)
            cPrime[i] =
                upper[i] / pivot;

        dPrime[i] =
            (
                rhs[i] -
                lower[i - 1] *
                dPrime[i - 1]
            ) / pivot;
    }

    solution[n - 1] =
        dPrime[n - 1];

    for (int i = n - 2;
         i >= 0;
         --i)
    {
        solution[i] =
            dPrime[i] -
            cPrime[i] *
            solution[i + 1];
    }

    return solution;
}

// ================================================================
// Inner product
// ================================================================

double innerProduct(
    const std::vector<double>& a,
    const std::vector<double>& b,
    double dr)
{
    if (a.size() != b.size())
        throw std::runtime_error(
            "Inner product size mismatch.");

    const int n =
        static_cast<int>(a.size());

    double result =
        0.0;

    for (int i = 0;
         i < n;
         ++i)
    {
        const double weight =
            (i == 0 ||
             i == n - 1)
            ? 0.5
            : 1.0;

        result +=
            weight *
            a[i] *
            b[i] *
            dr;
    }

    return result;
}

// ================================================================
// Normalize radial orbital
// ================================================================

void normalize(
    std::vector<double>& u,
    double dr)
{
    const double norm2 =
        innerProduct(
            u,
            u,
            dr);

    if (norm2 <= 0.0)
        throw std::runtime_error(
            "Cannot normalize zero orbital.");

    const double inverseNorm =
        1.0 /
        std::sqrt(norm2);

    for (double& value : u)
        value *= inverseNorm;
}

// ================================================================
// Orthogonalize against existing orbitals
// ================================================================

void orthogonalize(
    std::vector<double>& orbital,
    const std::vector<
        std::vector<double>>& previousOrbitals,
    double dr)
{
    for (const auto& previous :
         previousOrbitals)
    {
        const double projection =
            innerProduct(
                previous,
                orbital,
                dr);

        for (std::size_t i = 0;
             i < orbital.size();
             ++i)
        {
            orbital[i] -=
                projection *
                previous[i];
        }
    }
}

// ================================================================
// Apply radial Hamiltonian
// ================================================================

std::vector<double> applyHamiltonian(
    const std::vector<double>& u,
    const std::vector<double>& potential,
    double dr)
{
    const int n =
        static_cast<int>(u.size());

    if (potential.size() != u.size())
        throw std::runtime_error(
            "Hamiltonian size mismatch.");

    const double inverseDr2 =
        1.0 /
        (dr * dr);

    std::vector<double> result(
        n,
        0.0);

    for (int i = 0;
         i < n;
         ++i)
    {
        const double left =
            (i > 0)
            ? u[i - 1]
            : 0.0;

        const double center =
            u[i];

        const double right =
            (i + 1 < n)
            ? u[i + 1]
            : 0.0;

        const double secondDerivative =
            (
                left -
                2.0 * center +
                right
            ) *
            inverseDr2;

        result[i] =
            -0.5 *
            secondDerivative +
            potential[i] *
            center;
    }

    return result;
}

// ================================================================
// Rayleigh quotient
// ================================================================

double calculateRayleighQuotient(
    const std::vector<double>& u,
    const std::vector<double>& potential,
    double dr)
{
    const std::vector<double> Hu =
        applyHamiltonian(
            u,
            potential,
            dr);

    const double numerator =
        innerProduct(
            u,
            Hu,
            dr);

    const double denominator =
        innerProduct(
            u,
            u,
            dr);

    if (denominator <= 0.0)
        throw std::runtime_error(
            "Invalid Rayleigh quotient.");

    return numerator / denominator;
}

// ================================================================
// Sturm count
//
// Number of eigenvalues below E.
// ================================================================

int countEigenvaluesBelow(
    const std::vector<double>& diagonal,
    double offDiagonal,
    double energy)
{
    const int n =
        static_cast<int>(
            diagonal.size());

    if (n == 0)
        return 0;

    int count =
        0;

    double q =
        diagonal[0] -
        energy;

    if (q < 0.0)
        ++count;

    if (std::abs(q) < MIN_PIVOT)
        q =
            (q < 0.0)
            ? -MIN_PIVOT
            : MIN_PIVOT;

    for (int i = 1;
         i < n;
         ++i)
    {
        q =
            diagonal[i] -
            energy -
            (
                offDiagonal *
                offDiagonal
            ) /
            q;

        if (q < 0.0)
            ++count;

        if (std::abs(q) < MIN_PIVOT)
        {
            q =
                (q < 0.0)
                ? -MIN_PIVOT
                : MIN_PIVOT;
        }
    }

    return count;
}

// ================================================================
// Eigenvalue bracket
// ================================================================

double findEigenvalue(
    const std::vector<double>& diagonal,
    double offDiagonal,
    int state)
{
    if (state < 0)
        throw std::runtime_error(
            "Invalid eigenstate index.");

    double lower =
        -100.0;

    double upper =
        10.0;

    while (
        countEigenvaluesBelow(
            diagonal,
            offDiagonal,
            lower) > state)
    {
        lower *= 2.0;

        if (lower < -1.0e10)
            throw std::runtime_error(
                "Unable to bracket lower eigenvalue.");
    }

    while (
        countEigenvaluesBelow(
            diagonal,
            offDiagonal,
            upper) <= state)
    {
        upper *= 2.0;

        if (upper > 1.0e10)
            throw std::runtime_error(
                "Unable to bracket eigenvalue.");
    }

    double energy =
        0.0;

    for (int iteration = 0;
         iteration < 250;
         ++iteration)
    {
        const double middle =
            0.5 *
            (lower + upper);

        const int count =
            countEigenvaluesBelow(
                diagonal,
                offDiagonal,
                middle);

        if (count > state)
            upper = middle;
        else
            lower = middle;

        energy =
            0.5 *
            (lower + upper);

        if (std::abs(
                upper - lower) <
            EIGENVALUE_TOLERANCE)
        {
            break;
        }
    }

    return energy;
}

// ================================================================
// Solve radial eigenstate
//
// state = 0 -> 1s
// state = 1 -> 2s
//
// For the spherically symmetric l = 0 problem, these are the
// first and second radial eigenstates.
//
// ================================================================

OrbitalResult solveEigenstate(
    const std::vector<double>& potential,
    double dr,
    int state,
    std::vector<double> initialGuess,
    const std::vector<
        std::vector<double>>& previousOrbitals)
{
    const int n =
        static_cast<int>(potential.size());

    if (n == 0)
        throw std::runtime_error(
            "Empty potential.");

    if (static_cast<int>(
            initialGuess.size()) != n)
    {
        throw std::runtime_error(
            "Initial orbital size mismatch.");
    }

    const double inverseDr2 =
        1.0 /
        (dr * dr);

    const double diagonalKinetic =
        inverseDr2;

    const double offDiagonal =
        -0.5 *
        inverseDr2;

    std::vector<double> diagonal(
        n);

    for (int i = 0;
         i < n;
         ++i)
    {
        diagonal[i] =
            diagonalKinetic +
            potential[i];
    }

    const double eigenvalue =
        findEigenvalue(
            diagonal,
            offDiagonal,
            state);

    std::vector<double> orbital =
        std::move(initialGuess);

    orthogonalize(
        orbital,
        previousOrbitals,
        dr);

    normalize(
        orbital,
        dr);

    /*
        The shift is placed close to the requested eigenvalue.
        Inverse iteration converges to the eigenvector associated
        with that eigenvalue.
    */

    double shift =
        eigenvalue - 0.02;

    if (state > 0)
        shift =
            eigenvalue - 0.01;

    std::vector<double> lowerDiagonal(
        n - 1,
        offDiagonal);

    std::vector<double> upperDiagonal(
        n - 1,
        offDiagonal);

    std::vector<double> shiftedDiagonal(
        n);

    for (int i = 0;
         i < n;
         ++i)
    {
        shiftedDiagonal[i] =
            diagonal[i] -
            shift;
    }

    for (int iteration = 0;
         iteration < 300;
         ++iteration)
    {
        std::vector<double> next =
            solveTridiagonal(
                lowerDiagonal,
                shiftedDiagonal,
                upperDiagonal,
                orbital);

        orthogonalize(
            next,
            previousOrbitals,
            dr);

        normalize(
            next,
            dr);

        double overlap =
            innerProduct(
                next,
                orbital,
                dr);

        if (overlap < 0.0)
        {
            for (double& value : next)
                value = -value;
        }

        double difference2 =
            0.0;

        for (std::size_t i = 0;
             i < next.size();
             ++i)
        {
            const double difference =
                next[i] -
                orbital[i];

            difference2 +=
                difference *
                difference;
        }

        difference2 *= dr;

        orbital.swap(next);

        if (std::sqrt(
                difference2) <
            ORBITAL_TOLERANCE)
        {
            break;
        }
    }

    orthogonalize(
        orbital,
        previousOrbitals,
        dr);

    normalize(
        orbital,
        dr);

    const double finalEnergy =
        calculateRayleighQuotient(
            orbital,
            potential,
            dr);

    return
    {
        orbital,
        finalEnergy
    };
}

// ================================================================
// Construct initial orbital
// ================================================================

std::vector<double> makeInitialOrbital(
    int N,
    double dr,
    double exponent,
    int state)
{
    std::vector<double> orbital(
        N,
        0.0);

    for (int i = 0;
         i < N;
         ++i)
    {
        const double r =
            radiusAt(
                i,
                dr);

        if (state == 0)
        {
            orbital[i] =
                r *
                std::exp(
                    -exponent * r);
        }
        else
        {
            orbital[i] =
                r *
                (
                    1.0 -
                    0.5 *
                    exponent *
                    r
                ) *
                std::exp(
                    -0.5 *
                    exponent *
                    r);
        }
    }

    normalize(
        orbital,
        dr);

    return orbital;
}

// ================================================================
// Spin density
//
// rho_sigma(r) = N_sigma |u_sigma(r)|^2 / (4 pi r^2)
// ================================================================

std::vector<double> calculateSpinDensity(
    const std::vector<double>& orbital,
    double dr,
    double occupation)
{
    const int n =
        static_cast<int>(
            orbital.size());

    std::vector<double> density(
        n,
        0.0);

    if (occupation <= 0.0)
        return density;

    for (int i = 0;
         i < n;
         ++i)
    {
        const double r =
            radiusAt(
                i,
                dr);

        density[i] =
            occupation *
            orbital[i] *
            orbital[i] /
            (
                4.0 *
                PI *
                r *
                r
            );
    }

    return density;
}

// ================================================================
// Add density
// ================================================================

void addDensity(
    std::vector<double>& destination,
    const std::vector<double>& source)
{
    if (destination.size() != source.size())
        throw std::runtime_error(
            "Density size mismatch.");

    for (std::size_t i = 0;
         i < destination.size();
         ++i)
    {
        destination[i] +=
            source[i];
    }
}

// ================================================================
// Total spin densities
// ================================================================

SpinDensities calculateDensities(
    const SpinOrbitals& orbitals,
    double dr,
    double electronsUp,
    double electronsDown)
{
    const int n =
        static_cast<int>(
            orbitals.up1s.size());

    SpinDensities result;

    result.up.assign(
        n,
        0.0);

    result.down.assign(
        n,
        0.0);

    /*
        Occupations are represented explicitly by the orbital
        population:

        1s_up  + 2s_up
        1s_down + 2s_down
    */

    addDensity(
        result.up,
        calculateSpinDensity(
            orbitals.up1s,
            dr,
            std::min(
                electronsUp,
                1.0)));

    if (electronsUp > 1.0)
    {
        addDensity(
            result.up,
            calculateSpinDensity(
                orbitals.up2s,
                dr,
                electronsUp - 1.0));
    }

    addDensity(
        result.down,
        calculateSpinDensity(
            orbitals.down1s,
            dr,
            std::min(
                electronsDown,
                1.0)));

    if (electronsDown > 1.0)
    {
        addDensity(
            result.down,
            calculateSpinDensity(
                orbitals.down2s,
                dr,
                electronsDown - 1.0));
    }

    return result;
}

// ================================================================
// Electron number
// ================================================================

double calculateElectronNumber(
    const std::vector<double>& density,
    double dr)
{
    double number =
        0.0;

    const int n =
        static_cast<int>(
            density.size());

    for (int i = 0;
         i < n;
         ++i)
    {
        const double r =
            radiusAt(
                i,
                dr);

        const double weight =
            (i == 0 ||
             i == n - 1)
            ? 0.5
            : 1.0;

        number +=
            weight *
            density[i] *
            4.0 *
            PI *
            r *
            r *
            dr;
    }

    return number;
}

// ================================================================
// Hartree potential from total density
//
// For spherical density:
//
// V_H(r) = Q(r)/r
//        + integral_r^R 4 pi r' rho(r') / r' dr'
//
// ================================================================

std::vector<double> calculateHartreePotential(
    const std::vector<double>& density,
    double dr)
{
    const int n =
        static_cast<int>(
            density.size());

    std::vector<double> cumulative(
        n,
        0.0);

    std::vector<double> outer(
        n,
        0.0);

    std::vector<double> potential(
        n,
        0.0);

    for (int i = 0;
         i < n;
         ++i)
    {
        const double r =
            radiusAt(
                i,
                dr);

        const double integrand =
            4.0 *
            PI *
            r *
            r *
            density[i];

        if (i == 0)
        {
            cumulative[i] =
                0.5 *
                dr *
                integrand;
        }
        else
        {
            const double previousR =
                radiusAt(
                    i - 1,
                    dr);

            const double previousIntegrand =
                4.0 *
                PI *
                previousR *
                previousR *
                density[i - 1];

            cumulative[i] =
                cumulative[i - 1] +
                0.5 *
                dr *
                (
                    previousIntegrand +
                    integrand
                );
        }
    }

    for (int i = n - 1;
         i >= 0;
         --i)
    {
        const double r =
            radiusAt(
                i,
                dr);

        const double integrand =
            4.0 *
            PI *
            r *
            density[i] /
            r;

        if (i == n - 1)
        {
            outer[i] =
                0.5 *
                dr *
                integrand;
        }
        else
        {
            const double nextR =
                radiusAt(
                    i + 1,
                    dr);

            const double nextIntegrand =
                4.0 *
                PI *
                nextR *
                density[i + 1];

            outer[i] =
                outer[i + 1] +
                0.5 *
                dr *
                (
                    integrand +
                    nextIntegrand
                );
        }
    }

    for (int i = 0;
         i < n;
         ++i)
    {
        const double r =
            radiusAt(
                i,
                dr);

        potential[i] =
            cumulative[i] / r +
            outer[i];
    }

    return potential;
}

// ================================================================
// PW92 correlation energy per electron
// ================================================================

double calculatePW92Correlation(
    double rs,
    const PW92Parameters& p)
{
    if (rs <= 0.0)
        throw std::runtime_error(
            "Invalid rs.");

    const double sqrtRs =
        std::sqrt(rs);

    const double q0 =
        -2.0 *
        p.A *
        (
            1.0 +
            p.alpha1 *
            rs
        );

    const double q1 =
        2.0 *
        p.A *
        sqrtRs *
        (
            p.beta1 +
            sqrtRs *
            (
                p.beta2 +
                sqrtRs *
                (
                    p.beta3 +
                    p.beta4 *
                    sqrtRs
                )
            )
        );

    return
        q0 *
        std::log1p(
            1.0 / q1);
}

// ================================================================
// Spin interpolation of PW92 correlation
//
// zeta = (rho_up-rho_down)/rho
//
// epsilon_c(rs,zeta) =
//
// epsilon_c(rs,0)
// + f(zeta)
//   [epsilon_c(rs,1)-epsilon_c(rs,0)]
// ================================================================

double calculateSpinInterpolation(
    double zeta)
{
    zeta =
        std::max(
            -1.0,
            std::min(
                1.0,
                zeta));

    const double numerator =
        std::pow(
            1.0 + zeta,
            4.0 / 3.0)
        +
        std::pow(
            1.0 - zeta,
            4.0 / 3.0)
        -
        2.0;

    const double denominator =
        std::pow(
            2.0,
            4.0 / 3.0)
        -
        2.0;

    return
        numerator /
        denominator;
}

// ================================================================
// LSDA correlation energy per electron
// ================================================================

double calculateLSDACorrelationEnergyPerElectron(
    double rhoUp,
    double rhoDown)
{
    const double rho =
        rhoUp +
        rhoDown;

    if (rho <= MIN_DENSITY)
        return 0.0;

    const double rs =
        std::pow(
            3.0 /
            (
                4.0 *
                PI *
                rho
            ),
            1.0 / 3.0);

    const double zeta =
        (
            rhoUp -
            rhoDown
        ) / rho;

    const double epsilonUnpolarized =
        calculatePW92Correlation(
            rs,
            PW92_UNPOLARIZED);

    const double epsilonPolarized =
        calculatePW92Correlation(
            rs,
            PW92_POLARIZED);

    const double interpolation =
        calculateSpinInterpolation(
            zeta);

    return
        epsilonUnpolarized +
        interpolation *
        (
            epsilonPolarized -
            epsilonUnpolarized
        );
}

// ================================================================
// LSDA exchange energy density
//
// Exc = integral f_xc(r) d^3r
//
// For spin densities:
//
// f_x = -3/4 (3/pi)^(1/3)
//       [rho_up^(4/3)+rho_down^(4/3)]
// ================================================================

double calculateLSDAExchangeEnergyDensity(
    double rhoUp,
    double rhoDown)
{
    if (rhoUp < 0.0 ||
        rhoDown < 0.0)
        throw std::runtime_error(
            "Negative spin density.");

    const double coefficient =
        -0.75 *
        std::pow(
            3.0 / PI,
            1.0 / 3.0);

    return
        coefficient *
        (
            std::pow(
                rhoUp,
                4.0 / 3.0)
            +
            std::pow(
                rhoDown,
                4.0 / 3.0)
        );
}

// ================================================================
// LSDA correlation energy density
// ================================================================

double calculateLSDACorrelationEnergyDensity(
    double rhoUp,
    double rhoDown)
{
    const double rho =
        rhoUp +
        rhoDown;

    if (rho <= MIN_DENSITY)
        return 0.0;

    const double epsilonC =
        calculateLSDACorrelationEnergyPerElectron(
            rhoUp,
            rhoDown);

    return
        rho *
        epsilonC;
}

// ================================================================
// LSDA exchange-correlation energy density
// ================================================================

double calculateLSDAExchangeCorrelationEnergyDensity(
    double rhoUp,
    double rhoDown)
{
    return
        calculateLSDAExchangeEnergyDensity(
            rhoUp,
            rhoDown)
        +
        calculateLSDACorrelationEnergyDensity(
            rhoUp,
            rhoDown);
}

// ================================================================
// Numerical spin derivative of LSDA energy density
//
// v_xc^up = d f_xc / d rho_up
// v_xc^down = d f_xc / d rho_down
//
// Since LSDA has no density-gradient dependence, these are direct
// partial derivatives of the local energy density.
// ================================================================

double calculateLSDAExchangeCorrelationPotentialUp(
    double rhoUp,
    double rhoDown)
{
    const double rho =
        rhoUp +
        rhoDown;

    if (rho <= MIN_DENSITY)
        return 0.0;

    const double relativeStep =
        1.0e-5;

    const double step =
        std::max(
            1.0e-12,
            relativeStep *
            std::max(
                rhoUp,
                rho));

    const double rhoUpPlus =
        rhoUp +
        step;

    const double rhoUpMinus =
        std::max(
            0.0,
            rhoUp -
            step);

    const double fPlus =
        calculateLSDAExchangeCorrelationEnergyDensity(
            rhoUpPlus,
            rhoDown);

    const double fMinus =
        calculateLSDAExchangeCorrelationEnergyDensity(
            rhoUpMinus,
            rhoDown);

    if (rhoUpPlus == rhoUpMinus)
        return 0.0;

    return
        (
            fPlus -
            fMinus
        ) /
        (
            rhoUpPlus -
            rhoUpMinus
        );
}

// ================================================================
// Down-spin derivative
// ================================================================

double calculateLSDAExchangeCorrelationPotentialDown(
    double rhoUp,
    double rhoDown)
{
    const double rho =
        rhoUp +
        rhoDown;

    if (rho <= MIN_DENSITY)
        return 0.0;

    const double relativeStep =
        1.0e-5;

    const double step =
        std::max(
            1.0e-12,
            relativeStep *
            std::max(
                rhoDown,
                rho));

    const double rhoDownPlus =
        rhoDown +
        step;

    const double rhoDownMinus =
        std::max(
            0.0,
            rhoDown -
            step);

    const double fPlus =
        calculateLSDAExchangeCorrelationEnergyDensity(
            rhoUp,
            rhoDownPlus);

    const double fMinus =
        calculateLSDAExchangeCorrelationEnergyDensity(
            rhoUp,
            rhoDownMinus);

    if (rhoDownPlus == rhoDownMinus)
        return 0.0;

    return
        (
            fPlus -
            fMinus
        ) /
        (
            rhoDownPlus -
            rhoDownMinus
        );
}

// ================================================================
// LSDA potential
// ================================================================

LSDAPotential calculateLSDAPotential(
    const SpinDensities& density,
    double dr,
    double Z)
{
    const int n =
        static_cast<int>(
            density.up.size());

    if (density.down.size() !=
        density.up.size())
    {
        throw std::runtime_error(
            "Spin density size mismatch.");
    }

    LSDAPotential result;

    result.hartree =
        calculateHartreePotential(
            [&]()
            {
                std::vector<double> total(n);

                for (int i = 0;
                     i < n;
                     ++i)
                {
                    total[i] =
                        density.up[i] +
                        density.down[i];
                }

                return total;
            }(),
            dr);

    result.xcUp.assign(
        n,
        0.0);

    result.xcDown.assign(
        n,
        0.0);

    result.effectiveUp.assign(
        n,
        0.0);

    result.effectiveDown.assign(
        n,
        0.0);

    for (int i = 0;
         i < n;
         ++i)
    {
        const double rhoUp =
            density.up[i];

        const double rhoDown =
            density.down[i];

        const double r =
            radiusAt(
                i,
                dr);

        result.xcUp[i] =
            calculateLSDAExchangeCorrelationPotentialUp(
                rhoUp,
                rhoDown);

        result.xcDown[i] =
            calculateLSDAExchangeCorrelationPotentialDown(
                rhoUp,
                rhoDown);

        const double nuclear =
            -Z / r;

        result.effectiveUp[i] =
            nuclear +
            result.hartree[i] +
            result.xcUp[i];

        result.effectiveDown[i] =
            nuclear +
            result.hartree[i] +
            result.xcDown[i];
    }

    return result;
}

// ================================================================
// Kinetic energy of one radial orbital
// ================================================================

double calculateOneElectronKineticEnergy(
    const std::vector<double>& orbital,
    double dr)
{
    const int n =
        static_cast<int>(
            orbital.size());

    const double inverseDr2 =
        1.0 /
        (dr * dr);

    double energy =
        0.0;

    for (int i = 0;
         i < n;
         ++i)
    {
        const double left =
            (i > 0)
            ? orbital[i - 1]
            : 0.0;

        const double center =
            orbital[i];

        const double right =
            (i + 1 < n)
            ? orbital[i + 1]
            : 0.0;

        const double secondDerivative =
            (
                left -
                2.0 * center +
                right
            ) *
            inverseDr2;

        const double kineticOperator =
            -0.5 *
            secondDerivative;

        const double weight =
            (i == 0 ||
             i == n - 1)
            ? 0.5
            : 1.0;

        energy +=
            weight *
            orbital[i] *
            kineticOperator *
            dr;
    }

    return energy;
}

// ================================================================
// External nuclear energy
// ================================================================

double calculateExternalEnergy(
    const SpinDensities& density,
    double dr,
    double Z)
{
    const int n =
        static_cast<int>(
            density.up.size());

    double energy =
        0.0;

    for (int i = 0;
         i < n;
         ++i)
    {
        const double r =
            radiusAt(
                i,
                dr);

        const double rho =
            density.up[i] +
            density.down[i];

        const double volumeElement =
            4.0 *
            PI *
            r *
            r *
            dr;

        const double weight =
            (i == 0 ||
             i == n - 1)
            ? 0.5
            : 1.0;

        energy +=
            weight *
            rho *
            (
                -Z / r
            ) *
            volumeElement;
    }

    return energy;
}

// ================================================================
// Hartree energy
// ================================================================

double calculateHartreeEnergy(
    const SpinDensities& density,
    const std::vector<double>& hartree,
    double dr)
{
    const int n =
        static_cast<int>(
            density.up.size());

    double energy =
        0.0;

    for (int i = 0;
         i < n;
         ++i)
    {
        const double r =
            radiusAt(
                i,
                dr);

        const double rho =
            density.up[i] +
            density.down[i];

        const double volumeElement =
            4.0 *
            PI *
            r *
            r *
            dr;

        const double weight =
            (i == 0 ||
             i == n - 1)
            ? 0.5
            : 1.0;

        energy +=
            weight *
            0.5 *
            rho *
            hartree[i] *
            volumeElement;
    }

    return energy;
}

// ================================================================
// LSDA exchange and correlation energies
// ================================================================

void calculateLSDAExchangeCorrelationEnergies(
    const SpinDensities& density,
    double dr,
    double& exchange,
    double& correlation)
{
    const int n =
        static_cast<int>(
            density.up.size());

    exchange =
        0.0;

    correlation =
        0.0;

    for (int i = 0;
         i < n;
         ++i)
    {
        const double rhoUp =
            density.up[i];

        const double rhoDown =
            density.down[i];

        const double r =
            radiusAt(
                i,
                dr);

        const double volumeElement =
            4.0 *
            PI *
            r *
            r *
            dr;

        const double weight =
            (i == 0 ||
             i == n - 1)
            ? 0.5
            : 1.0;

        exchange +=
            weight *
            calculateLSDAExchangeEnergyDensity(
                rhoUp,
                rhoDown) *
            volumeElement;

        correlation +=
            weight *
            calculateLSDACorrelationEnergyDensity(
                rhoUp,
                rhoDown) *
            volumeElement;
    }
}

// ================================================================
// Total DFT energy
//
// E = Ts + Eext + EH + Exc
// ================================================================

LSDATotalEnergy calculateTotalEnergy(
    const SpinOrbitals& orbitals,
    const SpinDensities& density,
    const LSDAPotential& potential,
    double dr,
    double electronsUp,
    double electronsDown,
    double Z)
{
    LSDATotalEnergy result;

    /*
        Kinetic energy.

        For the systems treated here the occupations are:

        Li:
            up:   1s^1 2s^1
            down: 1s^1

        He triplet:
            up:   1s^1 2s^1
            down: none
    */

    result.kinetic =
        calculateOneElectronKineticEnergy(
            orbitals.up1s,
            dr);

    if (electronsUp > 1.0)
    {
        result.kinetic +=
            calculateOneElectronKineticEnergy(
                orbitals.up2s,
                dr);
    }

    if (electronsDown > 0.0)
    {
        result.kinetic +=
            calculateOneElectronKineticEnergy(
                orbitals.down1s,
                dr);

        if (electronsDown > 1.0)
        {
            result.kinetic +=
                calculateOneElectronKineticEnergy(
                    orbitals.down2s,
                    dr);
        }
    }

    result.external =
        calculateExternalEnergy(
            density,
            dr,
            Z);

    result.hartree =
        calculateHartreeEnergy(
            density,
            potential.hartree,
            dr);

    calculateLSDAExchangeCorrelationEnergies(
        density,
        dr,
        result.exchange,
        result.correlation);

    result.total =
        result.kinetic +
        result.external +
        result.hartree +
        result.exchange +
        result.correlation;

    return result;
}

// ================================================================
// Density mixing
// ================================================================

SpinDensities mixDensities(
    const SpinDensities& oldDensity,
    const SpinDensities& newDensity,
    double mixing)
{
    const int n =
        static_cast<int>(
            oldDensity.up.size());

    SpinDensities result;

    result.up.resize(
        n);

    result.down.resize(
        n);

    for (int i = 0;
         i < n;
         ++i)
    {
        result.up[i] =
            (
                1.0 -
                mixing
            ) *
            oldDensity.up[i]
            +
            mixing *
            newDensity.up[i];

        result.down[i] =
            (
                1.0 -
                mixing
            ) *
            oldDensity.down[i]
            +
            mixing *
            newDensity.down[i];
    }

    return result;
}

// ================================================================
// Normalize spin density to prescribed electron number
// ================================================================

void normalizeDensityToElectronNumber(
    std::vector<double>& density,
    double dr,
    double targetNumber)
{
    if (targetNumber <= 0.0)
    {
        std::fill(
            density.begin(),
            density.end(),
            0.0);

        return;
    }

    const double currentNumber =
        calculateElectronNumber(
            density,
            dr);

    if (currentNumber <= 0.0)
        throw std::runtime_error(
            "Invalid spin density normalization.");

    const double factor =
        targetNumber /
        currentNumber;

    for (double& value :
         density)
    {
        value *=
            factor;
    }
}

// ================================================================
// Density difference
// ================================================================

double calculateSpinDensityError(
    const SpinDensities& a,
    const SpinDensities& b,
    double dr)
{
    const int n =
        static_cast<int>(
            a.up.size());

    double error =
        0.0;

    for (int i = 0;
         i < n;
         ++i)
    {
        const double r =
            radiusAt(
                i,
                dr);

        const double volumeElement =
            4.0 *
            PI *
            r *
            r *
            dr;

        const double weight =
            (i == 0 ||
             i == n - 1)
            ? 0.5
            : 1.0;

        error +=
            weight *
            (
                std::abs(
                    a.up[i] -
                    b.up[i])
                +
                std::abs(
                    a.down[i] -
                    b.down[i])
            ) *
            volumeElement;
    }

    return error;
}

// ================================================================
// Residual of KS equation
// ================================================================

double calculateEigenstateResidual(
    const std::vector<double>& orbital,
    const std::vector<double>& potential,
    double eigenvalue,
    double dr)
{
    const std::vector<double> Hpsi =
        applyHamiltonian(
            orbital,
            potential,
            dr);

    double residual2 =
        0.0;

    for (std::size_t i = 0;
         i < orbital.size();
         ++i)
    {
        const double residual =
            Hpsi[i] -
            eigenvalue *
            orbital[i];

        residual2 +=
            residual *
            residual *
            dr;
    }

    return
        std::sqrt(
            residual2);
}

// ================================================================
// Build orbital set for an atomic configuration
//
// occupationsUp / occupationsDown:
//
// Lithium:
//
//     up   = 1s + 2s
//     down = 1s
//
// Helium triplet:
//
//     up   = 1s + 2s
//     down = empty
// ================================================================

void solveSpinOrbitals(
    const LSDAPotential& potential,
    double dr,
    int N,
    double Z,
    double electronsUp,
    double electronsDown,
    SpinOrbitals& orbitals)
{
    /*
        Up channel
    */

    std::vector<
        std::vector<double>>
        previousUp;

    const bool upHas2s =
        electronsUp > 1.0;

    const OrbitalResult up1s =
        solveEigenstate(
            potential.effectiveUp,
            dr,
            0,
            makeInitialOrbital(
                N,
                dr,
                Z,
                0),
            previousUp);

    orbitals.up1s =
        up1s.orbital;

    orbitals.up1sEnergy =
        up1s.eigenvalue;

    previousUp.push_back(
        orbitals.up1s);

    if (upHas2s)
    {
        const OrbitalResult up2s =
            solveEigenstate(
                potential.effectiveUp,
                dr,
                1,
                makeInitialOrbital(
                    N,
                    dr,
                    Z,
                    1),
                previousUp);

        orbitals.up2s =
            up2s.orbital;

        orbitals.up2sEnergy =
            up2s.eigenvalue;
    }
    else
    {
        orbitals.up2s.assign(
            N,
            0.0);

        orbitals.up2sEnergy =
            0.0;
    }

    /*
        Down channel
    */

    std::vector<
        std::vector<double>>
        previousDown;

    if (electronsDown > 0.0)
    {
        const OrbitalResult down1s =
            solveEigenstate(
                potential.effectiveDown,
                dr,
                0,
                makeInitialOrbital(
                    N,
                    dr,
                    Z,
                    0),
                previousDown);

        orbitals.down1s =
            down1s.orbital;

        orbitals.down1sEnergy =
            down1s.eigenvalue;

        previousDown.push_back(
            orbitals.down1s);

        if (electronsDown > 1.0)
        {
            const OrbitalResult down2s =
                solveEigenstate(
                    potential.effectiveDown,
                    dr,
                    1,
                    makeInitialOrbital(
                        N,
                        dr,
                        Z,
                        1),
                    previousDown);

            orbitals.down2s =
                down2s.orbital;

            orbitals.down2sEnergy =
                down2s.eigenvalue;
        }
        else
        {
            orbitals.down2s.assign(
                N,
                0.0);

            orbitals.down2sEnergy =
                0.0;
        }
    }
    else
    {
        orbitals.down1s.assign(
            N,
            0.0);

        orbitals.down2s.assign(
            N,
            0.0);

        orbitals.down1sEnergy =
            0.0;

        orbitals.down2sEnergy =
            0.0;
    }
}

// ================================================================
// SCF initial density
// ================================================================

SpinDensities initializeDensity(
    int N,
    double dr,
    double Z,
    double electronsUp,
    double electronsDown)
{
    SpinDensities density;

    density.up.assign(
        N,
        0.0);

    density.down.assign(
        N,
        0.0);

    const std::vector<double> initial1s =
        makeInitialOrbital(
            N,
            dr,
            Z,
            0);

    if (electronsUp > 0.0)
    {
        addDensity(
            density.up,
            calculateSpinDensity(
                initial1s,
                dr,
                std::min(
                    1.0,
                    electronsUp)));
    }

    if (electronsDown > 0.0)
    {
        addDensity(
            density.down,
            calculateSpinDensity(
                initial1s,
                dr,
                std::min(
                    1.0,
                    electronsDown)));
    }

    if (electronsUp > 1.0)
    {
        const std::vector<double> initial2s =
            makeInitialOrbital(
                N,
                dr,
                Z,
                1);

        addDensity(
            density.up,
            calculateSpinDensity(
                initial2s,
                dr,
                electronsUp - 1.0));
    }

    if (electronsDown > 1.0)
    {
        const std::vector<double> initial2s =
            makeInitialOrbital(
                N,
                dr,
                Z,
                1);

        addDensity(
            density.down,
            calculateSpinDensity(
                initial2s,
                dr,
                electronsDown - 1.0));
    }

    normalizeDensityToElectronNumber(
        density.up,
        dr,
        electronsUp);

    normalizeDensityToElectronNumber(
        density.down,
        dr,
        electronsDown);

    return density;
}

// ================================================================
// Run generic spin-polarized atomic SCF
// ================================================================

AtomicSCFResult runSpinPolarizedAtom(
    const std::string& element,
    double Z,
    double electronsUp,
    double electronsDown)
{
    const int N =
        2000;

    const double Rmax =
        30.0;

    const double dr =
        Rmax /
        static_cast<double>(
            N + 1);

    const int maxIterations =
        250;

    const double mixing =
        0.30;

    const double densityTolerance =
        1.0e-8;

    const double energyTolerance =
        1.0e-10;

    AtomicSCFResult result;

    result.element =
        element;

    result.Z =
        Z;

    result.electronsUp =
        electronsUp;

    result.electronsDown =
        electronsDown;

    result.density =
        initializeDensity(
            N,
            dr,
            Z,
            electronsUp,
            electronsDown);

    double previousEnergy =
        std::numeric_limits<double>::quiet_NaN();

    std::cout
        << "\n====================================================\n"
        << element
        << " — SPIN-POLARIZED KOHN-SHAM DFT / LSDA\n"
        << "====================================================\n\n";

    std::cout
        << "Grid points:          "
        << N
        << "\n";

    std::cout
        << "Rmax:                 "
        << Rmax
        << " Bohr\n";

    std::cout
        << "dr:                   "
        << dr
        << " Bohr\n\n";

    std::cout
        << "Nuclear charge:       Z = "
        << Z
        << "\n";

    std::cout
        << "Spin-up electrons:    "
        << electronsUp
        << "\n";

    std::cout
        << "Spin-down electrons:  "
        << electronsDown
        << "\n";

    std::cout
        << "Total electrons:      "
        << electronsUp +
           electronsDown
        << "\n";

    std::cout
        << "Spin moment Nup-Ndown:"
        << " "
        << electronsUp -
           electronsDown
        << "\n\n";

    std::cout
        << "Exchange:             LSDA\n"
        << "Correlation:          LSDA / PW92\n"
        << "SCF mixing:           "
        << mixing
        << "\n\n";

    for (int iteration = 1;
         iteration <= maxIterations;
         ++iteration)
    {
        // --------------------------------------------------------
        // Build spin-dependent KS potential.
        // --------------------------------------------------------

        const LSDAPotential potential =
            calculateLSDAPotential(
                result.density,
                dr,
                Z);

        // --------------------------------------------------------
        // Solve spin-up and spin-down KS equations.
        // --------------------------------------------------------

        SpinOrbitals orbitals;

        solveSpinOrbitals(
            potential,
            dr,
            N,
            Z,
            electronsUp,
            electronsDown,
            orbitals);

        // --------------------------------------------------------
        // Generate new spin densities.
        // --------------------------------------------------------

        const SpinDensities newDensity =
            calculateDensities(
                orbitals,
                dr,
                electronsUp,
                electronsDown);

        // --------------------------------------------------------
        // Density residual.
        // --------------------------------------------------------

        const double densityError =
            calculateSpinDensityError(
                result.density,
                newDensity,
                dr);

        // --------------------------------------------------------
        // Mix spin densities.
        // --------------------------------------------------------

        SpinDensities mixedDensity =
            mixDensities(
                result.density,
                newDensity,
                mixing);

        normalizeDensityToElectronNumber(
            mixedDensity.up,
            dr,
            electronsUp);

        normalizeDensityToElectronNumber(
            mixedDensity.down,
            dr,
            electronsDown);

        result.density =
            std::move(
                mixedDensity);

        result.orbitals =
            std::move(
                orbitals);

        // --------------------------------------------------------
        // Energy evaluated using the current SCF density and
        // current spin orbitals.
        // --------------------------------------------------------

        const LSDAPotential energyPotential =
            calculateLSDAPotential(
                result.density,
                dr,
                Z);

        const LSDATotalEnergy energy =
            calculateTotalEnergy(
                result.orbitals,
                result.density,
                energyPotential,
                dr,
                electronsUp,
                electronsDown,
                Z);

        double energyChange =
            std::numeric_limits<double>::infinity();

        if (std::isfinite(
                previousEnergy))
        {
            energyChange =
                std::abs(
                    energy.total -
                    previousEnergy);
        }

        previousEnergy =
            energy.total;

        result.potential =
            energyPotential;

        result.energy =
            energy;

        result.densityError =
            densityError;

        result.energyChange =
            energyChange;

        result.iterations =
            iteration;

        if (iteration == 1 ||
            iteration % 5 == 0)
        {
            std::cout
                << "Iteration "
                << std::setw(3)
                << iteration
                << "   "
                << "eps_up(1s) = "
                << std::setw(13)
                << result.orbitals.up1sEnergy
                << "   "
                << "eps_down(1s) = "
                << std::setw(13)
                << result.orbitals.down1sEnergy
                << "   "
                << "E = "
                << std::setw(15)
                << result.energy.total
                << "   "
                << "dE = "
                << energyChange
                << "   "
                << "dRho = "
                << densityError
                << "\n";
        }

        if (iteration > 1 &&
            densityError <
                densityTolerance &&
            energyChange <
                energyTolerance)
        {
            result.converged =
                true;

            break;
        }
    }

    // ============================================================
    // Final self-consistent rebuild
    // ============================================================

    result.potential =
        calculateLSDAPotential(
            result.density,
            dr,
            Z);

    solveSpinOrbitals(
        result.potential,
        dr,
        N,
        Z,
        electronsUp,
        electronsDown,
        result.orbitals);

    /*
        Rebuild the density from the final KS orbitals. At
        convergence this should agree with the SCF density.
    */

    const SpinDensities finalOrbitalDensity =
        calculateDensities(
            result.orbitals,
            dr,
            electronsUp,
            electronsDown);

    const double finalConsistencyError =
        calculateSpinDensityError(
            result.density,
            finalOrbitalDensity,
            dr);

    result.density =
        finalOrbitalDensity;

    result.potential =
        calculateLSDAPotential(
            result.density,
            dr,
            Z);

    result.energy =
        calculateTotalEnergy(
            result.orbitals,
            result.density,
            result.potential,
            dr,
            electronsUp,
            electronsDown,
            Z);

    result.spinMoment =
        calculateElectronNumber(
            result.density.up,
            dr)
        -
        calculateElectronNumber(
            result.density.down,
            dr);

    std::cout
        << "\n";

    if (result.converged)
    {
        std::cout
            << "SCF converged after "
            << result.iterations
            << " iterations.\n";
    }
    else
    {
        std::cout
            << "SCF did NOT converge within "
            << maxIterations
            << " iterations.\n";
    }

    std::cout
        << "Final orbital-density consistency: "
        << finalConsistencyError
        << "\n";

    return result;
}

// ================================================================
// Spin diagnostics
// ================================================================

void printSpinDiagnostics(
    const AtomicSCFResult& result,
    double dr)
{
    const double up =
        calculateElectronNumber(
            result.density.up,
            dr);

    const double down =
        calculateElectronNumber(
            result.density.down,
            dr);

    const double total =
        up +
        down;

    const double magnetization =
        up -
        down;

    std::cout
        << "\n====================================================\n"
        << "SPIN DENSITY DIAGNOSTICS — "
        << result.element
        << "\n"
        << "====================================================\n\n";

    std::cout
        << std::setprecision(15);

    std::cout
        << "Integrated rho_up:        "
        << up
        << "\n";

    std::cout
        << "Target N_up:              "
        << result.electronsUp
        << "\n";

    std::cout
        << "Integrated rho_down:      "
        << down
        << "\n";

    std::cout
        << "Target N_down:            "
        << result.electronsDown
        << "\n";

    std::cout
        << "Total electrons:          "
        << total
        << "\n";

    std::cout
        << "Target total electrons:   "
        << result.electronsUp +
           result.electronsDown
        << "\n";

    std::cout
        << "Spin magnetization:       "
        << magnetization
        << "\n";

    std::cout
        << "Expected N_up-N_down:     "
        << result.electronsUp -
           result.electronsDown
        << "\n";
}

// ================================================================
// Orbital diagnostics
// ================================================================

void printOrbitalDiagnostics(
    const AtomicSCFResult& result,
    double dr)
{
    std::cout
        << "\n====================================================\n"
        << "ORBITAL DIAGNOSTICS — "
        << result.element
        << "\n"
        << "====================================================\n\n";

    std::cout
        << std::setprecision(15);

    std::cout
        << "Spin-up 1s energy:        "
        << result.orbitals.up1sEnergy
        << " Ha\n";

    if (result.electronsUp > 1.0)
    {
        std::cout
            << "Spin-up 2s energy:        "
            << result.orbitals.up2sEnergy
            << " Ha\n";
    }

    if (result.electronsDown > 0.0)
    {
        std::cout
            << "Spin-down 1s energy:      "
            << result.orbitals.down1sEnergy
            << " Ha\n";
    }

    if (result.electronsDown > 1.0)
    {
        std::cout
            << "Spin-down 2s energy:      "
            << result.orbitals.down2sEnergy
            << " Ha\n";
    }

    const double up1sNorm =
        innerProduct(
            result.orbitals.up1s,
            result.orbitals.up1s,
            dr);

    std::cout
        << "\n";

    std::cout
        << "Norm up 1s:               "
        << up1sNorm
        << "\n";

    if (result.electronsUp > 1.0)
    {
        const double up2sNorm =
            innerProduct(
                result.orbitals.up2s,
                result.orbitals.up2s,
                dr);

        const double overlap =
            innerProduct(
                result.orbitals.up1s,
                result.orbitals.up2s,
                dr);

        std::cout
            << "Norm up 2s:               "
            << up2sNorm
            << "\n";

        std::cout
            << "Overlap up 1s/2s:         "
            << overlap
            << "\n";
    }

    if (result.electronsDown > 0.0)
    {
        const double down1sNorm =
            innerProduct(
                result.orbitals.down1s,
                result.orbitals.down1s,
                dr);

        std::cout
            << "Norm down 1s:             "
            << down1sNorm
            << "\n";
    }

    /*
        KS equation residuals.
    */

    const double up1sResidual =
        calculateEigenstateResidual(
            result.orbitals.up1s,
            result.potential.effectiveUp,
            result.orbitals.up1sEnergy,
            dr);

    std::cout
        << "\nKS residual up 1s:       "
        << up1sResidual
        << "\n";

    if (result.electronsUp > 1.0)
    {
        const double up2sResidual =
            calculateEigenstateResidual(
                result.orbitals.up2s,
                result.potential.effectiveUp,
                result.orbitals.up2sEnergy,
                dr);

        std::cout
            << "KS residual up 2s:       "
            << up2sResidual
            << "\n";
    }

    if (result.electronsDown > 0.0)
    {
        const double down1sResidual =
            calculateEigenstateResidual(
                result.orbitals.down1s,
                result.potential.effectiveDown,
                result.orbitals.down1sEnergy,
                dr);

        std::cout
            << "KS residual down 1s:     "
            << down1sResidual
            << "\n";
    }
}

// ================================================================
// Energy diagnostics
// ================================================================

void printEnergyDiagnostics(
    const AtomicSCFResult& result)
{
    std::cout
        << "\n====================================================\n"
        << "ENERGY COMPONENTS — "
        << result.element
        << "\n"
        << "====================================================\n\n";

    std::cout
        << std::setprecision(15);

    std::cout
        << "Kinetic Ts:               "
        << result.energy.kinetic
        << " Ha\n";

    std::cout
        << "External Eext:            "
        << result.energy.external
        << " Ha\n";

    std::cout
        << "Hartree EH:               "
        << result.energy.hartree
        << " Ha\n";

    std::cout
        << "LSDA exchange Ex:         "
        << result.energy.exchange
        << " Ha\n";

    std::cout
        << "LSDA correlation Ec:      "
        << result.energy.correlation
        << " Ha\n";

    std::cout
        << "LSDA XC Exc:              "
        << result.energy.exchange +
           result.energy.correlation
        << " Ha\n";

    std::cout
        << "---------------------------------------------\n";

    std::cout
        << "Total energy:              "
        << result.energy.total
        << " Ha\n";
}

// ================================================================
// Potential diagnostics
// ================================================================

void printPotentialDiagnostics(
    const AtomicSCFResult& result,
    double dr)
{
    const int n =
        static_cast<int>(
            result.density.up.size());

    std::cout
        << "\n====================================================\n"
        << "SPIN-DEPENDENT KS POTENTIAL — "
        << result.element
        << "\n"
        << "====================================================\n\n";

    std::cout
        << std::left
        << std::setw(12) << "r"
        << std::setw(17) << "V_H"
        << std::setw(17) << "V_XC_up"
        << std::setw(17) << "V_XC_down"
        << std::setw(17) << "V_eff_up"
        << std::setw(17) << "V_eff_down"
        << "\n";

    std::cout
        << std::string(
            97,
            '-')
        << "\n";

    const std::vector<double> requestedRadii =
    {
        0.015,
        0.030,
        0.075,
        0.150,
        0.300,
        0.500,
        0.750,
        1.000,
        1.500,
        2.000,
        3.000,
        5.000
    };

    for (double requested :
         requestedRadii)
    {
        int index =
            static_cast<int>(
                std::round(
                    requested /
                    dr)) - 1;

        index =
            std::max(
                0,
                std::min(
                    n - 1,
                    index));

        const double r =
            radiusAt(
                index,
                dr);

        std::cout
            << std::left
            << std::setw(12)
            << std::setprecision(8)
            << r
            << std::setw(17)
            << result.potential.hartree[index]
            << std::setw(17)
            << result.potential.xcUp[index]
            << std::setw(17)
            << result.potential.xcDown[index]
            << std::setw(17)
            << result.potential.effectiveUp[index]
            << std::setw(17)
            << result.potential.effectiveDown[index]
            << "\n";
    }
}

// ================================================================
// Lithium
//
// Ground-state configuration:
//
//     1s^2 2s^1
//
// Spin assignment:
//
//     1s_up
//     1s_down
//     2s_up
//
// Therefore:
//
//     N_up   = 2
//     N_down = 1
//
// ================================================================

AtomicSCFResult runLithium()
{
    return
        runSpinPolarizedAtom(
            "LITHIUM",
            3.0,
            2.0,
            1.0);
}

// ================================================================
// Helium triplet
//
// Excited triplet:
//
//     1s^1 2s^1
//
// We use the M_S = +1 component:
//
//     1s_up
//     2s_up
//
// Therefore:
//
//     N_up   = 2
//     N_down = 0
//
// ================================================================

AtomicSCFResult runHeliumTriplet()
{
    return
        runSpinPolarizedAtom(
            "HELIUM TRIPLET",
            2.0,
            2.0,
            0.0);
}

// ================================================================
// Final comparison
// ================================================================

void printFinalSummary(
    const AtomicSCFResult& lithium,
    const AtomicSCFResult& heliumTriplet)
{
    std::cout
        << "\n====================================================\n"
        << "FINAL SUMMARY — SPIN-POLARIZED DFT / LSDA\n"
        << "====================================================\n\n";

    std::cout
        << std::setprecision(12);

    std::cout
        << "Lithium\n"
        << "-------\n";

    std::cout
        << "Configuration:        1s^2 2s^1\n";

    std::cout
        << "N_up:                 "
        << lithium.electronsUp
        << "\n";

    std::cout
        << "N_down:               "
        << lithium.electronsDown
        << "\n";

    std::cout
        << "Spin moment:          "
        << lithium.spinMoment
        << "\n";

    std::cout
        << "E(1s up):             "
        << lithium.orbitals.up1sEnergy
        << " Ha\n";

    std::cout
        << "E(2s up):             "
        << lithium.orbitals.up2sEnergy
        << " Ha\n";

    std::cout
        << "E(1s down):           "
        << lithium.orbitals.down1sEnergy
        << " Ha\n";

    std::cout
        << "Total LSDA energy:    "
        << lithium.energy.total
        << " Ha\n";

    std::cout
        << "SCF converged:        "
        << (
            lithium.converged
            ? "YES"
            : "NO"
        )
        << "\n\n";

    std::cout
        << "Helium triplet\n"
        << "--------------\n";

    std::cout
        << "Configuration:        1s^1 2s^1\n";

    std::cout
        << "Spin component:       M_S = +1\n";

    std::cout
        << "N_up:                 "
        << heliumTriplet.electronsUp
        << "\n";

    std::cout
        << "N_down:               "
        << heliumTriplet.electronsDown
        << "\n";

    std::cout
        << "Spin moment:          "
        << heliumTriplet.spinMoment
        << "\n";

    std::cout
        << "E(1s up):             "
        << heliumTriplet.orbitals.up1sEnergy
        << " Ha\n";

    std::cout
        << "E(2s up):             "
        << heliumTriplet.orbitals.up2sEnergy
        << " Ha\n";

    std::cout
        << "Total LSDA energy:    "
        << heliumTriplet.energy.total
        << " Ha\n";

    std::cout
        << "SCF converged:        "
        << (
            heliumTriplet.converged
            ? "YES"
            : "NO"
        )
        << "\n\n";

    std::cout
        << "====================================================\n";
}

// ================================================================
// Main
// ================================================================

int main()
{
    try
    {
        std::cout
            << "======================================================\n"
            << "       SPIN-POLARIZED KOHN-SHAM DFT / LSDA\n"
            << "======================================================\n\n";

        std::cout
            << "This program solves spherical atomic Kohn-Sham\n"
            << "equations with separate spin-up and spin-down\n"
            << "densities.\n\n";

        // ========================================================
        // Lithium
        // ========================================================

        const AtomicSCFResult lithium =
            runLithium();

        const double lithiumRmax =
            30.0;

        const int lithiumN =
            2000;

        const double lithiumDr =
            lithiumRmax /
            static_cast<double>(
                lithiumN + 1);

        printSpinDiagnostics(
            lithium,
            lithiumDr);

        printOrbitalDiagnostics(
            lithium,
            lithiumDr);

        printEnergyDiagnostics(
            lithium);

        printPotentialDiagnostics(
            lithium,
            lithiumDr);

        // ========================================================
        // Helium triplet
        // ========================================================

        const AtomicSCFResult heliumTriplet =
            runHeliumTriplet();

        const double heliumRmax =
            30.0;

        const int heliumN =
            2000;

        const double heliumDr =
            heliumRmax /
            static_cast<double>(
                heliumN + 1);

        printSpinDiagnostics(
            heliumTriplet,
            heliumDr);

        printOrbitalDiagnostics(
            heliumTriplet,
            heliumDr);

        printEnergyDiagnostics(
            heliumTriplet);

        printPotentialDiagnostics(
            heliumTriplet,
            heliumDr);

        // ========================================================
        // Final summary
        // ========================================================

        printFinalSummary(
            lithium,
            heliumTriplet);

        std::cout
            << "\nProgram finished successfully.\n";
    }
    catch (const std::exception& exception)
    {
        std::cerr
            << "\nFATAL ERROR:\n"
            << exception.what()
            << "\n";

        return 1;
    }

    return 0;
}