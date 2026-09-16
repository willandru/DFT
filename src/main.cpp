#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
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

// ================================================================
// PBE constants
//
// Perdew, Burke and Ernzerhof.
// Phys. Rev. Lett. 77, 3865 (1996).
//
// Atomic units.
// ================================================================

constexpr double PBE_MU =
    0.2195149727645171;

constexpr double PBE_KAPPA =
    0.8040;

constexpr double PBE_BETA =
    0.06672455060314922;

constexpr double PBE_GAMMA =
    0.031090690869654895;

// ================================================================
// PW92 correlation constants
// ================================================================

constexpr double PW92_A =
    0.0310907;

constexpr double PW92_ALPHA1 =
    0.21370;

constexpr double PW92_BETA1 =
    7.5957;

constexpr double PW92_BETA2 =
    3.5876;

constexpr double PW92_BETA3 =
    1.6382;

constexpr double PW92_BETA4 =
    0.49294;

// ================================================================
// Radial grid
//
// r_i = (i + 1) dr
//
// u(0) = u(Rmax) = 0
//
// Rmax = (N + 1) dr
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
    std::vector<double> u;
    double eigenvalue;
};

struct SCFResult
{
    double totalEnergy;
    double orbitalEnergy;
    double densityError;
    int iterations;
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
    {
        cPrime[0] =
            upper[0] /
            pivot;
    }

    dPrime[0] =
        rhs[0] /
        pivot;

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
        {
            cPrime[i] =
                upper[i] /
                pivot;
        }

        dPrime[i] =
            (
                rhs[i] -
                lower[i - 1] *
                dPrime[i - 1]
            ) /
            pivot;
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
// Normalize radial orbital
//
// integral |u|² dr = 1
// ================================================================

void normalize(
    std::vector<double>& u,
    double dr)
{
    double norm2 =
        0.0;

    for (double value : u)
    {
        norm2 +=
            value *
            value *
            dr;
    }

    if (norm2 <= 0.0)
        throw std::runtime_error(
            "Cannot normalize zero orbital.");

    const double inverseNorm =
        1.0 /
        std::sqrt(norm2);

    for (double& value : u)
    {
        value *=
            inverseNorm;
    }
}

// ================================================================
// Apply radial Hamiltonian
//
// H = -1/2 d²/dr² + V(r)
//
// l = 0
// ================================================================

std::vector<double> applyHamiltonian(
    const std::vector<double>& u,
    const std::vector<double>& potential,
    double dr)
{
    const int n =
        static_cast<int>(u.size());

    if (static_cast<int>(potential.size()) != n)
        throw std::runtime_error(
            "Hamiltonian vector size mismatch.");

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

    double numerator =
        0.0;

    double denominator =
        0.0;

    for (std::size_t i = 0;
         i < u.size();
         ++i)
    {
        numerator +=
            u[i] *
            Hu[i] *
            dr;

        denominator +=
            u[i] *
            u[i] *
            dr;
    }

    if (denominator <= 0.0)
        throw std::runtime_error(
            "Invalid Rayleigh quotient denominator.");

    return
        numerator /
        denominator;
}

// ================================================================
// Sturm count
// ================================================================

int countEigenvaluesBelow(
    const std::vector<double>& diagonal,
    double offDiagonal,
    double energy)
{
    const int n =
        static_cast<int>(
            diagonal.size());

    int count =
        0;

    double q =
        diagonal[0] -
        energy;

    if (q < 0.0)
        ++count;

    if (std::abs(q) < MIN_PIVOT)
    {
        q =
            (q < 0.0)
            ? -MIN_PIVOT
            : MIN_PIVOT;
    }

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
// Ground-state radial solver
// ================================================================

OrbitalResult solveGroundState(
    const std::vector<double>& potential,
    double dr,
    const std::vector<double>& initialGuess)
{
    const int n =
        static_cast<int>(
            potential.size());

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

    // ------------------------------------------------------------
    // Eigenvalue bracket.
    // ------------------------------------------------------------

    double lower =
        -100.0;

    double upper =
        10.0;

    while (
        countEigenvaluesBelow(
            diagonal,
            offDiagonal,
            lower) > 0)
    {
        lower *=
            2.0;

        if (lower < -1.0e8)
            throw std::runtime_error(
                "Unable to bracket lower eigenvalue.");
    }

    while (
        countEigenvaluesBelow(
            diagonal,
            offDiagonal,
            upper) < 1)
    {
        upper *=
            2.0;

        if (upper > 1.0e8)
            throw std::runtime_error(
                "Unable to bracket ground-state eigenvalue.");
    }

    // ------------------------------------------------------------
    // Bisection.
    // ------------------------------------------------------------

    double energy =
        0.0;

    for (int iteration = 0;
         iteration < 160;
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

        if (count >= 1)
            upper =
                middle;
        else
            lower =
                middle;

        energy =
            0.5 *
            (lower + upper);

        if (std::abs(
                upper - lower) <
            1.0e-13)
        {
            break;
        }
    }

    // ------------------------------------------------------------
    // Initial orbital.
    // ------------------------------------------------------------

    std::vector<double> orbital =
        initialGuess;

    normalize(
        orbital,
        dr);

    // ------------------------------------------------------------
    // Inverse iteration.
    // ------------------------------------------------------------

    const double sigma =
        energy -
        0.05;

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
            sigma;
    }

    for (int iteration = 0;
         iteration < 80;
         ++iteration)
    {
        std::vector<double> next =
            solveTridiagonal(
                lowerDiagonal,
                shiftedDiagonal,
                upperDiagonal,
                orbital);

        normalize(
            next,
            dr);

        double overlap =
            0.0;

        for (int i = 0;
             i < n;
             ++i)
        {
            overlap +=
                next[i] *
                orbital[i] *
                dr;
        }

        if (overlap < 0.0)
        {
            for (double& value : next)
            {
                value =
                    -value;
            }
        }

        double difference2 =
            0.0;

        for (int i = 0;
             i < n;
             ++i)
        {
            const double difference =
                next[i] -
                orbital[i];

            difference2 +=
                difference *
                difference *
                dr;
        }

        orbital.swap(next);

        if (std::sqrt(
                difference2) <
            1.0e-12)
        {
            break;
        }
    }

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
// Electron density
//
// rho(r) = N |u(r)|² / (4 pi r²)
// ================================================================

std::vector<double> calculateDensity(
    const std::vector<double>& u,
    double dr,
    double electronCount)
{
    const int n =
        static_cast<int>(
            u.size());

    std::vector<double> density(
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

        density[i] =
            electronCount *
            u[i] *
            u[i] /
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
// Density derivatives
// ================================================================

struct DensityDerivatives
{
    std::vector<double> first;
    std::vector<double> second;
};

DensityDerivatives calculateDensityDerivatives(
    const std::vector<double>& density,
    double dr)
{
    const int n =
        static_cast<int>(
            density.size());

    if (n < 4)
        throw std::runtime_error(
            "At least four density points are required.");

    DensityDerivatives result;

    result.first.assign(
        n,
        0.0);

    result.second.assign(
        n,
        0.0);

    const double inverse2Dr =
        1.0 /
        (2.0 * dr);

    const double inverseDr2 =
        1.0 /
        (dr * dr);

    // ------------------------------------------------------------
    // First derivative.
    // ------------------------------------------------------------

    result.first[0] =
        (
            -3.0 * density[0] +
            4.0 * density[1] -
            density[2]
        ) *
        inverse2Dr;

    for (int i = 1;
         i < n - 1;
         ++i)
    {
        result.first[i] =
            (
                density[i + 1] -
                density[i - 1]
            ) *
            inverse2Dr;
    }

    result.first[n - 1] =
        (
            3.0 * density[n - 1] -
            4.0 * density[n - 2] +
            density[n - 3]
        ) *
        inverse2Dr;

    // ------------------------------------------------------------
    // Second derivative.
    // ------------------------------------------------------------

    result.second[0] =
        (
            density[0] -
            2.0 * density[1] +
            density[2]
        ) *
        inverseDr2;

    for (int i = 1;
         i < n - 1;
         ++i)
    {
        result.second[i] =
            (
                density[i + 1] -
                2.0 * density[i] +
                density[i - 1]
            ) *
            inverseDr2;
    }

    result.second[n - 1] =
        (
            density[n - 1] -
            2.0 * density[n - 2] +
            density[n - 3]
        ) *
        inverseDr2;

    return result;
}

// ================================================================
// Hartree potential
//
// VH(r) = N [
//     1/r integral_0^r |u(r')|² dr'
//     +
//     integral_r^inf |u(r')|²/r' dr'
// ]
// ================================================================

std::vector<double> calculateHartreePotential(
    const std::vector<double>& u,
    double dr,
    double electronCount)
{
    const int n =
        static_cast<int>(
            u.size());

    std::vector<double> cumulative(
        n,
        0.0);

    std::vector<double> outer(
        n,
        0.0);

    std::vector<double> potential(
        n,
        0.0);

    cumulative[0] =
        0.5 *
        dr *
        u[0] *
        u[0];

    for (int i = 1;
         i < n;
         ++i)
    {
        cumulative[i] =
            cumulative[i - 1] +
            0.5 *
            dr *
            (
                u[i - 1] *
                u[i - 1] +
                u[i] *
                u[i]
            );
    }

    outer[n - 1] =
        0.5 *
        dr *
        u[n - 1] *
        u[n - 1] /
        radiusAt(
            n - 1,
            dr);

    for (int i = n - 2;
         i >= 0;
         --i)
    {
        const double r1 =
            radiusAt(
                i,
                dr);

        const double r2 =
            radiusAt(
                i + 1,
                dr);

        outer[i] =
            outer[i + 1] +
            0.5 *
            dr *
            (
                u[i] *
                u[i] /
                r1
                +
                u[i + 1] *
                u[i + 1] /
                r2
            );
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
            electronCount *
            (
                cumulative[i] /
                r
                +
                outer[i]
            );
    }

    return potential;
}

// ================================================================
// PW92 correlation
// ================================================================

struct PW92Correlation
{
    double epsilon;
    double derivativeRs;
};

PW92Correlation calculatePW92Correlation(
    double rs)
{
    if (rs <= 0.0)
        throw std::runtime_error(
            "Invalid rs in PW92.");

    const double sqrtRs =
        std::sqrt(rs);

    const double q0 =
        -2.0 *
        PW92_A *
        (
            1.0 +
            PW92_ALPHA1 *
            rs
        );

    const double q1 =
        2.0 *
        PW92_A *
        sqrtRs *
        (
            PW92_BETA1 +
            sqrtRs *
            (
                PW92_BETA2 +
                sqrtRs *
                (
                    PW92_BETA3 +
                    PW92_BETA4 *
                    sqrtRs
                )
            )
        );

    const double q2 =
        std::log1p(
            1.0 / q1);

    const double epsilon =
        q0 *
        q2;

    const double q3 =
        PW92_A *
        (
            PW92_BETA1 /
            sqrtRs
            +
            2.0 *
            PW92_BETA2
            +
            sqrtRs *
            (
                3.0 *
                PW92_BETA3
                +
                4.0 *
                PW92_BETA4 *
                sqrtRs
            )
        );

    const double derivative =
        -2.0 *
        PW92_A *
        PW92_ALPHA1 *
        q2
        -
        q0 *
        q3 /
        (
            q1 *
            (1.0 + q1)
        );

    return
    {
        epsilon,
        derivative
    };
}

// ================================================================
// PBE exchange
//
// epsilon_x^PBE = epsilon_x^LDA F_x(s)
//
// s = |grad rho| / (2 kF rho)
//
// F_x(s) = 1 + kappa
//          - kappa / (1 + mu s²/kappa)
// ================================================================

double calculatePBEExchangeEnergyPerElectron(
    double rho,
    double gradientMagnitude)
{
    if (rho <= MIN_DENSITY)
        return 0.0;

    const double kF =
        std::pow(
            3.0 *
            PI *
            PI *
            rho,
            1.0 / 3.0);

    const double s =
        gradientMagnitude /
        (
            2.0 *
            kF *
            rho
        );

    const double s2 =
        s * s;

    const double denominator =
        1.0 +
        PBE_MU *
        s2 /
        PBE_KAPPA;

    const double enhancement =
        1.0 +
        PBE_KAPPA -
        PBE_KAPPA /
        denominator;

    const double ldaExchange =
        -0.75 *
        std::pow(
            3.0 / PI,
            1.0 / 3.0) *
        std::pow(
            rho,
            1.0 / 3.0);

    return
        ldaExchange *
        enhancement;
}

// ================================================================
// PBE correlation
// ================================================================

double calculatePBECorrelationEnergyPerElectron(
    double rho,
    double gradientMagnitude)
{
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

    const PW92Correlation uniform =
        calculatePW92Correlation(
            rs);

    const double epsilonC =
        uniform.epsilon;

    const double kF =
        std::pow(
            3.0 *
            PI *
            PI *
            rho,
            1.0 / 3.0);

    const double ks =
        std::sqrt(
            4.0 *
            kF /
            PI);

    const double t =
        gradientMagnitude /
        (
            2.0 *
            ks *
            rho
        );

    const double t2 =
        t * t;

    if (t2 < 1.0e-30)
        return epsilonC;

    const double denominatorA =
        std::expm1(
            -epsilonC /
            PBE_GAMMA);

    if (denominatorA <= 0.0)
        return epsilonC;

    const double A =
        (
            PBE_BETA /
            PBE_GAMMA
        ) /
        denominatorA;

    const double At2 =
        A *
        t2;

    const double A2t4 =
        A *
        A *
        t2 *
        t2;

    const double numerator =
        (
            PBE_BETA /
            PBE_GAMMA
        ) *
        t2 *
        (
            1.0 +
            At2
        );

    const double denominator =
        1.0 +
        At2 +
        A2t4;

    const double x =
        numerator /
        denominator;

    const double H =
        PBE_GAMMA *
        std::log1p(
            x);

    return
        epsilonC +
        H;
}

// ================================================================
// PBE functional density
//
// f(rho,g) = rho [epsilon_x + epsilon_c]
// ================================================================

double calculatePBEFunctionalDensity(
    double rho,
    double gradientMagnitude)
{
    if (rho <= MIN_DENSITY)
        return 0.0;

    const double exchange =
        calculatePBEExchangeEnergyPerElectron(
            rho,
            gradientMagnitude);

    const double correlation =
        calculatePBECorrelationEnergyPerElectron(
            rho,
            gradientMagnitude);

    return
        rho *
        (
            exchange +
            correlation
        );
}

// ================================================================
// Partial derivative:
//
// df / drho
// ================================================================

double calculatePartialDensityDerivative(
    double rho,
    double gradientMagnitude)
{
    if (rho <= MIN_DENSITY)
        return 0.0;

    constexpr double relativeStep =
        1.0e-5;

    const double rhoPlus =
        rho *
        (1.0 + relativeStep);

    const double rhoMinus =
        rho *
        (1.0 - relativeStep);

    const double fPlus =
        calculatePBEFunctionalDensity(
            rhoPlus,
            gradientMagnitude);

    const double fMinus =
        calculatePBEFunctionalDensity(
            rhoMinus,
            gradientMagnitude);

    return
        (
            fPlus -
            fMinus
        ) /
        (
            rhoPlus -
            rhoMinus
        );
}

// ================================================================
// Partial derivative:
//
// df / d|grad rho|
// ================================================================

double calculatePartialGradientDerivative(
    double rho,
    double gradientMagnitude)
{
    if (rho <= MIN_DENSITY)
        return 0.0;

    const double scale =
        std::max(
            gradientMagnitude,
            1.0e-12);

    const double step =
        1.0e-5 *
        scale;

    const double gradientPlus =
        gradientMagnitude +
        step;

    const double gradientMinus =
        std::max(
            0.0,
            gradientMagnitude -
            step);

    if (gradientPlus <= gradientMinus)
        return 0.0;

    const double fPlus =
        calculatePBEFunctionalDensity(
            rho,
            gradientPlus);

    const double fMinus =
        calculatePBEFunctionalDensity(
            rho,
            gradientMinus);

    return
        (
            fPlus -
            fMinus
        ) /
        (
            gradientPlus -
            gradientMinus
        );
}

// ================================================================
// PBE exchange-correlation energy
// ================================================================

struct PBEEnergy
{
    double exchange;
    double correlation;
    double total;
};

PBEEnergy calculatePBEExchangeCorrelationEnergy(
    const std::vector<double>& density,
    double dr)
{
    const int n =
        static_cast<int>(
            density.size());

    const DensityDerivatives derivatives =
        calculateDensityDerivatives(
            density,
            dr);

    double exchange =
        0.0;

    double correlation =
        0.0;

    for (int i = 0;
         i < n;
         ++i)
    {
        const double rho =
            density[i];

        if (rho <= MIN_DENSITY)
            continue;

        const double r =
            radiusAt(
                i,
                dr);

        const double gradient =
            std::abs(
                derivatives.first[i]);

        const double epsilonX =
            calculatePBEExchangeEnergyPerElectron(
                rho,
                gradient);

        const double epsilonC =
            calculatePBECorrelationEnergyPerElectron(
                rho,
                gradient);

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
            rho *
            epsilonX *
            volumeElement;

        correlation +=
            weight *
            rho *
            epsilonC *
            volumeElement;
    }

    return
    {
        exchange,
        correlation,
        exchange +
        correlation
    };
}

// ================================================================
// PBE exchange-correlation potential
//
// vxc = df/drho
//       -
//       1/r² d/dr [
//           r² df/d|grad rho|
//       ]
// ================================================================

std::vector<double>
calculatePBEExchangeCorrelationPotential(
    const std::vector<double>& density,
    double dr)
{
    const int n =
        static_cast<int>(
            density.size());

    const DensityDerivatives derivatives =
        calculateDensityDerivatives(
            density,
            dr);

    std::vector<double> partialRho(
        n,
        0.0);

    std::vector<double> partialGradient(
        n,
        0.0);

    // ------------------------------------------------------------
    // Local partial derivatives.
    // ------------------------------------------------------------

    for (int i = 0;
         i < n;
         ++i)
    {
        const double rho =
            density[i];

        if (rho <= MIN_DENSITY)
            continue;

        const double gradient =
            std::abs(
                derivatives.first[i]);

        partialRho[i] =
            calculatePartialDensityDerivative(
                rho,
                gradient);

        partialGradient[i] =
            calculatePartialGradientDerivative(
                rho,
                gradient);
    }

    // ------------------------------------------------------------
    // Radial flux.
    //
    // q(r) = r² df/d|grad rho|
    // ------------------------------------------------------------

    std::vector<double> radialFlux(
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

        radialFlux[i] =
            r *
            r *
            partialGradient[i];
    }

    // ------------------------------------------------------------
    // Divergence.
    // ------------------------------------------------------------

    std::vector<double> divergence(
        n,
        0.0);

    // First point.
    {
        const double derivativeAtFirstPoint =
            (
                -3.0 *
                radialFlux[0]
                +
                4.0 *
                radialFlux[1]
                -
                radialFlux[2]
            ) /
            (
                2.0 *
                dr
            );

        const double radiusAtFirstPoint =
            radiusAt(
                0,
                dr);

        divergence[0] =
            derivativeAtFirstPoint /
            (
                radiusAtFirstPoint *
                radiusAtFirstPoint
            );
    }

    // Interior.
    for (int i = 1;
         i < n - 1;
         ++i)
    {
        const double derivativeAtInterior =
            (
                radialFlux[i + 1] -
                radialFlux[i - 1]
            ) /
            (
                2.0 *
                dr
            );

        const double radiusAtInterior =
            radiusAt(
                i,
                dr);

        divergence[i] =
            derivativeAtInterior /
            (
                radiusAtInterior *
                radiusAtInterior
            );
    }

    // Last point.
    {
        const double derivativeAtLastPoint =
            (
                3.0 *
                radialFlux[n - 1]
                -
                4.0 *
                radialFlux[n - 2]
                +
                radialFlux[n - 3]
            ) /
            (
                2.0 *
                dr
            );

        const double radiusAtLastPoint =
            radiusAt(
                n - 1,
                dr);

        divergence[n - 1] =
            derivativeAtLastPoint /
            (
                radiusAtLastPoint *
                radiusAtLastPoint
            );
    }

    // ------------------------------------------------------------
    // Functional derivative.
    // ------------------------------------------------------------

    std::vector<double> potential(
        n,
        0.0);

    for (int i = 0;
         i < n;
         ++i)
    {
        if (density[i] <= MIN_DENSITY)
        {
            potential[i] =
                0.0;

            continue;
        }

        potential[i] =
            partialRho[i] -
            divergence[i];
    }

    return potential;
}

// ================================================================
// Kinetic energy
//
// IMPORTANT:
//
// u is normalized to ONE electron orbital.
//
// For a closed-shell orbital occupied by N electrons:
//
//     Ts = N <u| -1/2 d²/dr² |u>
//
// Therefore electronCount MUST multiply the one-orbital
// kinetic contribution.
// ================================================================

double calculateKineticEnergy(
    const std::vector<double>& u,
    double dr,
    double electronCount)
{
    const int n =
        static_cast<int>(
            u.size());

    const double inverseDr2 =
        1.0 /
        (dr * dr);

    double oneElectronKinetic =
        0.0;

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

        oneElectronKinetic +=
            u[i] *
            (
                -0.5 *
                secondDerivative
            ) *
            dr;
    }

    return
        electronCount *
        oneElectronKinetic;
}

// ================================================================
// External nuclear energy
//
// Eext = integral rho(-Z/r)d³r
//
// With rho = N|u|²/(4 pi r²):
//
// Eext = -ZN integral |u|²/r dr
// ================================================================

double calculateExternalEnergy(
    const std::vector<double>& u,
    double dr,
    double nuclearCharge,
    double electronCount)
{
    const int n =
        static_cast<int>(
            u.size());

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

        const double weight =
            (i == 0 ||
             i == n - 1)
            ? 0.5
            : 1.0;

        energy +=
            weight *
            (
                -nuclearCharge *
                electronCount *
                u[i] *
                u[i] /
                r
            ) *
            dr;
    }

    return energy;
}

// ================================================================
// Hartree energy
//
// EH = 1/2 integral rho VH d³r
// ================================================================

double calculateHartreeEnergy(
    const std::vector<double>& density,
    const std::vector<double>& hartree,
    double dr)
{
    const int n =
        static_cast<int>(
            density.size());

    if (static_cast<int>(
            hartree.size()) != n)
    {
        throw std::runtime_error(
            "Hartree energy size mismatch.");
    }

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

        const double weight =
            (i == 0 ||
             i == n - 1)
            ? 0.5
            : 1.0;

        energy +=
            weight *
            0.5 *
            density[i] *
            hartree[i] *
            4.0 *
            PI *
            r *
            r *
            dr;
    }

    return energy;
}

// ================================================================
// Total PBE energy
//
// E = Ts + Eext + EH + Exc
// ================================================================

double calculateTotalEnergy(
    const std::vector<double>& u,
    const std::vector<double>& density,
    const std::vector<double>& hartree,
    double dr,
    double nuclearCharge,
    double electronCount)
{
    const double kinetic =
        calculateKineticEnergy(
            u,
            dr,
            electronCount);

    const double external =
        calculateExternalEnergy(
            u,
            dr,
            nuclearCharge,
            electronCount);

    const double hartreeEnergy =
        calculateHartreeEnergy(
            density,
            hartree,
            dr);

    const PBEEnergy xc =
        calculatePBEExchangeCorrelationEnergy(
            density,
            dr);

    return
        kinetic +
        external +
        hartreeEnergy +
        xc.total;
}

// ================================================================
// Hydrogen
// ================================================================

double runHydrogen()
{
    std::cout
        << "====================================================\n"
        << "HYDROGEN\n"
        << "========\n\n";

    const int N =
        2000;

    const double Rmax =
        30.0;

    const double dr =
        Rmax /
        static_cast<double>(
            N + 1);

    const double Z =
        1.0;

    std::vector<double> potential(
        N);

    for (int i = 0;
         i < N;
         ++i)
    {
        const double r =
            radiusAt(
                i,
                dr);

        potential[i] =
            -Z / r;
    }

    std::vector<double> initialGuess(
        N);

    for (int i = 0;
         i < N;
         ++i)
    {
        const double r =
            radiusAt(
                i,
                dr);

        initialGuess[i] =
            r *
            std::exp(-r);
    }

    const OrbitalResult result =
        solveGroundState(
            potential,
            dr,
            initialGuess);

    const double exactEnergy =
        -0.5;

    const double absoluteError =
        std::abs(
            result.eigenvalue -
            exactEnergy);

    std::cout
        << std::setprecision(15);

    std::cout
        << "Grid points:       "
        << N
        << "\n";

    std::cout
        << "dr:                "
        << dr
        << " Bohr\n\n";

    std::cout
        << "Configuration:     1s^1\n";

    std::cout
        << "Numerical energy:  "
        << result.eigenvalue
        << " Ha\n";

    std::cout
        << "Exact energy:      "
        << exactEnergy
        << " Ha\n";

    std::cout
        << "Absolute error:    "
        << absoluteError
        << " Ha\n\n";

    if (absoluteError < 1.0e-4)
        std::cout
            << "PASS\n";
    else
        std::cout
            << "FAIL\n";

    std::cout
        << "\n";

    return result.eigenvalue;
}

// ================================================================
// Helium
//
// Closed shell:
//
//     1s²
//
// Spherical, spin-unpolarized PBE.
// ================================================================

SCFResult runHelium()
{
    std::cout
        << "====================================================\n"
        << "HELIUM\n"
        << "======\n\n";

    const int N =
        2000;

    const double Rmax =
        30.0;

    const double dr =
        Rmax /
        static_cast<double>(
            N + 1);

    const double Z =
        2.0;

    const double electronCount =
        2.0;

    const int maxIterations =
        100;

    const double mixing =
        0.30;

    const double densityTolerance =
        1.0e-9;

    const double energyTolerance =
        1.0e-11;

    // ------------------------------------------------------------
    // Initial hydrogenic orbital.
    // ------------------------------------------------------------

    std::vector<double> orbital(
        N);

    for (int i = 0;
         i < N;
         ++i)
    {
        const double r =
            radiusAt(
                i,
                dr);

        orbital[i] =
            r *
            std::exp(
                -Z * r);
    }

    normalize(
        orbital,
        dr);

    std::vector<double> density =
        calculateDensity(
            orbital,
            dr,
            electronCount);

    double previousEnergy =
        std::numeric_limits<double>::quiet_NaN();

    double finalDensityError =
        std::numeric_limits<double>::infinity();

    double finalEnergy =
        0.0;

    double finalOrbitalEnergy =
        0.0;

    int iterationsUsed =
        maxIterations;

    bool converged =
        false;

    std::cout
        << "System\n\n";

    std::cout
        << "Element:             He\n"
        << "Nuclear charge:      Z = 2\n"
        << "Electrons:           2\n"
        << "Configuration:       1s^2\n\n";

    std::cout
        << "Kohn-Sham model\n\n";

    std::cout
        << "Hartree:             ON\n"
        << "Exchange:            PBE-GGA\n"
        << "Correlation:         PBE-GGA\n"
        << "SCF mixing:          "
        << mixing
        << "\n\n";

    std::cout
        << "SCF iterations\n\n";

    std::cout
        << std::setprecision(15);

    for (int iteration = 1;
         iteration <= maxIterations;
         ++iteration)
    {
        // --------------------------------------------------------
        // Hartree.
        // --------------------------------------------------------

        const std::vector<double> hartree =
            calculateHartreePotential(
                orbital,
                dr,
                electronCount);

        // --------------------------------------------------------
        // PBE exchange-correlation potential.
        // --------------------------------------------------------

        const std::vector<double> exchangeCorrelation =
            calculatePBEExchangeCorrelationPotential(
                density,
                dr);

        // --------------------------------------------------------
        // Kohn-Sham potential.
        // --------------------------------------------------------

        std::vector<double> effectivePotential(
            N);

        for (int i = 0;
             i < N;
             ++i)
        {
            const double r =
                radiusAt(
                    i,
                    dr);

            effectivePotential[i] =
                -Z / r +
                hartree[i] +
                exchangeCorrelation[i];
        }

        // --------------------------------------------------------
        // Solve Kohn-Sham equation.
        // --------------------------------------------------------

        const OrbitalResult orbitalResult =
            solveGroundState(
                effectivePotential,
                dr,
                orbital);

        finalOrbitalEnergy =
            orbitalResult.eigenvalue;

        // --------------------------------------------------------
        // New density.
        // --------------------------------------------------------

        const std::vector<double> newDensity =
            calculateDensity(
                orbitalResult.u,
                dr,
                electronCount);

        // --------------------------------------------------------
        // Raw density residual.
        // --------------------------------------------------------

        double densityError =
            0.0;

        for (int i = 0;
             i < N;
             ++i)
        {
            const double r =
                radiusAt(
                    i,
                    dr);

            const double weight =
                (i == 0 ||
                 i == N - 1)
                ? 0.5
                : 1.0;

            densityError +=
                weight *
                std::abs(
                    newDensity[i] -
                    density[i]) *
                4.0 *
                PI *
                r *
                r *
                dr;
        }

        // --------------------------------------------------------
        // Density mixing.
        // --------------------------------------------------------

        std::vector<double> mixedDensity(
            N);

        for (int i = 0;
             i < N;
             ++i)
        {
            mixedDensity[i] =
                (
                    1.0 -
                    mixing
                ) *
                density[i]
                +
                mixing *
                newDensity[i];
        }

        // --------------------------------------------------------
        // Normalize mixed density to exactly two electrons.
        // --------------------------------------------------------

        double electronNumber =
            0.0;

        for (int i = 0;
             i < N;
             ++i)
        {
            const double r =
                radiusAt(
                    i,
                    dr);

            const double weight =
                (i == 0 ||
                 i == N - 1)
                ? 0.5
                : 1.0;

            electronNumber +=
                weight *
                mixedDensity[i] *
                4.0 *
                PI *
                r *
                r *
                dr;
        }

        if (electronNumber <= 0.0)
            throw std::runtime_error(
                "Invalid mixed electron density.");

        const double densityScale =
            electronCount /
            electronNumber;

        for (double& value :
             mixedDensity)
        {
            value *=
                densityScale;
        }

        // --------------------------------------------------------
        // Construct orbital from mixed density.
        // --------------------------------------------------------

        std::vector<double> mixedOrbital(
            N);

        for (int i = 0;
             i < N;
             ++i)
        {
            const double r =
                radiusAt(
                    i,
                    dr);

            const double value =
                mixedDensity[i] *
                4.0 *
                PI *
                r *
                r /
                electronCount;

            mixedOrbital[i] =
                std::sqrt(
                    std::max(
                        0.0,
                        value));
        }

        normalize(
            mixedOrbital,
            dr);

        // --------------------------------------------------------
        // Energy.
        // --------------------------------------------------------

        const std::vector<double> mixedHartree =
            calculateHartreePotential(
                mixedOrbital,
                dr,
                electronCount);

        const double totalEnergy =
            calculateTotalEnergy(
                mixedOrbital,
                mixedDensity,
                mixedHartree,
                dr,
                Z,
                electronCount);

        double energyChange =
            std::numeric_limits<double>::infinity();

        if (std::isfinite(
                previousEnergy))
        {
            energyChange =
                std::abs(
                    totalEnergy -
                    previousEnergy);
        }
        else
        {
            energyChange =
                std::abs(
                    totalEnergy);
        }

        // --------------------------------------------------------
        // Update.
        // --------------------------------------------------------

        density =
            std::move(
                mixedDensity);

        orbital =
            std::move(
                mixedOrbital);

        previousEnergy =
            totalEnergy;

        finalEnergy =
            totalEnergy;

        finalDensityError =
            densityError;

        // --------------------------------------------------------
        // Output.
        // --------------------------------------------------------

        if (iteration == 1 ||
            iteration % 5 == 0)
        {
            std::cout
                << "Iteration "
                << std::setw(3)
                << iteration
                << "   orbital = "
                << finalOrbitalEnergy
                << " Ha"
                << "   E = "
                << finalEnergy
                << " Ha"
                << "   dE = "
                << energyChange
                << "   dRho = "
                << finalDensityError
                << "\n";
        }

        // --------------------------------------------------------
        // Convergence.
        // --------------------------------------------------------

        if (iteration > 1 &&
            finalDensityError <
                densityTolerance &&
            energyChange <
                energyTolerance)
        {
            iterationsUsed =
                iteration;

            converged =
                true;

            break;
        }
    }

    // ------------------------------------------------------------
    // Final Kohn-Sham solution.
    // ------------------------------------------------------------

    const std::vector<double> finalHartree =
        calculateHartreePotential(
            orbital,
            dr,
            electronCount);

    const std::vector<double> finalXC =
        calculatePBEExchangeCorrelationPotential(
            density,
            dr);

    std::vector<double> finalPotential(
        N);

    for (int i = 0;
         i < N;
         ++i)
    {
        const double r =
            radiusAt(
                i,
                dr);

        finalPotential[i] =
            -Z / r +
            finalHartree[i] +
            finalXC[i];
    }

    const OrbitalResult finalOrbital =
        solveGroundState(
            finalPotential,
            dr,
            orbital);

    finalOrbitalEnergy =
        finalOrbital.eigenvalue;

    if (converged)
    {
        std::cout
            << "\nSCF converged after "
            << iterationsUsed
            << " iterations.\n";
    }
    else
    {
        std::cout
            << "\nSCF did NOT converge within "
            << maxIterations
            << " iterations.\n";
    }

    return
    {
        finalEnergy,
        finalOrbitalEnergy,
        finalDensityError,
        iterationsUsed
    };
}

// ================================================================
// Main
// ================================================================

int main()
{
    std::cout
        << "======================================================\n"
        << "          MINIMAL KOHN-SHAM DFT\n"
        << "======================================================\n\n";

    const double hydrogenEnergy =
        runHydrogen();

    const SCFResult helium =
        runHelium();

    std::cout
        << "====================================================\n"
        << "SUMMARY\n"
        << "=======\n\n";

    std::cout
        << std::setprecision(15);

    std::cout
        << "Hydrogen\n"
        << "--------\n";

    std::cout
        << "Numerical energy:     "
        << hydrogenEnergy
        << " Ha\n";

    std::cout
        << "Exact energy:         "
        << -0.5
        << " Ha\n\n";

    std::cout
        << "Helium\n"
        << "------\n";

    std::cout
        << "Configuration:        1s^2\n";

    std::cout
        << "Orbital energy:       "
        << helium.orbitalEnergy
        << " Ha\n";

    std::cout
        << "PBE total energy:     "
        << helium.totalEnergy
        << " Ha\n";

    std::cout
        << "SCF iterations:       "
        << helium.iterations
        << "\n";

    std::cout
        << "Final density change: "
        << helium.densityError
        << "\n\n";

    std::cout
        << "Exchange:             PBE-GGA\n"
        << "Correlation:          PBE-GGA\n";

    std::cout
        << "Reference He energy:  approximately -2.9037 Ha\n\n";

    std::cout
        << "====================================================\n";

    return 0;
}