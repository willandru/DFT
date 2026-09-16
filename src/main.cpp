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

// ================================================================
// PBE constants
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
// The physical interval is:
//
//     0 < r < Rmax
//
// The numerical points are located at:
//
//     r_i = (i + 1) dr
//
// with u(0) = 0 and u(Rmax) = 0.
//
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
    double energyChange;
    int iterations;
    bool converged;

    std::vector<double> orbital;
    std::vector<double> density;
    std::vector<double> hartree;
    std::vector<double> exchangeCorrelation;
    std::vector<double> effectivePotential;
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
// Normalize radial orbital
//
// For the reduced radial orbital:
//
//     integral |u(r)|^2 dr = 1
//
// ================================================================

void normalize(
    std::vector<double>& u,
    double dr)
{
    double norm2 =
        0.0;

    const int n =
        static_cast<int>(u.size());

    for (int i = 0;
         i < n;
         ++i)
    {
        const double weight =
            (i == 0 ||
             i == n - 1)
            ? 0.5
            : 1.0;

        norm2 +=
            weight *
            u[i] *
            u[i] *
            dr;
    }

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
// Apply radial Hamiltonian
//
// H = -1/2 d²/dr² + V(r)
//
// The reduced radial orbital satisfies:
//
//     u(0) = 0
//     u(Rmax) = 0
//
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

    const int n =
        static_cast<int>(u.size());

    for (int i = 0;
         i < n;
         ++i)
    {
        const double weight =
            (i == 0 ||
             i == n - 1)
            ? 0.5
            : 1.0;

        numerator +=
            weight *
            u[i] *
            Hu[i] *
            dr;

        denominator +=
            weight *
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
//
// First obtain the lowest eigenvalue through a Sturm bisection.
// Then construct the corresponding eigenvector using inverse
// iteration around the computed eigenvalue.
//
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
        lower *= 2.0;

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
        upper *= 2.0;

        if (upper > 1.0e8)
            throw std::runtime_error(
                "Unable to bracket ground-state eigenvalue.");
    }

    double energy =
        0.0;

    for (int iteration = 0;
         iteration < 200;
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

    std::vector<double> orbital =
        initialGuess;

    normalize(
        orbital,
        dr);

    /*
        Inverse iteration is performed with a shift slightly below
        the lowest eigenvalue.  The shift is chosen relative to the
        numerical spectral spacing so that the linear system remains
        well conditioned while still strongly selecting the ground
        state.
    */

    const double shift =
        energy - 0.02;

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
         iteration < 120;
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
            const double weight =
                (i == 0 ||
                 i == n - 1)
                ? 0.5
                : 1.0;

            overlap +=
                weight *
                next[i] *
                orbital[i] *
                dr;
        }

        if (overlap < 0.0)
        {
            for (double& value : next)
                value = -value;
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

            const double weight =
                (i == 0 ||
                 i == n - 1)
                ? 0.5
                : 1.0;

            difference2 +=
                weight *
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
//
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
// For spherical density:
//
// V_H(r) = Q(r)/r + integral_r^R [4 pi r' rho(r')] / r' dr'
//
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

    const double denominatorS =
        2.0 *
        kF *
        rho;

    if (denominatorS <= 0.0)
        return 0.0;

    const double s =
        gradientMagnitude /
        denominatorS;

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

    const double denominatorT =
        2.0 *
        ks *
        rho;

    if (denominatorT <= 0.0)
        return epsilonC;

    const double t =
        gradientMagnitude /
        denominatorT;

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
// f_xc(rho, |grad rho|) = rho * epsilon_xc
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
// Partial derivative df/drho
//
// Numerical derivative with respect to rho while keeping
// |grad rho| fixed.
//
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
// Partial derivative df/d|grad rho|
//
// Numerical derivative with respect to the gradient magnitude.
//
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
            1.0e-10);

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
// For:
//
//     E_xc = integral f(rho, |grad rho|) d^3r
//
// the functional derivative is:
//
//     v_xc = df/drho
//            - div(
//                df/d|grad rho| *
//                grad rho / |grad rho|
//              )
//
// For spherical symmetry:
//
//     v_xc(r) = df/drho
//              - 1/r² d/dr [
//                    r² df/d|grad rho| *
//                    sign(d rho / dr)
//                ]
//
// ================================================================

std::vector<double>
calculatePBEExchangeCorrelationPotential(
    const std::vector<double>& density,
    double dr)
{
    const int n =
        static_cast<int>(
            density.size());

    if (n < 4)
        throw std::runtime_error(
            "Insufficient density points for PBE potential.");

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

    std::vector<double> radialFlux(
        n,
        0.0);

    for (int i = 0;
         i < n;
         ++i)
    {
        const double rho =
            density[i];

        if (rho <= MIN_DENSITY)
            continue;

        const double densityGradient =
            derivatives.first[i];

        const double gradient =
            std::abs(
                densityGradient);

        partialRho[i] =
            calculatePartialDensityDerivative(
                rho,
                gradient);

        partialGradient[i] =
            calculatePartialGradientDerivative(
                rho,
                gradient);

        double sign =
            0.0;

        if (densityGradient > 0.0)
            sign = 1.0;
        else if (densityGradient < 0.0)
            sign = -1.0;

        radialFlux[i] =
            radiusAt(i, dr) *
            radiusAt(i, dr) *
            partialGradient[i] *
            sign;
    }

    std::vector<double> divergence(
        n,
        0.0);

    const double inverse2Dr =
        1.0 /
        (2.0 * dr);

    // ------------------------------------------------------------
    // First point
    // ------------------------------------------------------------

    {
        const double derivative =
            (
                -3.0 * radialFlux[0] +
                4.0 * radialFlux[1] -
                radialFlux[2]
            ) *
            inverse2Dr;

        const double r =
            radiusAt(
                0,
                dr);

        divergence[0] =
            derivative /
            (
                r * r
            );
    }

    // ------------------------------------------------------------
    // Interior
    // ------------------------------------------------------------

    for (int i = 1;
         i < n - 1;
         ++i)
    {
        const double derivative =
            (
                radialFlux[i + 1] -
                radialFlux[i - 1]
            ) *
            inverse2Dr;

        const double r =
            radiusAt(
                i,
                dr);

        divergence[i] =
            derivative /
            (
                r * r
            );
    }

    // ------------------------------------------------------------
    // Last point
    // ------------------------------------------------------------

    {
        const double derivative =
            (
                3.0 * radialFlux[n - 1] -
                4.0 * radialFlux[n - 2] +
                radialFlux[n - 3]
            ) *
            inverse2Dr;

        const double r =
            radiusAt(
                n - 1,
                dr);

        divergence[n - 1] =
            derivative /
            (
                r * r
            );
    }

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
// T_s = sum_i f_i * u_i * (-1/2 d²u/dr²) dr
//
// The same finite-difference kinetic operator used by the
// Kohn-Sham Hamiltonian is used here.
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

        const double kineticOperator =
            -0.5 *
            secondDerivative;

        const double weight =
            (i == 0 ||
             i == n - 1)
            ? 0.5
            : 1.0;

        oneElectronKinetic +=
            weight *
            u[i] *
            kineticOperator *
            dr;
    }

    return
        electronCount *
        oneElectronKinetic;
}

// ================================================================
// External nuclear energy
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
        200;

    const double mixing =
        0.30;

    const double densityTolerance =
        1.0e-9;

    const double energyTolerance =
        1.0e-11;

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

    double finalEnergyChange =
        std::numeric_limits<double>::infinity();

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
        // Build Kohn-Sham potential from current density.
        // --------------------------------------------------------

        const std::vector<double> currentHartree =
            calculateHartreePotential(
                orbital,
                dr,
                electronCount);

        const std::vector<double> currentXC =
            calculatePBEExchangeCorrelationPotential(
                density,
                dr);

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
                currentHartree[i] +
                currentXC[i];
        }

        // --------------------------------------------------------
        // Solve the Kohn-Sham equation.
        // --------------------------------------------------------

        const OrbitalResult orbitalResult =
            solveGroundState(
                effectivePotential,
                dr,
                orbital);

        finalOrbitalEnergy =
            orbitalResult.eigenvalue;

        // --------------------------------------------------------
        // Density generated by the new KS orbital.
        // --------------------------------------------------------

        const std::vector<double> newDensity =
            calculateDensity(
                orbitalResult.u,
                dr,
                electronCount);

        // --------------------------------------------------------
        // Density residual before mixing.
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
        // Linear density mixing.
        //
        // The orbital is NOT reconstructed from the mixed
        // density. The KS orbital remains the actual eigenvector
        // obtained from the current KS Hamiltonian.
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
        // Normalize the mixed density to the requested electron
        // number. This compensates only for numerical quadrature
        // error and does not alter its shape.
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
        // The actual KS orbital from this iteration is retained.
        // --------------------------------------------------------

        orbital =
            orbitalResult.u;

        // --------------------------------------------------------
        // The mixed density is the input density for the next
        // SCF iteration.
        // --------------------------------------------------------

        density =
            std::move(
                mixedDensity);

        // --------------------------------------------------------
        // Build Hartree from the actual orbital corresponding to
        // the current KS solution.
        // --------------------------------------------------------

        const std::vector<double> energyHartree =
            calculateHartreePotential(
                orbital,
                dr,
                electronCount);

        // --------------------------------------------------------
        // Total energy is evaluated from the current KS orbital
        // and its corresponding density.
        // --------------------------------------------------------

        const double totalEnergy =
            calculateTotalEnergy(
                orbital,
                density,
                energyHartree,
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

        previousEnergy =
            totalEnergy;

        finalEnergy =
            totalEnergy;

        finalDensityError =
            densityError;

        finalEnergyChange =
            energyChange;

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

        iterationsUsed =
            iteration;
    }

    // ============================================================
    // Rebuild the final self-consistent potential from the final
    // density.
    // ============================================================

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

    // ------------------------------------------------------------
    // Final eigenvalue in the final potential.
    // ------------------------------------------------------------

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
        finalEnergyChange,
        iterationsUsed,
        converged,
        finalOrbital.u,
        density,
        finalHartree,
        finalXC,
        finalPotential
    };
}

// ================================================================
// Diagnostics
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

double calculateDensityNormDifference(
    const std::vector<double>& a,
    const std::vector<double>& b,
    double dr)
{
    if (a.size() != b.size())
        throw std::runtime_error(
            "Density comparison size mismatch.");

    const int n =
        static_cast<int>(
            a.size());

    double result =
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

        result +=
            weight *
            std::abs(
                a[i] -
                b[i]) *
            4.0 *
            PI *
            r *
            r *
            dr;
    }

    return result;
}

// ================================================================
// Potential diagnostics
// ================================================================

void printPotentialDiagnostics(
    const SCFResult& result,
    double dr,
    double Z)
{
    const int n =
        static_cast<int>(
            result.density.size());

    std::cout
        << "\n====================================================\n"
        << "HELIUM POTENTIAL DIAGNOSTICS\n"
        << "============================\n\n";

    std::cout
        << std::setprecision(12);

    std::cout
        << std::left
        << std::setw(14) << "r [Bohr]"
        << std::setw(18) << "V_nuclear"
        << std::setw(18) << "V_Hartree"
        << std::setw(18) << "V_XC"
        << std::setw(18) << "V_eff"
        << "\n";

    std::cout
        << std::string(
            86,
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
        5.000,
        10.000
    };

    for (double requestedRadius :
         requestedRadii)
    {
        int index =
            static_cast<int>(
                std::round(
                    requestedRadius /
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

        const double nuclear =
            -Z / r;

        std::cout
            << std::left
            << std::setw(14)
            << r
            << std::setw(18)
            << nuclear
            << std::setw(18)
            << result.hartree[index]
            << std::setw(18)
            << result.exchangeCorrelation[index]
            << std::setw(18)
            << result.effectivePotential[index]
            << "\n";
    }
}

// ================================================================
// Density diagnostics
// ================================================================

void printDensityDiagnostics(
    const SCFResult& result,
    double dr)
{
    std::cout
        << "\n====================================================\n"
        << "DENSITY DIAGNOSTICS\n"
        << "===================\n\n";

    const double electronNumber =
        calculateElectronNumber(
            result.density,
            dr);

    std::cout
        << std::setprecision(15);

    std::cout
        << "Integrated electron number: "
        << electronNumber
        << "\n";

    std::cout
        << "Target electron number:     2.0\n";

    std::cout
        << "Normalization error:        "
        << std::abs(
               electronNumber - 2.0)
        << "\n";
}

// ================================================================
// Energy diagnostics
// ================================================================

void printEnergyDiagnostics(
    const SCFResult& result,
    double dr,
    double Z,
    double electronCount)
{
    const double kinetic =
        calculateKineticEnergy(
            result.orbital,
            dr,
            electronCount);

    const double external =
        calculateExternalEnergy(
            result.orbital,
            dr,
            Z,
            electronCount);

    const double hartreeEnergy =
        calculateHartreeEnergy(
            result.density,
            result.hartree,
            dr);

    const PBEEnergy xc =
        calculatePBEExchangeCorrelationEnergy(
            result.density,
            dr);

    const double total =
        kinetic +
        external +
        hartreeEnergy +
        xc.total;

    std::cout
        << "\n====================================================\n"
        << "ENERGY COMPONENTS\n"
        << "=================\n\n";

    std::cout
        << std::setprecision(15);

    std::cout
        << "Kinetic Ts:             "
        << kinetic
        << " Ha\n";

    std::cout
        << "External Eext:          "
        << external
        << " Ha\n";

    std::cout
        << "Hartree EH:             "
        << hartreeEnergy
        << " Ha\n";

    std::cout
        << "PBE exchange Ex:        "
        << xc.exchange
        << " Ha\n";

    std::cout
        << "PBE correlation Ec:     "
        << xc.correlation
        << " Ha\n";

    std::cout
        << "PBE XC Exc:             "
        << xc.total
        << " Ha\n";

    std::cout
        << "---------------------------------------------\n";

    std::cout
        << "Total energy:            "
        << total
        << " Ha\n";

    std::cout
        << "Stored SCF energy:       "
        << result.totalEnergy
        << " Ha\n";

    std::cout
        << "Difference:              "
        << std::abs(
               total -
               result.totalEnergy)
        << " Ha\n";

    std::cout
        << "\nReference He energy:     approximately -2.9037 Ha\n";
}

// ================================================================
// Orbital diagnostics
// ================================================================

void printOrbitalDiagnostics(
    const SCFResult& result,
    double dr,
    double Z)
{
    const int n =
        static_cast<int>(
            result.orbital.size());

    double maximumDensity =
        0.0;

    int maximumIndex =
        0;

    for (int i = 0;
         i < n;
         ++i)
    {
        if (result.density[i] >
            maximumDensity)
        {
            maximumDensity =
                result.density[i];

            maximumIndex =
                i;
        }
    }

    const double maximumRadius =
        radiusAt(
            maximumIndex,
            dr);

    double effectivePotentialMinimum =
        std::numeric_limits<double>::infinity();

    int effectiveMinimumIndex =
        0;

    for (int i = 0;
         i < n;
         ++i)
    {
        if (result.effectivePotential[i] <
            effectivePotentialMinimum)
        {
            effectivePotentialMinimum =
                result.effectivePotential[i];

            effectiveMinimumIndex =
                i;
        }
    }

    const double effectiveMinimumRadius =
        radiusAt(
            effectiveMinimumIndex,
            dr);

    const double effectiveChargeAtSmallR =
        -effectivePotentialMinimum *
        effectiveMinimumRadius;

    std::cout
        << "\n====================================================\n"
        << "ORBITAL / EFFECTIVE POTENTIAL DIAGNOSTICS\n"
        << "==========================================\n\n";

    std::cout
        << std::setprecision(15);

    std::cout
        << "Orbital energy:                 "
        << result.orbitalEnergy
        << " Ha\n";

    std::cout
        << "Maximum density:                "
        << maximumDensity
        << "\n";

    std::cout
        << "Radius of maximum density:     "
        << maximumRadius
        << " Bohr\n";

    std::cout
        << "Minimum effective potential:    "
        << effectivePotentialMinimum
        << " Ha\n";

    std::cout
        << "Radius of minimum V_eff:        "
        << effectiveMinimumRadius
        << " Bohr\n";

    std::cout
        << "Diagnostic -r V_eff:            "
        << effectiveChargeAtSmallR
        << "\n";
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
            << "             MINIMAL KOHN-SHAM DFT\n"
            << "======================================================\n\n";

        // --------------------------------------------------------
        // Hydrogen validation
        // --------------------------------------------------------

        const double hydrogenEnergy =
            runHydrogen();

        // --------------------------------------------------------
        // Helium self-consistent Kohn-Sham calculation
        // --------------------------------------------------------

        const SCFResult helium =
            runHelium();

        // --------------------------------------------------------
        // Grid parameters
        // --------------------------------------------------------

        const double Rmax =
            30.0;

        const int N =
            2000;

        const double dr =
            Rmax /
            static_cast<double>(
                N + 1);

        // --------------------------------------------------------
        // Diagnostics
        // --------------------------------------------------------

        printDensityDiagnostics(
            helium,
            dr);

        printEnergyDiagnostics(
            helium,
            dr,
            2.0,
            2.0);

        printOrbitalDiagnostics(
            helium,
            dr,
            2.0);

        printPotentialDiagnostics(
            helium,
            dr,
            2.0);

        // ========================================================
        // Final summary
        // ========================================================

        std::cout
            << "\n====================================================\n"
            << "FINAL SUMMARY\n"
            << "=============\n\n";

        std::cout
            << std::setprecision(15);

        // --------------------------------------------------------
        // Hydrogen
        // --------------------------------------------------------

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
            << " Ha\n";

        std::cout
            << "Absolute error:       "
            << std::abs(
                   hydrogenEnergy + 0.5)
            << " Ha\n\n";

        // --------------------------------------------------------
        // Helium
        // --------------------------------------------------------

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
            << "SCF converged:        "
            << (helium.converged
                ? "YES"
                : "NO")
            << "\n";

        std::cout
            << "Final density change: "
            << helium.densityError
            << "\n";

        std::cout
            << "Final energy change:  "
            << helium.energyChange
            << " Ha\n";

        std::cout
            << "\nExchange:             PBE-GGA\n"
            << "Correlation:          PBE-GGA\n";

        std::cout
            << "Reference He energy:  approximately -2.9037 Ha\n\n";

        std::cout
            << "====================================================\n";
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