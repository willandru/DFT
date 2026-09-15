#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

// ================================================================
// Constants
// ================================================================

constexpr double PI =
    3.1415926535897932384626433832795;

// ================================================================
// PW92 LDA correlation parameters
//
// Unpolarized electron gas.
//
// Perdew-Wang 1992 parameterization.
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
// Radial coordinate
//
// Physical grid:
//
//     r_i = (i + 1) dr
//
// Boundary conditions:
//
//     u(0)    = 0
//     u(Rmax) = 0
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
// Normalize radial orbital
//
//     integral |u(r)|² dr = 1
// ================================================================

void normalize(
    std::vector<double>& u,
    double dr)
{
    if (u.empty())
    {
        throw std::runtime_error(
            "Cannot normalize empty orbital.");
    }

    if (dr <= 0.0)
    {
        throw std::runtime_error(
            "Grid spacing must be positive.");
    }

    double norm = 0.0;

    for (double value : u)
        norm += value * value * dr;

    norm = std::sqrt(norm);

    if (!std::isfinite(norm) || norm <= 0.0)
    {
        throw std::runtime_error(
            "Cannot normalize zero or invalid orbital.");
    }

    for (double& value : u)
        value /= norm;
}

// ================================================================
// Tridiagonal solver
//
// Thomas algorithm.
// ================================================================

std::vector<double> solveTridiagonal(
    const std::vector<double>& lower,
    const std::vector<double>& diagonal,
    const std::vector<double>& upper,
    const std::vector<double>& rhs)
{
    const int N =
        static_cast<int>(diagonal.size());

    if (N == 0)
    {
        throw std::runtime_error(
            "Cannot solve empty tridiagonal system.");
    }

    if (
        static_cast<int>(lower.size()) != N ||
        static_cast<int>(upper.size()) != N ||
        static_cast<int>(rhs.size()) != N)
    {
        throw std::runtime_error(
            "Invalid tridiagonal matrix dimensions.");
    }

    std::vector<double> a = lower;
    std::vector<double> b = diagonal;
    std::vector<double> c = upper;
    std::vector<double> d = rhs;

    constexpr double MIN_PIVOT =
        1.0e-14;

    // ------------------------------------------------------------
    // Forward elimination
    // ------------------------------------------------------------

    for (int i = 1;
         i < N;
         ++i)
    {
        if (!std::isfinite(b[i - 1]))
        {
            throw std::runtime_error(
                "Non-finite pivot encountered.");
        }

        if (std::abs(b[i - 1]) < MIN_PIVOT)
        {
            throw std::runtime_error(
                "Singular or ill-conditioned tridiagonal system.");
        }

        const double multiplier =
            a[i] / b[i - 1];

        b[i] -=
            multiplier * c[i - 1];

        d[i] -=
            multiplier * d[i - 1];
    }

    if (!std::isfinite(b[N - 1]))
    {
        throw std::runtime_error(
            "Non-finite final pivot encountered.");
    }

    if (std::abs(b[N - 1]) < MIN_PIVOT)
    {
        throw std::runtime_error(
            "Singular or ill-conditioned tridiagonal system.");
    }

    // ------------------------------------------------------------
    // Back substitution
    // ------------------------------------------------------------

    std::vector<double> x(
        N,
        0.0);

    x[N - 1] =
        d[N - 1] / b[N - 1];

    for (int i = N - 2;
         i >= 0;
         --i)
    {
        if (std::abs(b[i]) < MIN_PIVOT)
        {
            throw std::runtime_error(
                "Singular or ill-conditioned tridiagonal system.");
        }

        x[i] =
            (
                d[i]
                - c[i] * x[i + 1]
            )
            / b[i];
    }

    return x;
}

// ================================================================
// Apply radial Hamiltonian
//
//     H = -1/2 d²/dr² + V(r)
//
// for l = 0.
// ================================================================

void applyHamiltonian(
    const std::vector<double>& u,
    const std::vector<double>& potential,
    std::vector<double>& Hu,
    double dr)
{
    const int N =
        static_cast<int>(u.size());

    if (
        static_cast<int>(potential.size()) != N ||
        static_cast<int>(Hu.size()) != N)
    {
        throw std::runtime_error(
            "Hamiltonian vector dimensions do not match.");
    }

    if (N == 0 || dr <= 0.0)
    {
        throw std::runtime_error(
            "Invalid radial grid.");
    }

    const double inverseDr2 =
        1.0 / (dr * dr);

    for (int i = 0;
         i < N;
         ++i)
    {
        const double left =
            (i > 0)
            ? u[i - 1]
            : 0.0;

        const double right =
            (i + 1 < N)
            ? u[i + 1]
            : 0.0;

        const double secondDerivative =
            (
                right
                - 2.0 * u[i]
                + left
            )
            * inverseDr2;

        Hu[i] =
            -0.5 * secondDerivative
            + potential[i] * u[i];
    }
}

// ================================================================
// Solve lowest eigenstate
//
// 1. Tridiagonal Hamiltonian.
// 2. Sturm count.
// 3. Bisection for lowest eigenvalue.
// 4. Inverse iteration for eigenvector.
// ================================================================

OrbitalResult solveGroundState(
    const std::vector<double>& potential,
    double dr)
{
    const int N =
        static_cast<int>(potential.size());

    if (N < 2)
    {
        throw std::runtime_error(
            "Ground-state solver requires at least two grid points.");
    }

    if (dr <= 0.0)
    {
        throw std::runtime_error(
            "Grid spacing must be positive.");
    }

    // ------------------------------------------------------------
    // Construct tridiagonal Hamiltonian
    // ------------------------------------------------------------

    const double diagonalKinetic =
        1.0 / (dr * dr);

    const double offDiagonal =
        -0.5 / (dr * dr);

    std::vector<double> diagonal(
        N,
        0.0);

    for (int i = 0;
         i < N;
         ++i)
    {
        diagonal[i] =
            diagonalKinetic
            + potential[i];
    }

    // ------------------------------------------------------------
    // Sturm count
    //
    // Returns number of eigenvalues strictly below energy.
    // ------------------------------------------------------------

    auto countBelow =
        [&](double energy)
    {
        int count = 0;

        constexpr double MIN_PIVOT =
            1.0e-14;

        double q =
            diagonal[0] - energy;

        if (q < 0.0)
            ++count;

        if (std::abs(q) < MIN_PIVOT)
        {
            q =
                q < 0.0
                ? -MIN_PIVOT
                : MIN_PIVOT;
        }

        const double off2 =
            offDiagonal * offDiagonal;

        for (int i = 1;
             i < N;
             ++i)
        {
            q =
                diagonal[i]
                - energy
                - off2 / q;

            if (q < 0.0)
                ++count;

            if (std::abs(q) < MIN_PIVOT)
            {
                q =
                    q < 0.0
                    ? -MIN_PIVOT
                    : MIN_PIVOT;
            }
        }

        return count;
    };

    // ------------------------------------------------------------
    // Energy bracket
    // ------------------------------------------------------------

    double lowerEnergy = -100.0;
    double upperEnergy = 10.0;

    while (countBelow(lowerEnergy) >= 1)
    {
        lowerEnergy *= 2.0;

        if (!std::isfinite(lowerEnergy))
        {
            throw std::runtime_error(
                "Unable to bracket lowest eigenvalue.");
        }
    }

    while (countBelow(upperEnergy) < 1)
    {
        upperEnergy *= 2.0;

        if (!std::isfinite(upperEnergy))
        {
            throw std::runtime_error(
                "Unable to bracket lowest eigenvalue.");
        }
    }

    // ------------------------------------------------------------
    // Bisection
    // ------------------------------------------------------------

    constexpr int MAX_BISECTION =
        120;

    constexpr double ENERGY_TOLERANCE =
        1.0e-12;

    double energy = 0.0;

    for (int iteration = 0;
         iteration < MAX_BISECTION;
         ++iteration)
    {
        const double middle =
            0.5 *
            (
                lowerEnergy
                + upperEnergy
            );

        if (countBelow(middle) < 1)
            lowerEnergy = middle;
        else
            upperEnergy = middle;

        energy =
            0.5 *
            (
                lowerEnergy
                + upperEnergy
            );

        if (
            upperEnergy
            - lowerEnergy
            < ENERGY_TOLERANCE
        )
        {
            break;
        }
    }

    // ------------------------------------------------------------
    // Initial orbital
    //
    // Hydrogen-like 1s radial form.
    // ------------------------------------------------------------

    std::vector<double> u(
        N,
        0.0);

    for (int i = 0;
         i < N;
         ++i)
    {
        const double r =
            radiusAt(i, dr);

        u[i] =
            r
            * std::exp(-r);
    }

    normalize(
        u,
        dr);

    // ------------------------------------------------------------
    // Inverse iteration
    // ------------------------------------------------------------

    const double sigma =
        energy - 0.05;

    constexpr int MAX_INVERSE_ITERATIONS =
        40;

    constexpr double ORBITAL_TOLERANCE =
        1.0e-12;

    for (int iteration = 0;
         iteration < MAX_INVERSE_ITERATIONS;
         ++iteration)
    {
        std::vector<double> lower(
            N,
            offDiagonal);

        std::vector<double> upper(
            N,
            offDiagonal);

        std::vector<double> shiftedDiagonal(
            N,
            0.0);

        lower[0] = 0.0;
        upper[N - 1] = 0.0;

        for (int i = 0;
             i < N;
             ++i)
        {
            shiftedDiagonal[i] =
                diagonal[i]
                - sigma;
        }

        const std::vector<double> next =
            solveTridiagonal(
                lower,
                shiftedDiagonal,
                upper,
                u);

        std::vector<double> normalized =
            next;

        normalize(
            normalized,
            dr);

        // --------------------------------------------------------
        // Fix global sign
        // --------------------------------------------------------

        double overlap = 0.0;

        for (int i = 0;
             i < N;
             ++i)
        {
            overlap +=
                u[i]
                * normalized[i]
                * dr;
        }

        if (overlap < 0.0)
        {
            for (double& value : normalized)
                value *= -1.0;
        }

        // --------------------------------------------------------
        // Orbital convergence
        // --------------------------------------------------------

        double difference = 0.0;

        for (int i = 0;
             i < N;
             ++i)
        {
            const double delta =
                normalized[i]
                - u[i];

            difference +=
                delta
                * delta
                * dr;
        }

        u =
            std::move(normalized);

        if (
            std::sqrt(difference)
            < ORBITAL_TOLERANCE
        )
        {
            break;
        }
    }

    // ------------------------------------------------------------
    // Rayleigh quotient
    // ------------------------------------------------------------

    std::vector<double> Hu(
        N,
        0.0);

    applyHamiltonian(
        u,
        potential,
        Hu,
        dr);

    double eigenvalue = 0.0;

    for (int i = 0;
         i < N;
         ++i)
    {
        eigenvalue +=
            u[i]
            * Hu[i]
            * dr;
    }

    if (!std::isfinite(eigenvalue))
    {
        throw std::runtime_error(
            "Ground-state eigenvalue is not finite.");
    }

    return
    {
        std::move(u),
        eigenvalue
    };
}

// ================================================================
// Hartree potential
//
// rho(r) = Ne |u(r)|² / (4 pi r²)
//
// V_H(r) = Ne [
//
//     1/r integral_0^r |u(r')|² dr'
//
//     +
//
//     integral_r^Rmax |u(r')|²/r' dr'
//
// ]
// ================================================================

std::vector<double> calculateHartreePotential(
    const std::vector<double>& u,
    double dr,
    int electrons)
{
    const int N =
        static_cast<int>(u.size());

    if (N == 0)
    {
        throw std::runtime_error(
            "Cannot calculate Hartree potential for empty orbital.");
    }

    if (dr <= 0.0)
    {
        throw std::runtime_error(
            "Grid spacing must be positive.");
    }

    if (electrons <= 0)
    {
        throw std::runtime_error(
            "Electron count must be positive.");
    }

    std::vector<double> cumulative(
        N,
        0.0);

    std::vector<double> outer(
        N,
        0.0);

    std::vector<double> VH(
        N,
        0.0);

    // ------------------------------------------------------------
    // Integral from 0 to r
    // ------------------------------------------------------------

    cumulative[0] =
        0.5
        * dr
        * u[0]
        * u[0];

    for (int i = 1;
         i < N;
         ++i)
    {
        cumulative[i] =
            cumulative[i - 1]
            +
            0.5
            * dr
            *
            (
                u[i - 1] * u[i - 1]
                +
                u[i] * u[i]
            );
    }

    // ------------------------------------------------------------
    // Integral from r to Rmax
    // ------------------------------------------------------------

    outer[N - 1] =
        0.5
        * dr
        *
        (
            u[N - 1]
            * u[N - 1]
            /
            radiusAt(
                N - 1,
                dr)
        );

    for (int i = N - 2;
         i >= 0;
         --i)
    {
        const double r1 =
            radiusAt(i, dr);

        const double r2 =
            radiusAt(i + 1, dr);

        const double f1 =
            u[i]
            * u[i]
            / r1;

        const double f2 =
            u[i + 1]
            * u[i + 1]
            / r2;

        outer[i] =
            outer[i + 1]
            +
            0.5
            * dr
            * (f1 + f2);
    }

    // ------------------------------------------------------------
    // Hartree potential
    // ------------------------------------------------------------

    for (int i = 0;
         i < N;
         ++i)
    {
        const double r =
            radiusAt(i, dr);

        VH[i] =
            static_cast<double>(electrons)
            *
            (
                cumulative[i] / r
                +
                outer[i]
            );
    }

    return VH;
}

// ================================================================
// Electron density
//
// rho(r) = Ne |u(r)|² / (4 pi r²)
// ================================================================

double calculateDensity(
    double u,
    double r,
    int electrons)
{
    if (r <= 0.0)
        throw std::runtime_error(
            "Radial coordinate must be positive.");

    return
        static_cast<double>(electrons)
        * u
        * u
        /
        (
            4.0
            * PI
            * r
            * r
        );
}

// ================================================================
// LDA Dirac exchange energy per electron
//
// epsilon_x(rho)
//     = -3/4 (3/pi)^(1/3) rho^(1/3)
// ================================================================

double exchangeEnergyPerParticle(
    double density)
{
    if (density <= 0.0)
        return 0.0;

    return
        -0.75
        * std::cbrt(3.0 / PI)
        * std::cbrt(density);
}

// ================================================================
// LDA Dirac exchange potential
//
// v_x(rho)
//     = -(3/pi)^(1/3) rho^(1/3)
// ================================================================

double exchangePotential(
    double density)
{
    if (density <= 0.0)
        return 0.0;

    return
        -std::cbrt(3.0 / PI)
        * std::cbrt(density);
}

// ================================================================
// PW92 correlation energy per particle
//
// r_s = (3 / (4 pi rho))^(1/3)
//
// epsilon_c(r_s) =
//
// -2 A (1 + alpha1 r_s)
//
// ln(
//     1 +
//     1 /
//     [2 A (beta1 sqrt(r_s)
//          + beta2 r_s
//          + beta3 r_s^(3/2)
//          + beta4 r_s²)]
// )
// ================================================================

double correlationEnergyPerParticle(
    double density)
{
    if (density <= 0.0)
        return 0.0;

    const double rs =
        std::cbrt(
            3.0
            /
            (
                4.0
                * PI
                * density
            )
        );

    const double sqrtRs =
        std::sqrt(rs);

    const double rs32 =
        rs * sqrtRs;

    const double rs2 =
        rs * rs;

    const double denominator =
        2.0
        * PW92_A
        *
        (
            PW92_BETA1 * sqrtRs
            +
            PW92_BETA2 * rs
            +
            PW92_BETA3 * rs32
            +
            PW92_BETA4 * rs2
        );

    const double logarithm =
        std::log(
            1.0
            + 1.0 / denominator
        );

    return
        -2.0
        * PW92_A
        *
        (
            1.0
            + PW92_ALPHA1 * rs
        )
        * logarithm;
}

// ================================================================
// PW92 correlation potential
//
// v_c(rho)
//     = epsilon_c
//       - (r_s / 3) d epsilon_c / d r_s
//
// The derivative is analytical.
// ================================================================

double correlationPotential(
    double density)
{
    if (density <= 0.0)
        return 0.0;

    const double rs =
        std::cbrt(
            3.0
            /
            (
                4.0
                * PI
                * density
            )
        );

    const double sqrtRs =
        std::sqrt(rs);

    const double rs32 =
        rs * sqrtRs;

    const double rs2 =
        rs * rs;

    const double denominatorCore =
        PW92_BETA1 * sqrtRs
        +
        PW92_BETA2 * rs
        +
        PW92_BETA3 * rs32
        +
        PW92_BETA4 * rs2;

    const double denominator =
        2.0
        * PW92_A
        * denominatorCore;

    const double logarithm =
        std::log(
            1.0
            + 1.0 / denominator
        );

    const double epsilonC =
        -2.0
        * PW92_A
        *
        (
            1.0
            + PW92_ALPHA1 * rs
        )
        * logarithm;

    // ------------------------------------------------------------
    // d[denominatorCore] / dr_s
    // ------------------------------------------------------------

    const double derivativeCore =
        PW92_BETA1
            / (2.0 * sqrtRs)
        +
        PW92_BETA2
        +
        1.5
        * PW92_BETA3
        * sqrtRs
        +
        2.0
        * PW92_BETA4
        * rs;

    // ------------------------------------------------------------
    // d[ln(1 + 1/denominator)] / dr_s
    // ------------------------------------------------------------

    const double derivativeLogarithm =
        -derivativeCore
        /
        (
            denominatorCore
            * (1.0 + denominator)
        );

    // ------------------------------------------------------------
    // d epsilon_c / dr_s
    // ------------------------------------------------------------

    const double derivativeEpsilon =
        -2.0
        * PW92_A
        *
        (
            PW92_ALPHA1 * logarithm
            +
            (
                1.0
                + PW92_ALPHA1 * rs
            )
            * derivativeLogarithm
        );

    // ------------------------------------------------------------
    // v_c = epsilon_c - r_s/3 * d epsilon_c/dr_s
    // ------------------------------------------------------------

    return
        epsilonC
        -
        (rs / 3.0)
        * derivativeEpsilon;
}

// ================================================================
// LDA exchange-correlation potential
//
//     V_xc = V_x + V_c
// ================================================================

std::vector<double> calculateExchangeCorrelationPotential(
    const std::vector<double>& u,
    double dr,
    int electrons)
{
    const int N =
        static_cast<int>(u.size());

    std::vector<double> Vxc(
        N,
        0.0);

    for (int i = 0;
         i < N;
         ++i)
    {
        const double r =
            radiusAt(i, dr);

        const double density =
            calculateDensity(
                u[i],
                r,
                electrons);

        const double Vx =
            exchangePotential(
                density);

        const double Vc =
            correlationPotential(
                density);

        Vxc[i] =
            Vx
            + Vc;
    }

    return Vxc;
}

// ================================================================
// Total DFT energy
//
//     E = T_s
//       + E_ext
//       + E_H
//       + E_x
//       + E_c
//
// For closed-shell He:
//
//     T_s = 2 T_orbital
// ================================================================

double calculateTotalEnergy(
    const std::vector<double>& u,
    const std::vector<double>& Vext,
    const std::vector<double>& VH,
    double dr,
    int electrons)
{
    const int N =
        static_cast<int>(u.size());

    if (
        static_cast<int>(Vext.size()) != N ||
        static_cast<int>(VH.size()) != N)
    {
        throw std::runtime_error(
            "Energy vectors have inconsistent dimensions.");
    }

    double orbitalKineticEnergy = 0.0;
    double externalEnergy = 0.0;
    double hartreeEnergy = 0.0;
    double exchangeEnergy = 0.0;
    double correlationEnergy = 0.0;

    // ------------------------------------------------------------
    // Kinetic energy of one spatial orbital
    //
    // Piecewise-linear derivative over the full radial interval:
    //
    //     [0, dr],
    //     [dr, 2dr],
    //     ...
    //     [Ndr, Rmax]
    //
    // with u(0)=u(Rmax)=0.
    // ------------------------------------------------------------

    for (int interval = 0;
         interval <= N;
         ++interval)
    {
        double derivative = 0.0;

        if (interval == 0)
        {
            derivative =
                u[0] / dr;
        }
        else if (interval == N)
        {
            derivative =
                -u[N - 1] / dr;
        }
        else
        {
            derivative =
                (
                    u[interval]
                    - u[interval - 1]
                )
                / dr;
        }

        orbitalKineticEnergy +=
            0.5
            * derivative
            * derivative
            * dr;
    }

    const double kineticEnergy =
        static_cast<double>(electrons)
        * orbitalKineticEnergy;

    // ------------------------------------------------------------
    // Other energy terms
    // ------------------------------------------------------------

    for (int i = 0;
         i < N;
         ++i)
    {
        const double r =
            radiusAt(i, dr);

        const double u2 =
            u[i] * u[i];

        // --------------------------------------------------------
        // Nuclear attraction
        //
        // E_ext = integral rho V_ext d^3r
        //       = Ne integral |u|² V_ext dr
        // --------------------------------------------------------

        externalEnergy +=
            static_cast<double>(electrons)
            * u2
            * Vext[i]
            * dr;

        // --------------------------------------------------------
        // Hartree energy
        //
        // E_H = 1/2 integral rho V_H d^3r
        // --------------------------------------------------------

        hartreeEnergy +=
            0.5
            * static_cast<double>(electrons)
            * u2
            * VH[i]
            * dr;

        // --------------------------------------------------------
        // Electron density
        // --------------------------------------------------------

        const double density =
            calculateDensity(
                u[i],
                r,
                electrons);

        // --------------------------------------------------------
        // Exchange
        // --------------------------------------------------------

        const double epsilonX =
            exchangeEnergyPerParticle(
                density);

        exchangeEnergy +=
            static_cast<double>(electrons)
            * u2
            * epsilonX
            * dr;

        // --------------------------------------------------------
        // Correlation
        // --------------------------------------------------------

        const double epsilonC =
            correlationEnergyPerParticle(
                density);

        correlationEnergy +=
            static_cast<double>(electrons)
            * u2
            * epsilonC
            * dr;
    }

    return
        kineticEnergy
        + externalEnergy
        + hartreeEnergy
        + exchangeEnergy
        + correlationEnergy;
}

// ================================================================
// HYDROGEN
//
// One electron:
//
//     H = -1/2 ∇² - 1/r
//
// Electron-electron terms are disabled for this validation.
// ================================================================

void runHydrogen()
{
    constexpr int N = 2000;
    constexpr double R_MAX = 30.0;
    constexpr double Z = 1.0;

    const double dr =
        R_MAX
        / static_cast<double>(N + 1);

    std::vector<double> potential(
        N,
        0.0);

    for (int i = 0;
         i < N;
         ++i)
    {
        const double r =
            radiusAt(i, dr);

        potential[i] =
            -Z / r;
    }

    const OrbitalResult result =
        solveGroundState(
            potential,
            dr);

    constexpr double exactEnergy =
        -0.5;

    const double error =
        std::abs(
            result.eigenvalue
            - exactEnergy);

    std::cout
        << "\n====================================================\n"
        << "HYDROGEN\n"
        << "====================================================\n\n";

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
        << error
        << " Ha\n";

    if (error < 1.0e-4)
        std::cout << "\nPASS\n";
    else
        std::cout << "\nWARNING\n";
}

// ================================================================
// HELIUM
//
// Restricted Kohn-Sham:
//
//     He = 1s²
//
// Hartree: ON
// Exchange: LDA Dirac
// Correlation: LDA PW92
// ================================================================

SCFResult runHelium()
{
    constexpr int N = 2000;
    constexpr double R_MAX = 30.0;

    constexpr double Z = 2.0;
    constexpr int electrons = 2;

    constexpr int MAX_SCF_ITERATIONS =
        100;

    constexpr double DENSITY_MIXING =
        0.30;

    constexpr double SCF_TOLERANCE =
        1.0e-9;

    const double dr =
        R_MAX
        / static_cast<double>(N + 1);

    // ------------------------------------------------------------
    // Nuclear potential
    // ------------------------------------------------------------

    std::vector<double> Vext(
        N,
        0.0);

    for (int i = 0;
         i < N;
         ++i)
    {
        const double r =
            radiusAt(i, dr);

        Vext[i] =
            -Z / r;
    }

    // ------------------------------------------------------------
    // Initial hydrogenic 1s orbital
    // ------------------------------------------------------------

    std::vector<double> u(
        N,
        0.0);

    for (int i = 0;
         i < N;
         ++i)
    {
        const double r =
            radiusAt(i, dr);

        u[i] =
            r
            * std::exp(-Z * r);
    }

    normalize(
        u,
        dr);

    double previousEnergy = 0.0;

    double densityError = 0.0;
    double orbitalEnergy = 0.0;
    double totalEnergy = 0.0;

    int completedIterations = 0;

    std::cout
        << "\n====================================================\n"
        << "HELIUM\n"
        << "====================================================\n\n";

    std::cout
        << "System\n"
        << "------\n";

    std::cout
        << "Element:             He\n";

    std::cout
        << "Nuclear charge:      Z = "
        << Z
        << "\n";

    std::cout
        << "Electrons:           "
        << electrons
        << "\n";

    std::cout
        << "Configuration:       1s^2\n\n";

    std::cout
        << "Kohn-Sham model\n"
        << "---------------\n";

    std::cout
        << "Hartree:             ON\n";

    std::cout
        << "Exchange:            LDA Dirac\n";

    std::cout
        << "Correlation:         LDA PW92\n";

    std::cout
        << "SCF mixing:          "
        << DENSITY_MIXING
        << "\n\n";

    std::cout
        << "SCF iterations\n"
        << "--------------\n";

    for (int iteration = 1;
         iteration <= MAX_SCF_ITERATIONS;
         ++iteration)
    {
        completedIterations =
            iteration;

        // --------------------------------------------------------
        // Hartree potential
        // --------------------------------------------------------

        const std::vector<double> VH =
            calculateHartreePotential(
                u,
                dr,
                electrons);

        // --------------------------------------------------------
        // Exchange-correlation potential
        //
        //     V_xc = V_x + V_c
        // --------------------------------------------------------

        const std::vector<double> Vxc =
            calculateExchangeCorrelationPotential(
                u,
                dr,
                electrons);

        // --------------------------------------------------------
        // Kohn-Sham potential
        //
        //     V_KS = V_ext + V_H + V_xc
        // --------------------------------------------------------

        std::vector<double> VKs(
            N,
            0.0);

        for (int i = 0;
             i < N;
             ++i)
        {
            VKs[i] =
                Vext[i]
                + VH[i]
                + Vxc[i];
        }

        // --------------------------------------------------------
        // Solve Kohn-Sham equation
        // --------------------------------------------------------

        const OrbitalResult orbital =
            solveGroundState(
                VKs,
                dr);

        orbitalEnergy =
            orbital.eigenvalue;

        // --------------------------------------------------------
        // Calculate raw density residual
        //
        // This is deliberately calculated BEFORE mixing:
        //
        //     dRho = integral |rho_new - rho_old| dr
        //
        // The previous implementation measured the change after
        // mixing, which is smaller by approximately the mixing
        // factor. The raw residual is the more direct SCF
        // convergence criterion.
        // --------------------------------------------------------

        densityError = 0.0;

        for (int i = 0;
             i < N;
             ++i)
        {
            const double oldDensity =
                u[i] * u[i];

            const double newDensity =
                orbital.u[i]
                * orbital.u[i];

            densityError +=
                std::abs(
                    newDensity
                    - oldDensity)
                * dr;
        }

        // --------------------------------------------------------
        // Mix densities
        // --------------------------------------------------------

        std::vector<double> mixedU(
            N,
            0.0);

        for (int i = 0;
             i < N;
             ++i)
        {
            const double oldDensity =
                u[i] * u[i];

            const double newDensity =
                orbital.u[i]
                * orbital.u[i];

            const double mixedDensity =
                (
                    1.0
                    - DENSITY_MIXING
                )
                * oldDensity
                +
                DENSITY_MIXING
                * newDensity;

            mixedU[i] =
                std::sqrt(
                    std::max(
                        0.0,
                        mixedDensity));
        }

        // --------------------------------------------------------
        // Normalize mixed orbital
        // --------------------------------------------------------

        normalize(
            mixedU,
            dr);

        u =
            std::move(mixedU);

        // --------------------------------------------------------
        // Recalculate Hartree from updated density
        // --------------------------------------------------------

        const std::vector<double> finalVH =
            calculateHartreePotential(
                u,
                dr,
                electrons);

        // --------------------------------------------------------
        // Total DFT energy
        // --------------------------------------------------------

        totalEnergy =
            calculateTotalEnergy(
                u,
                Vext,
                finalVH,
                dr,
                electrons);

        // --------------------------------------------------------
        // Energy change
        // --------------------------------------------------------

        double energyChange = 0.0;

        if (iteration > 1)
        {
            energyChange =
                std::abs(
                    totalEnergy
                    - previousEnergy);
        }
        else
        {
            energyChange =
                std::abs(totalEnergy);
        }

        // --------------------------------------------------------
        // Output
        // --------------------------------------------------------

        if (
            iteration == 1
            ||
            iteration % 5 == 0
        )
        {
            std::cout
                << "Iteration "
                << std::setw(3)
                << iteration

                << "   orbital = "
                << std::setw(14)
                << orbitalEnergy

                << " Ha   E = "
                << std::setw(14)
                << totalEnergy

                << " Ha   dE = "
                << energyChange

                << "   dRho = "
                << densityError
                << "\n";
        }

        // --------------------------------------------------------
        // SCF convergence
        //
        // Both the raw density residual and energy change must
        // satisfy the tolerance.
        // --------------------------------------------------------

        if (
            iteration > 1
            &&
            densityError
                < SCF_TOLERANCE
            &&
            energyChange
                < SCF_TOLERANCE
        )
        {
            std::cout
                << "\nSCF converged after "
                << iteration
                << " iterations.\n";

            return
            {
                totalEnergy,
                orbitalEnergy,
                densityError,
                iteration
            };
        }

        previousEnergy =
            totalEnergy;
    }

    std::cout
        << "\nSCF reached maximum iterations.\n";

    return
    {
        totalEnergy,
        orbitalEnergy,
        densityError,
        completedIterations
    };
}

// ================================================================
// MAIN
// ================================================================

int main()
{
    std::cout
        << std::setprecision(12);

    std::cout
        << "====================================================\n"
        << "          MINIMAL KOHN-SHAM DFT\n"
        << "====================================================\n";

    // ------------------------------------------------------------
    // Hydrogen validation
    // ------------------------------------------------------------

    runHydrogen();

    // ------------------------------------------------------------
    // Helium validation
    // ------------------------------------------------------------

    const SCFResult helium =
        runHelium();

    // ------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------

    std::cout
        << "\n====================================================\n"
        << "SUMMARY\n"
        << "====================================================\n\n";

    std::cout
        << "Hydrogen\n"
        << "--------\n";

    std::cout
        << "Expected ground state: "
        << "-0.5 Ha\n\n";

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
        << "DFT total energy:     "
        << helium.totalEnergy
        << " Ha\n";

    std::cout
        << "SCF iterations:       "
        << helium.iterations
        << "\n";

    std::cout
        << "Final density change: "
        << helium.densityError
        << "\n";

    std::cout
        << "\nFunctional:\n"
        << "LDA Dirac exchange + PW92 correlation\n";

    std::cout
        << "\nReference:\n"
        << "Exact non-relativistic He energy "
        << "approximately -2.9037 Ha.\n";

    std::cout
        << "\n====================================================\n";

    return 0;
}