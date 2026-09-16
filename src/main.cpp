#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using std::vector;

constexpr double PI = 3.1415926535897932384626433832795;
constexpr int N = 2000;
constexpr double RMAX = 30.0;
constexpr double MIXING = 0.25;
constexpr int MAX_SCF_ITER = 250;
constexpr double DENSITY_TOL = 1.0e-9;
constexpr double ENERGY_TOL = 1.0e-11;
constexpr double KS_RESIDUAL_TOL = 1.0e-7;
constexpr double EPS = 1.0e-14;
constexpr double RHO_FLOOR = 1.0e-20;
constexpr double H_EXACT = -0.5;

struct AtomicOrbital
{
    std::string label;
    int n = 0;
    int l = 0;
    double occupation = 0.0;
    int stateIndex = 0;
    vector<double> u;
    double eigenvalue = 0.0;
};

struct AtomResult
{
    std::string symbol;
    int Z = 0;
    int electrons = 0;
    double totalEnergy = 0.0;
    double kineticEnergy = 0.0;
    double externalEnergy = 0.0;
    double hartreeEnergy = 0.0;
    double xcEnergy = 0.0;
    double electronNumber = 0.0;
    double densityResidual = 0.0;
    double energyResidual = 0.0;
    double ksResidual = 0.0;
    int iterations = 0;
    bool converged = false;
    vector<AtomicOrbital> orbitals;
};

struct TridiagonalMatrix
{
    vector<double> lower;
    vector<double> diagonal;
    vector<double> upper;
};

double sqr(double x)
{
    return x * x;
}

double radialStep()
{
    return RMAX / static_cast<double>(N + 1);
}

vector<double> buildRadialGrid()
{
    const double dr = radialStep();
    vector<double> r(N);

    for (int i = 0; i < N; ++i)
        r[i] = (i + 1) * dr;

    return r;
}

std::string symbolForZ(int Z)
{
    static const std::map<int, std::string> symbols =
    {
        {1, "H"},
        {2, "He"},
        {3, "Li"},
        {4, "Be"},
        {5, "B"},
        {6, "C"}
    };

    const auto it = symbols.find(Z);

    if (it == symbols.end())
        throw std::runtime_error("Z fuera del rango H-C.");

    return it->second;
}

vector<AtomicOrbital> configurationForZ(int Z)
{
    switch (Z)
    {
        case 1:
            return {{"1s", 1, 0, 1.0, 0}};

        case 2:
            return {{"1s", 1, 0, 2.0, 0}};

        case 3:
            return
            {
                {"1s", 1, 0, 2.0, 0},
                {"2s", 2, 0, 1.0, 1}
            };

        case 4:
            return
            {
                {"1s", 1, 0, 2.0, 0},
                {"2s", 2, 0, 2.0, 1}
            };

        case 5:
            return
            {
                {"1s", 1, 0, 2.0, 0},
                {"2s", 2, 0, 2.0, 1},
                {"2p", 2, 1, 1.0, 0}
            };

        case 6:
            return
            {
                {"1s", 1, 0, 2.0, 0},
                {"2s", 2, 0, 2.0, 1},
                {"2p", 2, 1, 2.0, 0}
            };

        default:
            throw std::runtime_error("Z fuera del rango H-C.");
    }
}

void normalizeOrbital(vector<double>& u)
{
    const double dr = radialStep();
    double norm = 0.0;

    for (double value : u)
        norm += value * value * dr;

    if (!std::isfinite(norm) || norm <= EPS)
        throw std::runtime_error("Orbital con norma invalida.");

    const double factor = 1.0 / std::sqrt(norm);

    for (double& value : u)
        value *= factor;
}

double orbitalNorm(const vector<double>& u)
{
    const double dr = radialStep();
    double norm = 0.0;

    for (double value : u)
        norm += value * value * dr;

    return norm;
}

double dotProduct(
    const vector<double>& a,
    const vector<double>& b)
{
    if (a.size() != b.size())
        throw std::runtime_error("Vectores incompatibles.");

    const double dr = radialStep();
    double result = 0.0;

    for (int i = 0; i < N; ++i)
        result += a[i] * b[i] * dr;

    return result;
}

TridiagonalMatrix buildHamiltonian(
    const vector<double>& r,
    const vector<double>& potential,
    int l)
{
    const double dr = radialStep();
    const double invDr2 = 1.0 / sqr(dr);

    TridiagonalMatrix H;

    H.lower.resize(N - 1);
    H.diagonal.resize(N);
    H.upper.resize(N - 1);

    for (int i = 0; i < N; ++i)
    {
        const double centrifugal =
            0.5 *
            static_cast<double>(l * (l + 1)) /
            sqr(r[i]);

        H.diagonal[i] =
            invDr2 +
            centrifugal +
            potential[i];

        if (i < N - 1)
        {
            H.lower[i] = -0.5 * invDr2;
            H.upper[i] = -0.5 * invDr2;
        }
    }

    return H;
}

vector<double> applyHamiltonian(
    const TridiagonalMatrix& H,
    const vector<double>& u)
{
    vector<double> result(N, 0.0);

    for (int i = 0; i < N; ++i)
    {
        result[i] =
            H.diagonal[i] * u[i];

        if (i > 0)
            result[i] +=
                H.lower[i - 1] * u[i - 1];

        if (i < N - 1)
            result[i] +=
                H.upper[i] * u[i + 1];
    }

    return result;
}

double rayleighQuotient(
    const TridiagonalMatrix& H,
    const vector<double>& u)
{
    const vector<double> Hu =
        applyHamiltonian(H, u);

    const double denominator =
        dotProduct(u, u);

    if (denominator <= EPS)
        throw std::runtime_error(
            "Norma nula en cociente de Rayleigh.");

    return dotProduct(u, Hu) / denominator;
}

vector<double> solveTridiagonal(
    const vector<double>& a,
    const vector<double>& b,
    const vector<double>& c,
    const vector<double>& d)
{
    const int n =
        static_cast<int>(b.size());

    if (n < 2 ||
        static_cast<int>(a.size()) != n - 1 ||
        static_cast<int>(c.size()) != n - 1 ||
        static_cast<int>(d.size()) != n)
    {
        throw std::runtime_error(
            "Dimensiones invalidas en sistema tridiagonal.");
    }

    vector<double> cp(n - 1);
    vector<double> dp(n);

    double denominator = b[0];

    if (std::abs(denominator) < EPS)
        throw std::runtime_error(
            "Singularidad en sistema tridiagonal.");

    cp[0] = c[0] / denominator;
    dp[0] = d[0] / denominator;

    for (int i = 1; i < n; ++i)
    {
        denominator =
            b[i] -
            a[i - 1] * cp[i - 1];

        if (std::abs(denominator) < EPS)
            throw std::runtime_error(
                "Singularidad en sistema tridiagonal.");

        if (i < n - 1)
            cp[i] = c[i] / denominator;

        dp[i] =
            (
                d[i] -
                a[i - 1] * dp[i - 1]
            ) /
            denominator;
    }

    vector<double> x(n);

    x[n - 1] = dp[n - 1];

    for (int i = n - 2; i >= 0; --i)
        x[i] =
            dp[i] -
            cp[i] * x[i + 1];

    return x;
}

int countEigenvaluesBelow(
    const TridiagonalMatrix& H,
    double energy)
{
    int count = 0;

    double q =
        H.diagonal[0] -
        energy;

    if (q < 0.0)
        ++count;

    for (int i = 1; i < N; ++i)
    {
        double previous = q;

        if (std::abs(previous) < EPS)
            previous =
                previous < 0.0
                ? -EPS
                : EPS;

        q =
            H.diagonal[i] -
            energy -
            H.lower[i - 1] *
            H.upper[i - 1] /
            previous;

        if (q < 0.0)
            ++count;
    }

    return count;
}

double findEigenvalue(
    const TridiagonalMatrix& H,
    int stateIndex)
{
    double low = -2.0;
    double high = 1.0;

    while (
        countEigenvaluesBelow(H, low)
        > stateIndex)
    {
        low *= 2.0;
    }

    while (
        countEigenvaluesBelow(H, high)
        <= stateIndex)
    {
        high *= 2.0;
    }

    for (int iteration = 0;
         iteration < 200;
         ++iteration)
    {
        const double mid =
            0.5 * (low + high);

        if (
            countEigenvaluesBelow(H, mid)
            <= stateIndex)
        {
            low = mid;
        }
        else
        {
            high = mid;
        }

        if (std::abs(high - low) < 1.0e-12)
            break;
    }

    return 0.5 * (low + high);
}

vector<double> buildInitialOrbital(
    const vector<double>& r,
    double eigenvalue,
    int l,
    int stateIndex)
{
    vector<double> u(N, 0.0);

    const double alpha =
        std::max(
            0.1,
            std::sqrt(
                std::max(
                    0.01,
                    -2.0 * eigenvalue)));

    for (int i = 0; i < N; ++i)
    {
        const double x = r[i];

        double polynomial = 1.0;

        if (stateIndex > 0)
        {
            polynomial =
                1.0 -
                alpha *
                x /
                static_cast<double>(
                    stateIndex + 1);
        }

        u[i] =
            std::pow(
                x,
                static_cast<double>(l + 1)) *
            polynomial *
            std::exp(-alpha * x);
    }

    normalizeOrbital(u);

    return u;
}

vector<double> inverseIteration(
    const vector<double>& r,
    const TridiagonalMatrix& H,
    double eigenvalue,
    int l,
    int stateIndex)
{
    vector<double> u =
        buildInitialOrbital(
            r,
            eigenvalue,
            l,
            stateIndex);

    vector<double> a = H.lower;
    vector<double> b = H.diagonal;
    vector<double> c = H.upper;

    const double shift =
        eigenvalue +
        1.0e-10 *
        std::max(
            1.0,
            std::abs(eigenvalue));

    for (double& value : b)
        value -= shift;

    for (int iteration = 0;
         iteration < 100;
         ++iteration)
    {
        vector<double> next =
            solveTridiagonal(
                a,
                b,
                c,
                u);

        normalizeOrbital(next);

        if (dotProduct(u, next) < 0.0)
        {
            for (double& value : next)
                value = -value;
        }

        double difference = 0.0;

        for (int i = 0; i < N; ++i)
        {
            difference =
                std::max(
                    difference,
                    std::abs(
                        next[i] -
                        u[i]));
        }

        u =
            std::move(next);

        if (difference < 1.0e-12)
            break;
    }

    normalizeOrbital(u);

    return u;
}

vector<double> buildDensity(
    const vector<double>& r,
    const vector<AtomicOrbital>& orbitals)
{
    vector<double> density(N, 0.0);

    for (const auto& orbital :
         orbitals)
    {
        for (int i = 0; i < N; ++i)
        {
            density[i] +=
                orbital.occupation *
                sqr(orbital.u[i]) /
                (
                    4.0 *
                    PI *
                    sqr(r[i])
                );
        }
    }

    return density;
}

double electronNumber(
    const vector<double>& r,
    const vector<double>& density)
{
    const double dr =
        radialStep();

    double number = 0.0;

    for (int i = 0; i < N; ++i)
    {
        number +=
            4.0 *
            PI *
            sqr(r[i]) *
            density[i] *
            dr;
    }

    return number;
}

vector<double> hartreePotential(
    const vector<double>& r,
    const vector<double>& density)
{
    const double dr =
        radialStep();

    vector<double> Vh(N, 0.0);
    vector<double> enclosed(N, 0.0);

    double charge = 0.0;

    for (int i = 0; i < N; ++i)
    {
        charge +=
            4.0 *
            PI *
            sqr(r[i]) *
            density[i] *
            dr;

        enclosed[i] =
            charge;
    }

    double outer = 0.0;

    for (int i = N - 1; i >= 0; --i)
    {
        if (i < N - 1)
        {
            outer +=
                4.0 *
                PI *
                r[i + 1] *
                density[i + 1] *
                dr;
        }

        Vh[i] =
            enclosed[i] / r[i] +
            outer;
    }

    return Vh;
}

double exchangeEnergyDensity(double rho)
{
    constexpr double CX =
        0.7385587663820223;

    rho =
        std::max(
            rho,
            RHO_FLOOR);

    return
        -CX *
        std::pow(
            rho,
            4.0 / 3.0);
}

double exchangePotential(double rho)
{
    constexpr double CX =
        0.7385587663820223;

    rho =
        std::max(
            rho,
            RHO_FLOOR);

    return
        -(4.0 / 3.0) *
        CX *
        std::pow(
            rho,
            1.0 / 3.0);
}

double correlationEnergyPerParticle(
    double rs)
{
    rs =
        std::max(
            rs,
            1.0e-12);

    if (rs < 1.0)
    {
        constexpr double A = 0.0311;
        constexpr double B = -0.0480;
        constexpr double C = 0.0020;
        constexpr double D = -0.0116;

        return
            A * std::log(rs) +
            B +
            C * rs * std::log(rs) +
            D * rs;
    }

    constexpr double gamma = -0.1423;
    constexpr double beta1 = 1.0529;
    constexpr double beta2 = 0.3334;

    const double sqrtRs =
        std::sqrt(rs);

    return
        gamma /
        (
            1.0 +
            beta1 * sqrtRs +
            beta2 * rs
        );
}

double correlationPotential(double rho)
{
    rho =
        std::max(
            rho,
            RHO_FLOOR);

    const double rs =
        std::pow(
            3.0 /
            (4.0 * PI * rho),
            1.0 / 3.0);

    if (rs < 1.0)
    {
        constexpr double A = 0.0311;
        constexpr double B = -0.0480;
        constexpr double C = 0.0020;
        constexpr double D = -0.0116;

        const double eps =
            A * std::log(rs) +
            B +
            C * rs * std::log(rs) +
            D * rs;

        const double deps =
            A / rs +
            C * (std::log(rs) + 1.0) +
            D;

        return
            eps -
            (rs / 3.0) * deps;
    }

    constexpr double gamma = -0.1423;
    constexpr double beta1 = 1.0529;
    constexpr double beta2 = 0.3334;

    const double sqrtRs =
        std::sqrt(rs);

    const double denominator =
        1.0 +
        beta1 * sqrtRs +
        beta2 * rs;

    const double eps =
        gamma /
        denominator;

    const double dDenominator =
        beta1 /
        (2.0 * sqrtRs) +
        beta2;

    const double deps =
        -gamma *
        dDenominator /
        sqr(denominator);

    return
        eps -
        (rs / 3.0) * deps;
}

vector<double> exchangeCorrelationPotential(
    const vector<double>& density)
{
    vector<double> Vxc(N, 0.0);

    for (int i = 0; i < N; ++i)
    {
        const double rho =
            std::max(
                density[i],
                RHO_FLOOR);

        Vxc[i] =
            exchangePotential(rho) +
            correlationPotential(rho);
    }

    return Vxc;
}

double exchangeCorrelationEnergy(
    const vector<double>& r,
    const vector<double>& density)
{
    const double dr =
        radialStep();

    double energy = 0.0;

    for (int i = 0; i < N; ++i)
    {
        const double rho =
            std::max(
                density[i],
                RHO_FLOOR);

        const double rs =
            std::pow(
                3.0 /
                (4.0 * PI * rho),
                1.0 / 3.0);

        const double epsXC =
            exchangeEnergyDensity(rho) / rho +
            correlationEnergyPerParticle(rs);

        energy +=
            4.0 *
            PI *
            sqr(r[i]) *
            rho *
            epsXC *
            dr;
    }

    return energy;
}

double hartreeEnergy(
    const vector<double>& r,
    const vector<double>& density,
    const vector<double>& Vh)
{
    const double dr =
        radialStep();

    double energy = 0.0;

    for (int i = 0; i < N; ++i)
    {
        energy +=
            0.5 *
            4.0 *
            PI *
            sqr(r[i]) *
            density[i] *
            Vh[i] *
            dr;
    }

    return energy;
}

double orbitalKineticEnergy(
    const vector<double>& r,
    const vector<double>& u,
    int l)
{
    const double dr =
        radialStep();

    double energy = 0.0;

    for (int i = 1; i < N - 1; ++i)
    {
        const double du =
            (u[i + 1] - u[i - 1]) /
            (2.0 * dr);

        const double centrifugal =
            static_cast<double>(
                l * (l + 1)) *
            sqr(u[i]) /
            sqr(r[i]);

        energy +=
            0.5 *
            (
                sqr(du) +
                centrifugal
            ) *
            dr;
    }

    return energy;
}

double kineticEnergy(
    const vector<double>& r,
    const vector<AtomicOrbital>& orbitals)
{
    double energy = 0.0;

    for (const auto& orbital :
         orbitals)
    {
        energy +=
            orbital.occupation *
            orbitalKineticEnergy(
                r,
                orbital.u,
                orbital.l);
    }

    return energy;
}

double externalEnergy(
    const vector<double>& r,
    int Z,
    const vector<AtomicOrbital>& orbitals)
{
    const double dr =
        radialStep();

    double energy = 0.0;

    for (const auto& orbital :
         orbitals)
    {
        for (int i = 0; i < N; ++i)
        {
            energy +=
                orbital.occupation *
                sqr(orbital.u[i]) *
                (
                    -static_cast<double>(Z) /
                    r[i]
                ) *
                dr;
        }
    }

    return energy;
}

double maxKSResidual(
    const vector<double>& r,
    const vector<double>& Veff,
    const vector<AtomicOrbital>& orbitals)
{
    double maximum = 0.0;

    for (const auto& orbital :
         orbitals)
    {
        const TridiagonalMatrix H =
            buildHamiltonian(
                r,
                Veff,
                orbital.l);

        const vector<double> Hu =
            applyHamiltonian(
                H,
                orbital.u);

        for (int i = 0; i < N; ++i)
        {
            maximum =
                std::max(
                    maximum,
                    std::abs(
                        Hu[i] -
                        orbital.eigenvalue *
                        orbital.u[i]));
        }
    }

    return maximum;
}

double totalEnergy(
    double Ts,
    double Eext,
    double EH,
    double EXC)
{
    return
        Ts +
        Eext +
        EH +
        EXC;
}

void solveOrbitals(
    const vector<double>& r,
    const vector<double>& Veff,
    vector<AtomicOrbital>& orbitals)
{
    for (auto& orbital :
         orbitals)
    {
        const TridiagonalMatrix H =
            buildHamiltonian(
                r,
                Veff,
                orbital.l);

        orbital.eigenvalue =
            findEigenvalue(
                H,
                orbital.stateIndex);

        orbital.u =
            inverseIteration(
                r,
                H,
                orbital.eigenvalue,
                orbital.l,
                orbital.stateIndex);

        normalizeOrbital(
            orbital.u);

        orbital.eigenvalue =
            rayleighQuotient(
                H,
                orbital.u);
    }
}

double maxDensityDifference(
    const vector<double>& a,
    const vector<double>& b)
{
    if (a.size() != b.size())
        throw std::runtime_error(
            "Densidades incompatibles.");

    double difference = 0.0;

    for (int i = 0; i < N; ++i)
    {
        difference =
            std::max(
                difference,
                std::abs(
                    a[i] -
                    b[i]));
    }

    return difference;
}

AtomResult solveAtom(int Z)
{
    const vector<double> r =
        buildRadialGrid();

    AtomResult result;

    result.symbol =
        symbolForZ(Z);

    result.Z =
        Z;

    result.orbitals =
        configurationForZ(Z);

    for (const auto& orbital :
         result.orbitals)
    {
        result.electrons +=
            static_cast<int>(
                std::round(
                    orbital.occupation));
    }

    vector<double> density(N, 0.0);

    const double alpha =
        static_cast<double>(Z);

    for (int i = 0; i < N; ++i)
    {
        density[i] =
            static_cast<double>(
                result.electrons) *
            std::pow(alpha, 3.0) *
            std::exp(
                -2.0 *
                alpha *
                r[i]) /
            PI;
    }

    const double initialNumber =
        electronNumber(
            r,
            density);

    if (!std::isfinite(initialNumber) ||
        initialNumber <= EPS)
    {
        throw std::runtime_error(
            "Densidad inicial invalida.");
    }

    const double normalization =
        static_cast<double>(
            result.electrons) /
        initialNumber;

    for (double& value : density)
        value *= normalization;

    double previousEnergy =
        std::numeric_limits<double>::infinity();

    bool scfConverged = false;

    for (int iteration = 1;
         iteration <= MAX_SCF_ITER;
         ++iteration)
    {
        const vector<double> Vh =
            hartreePotential(
                r,
                density);

        const vector<double> Vxc =
            exchangeCorrelationPotential(
                density);

        vector<double> Veff(N);

        for (int i = 0; i < N; ++i)
        {
            Veff[i] =
                -static_cast<double>(Z) /
                r[i] +
                Vh[i] +
                Vxc[i];
        }

        vector<AtomicOrbital> newOrbitals =
            result.orbitals;

        solveOrbitals(
            r,
            Veff,
            newOrbitals);

        const vector<double> newDensity =
            buildDensity(
                r,
                newOrbitals);

        const double densityResidual =
            maxDensityDifference(
                density,
                newDensity);

        const vector<double> newVh =
            hartreePotential(
                r,
                newDensity);

        const double Ts =
            kineticEnergy(
                r,
                newOrbitals);

        const double Eext =
            externalEnergy(
                r,
                Z,
                newOrbitals);

        const double EH =
            hartreeEnergy(
                r,
                newDensity,
                newVh);

        const double EXC =
            exchangeCorrelationEnergy(
                r,
                newDensity);

        const double E =
            totalEnergy(
                Ts,
                Eext,
                EH,
                EXC);

        const double energyResidual =
            std::isfinite(previousEnergy)
            ? std::abs(
                E -
                previousEnergy)
            : std::numeric_limits<double>::infinity();

        vector<double> mixedDensity(N);

        for (int i = 0; i < N; ++i)
        {
            mixedDensity[i] =
                (1.0 - MIXING) *
                density[i] +
                MIXING *
                newDensity[i];
        }

        result.orbitals =
            std::move(newOrbitals);

        result.totalEnergy =
            E;

        result.kineticEnergy =
            Ts;

        result.externalEnergy =
            Eext;

        result.hartreeEnergy =
            EH;

        result.xcEnergy =
            EXC;

        result.electronNumber =
            electronNumber(
                r,
                newDensity);

        result.densityResidual =
            densityResidual;

        result.energyResidual =
            energyResidual;

        result.iterations =
            iteration;

        density =
            std::move(mixedDensity);

        if (densityResidual < DENSITY_TOL &&
            energyResidual < ENERGY_TOL)
        {
            scfConverged = true;
            break;
        }

        previousEnergy =
            E;
    }

    const vector<double> finalVh =
        hartreePotential(
            r,
            density);

    const vector<double> finalVxc =
        exchangeCorrelationPotential(
            density);

    vector<double> finalVeff(N);

    for (int i = 0; i < N; ++i)
    {
        finalVeff[i] =
            -static_cast<double>(Z) /
            r[i] +
            finalVh[i] +
            finalVxc[i];
    }

    vector<AtomicOrbital> finalOrbitals =
        result.orbitals;

    solveOrbitals(
        r,
        finalVeff,
        finalOrbitals);

    const vector<double> finalDensity =
        buildDensity(
            r,
            finalOrbitals);

    const double finalDensityResidual =
        maxDensityDifference(
            density,
            finalDensity);

    const vector<double> consistentVh =
        hartreePotential(
            r,
            finalDensity);

    const double Ts =
        kineticEnergy(
            r,
            finalOrbitals);

    const double Eext =
        externalEnergy(
            r,
            Z,
            finalOrbitals);

    const double EH =
        hartreeEnergy(
            r,
            finalDensity,
            consistentVh);

    const double EXC =
        exchangeCorrelationEnergy(
            r,
            finalDensity);

    result.orbitals =
        std::move(finalOrbitals);

    result.totalEnergy =
        totalEnergy(
            Ts,
            Eext,
            EH,
            EXC);

    result.kineticEnergy =
        Ts;

    result.externalEnergy =
        Eext;

    result.hartreeEnergy =
        EH;

    result.xcEnergy =
        EXC;

    result.electronNumber =
        electronNumber(
            r,
            finalDensity);

    result.densityResidual =
        finalDensityResidual;

    vector<double> consistentVxc =
        exchangeCorrelationPotential(
            finalDensity);

    vector<double> consistentPotential(N);

    for (int i = 0; i < N; ++i)
    {
        consistentPotential[i] =
            -static_cast<double>(Z) /
            r[i] +
            consistentVh[i] +
            consistentVxc[i];
    }

    result.ksResidual =
        maxKSResidual(
            r,
            consistentPotential,
            result.orbitals);

    result.converged =
        scfConverged &&
        result.densityResidual < DENSITY_TOL &&
        result.ksResidual < KS_RESIDUAL_TOL;

    return result;
}

double hydrogenCoulombEnergy()
{
    const vector<double> r =
        buildRadialGrid();

    vector<double> V(N);

    for (int i = 0; i < N; ++i)
        V[i] = -1.0 / r[i];

    const TridiagonalMatrix H =
        buildHamiltonian(
            r,
            V,
            0);

    return
        findEigenvalue(
            H,
            0);
}

bool validateOrbitalNorms(
    const AtomResult& atom)
{
    for (const auto& orbital :
         atom.orbitals)
    {
        const double norm =
            orbitalNorm(
                orbital.u);

        if (!std::isfinite(norm) ||
            std::abs(norm - 1.0) > 1.0e-6)
        {
            return false;
        }
    }

    return true;
}

bool validateElectronCount(
    const AtomResult& atom)
{
    int count = 0;

    for (const auto& orbital :
         atom.orbitals)
    {
        count +=
            static_cast<int>(
                std::round(
                    orbital.occupation));
    }

    return count == atom.electrons;
}

bool validateNumericalElectronNumber(
    const AtomResult& atom)
{
    return
        std::abs(
            atom.electronNumber -
            static_cast<double>(
                atom.electrons)) <
        1.0e-6;
}

bool validateEnergyDecomposition(
    const AtomResult& atom)
{
    const double reconstructed =
        atom.kineticEnergy +
        atom.externalEnergy +
        atom.hartreeEnergy +
        atom.xcEnergy;

    return
        std::abs(
            reconstructed -
            atom.totalEnergy) <
        1.0e-8;
}

bool validateSCF(
    const AtomResult& atom)
{
    return atom.converged;
}

int main()
{
    try
    {
        vector<AtomResult> results;

        for (int Z = 1; Z <= 6; ++Z)
            results.push_back(
                solveAtom(Z));

        std::cout
            << "\n"
            << "============================================================\n"
            << "                         DFT ATOMICO H -> C\n"
            << "============================================================\n\n";

        std::cout
            << std::left
            << std::setw(8)
            << "Atom"
            << std::right
            << std::setw(22)
            << "E_total"
            << std::setw(22)
            << "T_s"
            << std::setw(22)
            << "E_ext"
            << std::setw(22)
            << "E_H"
            << std::setw(22)
            << "E_XC"
            << std::setw(8)
            << "Iter"
            << std::setw(8)
            << "SCF"
            << '\n';

        std::cout
            << "------------------------------------------------------------------------------------------------------------------------\n";

        for (const auto& atom :
             results)
        {
            std::cout
                << std::left
                << std::setw(8)
                << atom.symbol
                << std::right
                << std::scientific
                << std::setprecision(10)
                << std::setw(22)
                << atom.totalEnergy
                << std::setw(22)
                << atom.kineticEnergy
                << std::setw(22)
                << atom.externalEnergy
                << std::setw(22)
                << atom.hartreeEnergy
                << std::setw(22)
                << atom.xcEnergy
                << std::fixed
                << std::setprecision(0)
                << std::setw(8)
                << atom.iterations
                << std::setw(8)
                << (
                    atom.converged
                    ? "YES"
                    : "NO"
                )
                << '\n';
        }

        const double hCoulomb =
            hydrogenCoulombEnergy();

        bool hSolverPass = true;
        bool scfPass = true;
        bool electronPass = true;
        bool numericalElectronPass = true;
        bool normPass = true;
        bool energyPass = true;

        hSolverPass =
            std::abs(
                hCoulomb -
                H_EXACT) <
            1.0e-4;

        for (const auto& atom :
             results)
        {
            scfPass =
                scfPass &&
                validateSCF(atom);

            electronPass =
                electronPass &&
                validateElectronCount(atom);

            numericalElectronPass =
                numericalElectronPass &&
                validateNumericalElectronNumber(atom);

            normPass =
                normPass &&
                validateOrbitalNorms(atom);

            energyPass =
                energyPass &&
                validateEnergyDecomposition(atom);
        }

        std::cout
            << "\n"
            << "============================================================\n"
            << "                         VALIDACION\n"
            << "============================================================\n\n";

        std::cout
            << std::left
            << std::setw(34)
            << "Solver Coulomb H"
            << (hSolverPass ? "PASS" : "FAIL")
            << '\n';

        std::cout
            << std::setw(34)
            << "SCF H-C"
            << (scfPass ? "PASS" : "FAIL")
            << '\n';

        std::cout
            << std::setw(34)
            << "Numero de electrones"
            << (electronPass ? "PASS" : "FAIL")
            << '\n';

        std::cout
            << std::setw(34)
            << "Numero electronico numerico"
            << (numericalElectronPass ? "PASS" : "FAIL")
            << '\n';

        std::cout
            << std::setw(34)
            << "Normas orbitales"
            << (normPass ? "PASS" : "FAIL")
            << '\n';

        std::cout
            << std::setw(34)
            << "Descomposicion energetica"
            << (energyPass ? "PASS" : "FAIL")
            << '\n';

        std::cout
            << "\n"
            << "E(H) Coulomb numerico = "
            << std::scientific
            << std::setprecision(10)
            << hCoulomb
            << " Ha\n";

        std::cout
            << "E(He) DFT-LDA-PZ81    = "
            << results[1].totalEnergy
            << " Ha\n";

        std::cout
            << "\n"
            << "============================================================\n"
            << (
                hSolverPass &&
                scfPass &&
                electronPass &&
                numericalElectronPass &&
                normPass &&
                energyPass
                ? "                    VALIDACION PASS\n"
                : "                    VALIDACION FAIL\n"
            )
            << "============================================================\n";

        return
            hSolverPass &&
            scfPass &&
            electronPass &&
            numericalElectronPass &&
            normPass &&
            energyPass
            ? 0
            : 1;
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "\nERROR: "
            << error.what()
            << '\n';

        return 1;
    }
}