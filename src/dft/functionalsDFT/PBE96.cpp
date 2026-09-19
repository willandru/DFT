#include "PBE96.h"

#include "DFTConstants.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr double THIRD =
    1.0 / 3.0;

constexpr double FOUR_THIRDS =
    4.0 / 3.0;

constexpr double TWO_THIRDS =
    2.0 / 3.0;

constexpr double SIXTH_NEGATIVE =
    -1.0 / 6.0;

constexpr double KAPPA =
    0.804;

constexpr double MU =
    0.2195149727645171;

constexpr double GAMMA =
    0.031090690869654895;

constexpr double BETA =
    0.06672455060314922;

constexpr double DELTA =
    BETA / GAMMA;

constexpr double SPIN_GAMMA =
    0.51984209978974632953;

constexpr double SPIN_FZZ =
    8.0 /
    (
        9.0 *
        SPIN_GAMMA
    );

constexpr double ETA =
    1.0e-12;

constexpr double EXCHANGE_A =
    -0.73855876638202240588;

constexpr double KF_COEFFICIENT =
    3.09366772628014;

struct PW92Result {
    double energy;
    double derivativeRs;
    double derivativeZeta;
};

struct PBEExchangeResult {
    double energyDensity;
    double densityDerivative;
    double gradientDerivative;
};

struct PBECorrelationResult {
    double energyPerElectron;
    double potentialAlpha;
    double potentialBeta;
    double gradientDerivative;
};

double clampZeta(
    double zeta
) {
    return std::clamp(
        zeta,
        -1.0,
        1.0
    );
}

double signOf(
    double value
) {
    if (value > 0.0) {
        return 1.0;
    }

    if (value < 0.0) {
        return -1.0;
    }

    return 0.0;
}

double calculateGcor2(
    double a,
    double a1,
    double b1,
    double b2,
    double b3,
    double b4,
    double sqrtRs,
    double& derivativeRs
) {
    const double sqrtRs2 =
        sqrtRs *
        sqrtRs;

    const double q0 =
        -2.0 *
        a *
        (
            1.0 +
            a1 *
            sqrtRs2
        );

    const double q1 =
        2.0 *
        a *
        sqrtRs *
        (
            b1 +
            sqrtRs *
            (
                b2 +
                sqrtRs *
                (
                    b3 +
                    b4 *
                    sqrtRs
                )
            )
        );

    const double q2 =
        std::log(
            1.0 +
            1.0 / q1
        );

    const double energy =
        q0 *
        q2;

    const double q3 =
        a *
        (
            b1 / sqrtRs +
            2.0 * b2 +
            sqrtRs *
            (
                3.0 * b3 +
                4.0 * b4 * sqrtRs
            )
        );

    derivativeRs =
        -2.0 *
        a *
        a1 *
        q2
        -
        q0 *
        q3 /
        (
            q1 *
            (
                1.0 +
                q1
            )
        );

    return energy;
}

PW92Result calculatePW92Correlation(
    double rs,
    double zeta
) {
    const double sqrtRs =
        std::sqrt(rs);

    double euDerivativeRs = 0.0;

    const double eu =
        calculateGcor2(
            0.0310907,
            0.21370,
            7.5957,
            3.5876,
            1.6382,
            0.49294,
            sqrtRs,
            euDerivativeRs
        );

    double epDerivativeRs = 0.0;

    const double ep =
        calculateGcor2(
            0.01554535,
            0.20548,
            14.1189,
            6.1977,
            3.3662,
            0.62517,
            sqrtRs,
            epDerivativeRs
        );

    double alphaDerivativeRs = 0.0;

    const double alpha =
        calculateGcor2(
            0.0168869,
            0.11125,
            10.357,
            3.6231,
            0.88026,
            0.49671,
            sqrtRs,
            alphaDerivativeRs
        );

    const double z4 =
        zeta *
        zeta *
        zeta *
        zeta;

    const double f =
        (
            std::pow(
                1.0 + zeta,
                FOUR_THIRDS
            )
            +
            std::pow(
                1.0 - zeta,
                FOUR_THIRDS
            )
            -
            2.0
        )
        /
        SPIN_GAMMA;

    const double energy =
        eu *
        (
            1.0 -
            f *
            z4
        )
        +
        ep *
        f *
        z4
        -
        alpha *
        f *
        (
            1.0 -
            z4
        )
        /
        SPIN_FZZ;

    const double derivativeRs =
        euDerivativeRs *
        (
            1.0 -
            f *
            z4
        )
        +
        epDerivativeRs *
        f *
        z4
        -
        alphaDerivativeRs *
        f *
        (
            1.0 -
            z4
        )
        /
        SPIN_FZZ;

    const double derivativeF =
        FOUR_THIRDS *
        (
            std::pow(
                1.0 + zeta,
                THIRD
            )
            -
            std::pow(
                1.0 - zeta,
                THIRD
            )
        )
        /
        SPIN_GAMMA;

    const double derivativeZeta =
        4.0 *
        zeta *
        zeta *
        zeta *
        f *
        (
            ep -
            eu +
            alpha /
            SPIN_FZZ
        )
        +
        derivativeF *
        (
            z4 *
            ep
            -
            z4 *
            eu
            -
            (
                1.0 -
                z4
            ) *
            alpha /
            SPIN_FZZ
        );

    return {
        energy,
        derivativeRs,
        derivativeZeta
    };
}

PBEExchangeResult calculatePBEExchange(
    double rho,
    double gradientMagnitude
) {
    if (rho <= 0.0) {
        return {
            0.0,
            0.0,
            0.0
        };
    }

    const double rhoThird =
        std::cbrt(rho);

    const double exchangeEnergyUniformPerElectron =
        EXCHANGE_A *
        rhoThird;

    const double exchangeEnergyUniformDensity =
        rho *
        exchangeEnergyUniformPerElectron;

    const double kF =
        KF_COEFFICIENT *
        rhoThird;

    const double denominator =
        2.0 *
        kF *
        rho;

    double s = 0.0;

    if (denominator > 0.0) {
        s =
            gradientMagnitude /
            denominator;
    }

    const double s2 =
        s *
        s;

    const double muOverKappa =
        MU /
        KAPPA;

    const double p0 =
        1.0 +
        muOverKappa *
        s2;

    const double enhancement =
        1.0 +
        KAPPA -
        KAPPA /
        p0;

    const double fs =
        2.0 *
        KAPPA *
        muOverKappa /
        (
            p0 *
            p0
        );

    const double densityDerivative =
        exchangeEnergyUniformPerElectron *
        FOUR_THIRDS *
        (
            enhancement -
            s2 *
            fs
        );

    double gradientDerivative =
        0.0;

    if (gradientMagnitude > 0.0) {
        const double al =
            0.161620459673995;

        gradientDerivative =
            EXCHANGE_A *
            al *
            s *
            fs;
    }

    return {
        exchangeEnergyUniformDensity *
            enhancement,
        densityDerivative,
        gradientDerivative
    };
}

PBECorrelationResult calculatePBECorrelation(
    double alphaDensity,
    double betaDensity,
    double alphaGradient,
    double betaGradient
) {
    const double density =
        alphaDensity +
        betaDensity;

    if (density <= 0.0) {
        return {
            0.0,
            0.0,
            0.0,
            0.0
        };
    }

    const double zeta =
        clampZeta(
            (
                alphaDensity -
                betaDensity
            )
            /
            density
        );

    const double gradientDensity =
        alphaGradient +
        betaGradient;

    const double gradientMagnitude =
        std::abs(
            gradientDensity
        );

    const double kF =
        std::cbrt(
            3.0 *
            DFTConstants::PI *
            DFTConstants::PI *
            density
        );

    const double rs =
        std::cbrt(
            3.0 /
            (
                4.0 *
                DFTConstants::PI *
                density
            )
        );

    const double phi =
        0.5 *
        (
            std::pow(
                1.0 + zeta,
                TWO_THIRDS
            )
            +
            std::pow(
                1.0 - zeta,
                TWO_THIRDS
            )
        );

    const double ks =
        std::sqrt(
            4.0 *
            kF /
            DFTConstants::PI
        );

    const double twoKsPhi =
        2.0 *
        ks *
        phi;

    double t =
        0.0;

    if (twoKsPhi > 0.0) {
        t =
            gradientMagnitude /
            (
                twoKsPhi *
                density
            );
    }

    const PW92Result pw92 =
        calculatePW92Correlation(
            rs,
            zeta
        );

    const double ec =
        pw92.energy;

    const double ecRs =
        pw92.derivativeRs;

    const double ecZeta =
        pw92.derivativeZeta;

    const double phi3 =
        phi *
        phi *
        phi;

    const double phi4 =
        phi3 *
        phi;

    const double exponent =
        -ec /
        (
            GAMMA *
            phi3
        );

    double b = 0.0;

    if (exponent < 700.0) {
        b =
            DELTA /
            std::expm1(
                exponent
            );
    } else {
        b =
            DELTA *
            std::exp(
                -exponent
            );
    }

    const double b2 =
        b *
        b;

    const double t2 =
        t *
        t;

    const double t4 =
        t2 *
        t2;

    const double t6 =
        t4 *
        t2;

    const double q4 =
        1.0 +
        b *
        t2;

    const double q5 =
        1.0 +
        b *
        t2 +
        b2 *
        t4;

    const double h =
        phi3 *
        (
            BETA /
            DELTA
        )
        *
        std::log(
            1.0 +
            DELTA *
            q4 *
            t2 /
            q5
        );

    const double correlationEnergyPerElectron =
        ec +
        h;

    const double rsthrd =
        rs /
        3.0;

    const double fac =
        DELTA /
        b +
        1.0;

    const double bec =
        b2 *
        fac /
        (
            BETA *
            phi3
        );

    const double q8 =
        q5 *
        q5
        +
        DELTA *
        q4 *
        q5 *
        t2;

    const double q9 =
        1.0 +
        2.0 *
        b *
        t2;

    const double hb =
        -BETA *
        phi3 *
        b *
        t6 *
        (
            2.0 +
            b *
            t2
        )
        /
        q8;

    const double hrs =
        -rsthrd *
        hb *
        bec *
        ecRs;

    const double ht =
        2.0 *
        BETA *
        phi3 *
        q9 /
        q8;

    const double comm =
        h +
        hrs -
        7.0 *
        t2 *
        ht /
        6.0;

    const double gZeta =
        (
            std::pow(
                1.0 + zeta + ETA,
                -THIRD
            )
            -
            std::pow(
                1.0 - zeta + ETA,
                -THIRD
            )
        )
        /
        3.0;

    const double phiDerivativeZeta =
        gZeta;

    const double bg =
        -3.0 *
        b2 *
        ec *
        fac /
        (
            BETA *
            phi4
        );

    const double hz =
        3.0 *
        phiDerivativeZeta *
        h /
        phi
        +
        hb *
        (
            bg *
            phiDerivativeZeta
            +
            bec *
            ecZeta
        );

    const double pref =
        hz -
        phiDerivativeZeta *
        t2 *
        ht /
        phi;

    const double potentialAlpha =
        (
            ec -
            rs *
            ecRs /
            3.0 -
            zeta *
            ecZeta
        )
        +
        (
            ecZeta
        )
        +
        comm
        +
        pref *
        (
            1.0 -
            zeta
        );

    const double potentialBeta =
        (
            ec -
            rs *
            ecRs /
            3.0 -
            zeta *
            ecZeta
        )
        -
        (
            ecZeta
        )
        +
        comm
        +
        pref *
        (
            -1.0 -
            zeta
        );

    double gradientDerivative =
        0.0;

    if (gradientMagnitude > 0.0) {
        gradientDerivative =
            t *
            ht /
            twoKsPhi;
    }

    return {
        correlationEnergyPerElectron,
        potentialAlpha,
        potentialBeta,
        gradientDerivative
    };
}

}

XCResult PBE96::evaluate(
    const XCInput& input
) const {
    const double alphaDensity =
        std::max(
            input.alphaDensity,
            0.0
        );

    const double betaDensity =
        std::max(
            input.betaDensity,
            0.0
        );

    const double density =
        alphaDensity +
        betaDensity;

    if (density <=
        DFTConstants::RHO_FLOOR) {

        return {
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        };
    }

    const double alphaGradient =
        input.alphaGradient;

    const double betaGradient =
        input.betaGradient;

    const double alphaGradientMagnitude =
        std::abs(
            alphaGradient
        );

    const double betaGradientMagnitude =
        std::abs(
            betaGradient
        );

    /*
     * PBE exchange with exact spin scaling:
     *
     * E_x[n_alpha,n_beta]
     * =
     * 1/2 E_x[2 n_alpha]
     * +
     * 1/2 E_x[2 n_beta].
     */
    const PBEExchangeResult exchangeAlpha =
        calculatePBEExchange(
            2.0 *
            alphaDensity,
            2.0 *
            alphaGradientMagnitude
        );

    const PBEExchangeResult exchangeBeta =
        calculatePBEExchange(
            2.0 *
            betaDensity,
            2.0 *
            betaGradientMagnitude
        );

    const PBECorrelationResult correlation =
        calculatePBECorrelation(
            alphaDensity,
            betaDensity,
            alphaGradient,
            betaGradient
        );

    const double exchangeEnergyDensity =
        0.5 *
        (
            exchangeAlpha.energyDensity +
            exchangeBeta.energyDensity
        );

    const double exchangePerElectron =
        exchangeEnergyDensity /
        density;

    const double energyPerElectron =
        exchangePerElectron +
        correlation.energyPerElectron;

    const double potentialAlpha =
        exchangeAlpha.densityDerivative +
        correlation.potentialAlpha;

    const double potentialBeta =
        exchangeBeta.densityDerivative +
        correlation.potentialBeta;

    /*
     * Exchange contribution:
     *
     * d f_x / d(grad n_sigma)
     * =
     * d f_x / d|grad(2 n_sigma)|
     * *
     * sign(grad n_sigma).
     *
     * Correlation depends on the total density gradient.
     */
    const double exchangeGradientAlpha =
        exchangeAlpha.gradientDerivative *
        signOf(
            alphaGradient
        );

    const double exchangeGradientBeta =
        exchangeBeta.gradientDerivative *
        signOf(
            betaGradient
        );

    const double correlationGradient =
        correlation.gradientDerivative;

    const double correlationSign =
        signOf(
            alphaGradient +
            betaGradient
        );

    const double gradientCoefficientAlpha =
        exchangeGradientAlpha +
        correlationGradient *
        correlationSign;

    const double gradientCoefficientBeta =
        exchangeGradientBeta +
        correlationGradient *
        correlationSign;

    return {
        energyPerElectron,
        potentialAlpha,
        potentialBeta,
        gradientCoefficientAlpha,
        gradientCoefficientBeta
    };
}