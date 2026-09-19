#include "SelfConsistentFieldMolecule.h"

#include "DFTConstants.h"
#include "ElectronDensityMolecule.h"
#include "ExchangeCorrelationMolecule.h"
#include "HartreePotentialMolecule.h"
#include "NuclearPotential.h"
#include "SelfConsistentFieldMoleculeHelper.h"
#include "SelfConsistentFieldMoleculeMath.h"
#include "TotalEnergyMolecule.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

constexpr double MIN_MIXING = 0.10;
constexpr double MAX_MIXING = 0.50;

constexpr double MIXING_INCREASE = 1.10;
constexpr double MIXING_DECREASE = 0.75;

constexpr double OSCILLATION_FACTOR = 1.05;

void buildInitialMolecularDensities(
    const CartesianGrid& grid,
    const Molecule& molecule,
    int alphaElectrons,
    int betaElectrons,
    std::vector<double>& alphaDensity,
    std::vector<double>& betaDensity,
    std::vector<MolecularOrbital>& initialAlphaOrbitals,
    std::vector<MolecularOrbital>& initialBetaOrbitals
)
{
    const std::vector<double> nuclearPotential =
        NuclearPotential::calculate(
            molecule,
            grid
        );

    const std::vector<int> alphaOccupations =
        MolecularSCFMath::buildSpinOccupations(
            alphaElectrons
        );

    const std::vector<int> betaOccupations =
        MolecularSCFMath::buildSpinOccupations(
            betaElectrons
        );

    initialAlphaOrbitals =
        MolecularSCFHelper::solveMolecularOrbitalsSilently(
            grid,
            nuclearPotential,
            alphaOccupations.size(),
            alphaOccupations,
            SpinChannel::Alpha,
            std::vector<MolecularOrbital>{}
        );

    if (alphaElectrons == betaElectrons) {

        initialBetaOrbitals =
            MolecularSCFMath::convertOrbitalsToSpin(
                initialAlphaOrbitals,
                SpinChannel::Beta
            );
    }
    else {

        initialBetaOrbitals =
            MolecularSCFHelper::solveMolecularOrbitalsSilently(
                grid,
                nuclearPotential,
                betaOccupations.size(),
                betaOccupations,
                SpinChannel::Beta,
                std::vector<MolecularOrbital>{}
            );
    }

    alphaDensity =
        calculateMolecularSpinDensity(
            initialAlphaOrbitals,
            SpinChannel::Alpha
        );

    betaDensity =
        calculateMolecularSpinDensity(
            initialBetaOrbitals,
            SpinChannel::Beta
        );
}

}

MolecularResult solveMolecularSelfConsistentField(
    const CartesianGrid& grid,
    const Molecule& molecule,
    int charge,
    int multiplicity,
    const XCFunctional& functional
)
{
    /*
     * ================================================================
     * 1. VALIDACION DEL SISTEMA
     * ================================================================
     */

    if (grid.getNx() < 2 ||
        grid.getNy() < 2 ||
        grid.getNz() < 2) {

        throw std::invalid_argument(
            "La malla cartesiana debe contener al menos dos puntos por dimension."
        );
    }

    if (molecule.getNucleusCount() == 0) {

        throw std::invalid_argument(
            "La molecula debe contener al menos un nucleo."
        );
    }

    if (multiplicity < 1) {

        throw std::invalid_argument(
            "La multiplicidad de espin debe ser un entero positivo."
        );
    }

    /*
     * ================================================================
     * 2. CARGA NUCLEAR Y NUMERO DE ELECTRONES
     * ================================================================
     */

    const int nuclearCharge =
        [&molecule]() {

            int total = 0;

            for (const Molecule::Nucleus& nucleus :
                 molecule.getNuclei()) {

                total +=
                    nucleus.atomicNumber;
            }

            return total;
        }();

    const int electronCount =
        nuclearCharge -
        charge;

    if (electronCount < 0) {

        throw std::invalid_argument(
            "La carga molecular produce un numero negativo de electrones."
        );
    }

    if (electronCount == 0) {

        throw std::invalid_argument(
            "El SCF molecular requiere al menos un electron."
        );
    }

    /*
     * ================================================================
     * 3. PARTICION DE ESPIN
     *
     * La multiplicidad satisface:
     *
     *     M = 2S + 1
     *
     * por lo tanto:
     *
     *     2S = M - 1
     *
     * y la particion electronica es:
     *
     *     N_alpha = (N + 2S) / 2
     *     N_beta  = (N - 2S) / 2
     *
     * equivalentemente:
     *
     *     N_alpha = (N + M - 1) / 2
     *     N_beta  = (N - M + 1) / 2
     * ================================================================
     */

    const int spinDifference =
        multiplicity - 1;

    if (spinDifference > electronCount) {

        throw std::invalid_argument(
            "La multiplicidad de espin es incompatible con el numero de electrones."
        );
    }

    /*
     * N + M - 1 debe ser par para que N_alpha y N_beta
     * sean enteros.
     */
    if ((electronCount + multiplicity - 1) % 2 != 0) {

        throw std::invalid_argument(
            "La multiplicidad de espin no es compatible con el numero de electrones."
        );
    }

    const int alphaElectrons =
        (electronCount + spinDifference) / 2;

    const int betaElectrons =
        (electronCount - spinDifference) / 2;

    if (alphaElectrons < 0 ||
        betaElectrons < 0 ||
        alphaElectrons + betaElectrons != electronCount) {

        throw std::invalid_argument(
            "La particion de electrones Alpha/Beta es invalida."
        );
    }

    const bool closedShell =
        multiplicity == 1 &&
        alphaElectrons == betaElectrons;

    /*
     * ================================================================
     * 4. ESTADO INICIAL
     * ================================================================
     */

    MolecularSCFHelper::Timer timer;

    MolecularSCFHelper::ConvergenceAnalysis convergenceAnalysis;

    timer.start();

    timer.begin(
        MolecularSCFHelper::TimingStage::Initialization
    );

    const std::vector<double> nuclearPotential =
        NuclearPotential::calculate(
            molecule,
            grid
        );

    std::vector<double> alphaDensity;

    std::vector<double> betaDensity;

    std::vector<MolecularOrbital> previousAlphaOrbitals;

    std::vector<MolecularOrbital> previousBetaOrbitals;

    buildInitialMolecularDensities(
        grid,
        molecule,
        alphaElectrons,
        betaElectrons,
        alphaDensity,
        betaDensity,
        previousAlphaOrbitals,
        previousBetaOrbitals
    );

    timer.end(
        MolecularSCFHelper::TimingStage::Initialization
    );

    if (alphaDensity.size() != grid.getSize() ||
        betaDensity.size() != grid.getSize()) {

        throw std::runtime_error(
            "Las densidades moleculares iniciales tienen un tamano incorrecto."
        );
    }

    /*
     * ================================================================
     * 5. OCUPACIONES
     * ================================================================
     */

    const std::vector<int> alphaOccupations =
        MolecularSCFMath::buildSpinOccupations(
            alphaElectrons
        );

    const std::vector<int> betaOccupations =
        MolecularSCFMath::buildSpinOccupations(
            betaElectrons
        );

    /*
     * ================================================================
     * 6. ESTADO SCF
     * ================================================================
     */

    double previousEnergy =
        std::numeric_limits<double>::infinity();

    double previousDensityDifference =
        std::numeric_limits<double>::infinity();

    double mixing =
        DFTConstants::MIXING;

    MolecularResult molecularResult;

    molecularResult.electrons =
        electronCount;

    molecularResult.charge =
        charge;

    SCFResult& result =
        molecularResult.scf;

    MolecularSCFHelper::printSCFHeader();

    /*
     * ================================================================
     * 7. CICLO SCF
     * ================================================================
     */

    for (int iteration = 1;
         iteration <= DFTConstants::MAX_SCF_ITERATIONS;
         ++iteration) {

        const double iterationMixing =
            mixing;

        timer.begin(
            MolecularSCFHelper::TimingStage::TotalIterations
        );

        /*
         * ------------------------------------------------------------
         * 7.1 DENSIDAD ELECTRONICA TOTAL
         * ------------------------------------------------------------
         */

        timer.begin(
            MolecularSCFHelper::TimingStage::Density
        );

        std::vector<double> density(
            grid.getSize(),
            0.0
        );

        for (std::size_t i = 0;
             i < grid.getSize();
             ++i) {

            density[i] =
                alphaDensity[i] +
                betaDensity[i];
        }

        timer.end(
            MolecularSCFHelper::TimingStage::Density
        );

        /*
         * ------------------------------------------------------------
         * 7.2 POTENCIAL DE HARTREE
         * ------------------------------------------------------------
         */

        timer.begin(
            MolecularSCFHelper::TimingStage::Hartree
        );

        const std::vector<double> hartreePotential =
            calculateHartreePotential(
                grid,
                density
            );

        timer.end(
            MolecularSCFHelper::TimingStage::Hartree
        );

        /*
         * ------------------------------------------------------------
         * 7.3 POTENCIAL DE INTERCAMBIO-CORRELACION ALPHA
         * ------------------------------------------------------------
         */

        timer.begin(
            MolecularSCFHelper::TimingStage::ExchangeCorrelationAlpha
        );

        const std::vector<double> alphaXCPotential =
            calculateMolecularSpinExchangeCorrelationPotential(
                functional,
                grid,
                alphaDensity,
                betaDensity,
                0
            );

        timer.end(
            MolecularSCFHelper::TimingStage::ExchangeCorrelationAlpha
        );

        /*
         * ------------------------------------------------------------
         * 7.4 POTENCIAL DE INTERCAMBIO-CORRELACION BETA
         * ------------------------------------------------------------
         */

        std::vector<double> betaXCPotential;

        if (closedShell) {

            betaXCPotential =
                alphaXCPotential;
        }
        else {

            timer.begin(
                MolecularSCFHelper::TimingStage::ExchangeCorrelationBeta
            );

            betaXCPotential =
                calculateMolecularSpinExchangeCorrelationPotential(
                    functional,
                    grid,
                    alphaDensity,
                    betaDensity,
                    1
                );

            timer.end(
                MolecularSCFHelper::TimingStage::ExchangeCorrelationBeta
            );
        }

        /*
         * ------------------------------------------------------------
         * 7.5 POTENCIALES EFECTIVOS DE KOHN-SHAM
         *
         * V_KS = V_nuclear + V_Hartree + V_XC
         * ------------------------------------------------------------
         */

        timer.begin(
            MolecularSCFHelper::TimingStage::PotentialConstruction
        );

        std::vector<double> alphaPotential(
            grid.getSize(),
            0.0
        );

        std::vector<double> betaPotential(
            grid.getSize(),
            0.0
        );

        for (std::size_t i = 0;
             i < grid.getSize();
             ++i) {

            alphaPotential[i] =
                nuclearPotential[i] +
                hartreePotential[i] +
                alphaXCPotential[i];

            betaPotential[i] =
                nuclearPotential[i] +
                hartreePotential[i] +
                betaXCPotential[i];
        }

        timer.end(
            MolecularSCFHelper::TimingStage::PotentialConstruction
        );

        /*
         * ------------------------------------------------------------
         * 7.6 RESOLUCION DE KOHN-SHAM ALPHA
         * ------------------------------------------------------------
         */

        timer.begin(
            MolecularSCFHelper::TimingStage::OrbitalAlpha
        );

        const std::vector<MolecularOrbital> alphaOrbitals =
            MolecularSCFHelper::solveMolecularOrbitalsSilently(
                grid,
                alphaPotential,
                alphaOccupations.size(),
                alphaOccupations,
                SpinChannel::Alpha,
                previousAlphaOrbitals
            );

        timer.end(
            MolecularSCFHelper::TimingStage::OrbitalAlpha
        );

        /*
         * ------------------------------------------------------------
         * 7.7 RESOLUCION DE KOHN-SHAM BETA
         * ------------------------------------------------------------
         */

        std::vector<MolecularOrbital> betaOrbitals;

        if (closedShell) {

            betaOrbitals =
                MolecularSCFMath::convertOrbitalsToSpin(
                    alphaOrbitals,
                    SpinChannel::Beta
                );
        }
        else {

            timer.begin(
                MolecularSCFHelper::TimingStage::OrbitalBeta
            );

            betaOrbitals =
                MolecularSCFHelper::solveMolecularOrbitalsSilently(
                    grid,
                    betaPotential,
                    betaOccupations.size(),
                    betaOccupations,
                    SpinChannel::Beta,
                    previousBetaOrbitals
                );

            timer.end(
                MolecularSCFHelper::TimingStage::OrbitalBeta
            );
        }

        /*
         * ------------------------------------------------------------
         * 7.8 COMBINACION DE ORBITALES
         * ------------------------------------------------------------
         */

        std::vector<MolecularOrbital> orbitals;

        orbitals.reserve(
            alphaOrbitals.size() +
            betaOrbitals.size()
        );

        orbitals.insert(
            orbitals.end(),
            alphaOrbitals.begin(),
            alphaOrbitals.end()
        );

        orbitals.insert(
            orbitals.end(),
            betaOrbitals.begin(),
            betaOrbitals.end()
        );

        /*
         * ------------------------------------------------------------
         * 7.9 NUEVA DENSIDAD ELECTRONICA
         * ------------------------------------------------------------
         */

        timer.begin(
            MolecularSCFHelper::TimingStage::OutputDensity
        );

        const std::vector<double> outputAlphaDensity =
            calculateMolecularSpinDensity(
                alphaOrbitals,
                SpinChannel::Alpha
            );

        const std::vector<double> outputBetaDensity =
            calculateMolecularSpinDensity(
                betaOrbitals,
                SpinChannel::Beta
            );

        std::vector<double> oldDensity(
            grid.getSize(),
            0.0
        );

        std::vector<double> outputDensity(
            grid.getSize(),
            0.0
        );

        for (std::size_t i = 0;
             i < grid.getSize();
             ++i) {

            oldDensity[i] =
                alphaDensity[i] +
                betaDensity[i];

            outputDensity[i] =
                outputAlphaDensity[i] +
                outputBetaDensity[i];
        }

        timer.end(
            MolecularSCFHelper::TimingStage::OutputDensity
        );

        /*
         * ------------------------------------------------------------
         * 7.10 NUEVO POTENCIAL DE HARTREE
         *
         * Se utiliza para evaluar la energia total con la densidad
         * obtenida de los orbitales actuales.
         * ------------------------------------------------------------
         */

        timer.begin(
            MolecularSCFHelper::TimingStage::OutputHartree
        );

        const std::vector<double> outputHartreePotential =
            calculateHartreePotential(
                grid,
                outputDensity
            );

        timer.end(
            MolecularSCFHelper::TimingStage::OutputHartree
        );

        /*
         * ------------------------------------------------------------
         * 7.11 ENERGIA TOTAL
         * ------------------------------------------------------------
         */

        timer.begin(
            MolecularSCFHelper::TimingStage::TotalEnergy
        );

        const EnergyComponents energy =
            calculateMolecularTotalEnergy(
                functional,
                molecule,
                grid,
                outputAlphaDensity,
                outputBetaDensity,
                orbitals,
                outputHartreePotential,
                nuclearPotential
            );

        timer.end(
            MolecularSCFHelper::TimingStage::TotalEnergy
        );

        /*
         * ------------------------------------------------------------
         * 7.12 CRITERIOS DE CONVERGENCIA EN DENSIDAD Y ENERGIA
         * ------------------------------------------------------------
         */

        timer.begin(
            MolecularSCFHelper::TimingStage::DensityDifference
        );

        const double densityDifference =
            MolecularSCFMath::calculateMolecularDensityDifference(
                grid,
                oldDensity,
                outputDensity
            );

        const double spinDensityDifference =
            MolecularSCFMath::calculateMolecularSpinDensityDifference(
                grid,
                alphaDensity,
                betaDensity,
                outputAlphaDensity,
                outputBetaDensity
            );

        const double effectiveDensityDifference =
            std::max(
                densityDifference,
                spinDensityDifference
            );

        const double energyDifference =
            std::isfinite(previousEnergy)
                ? std::abs(
                    energy.total -
                    previousEnergy
                )
                : std::numeric_limits<double>::infinity();

        timer.end(
            MolecularSCFHelper::TimingStage::DensityDifference
        );

        /*
         * ------------------------------------------------------------
         * 7.13 RESIDUO DE KOHN-SHAM
         * ------------------------------------------------------------
         */

        timer.begin(
            MolecularSCFHelper::TimingStage::KSResidual
        );

        const double residual =
            MolecularSCFMath::calculateMaximumMolecularKSResidual(
                grid,
                alphaPotential,
                betaPotential,
                orbitals
            );

        timer.end(
            MolecularSCFHelper::TimingStage::KSResidual
        );

        /*
         * ------------------------------------------------------------
         * 7.14 ACTUALIZACION DEL RESULTADO FISICO
         * ------------------------------------------------------------
         */

        result.orbitals.clear();

        result.molecularOrbitals =
            orbitals;

        result.alphaDensity =
            outputAlphaDensity;

        result.betaDensity =
            outputBetaDensity;

        result.density =
            outputDensity;

        result.alphaEffectivePotential =
            alphaPotential;

        result.betaEffectivePotential =
            betaPotential;

        result.effectivePotential =
            alphaPotential;

        result.energy =
            energy;

        result.densityDifference =
            effectiveDensityDifference;

        result.energyDifference =
            energyDifference;

        result.maxKSResidual =
            residual;

        result.iterations =
            iteration;

        /*
         * ------------------------------------------------------------
         * 7.15 EVALUACION DE CONVERGENCIA
         * ------------------------------------------------------------
         */

        const bool energyConverged =
            energyDifference <
            DFTConstants::ENERGY_TOL;

        const bool densityConverged =
            effectiveDensityDifference <
            DFTConstants::DENSITY_TOL;

        const bool ksConverged =
            residual <
            DFTConstants::KS_RESIDUAL_TOL;

        /*
         * Datos para reporte.
         * No intervienen en las ecuaciones del SCF.
         */

        if (energyConverged) {

            convergenceAnalysis.lastEnergyCriterionIteration =
                iteration;
        }

        if (densityConverged) {

            convergenceAnalysis.lastDensityCriterionIteration =
                iteration;
        }

        if (ksConverged) {

            convergenceAnalysis.lastKSCriterionIteration =
                iteration;
        }

        convergenceAnalysis.finalEnergyDifference =
            energyDifference;

        convergenceAnalysis.finalDensityDifference =
            effectiveDensityDifference;

        convergenceAnalysis.finalKSResidual =
            residual;

        convergenceAnalysis.finalMixing =
            iterationMixing;

        convergenceAnalysis.minimumEnergyDifference =
            std::min(
                convergenceAnalysis.minimumEnergyDifference,
                energyDifference
            );

        convergenceAnalysis.minimumDensityDifference =
            std::min(
                convergenceAnalysis.minimumDensityDifference,
                effectiveDensityDifference
            );

        convergenceAnalysis.minimumKSResidual =
            std::min(
                convergenceAnalysis.minimumKSResidual,
                residual
            );

        const bool converged =
            energyConverged &&
            densityConverged &&
            ksConverged;

        /*
         * ------------------------------------------------------------
         * 7.16 CONVERGENCIA
         * ------------------------------------------------------------
         */

        if (converged) {

            result.converged =
                true;

            timer.end(
                MolecularSCFHelper::TimingStage::TotalIterations
            );

            MolecularSCFHelper::printSCFIteration(
                iteration,
                energy.total,
                energyDifference,
                effectiveDensityDifference,
                residual,
                iterationMixing,
                timer.get(
                    MolecularSCFHelper::TimingStage::TotalIterations
                ),
                alphaOrbitals,
                betaOrbitals,
                timer.get(
                    MolecularSCFHelper::TimingStage::OrbitalAlpha
                ),
                timer.get(
                    MolecularSCFHelper::TimingStage::OrbitalBeta
                )
            );

            break;
        }

        /*
         * ------------------------------------------------------------
         * 7.17 ESTADOS ANTERIORES PARA LA SIGUIENTE ITERACION
         * ------------------------------------------------------------
         */

        previousAlphaOrbitals =
            alphaOrbitals;

        if (!closedShell) {

            previousBetaOrbitals =
                betaOrbitals;
        }

        /*
         * ------------------------------------------------------------
         * 7.18 ADAPTACION DEL MIXING
         *
         * El valor utilizado en la iteracion actual es iterationMixing.
         * El nuevo valor se calcula para la siguiente iteracion.
         * ------------------------------------------------------------
         */

        timer.begin(
            MolecularSCFHelper::TimingStage::Mixing
        );

        if (std::isfinite(previousDensityDifference)) {

            if (effectiveDensityDifference >
                previousDensityDifference *
                OSCILLATION_FACTOR) {

                mixing =
                    std::max(
                        MIN_MIXING,
                        mixing *
                        MIXING_DECREASE
                    );
            }
            else if (effectiveDensityDifference <
                     previousDensityDifference) {

                mixing =
                    std::min(
                        MAX_MIXING,
                        mixing *
                        MIXING_INCREASE
                    );
            }
        }

        MolecularSCFMath::mixDensity(
            alphaDensity,
            outputAlphaDensity,
            iterationMixing
        );

        MolecularSCFMath::mixDensity(
            betaDensity,
            outputBetaDensity,
            iterationMixing
        );

        timer.end(
            MolecularSCFHelper::TimingStage::Mixing
        );

        previousDensityDifference =
            effectiveDensityDifference;

        previousEnergy =
            energy.total;

        timer.end(
            MolecularSCFHelper::TimingStage::TotalIterations
        );

        MolecularSCFHelper::printSCFIteration(
            iteration,
            energy.total,
            energyDifference,
            effectiveDensityDifference,
            residual,
            iterationMixing,
            timer.get(
                MolecularSCFHelper::TimingStage::TotalIterations
            ),
            alphaOrbitals,
            betaOrbitals,
            timer.get(
                MolecularSCFHelper::TimingStage::OrbitalAlpha
            ),
            timer.get(
                MolecularSCFHelper::TimingStage::OrbitalBeta
            )
        );
    }

    /*
     * ================================================================
     * 8. FINALIZACION
     * ================================================================
     */

    timer.stop();

    MolecularSCFHelper::printSCFSummary(
        result.iterations,
        result.converged,
        result.energy.total
    );

    MolecularSCFHelper::printProfiling(
        timer
    );

    MolecularSCFHelper::printConvergenceAnalysis(
        convergenceAnalysis
    );

    MolecularSCFHelper::printSCFEndLine();

    return molecularResult;
}