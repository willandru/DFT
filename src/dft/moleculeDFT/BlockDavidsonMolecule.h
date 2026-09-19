#pragma once

#include "EigenvalueSolverMolecule.h"

#include <cstddef>
#include <vector>

class CartesianGrid;

/**
 * @brief Resuelve simultáneamente un conjunto de orbitales moleculares
 *        mediante el método Block-Davidson.
 *
 * A diferencia del Davidson secuencial, los orbitales solicitados se
 * representan como un bloque y se optimizan conjuntamente dentro de un
 * subespacio de búsqueda común.
 *
 * El solver no modifica el potencial efectivo ni realiza ninguna parte
 * del ciclo SCF. Su única responsabilidad es resolver el problema
 *
 *      H_KS psi_i = epsilon_i psi_i
 *
 * para los orbitales moleculares solicitados.
 *
 * @param grid
 *      Malla cartesiana sobre la que está discretizado el Hamiltoniano.
 *
 * @param effectivePotential
 *      Potencial efectivo evaluado en todos los puntos de la malla.
 *
 * @param numberOfOrbitals
 *      Número de orbitales moleculares que deben resolverse.
 *
 * @param occupations
 *      Ocupación electrónica asociada a cada orbital solicitado.
 *
 * @param spin
 *      Canal de espín que se está resolviendo.
 *
 * @param initialOrbitals
 *      Orbitales iniciales disponibles para construir el subespacio
 *      inicial. Puede estar vacío.
 *
 * @param maxIterations
 *      Número máximo de iteraciones del proceso Block-Davidson.
 *
 * @return
 *      Vector con los orbitales moleculares resueltos.
 *
 * @throws std::invalid_argument
 *      Si los tamaños de los argumentos son incompatibles o los datos
 *      de entrada no permiten construir el problema.
 *
 * @throws std::runtime_error
 *      Si el proceso Block-Davidson no consigue converger dentro del
 *      número máximo de iteraciones.
 */
std::vector<MolecularOrbital> solveMolecularOrbitalsBlockDavidson(
    const CartesianGrid& grid,
    const std::vector<double>& effectivePotential,
    std::size_t numberOfOrbitals,
    const std::vector<int>& occupations,
    SpinChannel spin,
    const std::vector<MolecularOrbital>& initialOrbitals,
    std::size_t maxIterations
);