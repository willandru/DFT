#pragma once

namespace PhysicalConstants
{
    inline constexpr double PI = 3.141592653589793238462643383279502884;

    // Atomic units
    inline constexpr double BOHR_TO_ANGSTROM = 0.529177210903;
    inline constexpr double ANGSTROM_TO_BOHR = 1.88972612462577;

    inline constexpr double HARTREE_TO_EV = 27.211386245988;
    inline constexpr double EV_TO_HARTREE = 1.0 / HARTREE_TO_EV;

    // SI
    inline constexpr double BOHR_RADIUS_M = 5.29177210903e-11;
    inline constexpr double HARTREE_ENERGY_J = 4.3597447222060e-18;
    inline constexpr double ELEMENTARY_CHARGE_C = 1.602176634e-19;
    inline constexpr double ELECTRON_MASS_KG = 9.1093837139e-31;
    inline constexpr double PROTON_MASS_KG = 1.67262192595e-27;
    inline constexpr double NEUTRON_MASS_KG = 1.67492750056e-27;
    inline constexpr double BOLTZMANN_CONSTANT_J_K = 1.380649e-23;
    inline constexpr double AVOGADRO_CONSTANT = 6.02214076e23;
}