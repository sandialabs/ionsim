#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

""" Counter-rotating laser couplings (include_counter_rotating=True). """

import dataclasses
import unittest

import numpy as np
from scipy import integrate

from ionsim.basis import StandardBasis
from ionsim.degree_of_freedom import AtomicStructure, MotionalMode
from ionsim.hamiltonian import Hamiltonian
from ionsim.ionsim_error import IonSimError
from ionsim.laser import Laser, Polarization

TPI = 2. * np.pi
X, Y, Z = np.eye(3)


def yb_s_p_f1(reverse: bool = False) -> AtomicStructure:
    """ 171Yb+ S1/2 F=1 and P1/2 F=1 (all mF), optionally with the excited levels first in the basis. """
    atom = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2'],
                                       level_names=['S1/2,1,-1', 'S1/2,1,0', 'S1/2,1,1', 'P1/2,1,-1', 'P1/2,1,0', 'P1/2,1,1'])
    if reverse:
        atom = dataclasses.replace(atom, energy_levels=list(reversed(atom.energy_levels)))
    return atom


class TestRealField(unittest.TestCase):
    """ With the counter-rotating terms, <e|H|g>(t) is the matrix element of the real field
            (1/2) [ (d.eps) e^{i phi} e^{-i omega t} + (d.eps*) e^{-i phi} e^{+i omega t} ],
        where the Cartesian vector d (Rabi frequency per unit polarization) is extracted from co-rotating Rabi frequencies
        of x, y and z polarized beams. This checks the conjugate polarization, phase, rates and both level orders
        independently of how the counter-rotating terms are built. """

    POLARIZATION = Polarization(np.array([np.cos(0.4), 1j * np.sin(0.4) * np.exp(0.9j), 0.]), Z)  # elliptical, along z
    PHASE = 0.7
    TIMES = (0., 1.3e-16, 7.7e-16, 2.1e-15, 5.3e-14)  # lab frame: sample the optical phase

    def dipole_vector(self, laser, g, e):
        """ d with laser.rabi_frequency(g, e) = d . eps for any polarization eps at the same field amplitude. """
        components = []
        for pol, n_hat in ((X, Z), (Y, Z), (Z, X)):
            basis_laser = dataclasses.replace(laser, polarization=Polarization(pol.astype(complex), n_hat), propagation_vector=n_hat)
            components.append(basis_laser.rabi_frequency(g, e, 1))
        return np.array(components)

    def check(self, reverse: bool, modulation_functions: dict | None = None):
        atom = yb_s_p_f1(reverse)
        basis = StandardBasis([atom])
        ground = [l for l in atom.energy_levels if l.term_symbol == 'S1/2']
        excited = [l for l in atom.energy_levels if l.term_symbol == 'P1/2']
        frequency = excited[0].energy - ground[0].energy + TPI * 3e9
        laser = Laser.gaussian_from_frequency(frequency, 1e-6, 20e-6, Z, self.POLARIZATION, self.PHASE,
                                              modulation_functions=modulation_functions)
        operators = laser.build_carrier_coupling_operators(basis, [atom], ground, excited, 1, include_counter_rotating=True)
        self.assertEqual(len(operators), 2 * len(laser.build_carrier_coupling_operators(basis, [atom], ground, excited, 1)))
        H_of_t = Hamiltonian(basis, operators, [0.] * len(basis.states)).hamiltonian_function
        amplitude = (modulation_functions or {}).get('amplitude', lambda t: 1.)
        phase = (modulation_functions or {}).get('phase', lambda t: 0.)
        offset = (modulation_functions or {}).get('frequency')
        eps = self.POLARIZATION.vector
        checked = 0
        for g in ground:
            for e in excited:
                d = self.dipole_vector(laser, g, e)
                co, counter = d @ eps, d @ np.conj(eps)
                if abs(co) < 1. and abs(counter) < 1.:
                    continue
                i_e, i_g = atom.energy_levels.index(e), atom.energy_levels.index(g)
                for t in self.TIMES:
                    optical = frequency * t + (integrate.quad(offset, 0., t)[0] if offset else 0.)
                    total_phase = self.PHASE + phase(t)
                    expected = 0.5 * amplitude(t) * (co * np.exp(1j * total_phase - 1j * optical)
                                                     + counter * np.exp(-1j * total_phase + 1j * optical))
                    H = np.asarray(H_of_t(t))
                    self.assertAlmostEqual(abs(H[i_e, i_g] - expected) / max(abs(co), abs(counter)), 0., delta=1e-6,
                                           msg=f'{g.name} -> {e.name}, t = {t}')
                    self.assertAlmostEqual(abs(H[i_g, i_e] - np.conj(expected)) / max(abs(co), abs(counter)), 0., delta=1e-6)
                checked += 1
        self.assertGreater(checked, 3)

    def test_real_field_ground_first(self):
        self.check(reverse=False)

    def test_real_field_excited_first(self):
        self.check(reverse=True)

    def test_real_field_with_modulation(self):
        """ Amplitude, phase and frequency modulation of the real field: A(t) cos(omega t + int delta - phi - phi(t)). """
        modulations = {'amplitude': lambda t: 0.6 + 0.3 * np.sin(3e14 * t), 'phase': lambda t: 2e14 * t, 'frequency': lambda t: TPI * 40e12}
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                self.check(reverse, modulations)


class TestCounterRotatingCouplings(unittest.TestCase):

    def test_default_is_unchanged(self):
        atom = yb_s_p_f1()
        basis = StandardBasis([atom])
        ground, excited = atom.energy_levels[:3], atom.energy_levels[3:]
        laser = Laser.gaussian_from_frequency(excited[0].energy - ground[0].energy, 1e-6, 20e-6, Z, Polarization.circular(Z, '+'), 0.)
        default = laser.build_carrier_coupling_operators(basis, [atom], ground, excited, 1)
        explicit = laser.build_carrier_coupling_operators(basis, [atom], ground, excited, 1, include_counter_rotating=False)
        self.assertEqual(len(default), len(explicit))
        for a, b in zip(default, explicit):
            np.testing.assert_array_equal(a.static_matrix.toarray(), b.static_matrix.toarray())
        with self.assertRaises(IonSimError):
            laser.build_carrier_coupling_operators(basis, [atom], ground, excited, 1, include_counter_rotating='yes')

    def test_sigma_plus_counter_rotating_term_drives_delta_m_minus_one(self):
        """ eps* of a sigma+ beam is sigma-: the counter-rotating couplings connect m_e = m_g - 1. """
        atom = yb_s_p_f1()
        basis = StandardBasis([atom])
        ground, excited = atom.energy_levels[:3], atom.energy_levels[3:]
        laser = Laser.gaussian_from_frequency(excited[0].energy - ground[0].energy, 1e-6, 20e-6, Z, Polarization.circular(Z, '+'), 0.)
        n_co = len(laser.build_carrier_coupling_operators(basis, [atom], ground, excited, 1))
        operators = laser.build_carrier_coupling_operators(basis, [atom], ground, excited, 1, include_counter_rotating=True)
        for index, operator in enumerate(operators):
            expected_dm = 1 if index < n_co else -1
            for coupling in operator.couplings:
                levels = {coupling.row_state.components[0], coupling.column_state.components[0]}
                g = next(l for l in levels if l.term_symbol == 'S1/2')
                e = next(l for l in levels if l.term_symbol == 'P1/2')
                self.assertEqual(round(e.mf - g.mf), expected_dm)
        # The conjugate of the sigma+ Jones vector is the sigma- one: Omega~ of sigma+ equals Omega of sigma-, pair by pair.
        sigma_minus = dataclasses.replace(laser, polarization=Polarization.circular(Z, '-'))
        scale = abs(laser.rabi_frequency(ground[1], excited[2], 1))
        for g in ground:
            for e in excited:
                self.assertAlmostEqual(abs(laser.rabi_frequency(g, e, 1, counter_rotating=True) - sigma_minus.rabi_frequency(g, e, 1)) / scale,
                                       0., places=12)

    def test_rates_and_sidebands(self):
        """ For a real (linear) polarization and zero phase: counter-rotating carrier elements equal the co-rotating ones,
            sideband elements change sign (eta -> -eta, also with Debye-Waller factors), and every rate is negated. """
        atom = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2'], level_names=['S1/2,0,0', 'P1/2,1,0'])
        mode = MotionalMode.from_frequency(TPI * 2e6, 4)
        g, e = atom.energy_levels
        laser = Laser.gaussian_from_frequency(e.energy - g.energy, 1e-6, 20e-6, X, Polarization.linear(X, 0., ref_axis=Z), 0.)
        for basis in (StandardBasis([atom, mode]), StandardBasis([mode, atom])):
            for builder, sign in (('carrier', 1.), ('red', -1.), ('blue', -1.)):
                for debye_waller in (False, True):
                    if builder == 'carrier':
                        if debye_waller:
                            continue
                        operators = laser.build_carrier_coupling_operators(basis, [atom], [g], [e], 1, include_counter_rotating=True)
                    else:
                        operators = laser.build_sideband_coupling_operators(builder, basis, [atom], [mode], [g], [e], 1, 0.08,
                                                                            debye_waller=debye_waller, include_counter_rotating=True)
                    with self.subTest(order=[d.name for d in basis.degrees_of_freedom], builder=builder, debye_waller=debye_waller):
                        self.assertEqual(len(operators), 2)
                        co, counter = operators
                        np.testing.assert_allclose(counter.static_matrix.toarray(), sign * co.static_matrix.toarray(), atol=1e-6)
                        np.testing.assert_allclose(counter.rate_matrix.toarray(), -co.rate_matrix.toarray())

    def test_e2_counter_rotating_rabi_frequency(self):
        """ E2: the counter-rotating term has the conjugate polarization and the opposite field gradient (-i k). For a real
            polarization this is -Omega; in general it is the co-rotating Rabi frequency of eps* with k -> -k. """
        ca = AtomicStructure.from_species(species='40Ca+', term_symbols=['S1/2', 'D5/2'],
                                          quantum_numbers=[{'n': 4, 'l': 0, 'mj': 0.5}, {'n': 3, 'l': 2, 'j': 2.5, 'mj': 1.5}])
        g, e = ca.energy_levels
        n_hat = (X + Z) / np.sqrt(2.)
        linear = Polarization.linear(n_hat, 0.3)
        laser = Laser.gaussian_from_frequency(e.energy - g.energy, 1e-3, 20e-6, n_hat, linear, 0.)
        co, counter = laser.rabi_frequency(g, e, 2), laser.rabi_frequency(g, e, 2, counter_rotating=True)
        self.assertGreater(abs(co), 1.)
        self.assertAlmostEqual(abs(counter + co) / abs(co), 0., places=12)
        elliptical = Polarization((np.cos(0.3) * linear.vector + 1j * np.sin(0.3) * np.cross(n_hat, linear.vector)), n_hat)
        laser = dataclasses.replace(laser, polarization=elliptical)
        conjugate = dataclasses.replace(laser, polarization=Polarization(np.conj(elliptical.vector), n_hat))
        self.assertAlmostEqual(abs(laser.rabi_frequency(g, e, 2, counter_rotating=True) + conjugate.rabi_frequency(g, e, 2))
                               / abs(conjugate.rabi_frequency(g, e, 2)), 0., places=12)


if __name__ == '__main__':
    unittest.main()
