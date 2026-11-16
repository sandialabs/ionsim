#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

import inspect
import itertools
import unittest
import warnings

import numpy as np
from scipy import constants as const
from scipy.special import genlaguerre

from ionsim.degree_of_freedom import AtomicStructure, MotionalMode
from ionsim.basis import StandardBasis
from ionsim.laser import Laser, SidebandType, GaussianBeam, PlaneWave
from ionsim.polarization import Polarization, LinearPolarization, CircularPolarization, SphericalPolarization
from ionsim.hamiltonian import Hamiltonian
from ionsim.atomic_internal_energy_level import (compute_coupling_amplitude_between_atomic_levels, compute_multipole_amplitude,
                                                 reduced_dipole_matrix_element)
from ionsim.ionsim_error import IonSimError
from ionsim.trapped_ion_mode_analysis import IonChain

TPI = 2. * np.pi
X, Y, Z = np.eye(3)


# ---- shared builders and helpers ----
def Yb_s_to_p(ground_first: bool = True):
    """171Yb+ |S1/2, F=1, mF=0> and |P1/2, F=1, mF=1> (E1, 369.5 nm, driven by sigma+ along z), in either level order."""
    qn_g, qn_e = {'n': 6, 'l': 0, 'f': 1, 'mf': 0}, {'n': 6, 'l': 1, 'f': 1, 'mf': 1}
    atom = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2'], quantum_numbers=[qn_g, qn_e] if ground_first else [qn_e, qn_g])
    g = next(level for level in atom.energy_levels if level.term_symbol == 'S1/2')
    e = next(level for level in atom.energy_levels if level.term_symbol == 'P1/2')
    return atom, g, e

def Ca_s_to_d(name: str = 'ion', ground_first: bool = True):
    """40Ca+ |S1/2, mJ=1/2> and |D5/2, mJ=3/2> (E2, 729 nm), in either level order. Driven (Delta m = +1) by light along x
    polarized along z, or by light in the xz plane polarized along y."""
    qn_g, qn_e = {'n': 4, 'l': 0, 'mj': 0.5}, {'n': 3, 'l': 2, 'j': 2.5, 'mj': 1.5}
    atom = AtomicStructure.from_species(species='40Ca+', term_symbols=['S1/2', 'D5/2'], name=name,
                                        quantum_numbers=[qn_g, qn_e] if ground_first else [qn_e, qn_g])
    g = next(level for level in atom.energy_levels if level.term_symbol == 'S1/2')
    e = next(level for level in atom.energy_levels if level.term_symbol == 'D5/2')
    return atom, g, e

def state_index(basis, level, n: int | None = None) -> int:
    """Basis index of the state containing level (and, if n is given, a mode in Fock state n)."""
    return next(i for i, state in enumerate(basis.states) if level in state.components
                and (n is None or any(getattr(c, 'fock_number', None) == n for c in state.components)))

def evolve(test: unittest.TestCase, basis, operators, initial_index: int, duration: float):
    """Final state after evolving basis state initial_index for duration in the frame rotating_frame_energies = -E.
    Dense output: odeint's step limit applies per output interval, and when it is exceeded odeint only warns and returns an
    invalid state, so the norm is checked (odeint's default rtol allows ~1e-6 drift)."""
    hamiltonian = Hamiltonian(basis, operators, [-state.energy for state in basis.states])
    psi0 = np.zeros(len(basis.states), dtype=complex)
    psi0[initial_index] = 1.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, psis = hamiltonian.evolve_wavefunction(psi0, duration, np.linspace(0., duration, 2001))
    test.assertAlmostEqual(np.linalg.norm(psis[-1]), 1., delta=1e-5)
    return psis[-1]

def coupling_elements(basis, operators) -> dict:
    """{(row index, column index): summed strength} over all couplings of the operators."""
    elements = {}
    for operator in operators:
        for coupling in operator.couplings:
            key = (basis.states.index(coupling.row_state), basis.states.index(coupling.column_state))
            elements[key] = elements.get(key, 0.) + coupling.strength
    return elements


class TestLaserConstructors(unittest.TestCase):

    def setUp(self):
        self.wavelength = 729e-9  # m; arbitrary optical wavelength
        self.power, self.waist = 5e-3, 15e-6  # W, m; arbitrary beam parameters

    def test_gaussian_from_frequency_matches_from_wavelength(self):
        frequency = 2 * np.pi * const.c / self.wavelength
        pol = LinearPolarization()
        from_freq = Laser.gaussian_from_frequency(frequency, self.power, self.waist, Z, pol, 0.)
        from_wl = Laser.gaussian_from_wavelength(self.wavelength, self.power, self.waist, Z, pol, 0.)
        self.assertIsInstance(from_freq.beam_profile, GaussianBeam)
        self.assertAlmostEqual(from_freq.wavelength_m / from_wl.wavelength_m, 1., places=12)
        self.assertAlmostEqual(from_freq.beam_profile.waist_m, self.waist)
        self.assertAlmostEqual(from_freq.peak_electric_field_magnitude_V_per_m / from_wl.peak_electric_field_magnitude_V_per_m, 1., places=12)

    def test_peak_intensity(self):
        """Gaussian peak intensity I0 = 2 P / (pi w^2)."""
        laser = Laser.gaussian_from_wavelength(self.wavelength, self.power, self.waist, Z, LinearPolarization(), 0.)
        self.assertAlmostEqual(laser.peak_intensity / (2 * self.power / (np.pi * self.waist**2)), 1., places=12)

    def test_peak_field_and_detuning(self):
        """556 nm, 1 mW, waist 0.5 mm: peak field ~1.39 kV/m. Detuning from 171Yb 1S0 -> 3P1 F=3/2 is negative (-195 GHz)
        because the F=3/2 hyperfine level lies above the fine-structure energy."""
        yb = AtomicStructure.from_species(species='171Yb', term_symbols=['S0', 'P1'], level_names=['S0,1/2,1/2', 'P1,3/2,3/2'])
        g, e = yb.energy_levels
        laser = Laser.gaussian_from_wavelength(556e-9, 1e-3, 0.5e-3, Z, LinearPolarization(angle=0.), np.pi)
        self.assertAlmostEqual(laser.peak_electric_field_magnitude_V_per_m * 1e-3, 1.3851612653205, places=10)
        self.assertAlmostEqual(laser.beam_profile.peak_electric_field_magnitude_V_per_m(1e-3) * 1e-3, 1.3851612653205, places=10)
        detuning = laser.detuning_from_level_transition(g, e)
        self.assertEqual(detuning, laser.detuning_from_transition_frequency(e.energy - g.energy))
        self.assertAlmostEqual(detuning, TPI * const.c / laser.wavelength_m - (e.energy - g.energy), places=10)
        self.assertAlmostEqual(detuning / TPI / 1e9, -195.33742087632817, places=8)


class TestPolarizationAndQuantizationAxis(unittest.TestCase):
    """The laser resolves its polarization spec against its propagation direction; couplings depend on the quantization
    axis only through the polarization's spherical components. (The polarization classes alone: test_polarization.py.)"""

    def test_ca40_729nm_coupling(self):
        """40Ca+ S1/2 -> D5/2 (E2) with n along x and polarization along z: |S1/2, 1/2> couples only to mJ = 1/2 +/- 1."""
        ca = AtomicStructure.from_species(species='40Ca+', term_symbols=['S1/2', 'D5/2'])
        ground = [level for level in ca.energy_levels if level.name == 'S1/2,1/2']
        excited = [level for level in ca.energy_levels if level.term_symbol == 'D5/2']
        laser = Laser.gaussian_from_wavelength(729e-9, 1e-3, 10e-6, X, Polarization(Z), 0.)
        operators = laser.build_carrier_coupling_operators(StandardBasis([ca]), [ca], ground, excited, 2, Z)
        coupled = sorted(next(state.name for state in (op.couplings[0].row_state, op.couplings[0].column_state) if 'D5/2' in state.name)
                         for op in operators)
        self.assertEqual(coupled, ['D5/2,-1/2', 'D5/2,3/2'])

    def test_laser_resolves_its_polarization_spec(self):
        """The laser's direction (any length) sets k and resolves the polarization, which must be transverse to it
        (n_hat . eps = 0); an invalid handedness is rejected when the spec is built, and other types by the laser."""
        laser = Laser.gaussian_from_wavelength(729e-9, 1e-3, 10e-6, np.array([2., 0., 0.]), Polarization(np.array([0., 3., 4.j])))
        np.testing.assert_allclose(laser.wavevector, X * TPI / 729e-9)
        np.testing.assert_allclose(laser.polarization.jones_vector, np.array([0., 0.6, 0.8j]))
        with self.subTest('not transverse'), self.assertRaises(IonSimError):
            Laser.gaussian_from_wavelength(729e-9, 1e-3, 10e-6, X, Polarization(X + Y))
        with self.subTest('pure sigma+ about z on a beam along x'), self.assertRaises(IonSimError):
            Laser.gaussian_from_wavelength(729e-9, 1e-3, 10e-6, X, SphericalPolarization({1: 1.}, Z))
        with self.subTest('handedness'), self.assertRaises(IonSimError):
            CircularPolarization('left')
        with self.subTest('not one of the polarization types'), self.assertRaises(IonSimError):
            Laser.gaussian_from_wavelength(729e-9, 1e-3, 10e-6, X, np.array([0., 0., 1.]))

    def test_rotating_everything_preserves_rabi_frequencies(self):
        """Rotating k, the polarization and the quantization axis together leaves every |Omega| unchanged (E1 and E2)."""
        rotation = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))[0]   # arbitrary orthogonal matrix
        rotation *= np.linalg.det(rotation)                                        # proper rotation
        yb = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2'],
                                          level_names=['S1/2,1,0', 'P1/2,1,-1', 'P1/2,0,0', 'P1/2,1,1'])
        ca = AtomicStructure.from_species(species='40Ca+', term_symbols=['S1/2', 'D5/2'])
        vector = np.array([0.3, 0.5 + 0.2j, -0.4j])                               # arbitrary elliptical polarization
        for atom, order, wavelength, n_hat in ((yb, 1, 369.5e-9, Z), (ca, 2, 729e-9, np.array([1., 0., 1.]) / np.sqrt(2.))):
            g = atom.energy_levels[0]
            excited = [level for level in atom.energy_levels if level.term_symbol != g.term_symbol]
            transverse = vector - np.dot(vector, n_hat) * n_hat
            def rabis(r):
                laser = Laser.gaussian_from_wavelength(wavelength, 1e-3, 10e-6, r @ n_hat, Polarization(r @ transverse), 0.)
                return np.array([abs(laser.rabi_frequency(g, e, order, r @ Z)) for e in excited])
            with self.subTest(multipole_order=order):
                reference, rotated = rabis(np.eye(3)), rabis(rotation)
                self.assertGreater(np.count_nonzero(reference > 1e-6 * reference.max()), 1)
                np.testing.assert_allclose(rotated, reference, rtol=1e-10, atol=1e-10 * reference.max())

    def test_reversed_quantization_axis_swaps_sigma_plus_and_minus(self):
        """sigma+ light along z is sigma- relative to a -z quantization axis: |F=1, mF=0> couples to mF' = -1 instead of +1,
        in the carrier and sideband builders and the target-Rabi constructor."""
        yb = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2'], level_names=['S1/2,1,0', 'P1/2,1,-1', 'P1/2,1,1'])
        mode = MotionalMode.from_frequency(TPI * 1e6, 3)
        basis = StandardBasis([yb, mode])
        g, minus, plus = yb.energy_levels
        sigma_plus = CircularPolarization('+')
        laser = Laser.gaussian_from_wavelength(369.5e-9, 1e-3, 20e-6, Z, sigma_plus, 0.)
        def coupled(operators):
            return sorted({state.components[0].name for operator in operators for coupling in operator.couplings
                           for state in (coupling.row_state, coupling.column_state) if state.components[0] is not g})
        for axis, expected in ((Z, ['P1/2,1,1']), (-Z, ['P1/2,1,-1'])):
            with self.subTest(quantization_axis=axis):
                self.assertEqual(coupled(laser.build_carrier_coupling_operators(basis, [yb], [g], [minus, plus], 1, axis)), expected)
                self.assertEqual(coupled(laser.build_sideband_coupling_operators(SidebandType.RED, basis, [yb], [mode], [g], [minus, plus], 1, 0.1, axis)), expected)
        Laser.from_target_rabi_frequency(TPI * 1e6, minus.energy - g.energy, g, minus, 1, Z, sigma_plus, -Z)
        with self.assertRaises(IonSimError):   # sigma+ relative to z does not drive mF = 0 -> -1
            Laser.from_target_rabi_frequency(TPI * 1e6, minus.energy - g.energy, g, minus, 1, Z, sigma_plus, Z)

    def test_quantization_axis_is_required(self):
        """The couplings depend strongly on the quantization axis, so no Laser method gives it a default value (polarization
        methods: test_polarization.py)."""
        for function in (Laser.rabi_frequency, Laser.build_atom_laser_ge_coupling_matrix, Laser.build_carrier_coupling_operators,
                         Laser.build_sideband_coupling_operators, Laser.build_sideband_coupling_operators_from_trap,
                         Laser.from_target_rabi_frequency, Laser.gaussian_from_target_rabi_frequency):
            with self.subTest(function=function.__qualname__):
                self.assertIs(inspect.signature(function).parameters['quantization_axis'].default, inspect.Parameter.empty)
        _, g, e = Yb_s_to_p()
        with self.assertRaises(TypeError):
            Laser.gaussian_from_wavelength(369.5e-9, 1e-3, 20e-6, Z, CircularPolarization('+'), 0.).rabi_frequency(g, e, 1)


class TestLaserDynamicsInInteractionPicture(unittest.TestCase):
    """ Two-level dynamics driven by Laser-built carrier couplings, in the frame rotating_frame_energies = -E, for either
        level order in the basis. The lasers are built for a target Rabi frequency (plane wave, sigma+ along z).

        These check the physics (detuning value and sign), not only that a resonant coupling is static.
    """
    RABI = TPI * 1e6   # rad/s; arbitrary, far below the optical frequency

    def laser(self, g, e, detuning=0., **kwargs):
        return Laser.from_target_rabi_frequency(self.RABI, e.energy - g.energy + detuning, g, e, 1, Z, CircularPolarization('+'), Z, **kwargs)

    def ground_population(self, laser, ground_first, duration, frequency_shift=0.):
        atom, g, e = Yb_s_to_p(ground_first)
        basis = StandardBasis([atom])
        operators = laser.build_carrier_coupling_operators(basis, [atom], [g], [e], 1, Z, frequency_shift_angular=frequency_shift)
        psi = evolve(self, basis, operators, state_index(basis, g), duration)
        return psi[state_index(basis, g)]

    def test_resonant_pi_pulse(self):
        """A resonant laser built for Omega transfers |g> -> |e> in pi / Omega."""
        for ground_first in (True, False):
            with self.subTest(ground_first=ground_first):
                _, g, e = Yb_s_to_p(ground_first)
                self.assertLess(abs(self.ground_population(self.laser(g, e), ground_first, np.pi / self.RABI))**2, 1e-6)

    def test_detuning_is_compensated_by_frequency_shift_and_modulation(self):
        """A laser detuned by +delta driven with frequency_shift = -delta (or a constant 'frequency' modulation of -delta)
        is resonant again: a pi pulse transfers all population. Without the compensation it does not."""
        delta = 2. * self.RABI  # rad/s; large enough that an uncompensated pi pulse clearly fails
        pi_time = np.pi / self.RABI
        for ground_first in (True, False):
            _, g, e = Yb_s_to_p(ground_first)
            detuned = self.laser(g, e, delta)
            modulated = self.laser(g, e, delta, frequency_modulation_function=lambda t: -delta)
            with self.subTest(ground_first=ground_first, case='uncompensated'):
                self.assertGreater(abs(self.ground_population(detuned, ground_first, pi_time))**2, 0.1)
            with self.subTest(ground_first=ground_first, case='frequency_shift'):
                self.assertLess(abs(self.ground_population(detuned, ground_first, pi_time, frequency_shift=-delta))**2, 1e-6)
            with self.subTest(ground_first=ground_first, case='frequency modulation'):
                # the 'frequency' phase is integrated numerically
                self.assertLess(abs(self.ground_population(modulated, ground_first, pi_time))**2, 1e-4)

    def test_constant_phase_modulation_matches_static_phase(self):
        """A constant 'phase' modulation gives the same coupling as the same static laser phase."""
        phase = 0.83   # rad; arbitrary
        for ground_first in (True, False):
            with self.subTest(ground_first=ground_first):
                atom, g, e = Yb_s_to_p(ground_first)
                basis = StandardBasis([atom])
                frame = [-state.energy for state in basis.states]
                def coupling(laser, t):
                    operators = laser.build_carrier_coupling_operators(basis, [atom], [g], [e], 1, Z)
                    return np.asarray(Hamiltonian(basis, operators, frame).hamiltonian_function(t))[0, 1]
                static = self.laser(g, e, phase=phase)
                modulated = self.laser(g, e, phase_modulation_function=lambda t: phase)
                for t in (0., 1e-9, 2.3e-9):   # s; arbitrary
                    self.assertLess(abs(coupling(modulated, t) - coupling(static, t)) / abs(coupling(static, t)), 1e-7)

    def test_light_shift_sign(self):
        """A far red-detuned laser (omega < omega_0) lowers |g> by |Omega|^2 / (4 |delta|): in the interaction picture the
        ground amplitude acquires the phase +|Omega|^2 t / (4 |delta|)."""
        delta = -50. * self.RABI  # red detuning
        duration = 40. * np.pi / self.RABI
        expected_phase = -(self.RABI**2 / (4. * delta)) * duration    # = -LS t with LS = Omega^2 / (4 delta) < 0
        self.assertGreater(expected_phase, 0.)
        for ground_first in (True, False):
            with self.subTest(ground_first=ground_first):
                _, g, e = Yb_s_to_p(ground_first)
                amplitude = self.ground_population(self.laser(g, e, delta), ground_first, duration)
                self.assertAlmostEqual(np.angle(amplitude), expected_phase, delta=0.05 * expected_phase)


class TestLaserMultipleAtoms(unittest.TestCase):
    """Coupling operators for several atoms, including separately built (distinct but identical) atoms."""

    def setUp(self):
        levels = ['S1/2,1,0', 'P1/2,1,-1', 'P1/2,1,0', 'P1/2,1,1']
        self.atom_a = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2'], level_names=levels, name='a')
        self.atom_b = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2'], level_names=levels, name='b')
        self.basis = StandardBasis([self.atom_a, self.atom_b])
        wavelength = 369.5e-9; power = 1e-3; waist = 20e-6; phase = 0.
        self.laser = Laser.gaussian_from_wavelength(wavelength, power, waist, Z, CircularPolarization('+'), phase)
        # Levels taken from atom a only; sigma+ light couples |S1/2,1,0> to |P1/2,1,1> only.
        self.ground = self.atom_a.energy_levels[:1]
        self.excited = self.atom_a.energy_levels[1:]

    def addressed_atoms(self, operators):
        """Index of the atom each operator acts on (the basis component that differs between its row and column states)."""
        indices = []
        for operator in operators:
            coupling = operator.couplings[0]
            changed = [k for k, (row, column) in enumerate(zip(coupling.row_state.components, coupling.column_state.components)) if row is not column]
            self.assertEqual(len(changed), 1)
            indices.append(changed[0])
        return indices

    def test_all_atoms_by_default(self):
        """Both atoms are addressed, with the same strengths: 4 basis states of the other atom x (|g><e| and |e><g|)."""
        operators = self.laser.build_carrier_coupling_operators(self.basis, None, self.ground, self.excited, 1, Z)
        self.assertEqual(sorted(self.addressed_atoms(operators)), [0, 1])
        strengths = [sorted(abs(coupling.strength) for coupling in operator.couplings) for operator in operators]
        self.assertEqual(len(strengths[0]), 8)
        np.testing.assert_allclose(strengths[0], strengths[1], rtol=1e-12)

    def test_selected_atom_with_levels_from_another_atom(self):
        operators = self.laser.build_carrier_coupling_operators(self.basis, [self.atom_b], self.ground, self.excited, 1, Z)
        self.assertEqual(self.addressed_atoms(operators), [1])


class TestLaserSidebandCouplings(unittest.TestCase):
    """ Red/blue sideband couplings on 40Ca+ S1/2 -> D5/2 (729 nm, E2), the standard single-laser sideband transition.
        The laser propagates along x with polarization along z, which couples |S1/2, mJ=1/2> to |D5/2, mJ=3/2> (and -1/2). """
    MODE_FREQUENCY = 2. * np.pi * 1e6   # rad/s; arbitrary trap frequency
    FOCK_DIMENSION = 5
    ETA = 0.08                          # arbitrary Lamb-Dicke parameter

    def build(self, ground_first: bool = True, mode_first: bool = False, n_atoms: int = 1):
        built = [Ca_s_to_d(f'ion{i}', ground_first) for i in range(n_atoms)]
        atoms, (_, g, e) = [atom for atom, _, _ in built], built[0]
        mode = MotionalMode.from_frequency(self.MODE_FREQUENCY, self.FOCK_DIMENSION, name='mode')
        basis = StandardBasis([mode, *atoms] if mode_first else [*atoms, mode])
        return atoms, mode, basis, g, e

    @staticmethod
    def laser(frequency):
        power = 1e-3; waist = 10e-6; phase = 0.
        return Laser.gaussian_from_frequency(frequency, power, waist, X, Polarization(Z), phase)

    def test_resonant_sideband_pi_pulses_for_any_basis_ordering(self):
        """Blue: |g,0> -> |e,1> in pi/(eta Omega) and |g,1> -> |e,2> sqrt(2) faster. Red: |g,1> -> |e,0>; |g,0> is dark.
        The mode-first ordering flips the kept element of the red sideband (|e,n-1> precedes |g,n>), which needs the
        opposite oscillation-rate sign; with the wrong sign the drive is off resonance by twice the laser frequency."""
        for ground_first in (True, False):
            for mode_first in (False, True):
                with self.subTest(ground_first=ground_first, mode_first=mode_first):
                    (atom,), mode, basis, g, e = self.build(ground_first, mode_first)
                    omega_0 = e.energy - g.energy
                    blue, red = self.laser(omega_0 + self.MODE_FREQUENCY), self.laser(omega_0 - self.MODE_FREQUENCY)
                    pi_time = np.pi / (self.ETA * abs(blue.rabi_frequency(g, e, 2, Z)))
                    blue_ops = blue.build_sideband_coupling_operators(SidebandType.BLUE, basis, [atom], [mode], [g], [e], 2, self.ETA, Z)
                    red_ops = red.build_sideband_coupling_operators(SidebandType.RED, basis, [atom], [mode], [g], [e], 2, self.ETA, Z)
                    for operators, (level, n), duration, (final_level, final_n) in (
                            (blue_ops, (g, 0), pi_time, (e, 1)),
                            (blue_ops, (g, 1), pi_time / np.sqrt(2.), (e, 2)),
                            (red_ops, (g, 1), pi_time, (e, 0)),
                            (red_ops, (g, 0), pi_time, (g, 0))):
                        psi = evolve(self, basis, operators, state_index(basis, level, n), duration)
                        self.assertGreater(abs(psi[state_index(basis, final_level, final_n)])**2, 1. - 1e-5)

    def test_sideband_matrix_elements(self):
        """Raising elements are (carrier raising element) x i eta^* sqrt(n+1) on |e,n+1><g,n| (blue) and
        (carrier raising element) x i eta sqrt(n+1) on |e,n><g,n+1| (red), and nothing else, for real and complex eta."""
        (atom,), mode, basis, g, e = self.build()
        laser = self.laser(e.energy - g.energy)
        carrier_raising = laser.build_atom_laser_ge_coupling_matrix(g, e, atom.energy_levels, 2, Z)[
            atom.energy_levels.index(e), atom.energy_levels.index(g)]
        for (sideband, shift), eta in itertools.product((('blue', +1), ('red', -1)), (self.ETA, self.ETA * np.exp(0.7j))):
            with self.subTest(sideband=sideband, eta=eta):
                operators = laser.build_sideband_coupling_operators(sideband, basis, [atom], [mode], [g], [e], 2, eta, Z)
                raising = {}
                for operator in operators:
                    for coupling in operator.couplings:
                        if e in coupling.row_state.components:
                            n_e, n_g = (next(c.fock_number for c in s.components if hasattr(c, 'fock_number'))
                                        for s in (coupling.row_state, coupling.column_state))
                            raising[(n_g, n_e)] = coupling.strength
                prefactor = 1j * (np.conj(eta) if sideband == 'blue' else eta)
                expected = {(n, n + shift): carrier_raising * prefactor * np.sqrt(max(n, n + shift))
                            for n in range(self.FOCK_DIMENSION) if 0 <= n + shift < self.FOCK_DIMENSION}
                self.assertEqual(set(raising), set(expected))
                for key, value in expected.items():
                    self.assertAlmostEqual(abs(raising[key] / value - 1.), 0., places=12)

    def test_debye_waller_matrix_elements(self):
        """debye_waller=True: blue element eta^* e^{-|eta|^2/2} L_n^1(|eta|^2) / sqrt(n+1) instead of eta^* sqrt(n+1), for real
        and complex eta (the correction depends only on |eta|)."""
        (atom,), mode, basis, g, e = self.build()
        laser = self.laser(e.energy - g.energy)
        def by_n(operator):
            return {next(c.fock_number for c in coupling.column_state.components if hasattr(c, 'fock_number')): coupling.strength
                    for coupling in operator.couplings if e in coupling.row_state.components}
        for eta in (0.3, 0.3 * np.exp(0.7j)):  # large enough for a visible correction
            with self.subTest(eta=eta):
                first_order, exact = (by_n(laser.build_sideband_coupling_operators(SidebandType.BLUE, basis, [atom], [mode], [g], [e], 2,
                                                                                   eta, Z, debye_waller=dw)[0]) for dw in (False, True))
                for n in range(self.FOCK_DIMENSION - 1):
                    expected_ratio = np.exp(-abs(eta)**2 / 2.) * genlaguerre(n, 1)(abs(eta)**2) / (n + 1)
                    self.assertAlmostEqual(abs(exact[n] / first_order[n] - expected_ratio), 0., places=12)

    def test_complex_eta_is_a_rephased_mode(self):
        """eta = |eta| e^{i theta} is |eta| with the mode rephased (a -> a e^{-i theta}): starting from |g, 0>, the state under
        eta is the state under |eta| with each Fock-n amplitude multiplied by e^{-i n theta}. Detuned red and blue tones
        together (a spin-dependent force) spread the state over several Fock states, so this needs i eta on the red and
        i eta^* on the blue sideband (here with the Debye-Waller elements, which depend on |eta|)."""
        (atom,), mode, basis, g, e = self.build()
        omega_0 = e.energy - g.energy
        rabi = abs(self.laser(omega_0).rabi_frequency(g, e, 2, Z))
        delta = 2. * self.ETA * rabi                      # spin-dependent displacement of ~0.5 after half a loop
        blue = self.laser(omega_0 + self.MODE_FREQUENCY + delta)
        red = self.laser(omega_0 - self.MODE_FREQUENCY - delta)
        theta = 0.7                                        # rad; arbitrary
        def final_state(eta):
            operators = [operator for laser, sideband in ((blue, 'blue'), (red, 'red'))
                         for operator in laser.build_sideband_coupling_operators(sideband, basis, [atom], [mode], [g], [e], 2, eta, Z,
                                                                                 debye_waller=True)]
            return evolve(self, basis, operators, state_index(basis, g, 0), np.pi / delta)
        real, rotated = final_state(self.ETA), final_state(self.ETA * np.exp(1j * theta))
        for level, n in itertools.product((g, e), range(self.FOCK_DIMENSION)):
            index = state_index(basis, level, n)
            self.assertAlmostEqual(abs(rotated[index] - real[index] * np.exp(-1j * n * theta)), 0., delta=1e-5)
        self.assertGreater(sum(abs(real[state_index(basis, level, 2)])**2 for level in (g, e)), 1e-3)   # nontrivial

    def test_matrix_rows_follow_target_atoms(self):
        """eta[i, j] belongs to (target_atoms[i], target_modes[j]): a stretch-like sign pattern [[a], [-a]] gives atom 1 the
        opposite sideband phase of atom 0, and reordering target_atoms together with the rows gives the same operators.
        A matrix of the wrong shape, a scalar for two atoms, or a non-array input is rejected."""
        atoms, mode, basis, g, e = self.build(n_atoms=2)
        laser = self.laser(e.energy - g.energy + self.MODE_FREQUENCY)
        a = self.ETA
        build = lambda target_atoms, eta: coupling_elements(basis, laser.build_sideband_coupling_operators(
            SidebandType.BLUE, basis, target_atoms, [mode], [g], [e], 2, eta, Z))
        stretch = build(atoms, np.array([[a], [-a]]))
        expected = build([atoms[0]], a)
        for key, value in build([atoms[1]], -a).items():
            expected[key] = expected.get(key, 0.) + value
        for other in (expected, build(atoms[::-1], np.array([[-a], [a]]))):
            self.assertEqual(set(other), set(stretch))
            for key in stretch:
                self.assertAlmostEqual(abs(other[key] - stretch[key]), 0., delta=1e-12 * abs(stretch[key]))
        for bad_eta in (a, np.array([[a, a]]), np.array([a, -a]), [[a], [-a]]):   # scalar for 2 atoms, wrong shapes, not an array
            with self.subTest(eta=bad_eta), self.assertRaises(IonSimError):
                build(atoms, bad_eta)


class TestLaserSidebandsFromTrap(unittest.TestCase):
    """Sideband couplings with Lamb-Dicke parameters from an IonChain (40Ca+ S1/2 -> D5/2, 729 nm, E2), with the trap ion
    and mode of each target DOF given as indices (stored ion indices, global mode indices)."""
    TRAP = (3. * TPI * 1e6, 3.2 * TPI * 1e6, 0.8 * TPI * 1e6)   # rad/s; radial >> axial: a chain along z
    FOCK_DIMENSION = 5
    power = 1e-3
    waist = 10e-6
    phase = 0. 

    def setup_chain(self, n_ions, n_hat):
        """Chain of n_ions 40Ca+ with its axial modes. Returns the atoms in stored (chain) order, i.e. atoms[k] is stored
        ion k, so ion_indices = range(n_ions); and the axial modes with their global indices."""
        input_atoms = [Ca_s_to_d(f'ion{i}')[0] for i in range(n_ions)]
        chain = IonChain.from_atomic_structure_basis(StandardBasis(input_atoms), *self.TRAP)
        atoms = [input_atoms[p] for p in chain.ion_indices]                        # stored order
        mode_indices = [chain.get_global_mode_index('axial', j) for j in range(n_ions)]
        modes = chain.build_mode_DOFs(mode_indices, self.FOCK_DIMENSION)
        basis = StandardBasis([*atoms, *modes])
        g, e = atoms[0].energy_levels
        n_hat = np.asarray(n_hat, dtype=float) / np.linalg.norm(n_hat)
        laser = Laser.gaussian_from_frequency(e.energy - g.energy, self.power, self.waist, n_hat, Polarization(Y), self.phase)
        return atoms, chain, modes, mode_indices, basis, g, e, laser

    def test_from_trap_equals_matrix_path(self):
        """from_trap builds exactly the operators of the matrix path with
        eta = ion_trap.get_lamb_dicke_parameters_for_ions_and_modes(ion_indices, mode_indices, k), for any target order;
        the stretch mode has opposite-sign eta on the two ions."""
        atoms, chain, modes, mode_indices, basis, g, e, laser = self.setup_chain(2, [1., 0., 1.])
        for ion_positions, mode_positions in (([1, 0], [1, 0]), (None, [1])):     # (target ions, target modes) by position
            target_atoms = None if ion_positions is None else [atoms[k] for k in ion_positions]
            ion_indices = [0, 1] if ion_positions is None else ion_positions          # None -> basis order = stored order
            target_modes = [modes[j] for j in mode_positions]
            target_mode_indices = [mode_indices[j] for j in mode_positions]
            for sideband in ('red', 'blue'):
                with self.subTest(sideband=sideband, ions=ion_positions, modes=mode_positions):
                    eta = chain.get_lamb_dicke_parameters_for_ions_and_modes(ion_indices, target_mode_indices, laser.wavevector)
                    from_trap = laser.build_sideband_coupling_operators_from_trap(sideband, basis, target_atoms, target_modes, [g], [e], 2,
                                                                                  chain, ion_indices, target_mode_indices, Z)
                    from_matrix = laser.build_sideband_coupling_operators(sideband, basis, target_atoms, target_modes, [g], [e], 2, eta, Z)
                    self.assertTrue(from_trap)
                    self.assertEqual(coupling_elements(basis, from_trap), coupling_elements(basis, from_matrix))
        stretch = chain.get_lamb_dicke_parameters_for_ions_and_modes([0, 1], [mode_indices[1]], laser.wavevector)[:, 0]
        self.assertGreater(abs(stretch[0]), 1e-3)
        self.assertAlmostEqual(stretch[0] / stretch[1], -1., places=8)

    def test_single_ion_blue_sideband_pi_pulse(self):
        """One ion, k along the axial mode: |eta| = k sqrt(hbar / 2 m omega_z) (compute_reference_single_ion_lamb_dicke_factors),
        and the from_trap blue sideband moves |g, 0> to |e, 1> in pi / (|eta| Omega)."""
        (atom,), chain, (mode,), (mode_index,), basis, g, e, carrier_laser = self.setup_chain(1, Z)
        reference_eta = chain.compute_reference_single_ion_lamb_dicke_factors(np.linalg.norm(carrier_laser.wavevector))[2]
        mode_frequency = mode.energy_levels[1].energy - mode.energy_levels[0].energy
        self.assertAlmostEqual(mode_frequency, self.TRAP[2], delta=1e-6 * self.TRAP[2])
        laser = Laser.gaussian_from_frequency(carrier_laser.frequency_angular + mode_frequency, self.power, self.waist, Z, Polarization(X), self.phase)
        eta = chain.get_lamb_dicke_parameters_for_ions_and_modes([0], [mode_index], laser.wavevector)[0, 0]
        self.assertAlmostEqual(abs(eta) / reference_eta, 1., places=6)
        operators = laser.build_sideband_coupling_operators_from_trap(SidebandType.BLUE, basis, [atom], [mode], [g], [e], 2, chain, [0], [mode_index], Z)
        psi = evolve(self, basis, operators, state_index(basis, g, 0), np.pi / (abs(reference_eta) * abs(laser.rabi_frequency(g, e, 2, Z))))
        self.assertGreater(abs(psi[state_index(basis, e, 1)])**2, 1. - 1e-5)

    def test_consistency_checks(self):
        """Mode indices pointing at modes of another frequency, or ion indices at ions of another species, are rejected."""
        atoms, chain, modes, mode_indices, basis, g, e, laser = self.setup_chain(2, [1., 0., 1.])
        with self.subTest('mode index of a different-frequency mode'), self.assertRaises(IonSimError):
            laser.build_sideband_coupling_operators_from_trap(SidebandType.BLUE, basis, atoms, modes, [g], [e], 2, chain, [0, 1], mode_indices[::-1], Z)
        with self.subTest('ion of a different species'), self.assertRaises(IonSimError):
            yb_chain = IonChain.from_species('171Yb+', 2, *self.TRAP)      # Yb modes (frequencies match), Ca atoms
            yb_modes = yb_chain.build_mode_DOFs(mode_indices, self.FOCK_DIMENSION)
            laser.build_sideband_coupling_operators_from_trap(SidebandType.BLUE, StandardBasis([*atoms, *yb_modes]), atoms, yb_modes,
                                                              [g], [e], 2, yb_chain, [0, 1], mode_indices, Z)


class TestLaserTargetRabiFrequency(unittest.TestCase):
    """Lasers constructed from a laser frequency and a target Rabi frequency on one transition or a set of them, and
    Laser.rabi_frequency(). The target-Rabi
    constructors are also used throughout TestLaserDynamicsInInteractionPicture."""

    def test_round_trip_E1(self):
        atom, g, e = Yb_s_to_p()
        polarization = CircularPolarization('+')
        target, detuning = TPI * 2e6, TPI * 10e6   # rad/s; arbitrary
        gaussian_waist = 20e-6
        lasers = {
            'plane wave': Laser.from_target_rabi_frequency(target, e.energy - g.energy, g, e, 1, Z, polarization, Z),
            'gaussian': Laser.gaussian_from_target_rabi_frequency(target, e.energy - g.energy, g, e, 1, gaussian_waist, Z, polarization, Z),
            'plane wave, detuned': Laser.from_target_rabi_frequency(target, e.energy - g.energy + detuning, g, e, 1, Z, polarization, Z),
            'gaussian, detuned': Laser.gaussian_from_target_rabi_frequency(target, e.energy - g.energy + detuning, g, e, 1, gaussian_waist, Z, polarization, Z),
        }
        for name, laser in lasers.items():
            with self.subTest(name):
                self.assertAlmostEqual(abs(laser.rabi_frequency(g, e, 1, Z)) / target, 1., places=12)
                expected_frequency = (e.energy - g.energy) + (detuning if 'detuned' in name else 0.)
                self.assertAlmostEqual(laser.frequency_angular / expected_frequency, 1., places=14)
                self.assertIsInstance(laser.beam_profile, GaussianBeam if 'gaussian' in name else PlaneWave)
                # The coupling builder uses the same Rabi frequency: absorption element strength Omega / 2.
                operator, = laser.build_carrier_coupling_operators(StandardBasis([atom]), [atom], [g], [e], 1, Z)
                self.assertAlmostEqual(abs(operator.couplings[0].strength) / (target / 2.), 1., places=12)

    def test_round_trip_E2(self):
        """E2 couplings depend on |k|, which the constructor evaluates at the laser's own frequency (including detuning)."""
        _, g, e = Ca_s_to_d()
        polarization = Polarization(Z)
        target = TPI * 100e3   # rad/s; arbitrary
        gaussian_waist = 10e-6
        for detuning in (0., TPI * 1e9):
            with self.subTest(detuning=detuning):
                for laser in (Laser.from_target_rabi_frequency(target, e.energy - g.energy + detuning, g, e, 2, X, polarization, Z),
                              Laser.gaussian_from_target_rabi_frequency(target, e.energy - g.energy + detuning, g, e, 2, gaussian_waist, X, polarization, Z)):
                    self.assertAlmostEqual(abs(laser.rabi_frequency(g, e, 2, Z)) / target, 1., places=12)

    def test_gaussian_power_consistency(self):
        """The derived power reproduces the target through the power-based constructor, and scales as Omega^2."""
        _, g, e = Yb_s_to_p()
        polarization = CircularPolarization('+')
        target, waist = TPI * 1e6, 15e-6
        laser = Laser.gaussian_from_target_rabi_frequency(target, e.energy - g.energy, g, e, 1, waist, Z, polarization, Z)
        rebuilt = Laser.gaussian_from_frequency(laser.frequency_angular, laser.power_W, waist, Z, polarization, 0.)
        self.assertAlmostEqual(abs(rebuilt.rabi_frequency(g, e, 1, Z)) / target, 1., places=12)
        doubled = Laser.gaussian_from_target_rabi_frequency(2. * target, e.energy - g.energy, g, e, 1, waist, Z, polarization, Z)
        self.assertAlmostEqual(doubled.power_W / laser.power_W, 4., places=12)
        self.assertAlmostEqual(laser.beam_profile.waist_m, waist)

    def test_root_sum_square_over_unresolved_levels(self):
        """ With lists of levels the target is sqrt(sum |Omega_ge|^2) over all pairs (the light-shift convention), for one
        ground level to several excited levels (87Rb 420 nm: 5S1/2 F=2 -> 6P3/2 F'=1,2,3) and several ground levels to one
        excited level (1013 nm: 6P3/2 F'=1,2,3 -> 53S1/2). The per-path ratios are those of a single-transition laser, a
        one-element list equals the single level, and far from resonance sum_k |Omega_k|^2 / (4 Delta_k) is the light shift
        Omega^2 / (4 Delta) up to the hyperfine splitting over Delta. """
        quantum_numbers = [{'n': 5, 'l': 0, 'f': 2, 'mf': 0}, {'n': 6, 'l': 1, 'f': 1, 'mf': 1}, {'n': 6, 'l': 1, 'f': 2, 'mf': 1},
                           {'n': 6, 'l': 1, 'f': 3, 'mf': 1}, {'n': 53, 'l': 0, 'mj': 0.5, 'mi': -0.5}]
        rb = AtomicStructure.from_species('87Rb', term_symbols=['S1/2', '6 P3/2', '53 S1/2'], quantum_numbers=quantum_numbers,
                                          magnetic_field=8.5)
        g, *intermediates, r = rb.energy_levels
        target, detuning = TPI * 237e6, -TPI * 7.8e9
        omega_420 = intermediates[0].fine_energy - g.energy + detuning
        laser_420 = Laser.gaussian_from_target_rabi_frequency(target, omega_420, g, intermediates, 1, 10e-6, Z, CircularPolarization('+'), Z)
        rabis = np.array([abs(laser_420.rabi_frequency(g, e, 1, Z)) for e in intermediates])
        self.assertAlmostEqual(np.sqrt(np.sum(rabis**2)) / target, 1., places=12)
        single = Laser.from_target_rabi_frequency(1., omega_420, g, intermediates[1], 1, Z, CircularPolarization('+'), Z)
        np.testing.assert_allclose(rabis / rabis[1], [abs(single.rabi_frequency(g, e, 1, Z)) for e in intermediates], rtol=1e-12)
        light_shift = np.sum(rabis**2 / (4. * np.array([laser_420.detuning_from_level_transition(g, e) for e in intermediates])))
        self.assertAlmostEqual(light_shift / (target**2 / (4. * detuning)), 1., delta=0.02)

        laser_1013 = Laser.from_target_rabi_frequency(TPI * 303e6, r.energy - g.energy - omega_420, intermediates, r, 1, -Z,
                                                      CircularPolarization('+'), Z)
        self.assertAlmostEqual(np.sqrt(sum(abs(laser_1013.rabi_frequency(e, r, 1, Z))**2 for e in intermediates)) / (TPI * 303e6), 1.,
                               places=12)
        one, plain = (Laser.from_target_rabi_frequency(target, omega_420, ground, excited, 1, Z, CircularPolarization('+'), Z)
                      for ground, excited in (([g], [intermediates[2]]), (g, intermediates[2])))
        self.assertEqual(one.peak_electric_field_magnitude_V_per_m, plain.peak_electric_field_magnitude_V_per_m)

    def test_undrivable_transitions_are_rejected(self):
        _, g, e = Yb_s_to_p()
        with self.subTest('polarization does not drive the transition'), self.assertRaises(IonSimError):
            Laser.from_target_rabi_frequency(TPI * 1e6, e.energy - g.energy, g, e, 1, Z, CircularPolarization('-'), Z)
        with self.subTest('excited level below ground level'), self.assertRaises(IonSimError):
            Laser.from_target_rabi_frequency(TPI * 1e6, e.energy - g.energy, e, g, 1, Z, CircularPolarization('+'), Z)


class TestPhysicalDipoleCouplings(unittest.TestCase):
    """E1 couplings with tabulated reduced matrix elements <J||er||J'> (Steck convention), benchmarked against
    D. A. Steck, "Rubidium 87 D Line Data", rev. 1.6 (the reference of the 87Rb config file), and R. M. Kroeze et al.,
    "171Yb Reference Data" (arXiv:2509.04416v1, the reference of the 171Yb config file)."""
    E_A0_OVER_HBAR = const.e * const.value('Bohr radius') / const.hbar

    def test_rb87_d2_cycling_saturation_intensity(self):
        """sigma+ |F=2, mF=2> -> |F'=3, mF'=3> at I_sat = 1.669 mW/cm^2 (Steck Table 7) has Omega = Gamma/sqrt(2)
        (I/I_sat = 2 Omega^2/Gamma^2, Gamma = 2pi x 6.065 MHz, Table 3), and effective dipole 2.989 e a0 (Table 7)."""
        rb = AtomicStructure.from_species(species='87Rb', term_symbols=['S1/2', 'P3/2'], level_names=['S1/2,2,2', 'P3/2,3,3'])
        g, e = rb.energy_levels
        laser = Laser(e.energy - g.energy, Z, CircularPolarization('+'), PlaneWave(1.669e-3 / 1e-4))
        rabi = abs(laser.rabi_frequency(g, e, 1, Z))
        self.assertAlmostEqual(rabi / (TPI * 6.065e6 / np.sqrt(2.)), 1., delta=2e-3)    # Steck values: ~0.15% uncertainty
        dipole = rabi / laser.peak_electric_field_magnitude_V_per_m / self.E_A0_OVER_HBAR             # |<e|r.eps|g>| / a0
        self.assertAlmostEqual(dipole / 2.989, 1., delta=1e-3)

    def test_reduced_elements_reproduce_config_decay_rates(self):
        """Partial decay rate omega_0^3 / (3 pi eps0 hbar c^3) (2J+1)/(2J'+1) |<J||er||J'>|^2 = branching_ratio / lifetime
        (Steck Eq. 38; Eq. 52 of the Yb reference), with each config's reduced elements, fine-structure energies,
        lifetimes and branching ratios: 87Rb D1 and D2, 171Yb 3P1 -> 1S0 and 3D1 -> 3P0."""
        cases = (('87Rb', ['S1/2', 'P1/2', 'P3/2'], [('S1/2', 'P1/2'), ('S1/2', 'P3/2')], 5e-3),
                 ('171Yb', ['S0', 'P0', 'P1', 'D1'], [('S0', 'P1'), ('P0', 'D1')], 1e-2))
        for species, term_symbols, transitions, tolerance in cases:
            manifold = {level.term_symbol: level for level in
                        AtomicStructure.from_species(species=species, term_symbols=term_symbols).energy_levels}
            for lower, upper in transitions:
                with self.subTest(species=species, transition=f'{upper} -> {lower}'):
                    g, e = manifold[lower], manifold[upper]
                    reduced = reduced_dipole_matrix_element(g, e) * const.e * const.value('Bohr radius')
                    omega_0 = e.fine_energy - g.fine_energy
                    rate = (omega_0**3 / (3. * np.pi * const.epsilon_0 * const.hbar * const.c**3)
                            * (2. * g.j + 1.) / (2. * e.j + 1.) * reduced**2)
                    branching_ratio = (e.branching_ratios or {lower: 1.})[lower]
                    self.assertAlmostEqual(rate * e.lifetime / branching_ratio, 1., delta=tolerance)

    def test_Yb171_rabi_frequencies_match_reference_tables(self):
        """Reference Rabi frequencies for P = 1 mW, D = 1 mm (taken as the 1/e^2 diameter, waist 0.5 mm) on
        |F=1/2, mF=1/2> -> |F'=3/2, mF'=3/2> with sigma+ light: Table 10 (1S0 -> 3P1, 556 nm) 2pi x 5.523(6) MHz,
        Table 12 (3P0 -> 3D1, 1389 nm) 2pi x 28.3(8) MHz."""
        for lower, upper, reference, tolerance in (('S0', 'P1', 5.523, 2e-3), ('P0', 'D1', 28.3, 3e-2)):
            with self.subTest(transition=f'{lower} -> {upper}'):
                yb = AtomicStructure.from_species(species='171Yb', term_symbols=[lower, upper], level_names=[f'{lower},1/2,1/2', f'{upper},3/2,3/2'])
                g, e = yb.energy_levels
                laser = Laser.gaussian_from_frequency(e.energy - g.energy, 1e-3, 0.5e-3, Z, CircularPolarization('+'), 0.)
                rabi = abs(laser.rabi_frequency(g, e, 1, Z)) / TPI / 1e6
                self.assertAlmostEqual(rabi / reference, 1., delta=tolerance)

    def test_e2_normalization_unchanged(self):
        """E2 keeps its previous absolute normalization: 2 x angular x (e a0/hbar) x i k a0 sqrt(2/3), in orbital units."""
        #this normalization has not yet been verified against a reference
        _, g, e = Ca_s_to_d()
        components = Polarization(Z).quadrupole_components(X, Z)
        k = 2. * np.pi / 729e-9
        angular = sum(components[q] * compute_multipole_amplitude(g, e, 2, q, include_J_J_prime_matrix_element=True)
                      for q in range(-2, 3))
        previous = 2. * angular * self.E_A0_OVER_HBAR * 1j * k * const.value('Bohr radius') * np.sqrt(2. / 3.)
        coupling = compute_coupling_amplitude_between_atomic_levels(g, e, components, [g, e], 2, wavenumber=k)
        self.assertAlmostEqual(abs(coupling / previous - 1.), 0., places=12)


if __name__ == '__main__':
    unittest.main()
