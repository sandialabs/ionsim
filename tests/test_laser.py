#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

import dataclasses
import unittest

import numpy as np
from scipy import constants as const 

from ionsim.process import Gate, Circuit
import warnings

from scipy.special import genlaguerre

from ionsim.degree_of_freedom import AtomicStructure, MotionalMode
from ionsim.state import State
from ionsim.basis import StandardBasis
from ionsim.laser import Laser, Polarization, GaussianBeam, PlaneWave, _perpendicular_basis, _spherical_basis_vectors
from ionsim.hamiltonian import Hamiltonian
from ionsim.atomic_internal_energy_level import (LSFineLevel, compute_coupling_amplitude_between_atomic_levels,
                                                 compute_multipole_amplitude, reduced_dipole_matrix_element)
from ionsim.ionsim_error import IonSimError
from ionsim.named_operators import Unitary
from ionsim.noise import Noise

class TestProcess(unittest.TestCase):

    def setUp(self):
        """Set up the necessary objects for testing and test constructors."""
        a_levels = ['S1/2,0,0', 'S1/2,1,-1', 'S1/2,1,0', 'S1/2,1,1']
        #levels = ['S1/2,1,0', 'P1/2,1,-1', 'P1/2,1,0', 'P1/2,1,1']
        self.atom_a = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=a_levels)
        #self.atom_a = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        b_levels = ['S1/2,1,0', 'P1/2,1,-1', 'P1/2,1,0', 'P1/2,1,1']
        self.atom_b = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2'], level_names=b_levels)
        self.basis = StandardBasis([self.atom_a, self.atom_b])

        c_levels = ['S0,1/2,1/2', 'P1,3/2,3/2']
        self.atom_c = AtomicStructure.from_species(species='171Yb', term_symbols=['S0', 'P1'], level_names=c_levels) # 3P1
        self.neutral_basis = StandardBasis([self.atom_c]) 

        propagation_vector = np.array([0., 0., 1.])
        #propagation_vector = np.array([np.cos(np.pi/4.), np.sin(np.pi/4.), 0.])
        phase = np.pi 
        # Create polarization 
        laser_polarization = Polarization.circular(propagation_vector, '+')

        # Create Gaussian beam profile  
        wavelength = 355*1E-9 # nm -> meters 
        beam_waist = 3 * 1E-6 # 3 µm -> meters 
        laser_power = 20e-3 # 20 mWatt -> Watt  
        self.laser = Laser.gaussian_from_wavelength(wavelength, laser_power, beam_waist, propagation_vector, laser_polarization, phase) 

        wavelength_laser_c = 556*1E-9 # nm -> meters
        laser_c_power = 1e-3 # Watt 
        laser_c_waist = (1e-3)/2 # m 
        laser_c_polarization = Polarization.linear(propagation_vector, angle=0.)
        phase = np.pi
        self.laser_c = Laser.gaussian_from_wavelength(wavelength_laser_c, laser_c_power, laser_c_waist, propagation_vector, laser_c_polarization, phase) 

        # Test attributes:
        self.assertAlmostEqual(beam_waist, self.laser.beam_profile.waist, places=10)
        # Expect about 1.4kV/m 
        self.assertAlmostEqual(self.laser_c.peak_electric_field_magnitude*1E-3, 1.3851612653205, places=10)
        self.assertAlmostEqual(self.laser_c.beam_profile.peak_electric_field_magnitude(laser_c_power)*1E-3, 1.3851612653205, places=10)

    def test_laser_coupling_builder(self):
        """Test the process fidelity of the extra noisy gate."""
        # Test building coupling operators from laser and polarization information  
        ground_levels = [self.atom_a.energy_levels[0]] 
        excited_levels = [*self.atom_a.energy_levels[1:]] 
        atom_a_coupling_operators = self.laser.build_carrier_coupling_operators(self.basis, [self.atom_a], ground_levels, excited_levels, 1) 
        # These are not dipole allowed: should be zero operators 
        self.assertEqual(len(atom_a_coupling_operators), 0)

        ground_levels = [self.atom_b.energy_levels[0]] 
        excited_levels = [*self.atom_b.energy_levels[1:]] 
        # Addresses atom B in a 2-atom basis 
        atom_b_coupling_operators = self.laser.build_carrier_coupling_operators(self.basis, [self.atom_b], ground_levels, excited_levels, 1) 
        # Expect one transition |s1/2, 1,0> --> |P1/2,1,+1> transition, and 4 couplings since atom A has 4 levels, so the 2-qubit basis has 4 couplings  
        self.assertEqual(len(atom_b_coupling_operators), 1)
        # For 2-qubits and a + transition on qubit 2 and 4 levels on qubit 1, there are 4 unique couplings and 8 total couplings (h.c.) 
        self.assertEqual(len(atom_b_coupling_operators[0].couplings), 8)

        # Check that the transition is with P1/2,1,+1;
        excited_level_name = atom_b_coupling_operators[0].couplings[0].column_state.name[-8:]
        expected_name = 'P1/2,1,1'
        self.assertEqual(expected_name, excited_level_name)


    def test_1S0_3P1_transition_coupling_from_laser(self):
        """ Following the reference https://arxiv.org/pdf/2509.04416v1, testing rabi frequency for 556 nm laser on 3P1 transition"""
        TPI = 2.*np.pi
        ground_levels = [self.atom_c.energy_levels[0]] 
        excited_levels = [self.atom_c.energy_levels[1]] 
        energy_difference = excited_levels[0].energy - ground_levels[0].energy 

        laser_detuning = self.laser_c.detuning_from_transition_frequency(energy_difference) 
        laser_detuning_v2 = self.laser_c.detuning_from_level_transition(ground_levels[0], excited_levels[0]) 
        self.assertEqual(laser_detuning, laser_detuning_v2)
        expected_detuning = (const.c / self.laser_c.wavelength)*2.*np.pi - energy_difference
        self.assertAlmostEqual(laser_detuning, expected_detuning, places=10)
        # Detuning should be negative because F=3/2 is shifted above the fine energy 
        self.assertAlmostEqual(laser_detuning/TPI/1E9, -195.33742087632817, places=8) 

        # Expecting 5.523 MHz Rabi frequency  
        atom_c_coupling_operators = self.laser_c.build_carrier_coupling_operators(self.neutral_basis, [self.atom_c], ground_levels, excited_levels, 1) 
        assert len(atom_c_coupling_operators) == 1

        # We expect a Rabi frequency of about ~5 MHz, but this is not precise  
        dipole_factor = 0.5398 # Table 10 of reference 
        # Note: couplings[0].strength is the matrix element Omega/2, not Omega. Regression value halved when the E1 prefactor
        # was corrected to Omega = |<e|d.eps|g>| E0 / hbar (E0 the real field amplitude); previously 3.905827783900461.
        rabi_frequency = dipole_factor * atom_c_coupling_operators[0].couplings[0].strength/TPI/1E6
        self.assertAlmostEqual(np.abs(rabi_frequency), 1.9529138919502305, places = 8)


class TestLaserConstructors(unittest.TestCase):

    def setUp(self):
        self.n_hat = np.array([0., 0., 1.])
        self.wavelength = 729e-9  # m; arbitrary optical wavelength
        self.power, self.waist = 5e-3, 15e-6  # W, m; arbitrary beam parameters

    def test_gaussian_from_frequency_matches_from_wavelength(self):
        frequency = 2 * np.pi * const.c / self.wavelength
        pol = Polarization.linear(self.n_hat)
        from_freq = Laser.gaussian_from_frequency(frequency, self.power, self.waist, self.n_hat, pol, 0.)
        from_wl = Laser.gaussian_from_wavelength(self.wavelength, self.power, self.waist, self.n_hat, pol, 0.)
        self.assertIsInstance(from_freq.beam_profile, GaussianBeam)
        self.assertAlmostEqual(from_freq.wavelength / from_wl.wavelength, 1., places=12)
        self.assertAlmostEqual(from_freq.beam_profile.waist, self.waist)
        self.assertAlmostEqual(from_freq.peak_electric_field_magnitude / from_wl.peak_electric_field_magnitude, 1., places=12)

    def test_peak_intensity(self):
        """Gaussian peak intensity I0 = 2 P / (pi w^2)."""
        laser = Laser.gaussian_from_wavelength(self.wavelength, self.power, self.waist, self.n_hat, Polarization.linear(self.n_hat), 0.)
        self.assertAlmostEqual(laser.peak_intensity / (2 * self.power / (np.pi * self.waist**2)), 1., places=12)

    def test_polarization_repr(self):
        self.assertIn('Polarization(vector=', repr(Polarization.linear(self.n_hat)))

    def test_unknown_modulation_key(self):
        laser = Laser.gaussian_from_wavelength(self.wavelength, self.power, self.waist, self.n_hat, Polarization.linear(self.n_hat), 0.,
                                               modulation_functions={'phse': lambda t: 0.})
        with self.assertRaises(IonSimError):
            laser.modulation_function


class TestPolarizationComponents(unittest.TestCase):
    """Angular conventions of E1 and E2 couplings.

    For a J = 0 -> J = k transition (no spin or nuclear spin), <k m| T^(k)_q |0 0> is the same for every m = q, so the
    coupling to |k, m> must be proportional to the polarization's rank-k component with q = m.
    """

    def setUp(self):
        # Arbitrary elliptical polarization for an arbitrary propagation direction.
        n_hat = np.array([0.3, -0.5, 0.8]); n_hat /= np.linalg.norm(n_hat)
        e1 = np.cross(n_hat, [1., 0., 0.]); e1 /= np.linalg.norm(e1)
        e2 = np.cross(n_hat, e1)
        self.polarization = Polarization(np.cos(0.4) * e1 + np.exp(0.9j) * np.sin(0.4) * e2, n_hat)
        self.ground = LSFineLevel(n=1, j=0, term_symbol='S0', fine_energy=0., hyperfine_A=0., l=0, s=0, mj=0)

    def assert_couplings_follow_components(self, k, components):
        term = {1: 'P1', 2: 'D2'}[k]
        excited = [LSFineLevel(n=1, j=k, term_symbol=term, fine_energy=1., hyperfine_A=0., l=k, s=0, mj=m) for m in range(-k, k + 1)]
        levels = [self.ground] + excited
        couplings = np.array([compute_coupling_amplitude_between_atomic_levels(self.ground, e, components, levels, k, wavenumber=1.)
                              for e in excited])
        expected = np.array([components[m] for m in range(-k, k + 1)])
        ratio = couplings / expected
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-10)  # proportional, with the same relative phases

    def test_dipole(self):
        self.assert_couplings_follow_components(1, self.polarization.dipole_components())

    def test_quadrupole(self):
        self.assert_couplings_follow_components(2, self.polarization.quadrupole_components())

    def test_quadrupole_selection_rule(self):
        """n along x and polarization along z (quantization axis): the E2 tensor has only q = +1, -1 components."""
        polarization = Polarization(np.array([0., 0., 1.]), np.array([1., 0., 0.]))
        components = polarization.quadrupole_components()
        # sum_q |C_q|^2 = |symmetric part of eps (x) n|^2 = 1/2 for perpendicular unit vectors
        self.assertAlmostEqual(sum(abs(components[q])**2 for q in (-1, 1)), 0.5, places=12)
        for q in (-2, 0, 2):
            self.assertAlmostEqual(abs(components[q]), 0., places=12)


class TestLaserQuadrupoleCoupling(unittest.TestCase):

    def test_ca40_729nm_coupling(self):
        """40Ca+ S1/2 -> D5/2 (E2) with n along x and polarization along z: |S1/2, 1/2> couples only to mJ = 1/2 +/- 1."""
        ca = AtomicStructure.from_species(species='40Ca+', term_symbols=['S1/2', 'D5/2'])
        basis = StandardBasis([ca])
        ground = [level for level in ca.energy_levels if level.name == 'S1/2,1/2']
        excited = [level for level in ca.energy_levels if level.term_symbol == 'D5/2']
        n_hat = np.array([1., 0., 0.])
        laser = Laser.gaussian_from_wavelength(729e-9, 1e-3, 10e-6, n_hat, Polarization(np.array([0., 0., 1.]), n_hat), 0.)
        operators = laser.build_carrier_coupling_operators(basis, [ca], ground, excited, 2)
        coupled = sorted(next(state.name for state in (op.couplings[0].row_state, op.couplings[0].column_state) if 'D5/2' in state.name)
                         for op in operators)
        self.assertEqual(coupled, ['D5/2,-1/2', 'D5/2,3/2'])


class TestLaserFrequencyAndModulation(unittest.TestCase):
    """Coupling phases and frequencies must not depend on whether the ground or excited level comes first in the basis."""

    TIMES = (0., 1e-9, 2.3e-9)  # s; arbitrary evaluation times

    def build(self, ground_first: bool):
        qn_g, qn_e = {'n': 6, 'l': 0, 'f': 1, 'mf': 0}, {'n': 6, 'l': 1, 'f': 1, 'mf': 1}
        atom = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2'],
                                            quantum_numbers=[qn_g, qn_e] if ground_first else [qn_e, qn_g])
        g = next(level for level in atom.energy_levels if level.term_symbol == 'S1/2')
        e = next(level for level in atom.energy_levels if level.term_symbol == 'P1/2')
        return atom, StandardBasis([atom]), g, e

    def coupling(self, laser, atom, basis, g, e, t, frame=None, frequency_shift=0.):
        """The upper-triangle coupling element of the Hamiltonian at time t."""
        operators = laser.build_carrier_coupling_operators(basis, [atom], [g], [e], 1, frequency_shift=frequency_shift)
        frame = [0.] * len(basis.states) if frame is None else frame
        return np.asarray(Hamiltonian(basis, operators, frame).hamiltonian_function(t))[0, 1]

    def test_resonant_laser_is_static_in_level_frame(self):
        """In IonSim's interaction picture (rotating_frame_energies = -E, which removes H0) a resonant laser is static."""
        for ground_first in (True, False):
            with self.subTest(ground_first=ground_first):
                atom, basis, g, e = self.build(ground_first)
                n_hat = np.array([0., 0., 1.])
                laser = Laser.gaussian_from_frequency(e.energy - g.energy, 1e-3, 20e-6, n_hat, Polarization.circular(n_hat, '+'), 0.)
                frame = [-state.energy for state in basis.states]
                operators = laser.build_carrier_coupling_operators(basis, [atom], [g], [e], 1)
                H0 = Hamiltonian(basis, operators, frame).H0_H0Shifts_Hints_and_Rates[0].toarray()
                np.testing.assert_allclose(H0, 0., atol=0.)
                values = [self.coupling(laser, atom, basis, g, e, t, frame) for t in self.TIMES]
                np.testing.assert_allclose(values, values[0], rtol=1e-6)

    def test_constant_modulations_match_static_parameters(self):
        """A constant phase (frequency) modulation equals the same static phase (frequency_shift)."""
        phase, delta = 0.83, 2 * np.pi * 37e6  # rad, rad/s; arbitrary
        for ground_first in (True, False):
            atom, basis, g, e = self.build(ground_first)
            n_hat = np.array([0., 0., 1.])
            laser = Laser.gaussian_from_wavelength(369.5e-9, 1e-3, 20e-6, n_hat, Polarization.circular(n_hat, '+'), 0.)
            cases = {
                'phase': (dataclasses.replace(laser, phase=phase), {}, dataclasses.replace(laser, modulation_functions={'phase': lambda t: phase})),
                'frequency': (laser, {'frequency_shift': delta}, dataclasses.replace(laser, modulation_functions={'frequency': lambda t: delta})),
            }
            for key, (static_laser, static_kwargs, modulated_laser) in cases.items():
                with self.subTest(ground_first=ground_first, modulation=key):
                    for t in self.TIMES:
                        static = self.coupling(static_laser, atom, basis, g, e, t, **static_kwargs)
                        modulated = self.coupling(modulated_laser, atom, basis, g, e, t)
                        # Lab-frame phases are ~1e7 rad at these times, so allow floating-point phase error.
                        self.assertLess(abs(modulated - static) / abs(static), 1e-7)


class TestLaserDynamicsInInteractionPicture(unittest.TestCase):
    """Two-level dynamics driven by Laser-built couplings, in the frame rotating_frame_energies = -E.

    These check the physics (detuning value and sign), not only that a resonant coupling is static.
    """

    def build(self, ground_first: bool):
        qn_g, qn_e = {'n': 6, 'l': 0, 'f': 1, 'mf': 0}, {'n': 6, 'l': 1, 'f': 1, 'mf': 1}
        atom = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2'],
                                            quantum_numbers=[qn_g, qn_e] if ground_first else [qn_e, qn_g])
        g = next(level for level in atom.energy_levels if level.term_symbol == 'S1/2')
        e = next(level for level in atom.energy_levels if level.term_symbol == 'P1/2')
        return atom, StandardBasis([atom]), g, e

    def laser(self, frequency, **kwargs):
        n_hat = np.array([0., 0., 1.])
        # Low power so the Rabi frequency (~2pi x 1 MHz scale) is far below the optical frequency.
        return Laser.gaussian_from_frequency(frequency, 1e-9, 20e-6, n_hat, Polarization.circular(n_hat, '+'), 0., **kwargs)

    def evolve(self, laser, atom, basis, g, e, duration, frequency_shift=0.):
        operators = laser.build_carrier_coupling_operators(basis, [atom], [g], [e], 1, frequency_shift=frequency_shift)
        rabi = 2. * abs(operators[0].couplings[0].strength)
        hamiltonian = Hamiltonian(basis, operators, [-state.energy for state in basis.states])
        psi0 = np.array([1. if state.components[0] is g else 0. for state in basis.states], dtype=complex)
        # Dense output: odeint's step limit applies per output interval, and when it is exceeded odeint only warns and
        # returns an invalid state. The norm check below guards against such silent solver failures.
        _, psis = hamiltonian.evolve_wavefunction(psi0, duration(rabi), np.linspace(0., duration(rabi), 2001))
        self.assertAlmostEqual(np.linalg.norm(psis[-1]), 1., delta=1e-5)  # odeint's default rtol allows ~1e-6 drift
        index_g = [state.components[0] for state in basis.states].index(g)
        return psis[-1], index_g, rabi

    def test_resonant_pi_pulse(self):
        for ground_first in (True, False):
            with self.subTest(ground_first=ground_first):
                atom, basis, g, e = self.build(ground_first)
                psi, index_g, _ = self.evolve(self.laser(e.energy - g.energy), atom, basis, g, e, lambda rabi: np.pi / rabi)
                self.assertLess(abs(psi[index_g])**2, 1e-6)

    def test_detuning_is_compensated_by_frequency_shift_and_modulation(self):
        """A laser detuned by +delta driven with frequency_shift = -delta (or a constant 'frequency' modulation of -delta)
        is resonant again: a pi pulse transfers all population. Without the compensation it does not."""
        for ground_first in (True, False):
            atom, basis, g, e = self.build(ground_first)
            rabi = 2. * abs(self.laser(e.energy - g.energy).build_carrier_coupling_operators(
                basis, [atom], [g], [e], 1)[0].couplings[0].strength)
            delta = 2. * rabi  # rad/s; large enough that an uncompensated pi pulse clearly fails
            detuned = self.laser(e.energy - g.energy + delta)
            pi_time = lambda r: np.pi / r
            with self.subTest(ground_first=ground_first, case='uncompensated'):
                psi, index_g, _ = self.evolve(detuned, atom, basis, g, e, pi_time)
                self.assertGreater(abs(psi[index_g])**2, 0.1)
            with self.subTest(ground_first=ground_first, case='frequency_shift'):
                psi, index_g, _ = self.evolve(detuned, atom, basis, g, e, pi_time, frequency_shift=-delta)
                self.assertLess(abs(psi[index_g])**2, 1e-6)
            with self.subTest(ground_first=ground_first, case='frequency modulation'):
                modulated = dataclasses.replace(detuned, modulation_functions={'frequency': lambda t: -delta})
                psi, index_g, _ = self.evolve(modulated, atom, basis, g, e, pi_time)
                self.assertLess(abs(psi[index_g])**2, 1e-4)  # 'frequency' phase is integrated numerically

    def test_light_shift_sign(self):
        """A far red-detuned laser (omega < omega_0) lowers |g> by |Omega|^2 / (4 |delta|): in the interaction picture the
        ground amplitude acquires the phase +|Omega|^2 t / (4 |delta|)."""
        for ground_first in (True, False):
            with self.subTest(ground_first=ground_first):
                atom, basis, g, e = self.build(ground_first)
                rabi = 2. * abs(self.laser(e.energy - g.energy).build_carrier_coupling_operators(
                    basis, [atom], [g], [e], 1)[0].couplings[0].strength)
                delta = -50. * rabi  # red detuning
                duration = lambda r: 40. * np.pi / r
                psi, index_g, _ = self.evolve(self.laser(e.energy - g.energy + delta), atom, basis, g, e, duration)
                expected_phase = -(rabi**2 / (4. * delta)) * duration(rabi)    # = -LS t with LS = Omega^2 / (4 delta) < 0
                self.assertGreater(expected_phase, 0.)
                self.assertAlmostEqual(np.angle(psi[index_g]), expected_phase, delta=0.05 * expected_phase)


class TestTransverseBasis(unittest.TestCase):
    """Transverse unit vectors used for polarizations and for the spherical basis of a quantization axis."""

    X, Y, Z = np.eye(3)

    def test_z_gives_x_and_y(self):
        e1, e2 = _perpendicular_basis(self.Z)
        np.testing.assert_allclose(e1, self.X, atol=1e-15)
        np.testing.assert_allclose(e2, self.Y, atol=1e-15)

    def test_standard_spherical_basis_for_z(self):
        """e_{+1} = -(x + iy)/sqrt(2), e_0 = z, e_{-1} = (x - iy)/sqrt(2) (Steck, eq. 7.188)."""
        e = _spherical_basis_vectors(self.Z)
        np.testing.assert_allclose(e[1], -(self.X + 1j * self.Y) / np.sqrt(2), atol=1e-15)
        np.testing.assert_allclose(e[0], self.Z, atol=1e-15)
        np.testing.assert_allclose(e[-1], (self.X - 1j * self.Y) / np.sqrt(2), atol=1e-15)

    def test_polarizations_along_z(self):
        np.testing.assert_allclose(Polarization.linear(self.Z, angle=0.).vector, self.X, atol=1e-15)
        np.testing.assert_allclose(Polarization.linear(self.Z, angle=np.pi / 2).vector, self.Y, atol=1e-15)
        # (x + iy)/sqrt(2) = -e_{+1}: pure sigma+ for a z quantization axis.
        components = Polarization.circular(self.Z, '+').dipole_components()
        self.assertAlmostEqual(abs(components[1]), 1., places=12)
        self.assertAlmostEqual(abs(components[0]) + abs(components[-1]), 0., places=12)

    def test_right_handed_and_orthonormal(self):
        rng = np.random.default_rng(3)  # arbitrary seed
        for n in [*rng.normal(size=(5, 3)), -self.Z, self.X]:
            with self.subTest(n=n):
                n_hat = n / np.linalg.norm(n)
                e1, e2 = _perpendicular_basis(n_hat)
                np.testing.assert_allclose(np.array([e1, e2, n_hat]) @ np.array([e1, e2, n_hat]).T, np.eye(3), atol=1e-12)
                np.testing.assert_allclose(np.cross(e1, e2), n_hat, atol=1e-12)

    def test_roundoff_near_pole(self):
        """A direction that differs from z only by roundoff gives the same basis as z."""
        e1, e2 = _perpendicular_basis(np.array([1e-17, -1e-17, 1.]))
        np.testing.assert_allclose(e1, self.X, atol=1e-12)
        np.testing.assert_allclose(e2, self.Y, atol=1e-12)

    def test_reference_axis(self):
        """With a reference axis, angle 0 polarizes along the axis's component perpendicular to propagation."""
        polarization = Polarization.linear(self.Y, angle=0., ref_axis=np.array([1., 1., 0.]))
        np.testing.assert_allclose(polarization.vector, self.X, atol=1e-15)
        with self.assertRaises(IonSimError):
            _perpendicular_basis(self.Y, ref_axis=self.Y)


class TestLaserMultipleAtoms(unittest.TestCase):
    """Coupling operators for several atoms, including separately built (distinct but identical) atoms."""

    def setUp(self):
        levels = ['S1/2,1,0', 'P1/2,1,-1', 'P1/2,1,0', 'P1/2,1,1']
        self.atom_a = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2'], level_names=levels, name='a')
        self.atom_b = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2'], level_names=levels, name='b')
        self.basis = StandardBasis([self.atom_a, self.atom_b])
        n_hat = np.array([0., 0., 1.])
        self.laser = Laser.gaussian_from_wavelength(369.5e-9, 1e-3, 20e-6, n_hat, Polarization.circular(n_hat, '+'), 0.)
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
        operators = self.laser.build_carrier_coupling_operators(self.basis, None, self.ground, self.excited, 1)
        self.assertEqual(sorted(self.addressed_atoms(operators)), [0, 1])

    def test_selected_atom(self):
        operators = self.laser.build_carrier_coupling_operators(self.basis, [self.atom_b], self.ground, self.excited, 1)
        self.assertEqual(self.addressed_atoms(operators), [1])

    def test_same_couplings_for_identical_atoms(self):
        """Each atom's operator has the same strengths: 4 basis states of the other atom x (|g><e| and |e><g|)."""
        operators = self.laser.build_carrier_coupling_operators(self.basis, None, self.ground, self.excited, 1)
        strengths = [sorted(abs(coupling.strength) for coupling in operator.couplings) for operator in operators]
        self.assertEqual(len(strengths[0]), 8)
        np.testing.assert_allclose(strengths[0], strengths[1], rtol=1e-12)

    def test_individual_atom_with_levels_from_another_atom(self):
        operators = self.laser.build_carrier_coupling_operators(self.basis, [self.atom_b], self.ground, self.excited, 1)
        self.assertEqual(self.addressed_atoms(operators), [1])

    def test_errors(self):
        other = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,1,0'])
        with self.subTest('atom not in basis'), self.assertRaises(ValueError):
            self.laser.build_carrier_coupling_operators(self.basis, [other], self.ground, self.excited, 1)
        missing_level = AtomicStructure.from_species(species='171Yb+', term_symbols=['P3/2'], level_names=['P3/2,1,1']).energy_levels
        with self.subTest('level missing from an atom'), self.assertRaises(ValueError):
            self.laser.build_carrier_coupling_operators(self.basis, None, self.ground, missing_level, 1)


class TestLaserSidebandCouplings(unittest.TestCase):
    """Red/blue sideband couplings on 40Ca+ S1/2 -> D5/2 (729 nm, E2), the standard single-laser sideband transition.

    The laser propagates along x with polarization along z, which couples |S1/2, mJ=1/2> to |D5/2, mJ=3/2> (and -1/2).
    """
    TPI = 2. * np.pi
    MODE_FREQUENCY = 2. * np.pi * 1e6   # rad/s; arbitrary trap frequency
    FOCK_DIMENSION = 6
    ETA = 0.08                          # arbitrary Lamb-Dicke parameter
    N_HAT = np.array([1., 0., 0.])
    QN_G, QN_E = {'n': 4, 'l': 0, 'mj': 0.5}, {'n': 3, 'l': 2, 'j': 2.5, 'mj': 1.5}

    def build(self, ground_first: bool = True, mode_first: bool = False, n_atoms: int = 1, fock_dimension: int | None = None):
        atoms = [AtomicStructure.from_species(species='40Ca+', term_symbols=['S1/2', 'D5/2'], name=f'ion{i}',
                                              quantum_numbers=[self.QN_G, self.QN_E] if ground_first else [self.QN_E, self.QN_G])
                 for i in range(n_atoms)]
        mode = MotionalMode.from_frequency(self.MODE_FREQUENCY, fock_dimension or self.FOCK_DIMENSION, name='mode')
        basis = StandardBasis([mode, *atoms] if mode_first else [*atoms, mode])
        g = next(level for level in atoms[0].energy_levels if level.term_symbol == 'S1/2')
        e = next(level for level in atoms[0].energy_levels if level.term_symbol == 'D5/2')
        return atoms, mode, basis, g, e

    def laser(self, frequency, power=1e-3):
        return Laser.gaussian_from_frequency(frequency, power, 10e-6, self.N_HAT, Polarization(np.array([0., 0., 1.]), self.N_HAT), 0.)

    @staticmethod
    def index(basis, level, n):
        """Basis index of the single-atom state |level, n> (first atom carries the level, the mode the Fock number)."""
        for i, state in enumerate(basis.states):
            if level in state.components and any(getattr(c, 'fock_number', None) == n for c in state.components):
                return i
        raise ValueError(f"No state |{level.name}, {n}>")

    def carrier_rabi(self, laser, atom, basis, g, e):
        return 2. * abs(laser.build_carrier_coupling_operators(basis, [atom], [g], [e], 2)[0].couplings[0].strength)

    def evolve(self, basis, operators, initial_index, duration):
        hamiltonian = Hamiltonian(basis, operators, [-state.energy for state in basis.states])
        psi0 = np.zeros(len(basis.states), dtype=complex)
        psi0[initial_index] = 1.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # Dense output: odeint fails silently (warning only) when its per-interval step limit is exceeded.
            _, psis = hamiltonian.evolve_wavefunction(psi0, duration, np.linspace(0., duration, 2001))
        self.assertAlmostEqual(np.linalg.norm(psis[-1]), 1., delta=1e-5)
        return psis[-1]

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
                    rabi = self.carrier_rabi(blue, atom, basis, g, e)
                    pi_time = np.pi / (self.ETA * rabi)
                    blue_ops = blue.build_blue_sideband_coupling_operators(basis, [atom], [mode], [g], [e], 2, self.ETA)
                    red_ops = red.build_red_sideband_coupling_operators(basis, [atom], [mode], [g], [e], 2, self.ETA)
                    psi = self.evolve(basis, blue_ops, self.index(basis, g, 0), pi_time)
                    self.assertGreater(abs(psi[self.index(basis, e, 1)])**2, 1. - 1e-5)
                    psi = self.evolve(basis, blue_ops, self.index(basis, g, 1), pi_time / np.sqrt(2.))
                    self.assertGreater(abs(psi[self.index(basis, e, 2)])**2, 1. - 1e-5)
                    psi = self.evolve(basis, red_ops, self.index(basis, g, 1), pi_time)
                    self.assertGreater(abs(psi[self.index(basis, e, 0)])**2, 1. - 1e-5)
                    psi = self.evolve(basis, red_ops, self.index(basis, g, 0), pi_time)
                    self.assertGreater(abs(psi[self.index(basis, g, 0)])**2, 1. - 1e-5)

    def test_sideband_matrix_elements(self):
        """Raising elements are (carrier raising element) x i eta sqrt(n+1), on |e,n+1><g,n| (blue) and
        |e,n><g,n+1| (red), and nothing else."""
        (atom,), mode, basis, g, e = self.build()
        laser = self.laser(e.energy - g.energy)
        carrier_raising = laser.build_atom_laser_ge_coupling_matrix(g, e, atom.energy_levels, 2)[
            atom.energy_levels.index(e), atom.energy_levels.index(g)]
        for sideband, shift in (('blue', +1), ('red', -1)):
            with self.subTest(sideband=sideband):
                operators = laser.build_sideband_coupling_operators(sideband, basis, [atom], [mode], [g], [e], 2, self.ETA)
                raising = {}
                for operator in operators:
                    for coupling in operator.couplings:
                        if e in coupling.row_state.components:
                            n_e, n_g = (next(c.fock_number for c in s.components if hasattr(c, 'fock_number'))
                                        for s in (coupling.row_state, coupling.column_state))
                            raising[(n_g, n_e)] = coupling.strength
                expected = {(n, n + shift): carrier_raising * 1j * self.ETA * np.sqrt(max(n, n + shift))
                            for n in range(self.FOCK_DIMENSION) if 0 <= n + shift < self.FOCK_DIMENSION}
                self.assertEqual(set(raising), set(expected))
                for key, value in expected.items():
                    self.assertAlmostEqual(abs(raising[key] / value - 1.), 0., places=12)

    def test_debye_waller_matrix_elements(self):
        """debye_waller=True: blue element eta e^{-eta^2/2} L_n^1(eta^2) / sqrt(n+1) instead of eta sqrt(n+1)."""
        eta = 0.3  # large enough for a visible correction
        (atom,), mode, basis, g, e = self.build()
        laser = self.laser(e.energy - g.energy)
        first_order, exact = (laser.build_blue_sideband_coupling_operators(basis, [atom], [mode], [g], [e], 2, eta, debye_waller=dw)[0]
                              for dw in (False, True))
        def by_n(operator):
            return {next(c.fock_number for c in coupling.column_state.components if hasattr(c, 'fock_number')): coupling.strength
                    for coupling in operator.couplings if e in coupling.row_state.components}
        first_order, exact = by_n(first_order), by_n(exact)
        for n in range(self.FOCK_DIMENSION - 1):
            expected_ratio = np.exp(-eta**2 / 2.) * genlaguerre(n, 1)(eta**2) / (n + 1)
            self.assertAlmostEqual(abs(exact[n] / first_order[n] - expected_ratio), 0., places=12)

    def test_detuned_sideband_compensated_by_frequency_shift(self):
        (atom,), mode, basis, g, e = self.build()
        omega_0 = e.energy - g.energy
        rabi = self.carrier_rabi(self.laser(omega_0), atom, basis, g, e)
        delta = 3. * self.ETA * rabi
        detuned = self.laser(omega_0 + self.MODE_FREQUENCY + delta)
        pi_time = np.pi / (self.ETA * rabi)
        for shift, transferred in ((0., False), (-delta, True)):
            with self.subTest(frequency_shift=shift):
                operators = detuned.build_blue_sideband_coupling_operators(basis, [atom], [mode], [g], [e], 2, self.ETA, frequency_shift=shift)
                population = abs(self.evolve(basis, operators, self.index(basis, g, 0), pi_time)[self.index(basis, e, 1)])**2
                if transferred:
                    self.assertGreater(population, 1. - 1e-5)
                else:
                    self.assertLess(population, 0.2)

    def test_molmer_sorensen_gate(self):
        """Two ions, one mode, bichromatic tones at omega_0 +/- (omega_mode + delta) built from two Laser objects (blue
        sideband of the blue tone, red sideband of the red tone). With eta Omega = delta / 2 and duration 2 pi / delta,
        |gg, 0> -> (|gg> + e^{i theta}|ee>)/sqrt(2) with the mode disentangled, for either basis ordering."""
        duration = 100e-6                       # s; arbitrary
        delta = self.TPI / duration             # one loop in phase space
        sideband_rabi = delta / 2.              # eta * Omega for a maximally entangling single-loop gate
        for mode_first in (False, True):
            with self.subTest(mode_first=mode_first):
                atoms, mode, basis, g, e = self.build(mode_first=mode_first, n_atoms=2, fock_dimension=10)
                omega_0 = e.energy - g.energy
                reference_power = 1e-3
                rabi_per_reference = self.carrier_rabi(self.laser(omega_0, reference_power), atoms[0], basis, g, e)
                power = reference_power * (sideband_rabi / self.ETA / rabi_per_reference)**2   # Omega scales as sqrt(P)
                blue = self.laser(omega_0 + self.MODE_FREQUENCY + delta, power)
                red = self.laser(omega_0 - self.MODE_FREQUENCY - delta, power)
                etas = {(atom, mode): self.ETA for atom in atoms}
                operators = (blue.build_blue_sideband_coupling_operators(basis, None, [mode], [g], [e], 2, etas)
                             + red.build_red_sideband_coupling_operators(basis, None, [mode], [g], [e], 2, etas))
                initial = next(i for i, state in enumerate(basis.states)                       # |g, g, n=0>
                               if all(getattr(c, 'fock_number', 0) == 0 and getattr(c, 'term_symbol', 'S1/2') == 'S1/2'
                                      for c in state.components))
                psi = self.evolve(basis, operators, initial, duration)
                spins = State.from_wavefunction(basis, psi).trace_out_degree_of_freedom(mode)
                rho = spins.density_matrix
                labels = [''.join('g' if c.term_symbol == 'S1/2' else 'e' for c in s.components) for s in spins.basis.states]
                gg, ee = labels.index('gg'), labels.index('ee')
                bell_fidelity = 0.5 * (rho[gg, gg] + rho[ee, ee]).real + abs(rho[gg, ee])   # best phase theta
                self.assertGreater(bell_fidelity, 1. - 1e-4)
                self.assertGreater(np.trace(rho @ rho).real, 1. - 1e-4)                 # mode disentangled

    def test_lamb_dicke_parameter(self):
        laser = self.laser(2. * np.pi * const.c / 729e-9)
        mass = 39.96259087  # daltons, 40Ca
        expected = laser.wavevector[0] * np.sqrt(const.hbar / (2. * mass * const.value('atomic mass constant') * self.MODE_FREQUENCY))
        self.assertAlmostEqual(laser.lamb_dicke_parameter(self.MODE_FREQUENCY, mass, [1., 0., 0.]), expected, places=14)
        self.assertAlmostEqual(expected, 0.0969, places=4)  # 729 nm, 1 MHz, k along the mode
        self.assertAlmostEqual(laser.lamb_dicke_parameter(self.MODE_FREQUENCY, mass, [-1. / np.sqrt(2.), 0., 0.]),
                               -expected / np.sqrt(2.), places=14)
        self.assertAlmostEqual(laser.lamb_dicke_parameter(self.MODE_FREQUENCY, mass, [0., 0., 1.]), 0., places=14)

    def test_coupling_operator_from_raising_matrix(self):
        """The helper returns one operator whose rate sign follows the triangle of the raising elements, None when every
        element is below SMALLEST_ENERGY_SCALE, and raises for diagonal elements or elements in both triangles."""
        (atom,), mode, basis, g, e = self.build(ground_first=True, mode_first=True)
        laser = self.laser(e.energy - g.energy)
        dimension = len(basis.states)
        strength = 2. * np.pi * 1e3  # rad/s; arbitrary, well above SMALLEST_ENERGY_SCALE
        def matrix(*elements):
            m = np.zeros((dimension, dimension), dtype=complex)
            for row, column, value in elements:
                m[row, column] = value
            return m
        with self.subTest('lower triangle: ground first, rate -omega'):
            operator = laser._coupling_operator_from_raising_matrix(basis, matrix((1, 0, strength), (3, 2, strength)), 0.)
            self.assertEqual({c.oscillation_rate for c in operator.couplings}, {-laser.frequency})
            self.assertEqual(len(operator.couplings), 4)  # 2 raising elements + Hermitian conjugates
        with self.subTest('upper triangle: excited first, rate +omega'):
            operator = laser._coupling_operator_from_raising_matrix(basis, matrix((0, 1, strength), (2, 3, strength)), 0.)
            self.assertEqual({c.oscillation_rate for c in operator.couplings}, {laser.frequency})
        with self.subTest('below threshold -> None'):
            self.assertIsNone(laser._coupling_operator_from_raising_matrix(basis, matrix((1, 0, 1e-6)), 0.))
        with self.subTest('sub-threshold element in the other triangle is ignored'):
            operator = laser._coupling_operator_from_raising_matrix(basis, matrix((1, 0, strength), (0, 3, 1e-6)), 0.)
            self.assertEqual(len(operator.couplings), 2)
        with self.subTest('both triangles'), self.assertRaises(IonSimError):
            laser._coupling_operator_from_raising_matrix(basis, matrix((1, 0, strength), (0, 3, strength)), 0.)
        with self.subTest('diagonal element'), self.assertRaises(IonSimError):
            laser._coupling_operator_from_raising_matrix(basis, matrix((1, 1, strength)), 0.)
        with self.subTest('carrier + red sideband in one matrix, mode first'), self.assertRaises(IonSimError):
            levels = atom.energy_levels
            atomic = np.zeros((len(levels), len(levels)), dtype=complex)
            atomic[levels.index(e), levels.index(g)] = strength
            carrier = basis.enlarge_matrix(atomic, [atom])
            red = carrier @ basis.enlarge_matrix(laser._sideband_motional_matrix('red', mode, self.ETA, False), [mode])
            laser._coupling_operator_from_raising_matrix(basis, carrier + red, 0.)

    def test_target_modes_are_required(self):
        """target_modes must be a non-empty list of distinct MotionalMode DOFs of the basis; there is no default."""
        (atom,), mode, basis, g, e = self.build()
        laser = self.laser(e.energy - g.energy)
        build = lambda target_modes: laser.build_blue_sideband_coupling_operators(basis, [atom], target_modes, [g], [e], 2, self.ETA)
        for case, target_modes in (('None', None), ('empty list', []), ('bare mode, not a list', mode),
                                   ('non-mode entry', [atom]), ('duplicate mode', [mode, mode])):
            with self.subTest(case), self.assertRaises(IonSimError):
                build(target_modes)
        with self.subTest('tuple accepted'):
            self.assertEqual(len(build((mode,))), len(build([mode])))

    def test_input_errors(self):
        (atom,), mode, basis, g, e = self.build()
        laser = self.laser(e.energy - g.energy)
        no_mode_basis = StandardBasis([atom])
        other_mode = MotionalMode.from_frequency(self.MODE_FREQUENCY, 3)
        with self.subTest('no modes in basis'), self.assertRaises(ValueError):
            laser.build_red_sideband_coupling_operators(no_mode_basis, [atom], [other_mode], [g], [e], 2, self.ETA)
        with self.subTest('mode not in basis'), self.assertRaises(ValueError):
            laser.build_red_sideband_coupling_operators(basis, [atom], [other_mode], [g], [e], 2, self.ETA)
        with self.subTest('missing (atom, mode) entry'), self.assertRaises(IonSimError):
            laser.build_red_sideband_coupling_operators(basis, [atom], [mode], [g], [e], 2, {})
        with self.subTest('unknown sideband'), self.assertRaises(IonSimError):
            laser.build_sideband_coupling_operators('green', basis, [atom], [mode], [g], [e], 2, self.ETA)
        with self.subTest('eta = 0 skips the pair'):
            self.assertEqual(laser.build_blue_sideband_coupling_operators(basis, [atom], [mode], [g], [e], 2, {(atom, mode): 0.}), [])


class TestLaserTargetRabiFrequency(unittest.TestCase):
    """Lasers constructed from a target Rabi frequency on one transition, and Laser.rabi_frequency()."""
    TPI = 2. * np.pi
    Z = np.array([0., 0., 1.])
    X = np.array([1., 0., 0.])

    def yb_transition(self):
        """171Yb+ |S1/2, F=1, mF=0> -> |P1/2, F=1, mF=1> (E1), driven by sigma+ light along z."""
        atom = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2'],
                                            quantum_numbers=[{'n': 6, 'l': 0, 'f': 1, 'mf': 0}, {'n': 6, 'l': 1, 'f': 1, 'mf': 1}])
        g, e = atom.energy_levels
        return atom, g, e, Polarization.circular(self.Z, '+')

    def ca_transition(self):
        """40Ca+ |S1/2, mJ=1/2> -> |D5/2, mJ=3/2> (E2, 729 nm), light along x polarized along z."""
        atom = AtomicStructure.from_species(species='40Ca+', term_symbols=['S1/2', 'D5/2'],
                                            quantum_numbers=[{'n': 4, 'l': 0, 'mj': 0.5}, {'n': 3, 'l': 2, 'j': 2.5, 'mj': 1.5}])
        g, e = atom.energy_levels
        return atom, g, e, Polarization(np.array([0., 0., 1.]), self.X)

    def test_round_trip_E1(self):
        atom, g, e, polarization = self.yb_transition()
        target, detuning = self.TPI * 2e6, self.TPI * 10e6   # rad/s; arbitrary
        lasers = {
            'plane wave': Laser.from_target_rabi_frequency(target, g, e, 1, self.Z, polarization),
            'gaussian': Laser.gaussian_from_target_rabi_frequency(target, g, e, 1, 20e-6, self.Z, polarization),
            'plane wave, detuned': Laser.from_target_rabi_frequency(target, g, e, 1, self.Z, polarization, detuning=detuning),
            'gaussian, detuned': Laser.gaussian_from_target_rabi_frequency(target, g, e, 1, 20e-6, self.Z, polarization, detuning=detuning),
        }
        for name, laser in lasers.items():
            with self.subTest(name):
                self.assertAlmostEqual(abs(laser.rabi_frequency(g, e, 1)) / target, 1., places=12)
                expected_frequency = (e.energy - g.energy) + (detuning if 'detuned' in name else 0.)
                self.assertAlmostEqual(laser.frequency / expected_frequency, 1., places=14)
                self.assertIsInstance(laser.beam_profile, GaussianBeam if 'gaussian' in name else PlaneWave)
                # The coupling builder uses the same Rabi frequency: absorption element strength Omega / 2.
                operator, = laser.build_carrier_coupling_operators(StandardBasis([atom]), [atom], [g], [e], 1)
                self.assertAlmostEqual(abs(operator.couplings[0].strength) / (target / 2.), 1., places=12)

    def test_round_trip_E2(self):
        """E2 couplings depend on |k|, which the constructor evaluates at the laser's own frequency (including detuning)."""
        atom, g, e, polarization = self.ca_transition()
        target = self.TPI * 100e3   # rad/s; arbitrary
        for detuning in (0., self.TPI * 1e9):
            with self.subTest(detuning=detuning):
                for laser in (Laser.from_target_rabi_frequency(target, g, e, 2, self.X, polarization, detuning=detuning),
                              Laser.gaussian_from_target_rabi_frequency(target, g, e, 2, 10e-6, self.X, polarization, detuning=detuning)):
                    self.assertAlmostEqual(abs(laser.rabi_frequency(g, e, 2)) / target, 1., places=12)

    def test_gaussian_power_consistency(self):
        """The derived power reproduces the target through the power-based constructor, and scales as Omega^2."""
        atom, g, e, polarization = self.yb_transition()
        target, waist = self.TPI * 1e6, 15e-6
        laser = Laser.gaussian_from_target_rabi_frequency(target, g, e, 1, waist, self.Z, polarization)
        rebuilt = Laser.gaussian_from_frequency(laser.frequency, laser.power, waist, self.Z, polarization, 0.)
        self.assertAlmostEqual(abs(rebuilt.rabi_frequency(g, e, 1)) / target, 1., places=12)
        doubled = Laser.gaussian_from_target_rabi_frequency(2. * target, g, e, 1, waist, self.Z, polarization)
        self.assertAlmostEqual(doubled.power / laser.power, 4., places=12)
        self.assertAlmostEqual(laser.beam_profile.waist, waist)

    def test_phase_and_modulation_are_passed_through(self):
        atom, g, e, polarization = self.yb_transition()
        modulation = {'amplitude': lambda t: 0.5}
        laser = Laser.gaussian_from_target_rabi_frequency(self.TPI * 1e6, g, e, 1, 15e-6, self.Z, polarization, phase=0.7,
                                                          modulation_functions=modulation)
        self.assertEqual(laser.phase, 0.7)
        self.assertIs(laser.modulation_functions, modulation)
        # rabi_frequency() excludes the static laser phase and the modulation (peak field, amplitude 1).
        unphased = Laser.gaussian_from_target_rabi_frequency(self.TPI * 1e6, g, e, 1, 15e-6, self.Z, polarization)
        self.assertAlmostEqual(abs(laser.rabi_frequency(g, e, 1) - unphased.rabi_frequency(g, e, 1)), 0., delta=1e-6)

    def test_resonant_pi_pulse(self):
        """A resonant laser built for Omega transfers |g> -> |e> in a pi pulse of duration pi / Omega."""
        atom, g, e, polarization = self.yb_transition()
        target = self.TPI * 1e6
        laser = Laser.from_target_rabi_frequency(target, g, e, 1, self.Z, polarization)
        basis = StandardBasis([atom])
        hamiltonian = Hamiltonian(basis, laser.build_carrier_coupling_operators(basis, [atom], [g], [e], 1),
                                  [-state.energy for state in basis.states])
        index_g = [state.components[0] for state in basis.states].index(g)
        psi0 = np.zeros(len(basis.states), dtype=complex)
        psi0[index_g] = 1.
        duration = np.pi / target
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, psis = hamiltonian.evolve_wavefunction(psi0, duration, np.linspace(0., duration, 2001))
        self.assertAlmostEqual(np.linalg.norm(psis[-1]), 1., delta=1e-5)
        self.assertLess(abs(psis[-1][index_g])**2, 1e-6)

    def test_errors(self):
        atom, g, e, polarization = self.yb_transition()
        target = self.TPI * 1e6
        cases = {
            'polarization does not drive the transition': lambda: Laser.from_target_rabi_frequency(
                target, g, e, 1, self.Z, Polarization.circular(self.Z, '-')),
            'non-positive target': lambda: Laser.from_target_rabi_frequency(0., g, e, 1, self.Z, polarization),
            'complex target': lambda: Laser.from_target_rabi_frequency(target * 1j, g, e, 1, self.Z, polarization),
            'excited level below ground level': lambda: Laser.from_target_rabi_frequency(target, e, g, 1, self.Z, polarization),
            'negative laser frequency': lambda: Laser.from_target_rabi_frequency(target, g, e, 1, self.Z, polarization,
                                                                                 detuning=-2. * (e.energy - g.energy)),
            'invalid multipole order': lambda: Laser.gaussian_from_target_rabi_frequency(target, g, e, 3, 10e-6, self.Z, polarization),
        }
        for case, build in cases.items():
            with self.subTest(case), self.assertRaises(IonSimError):
                build()
        laser = Laser.from_target_rabi_frequency(target, g, e, 1, self.Z, polarization)
        with self.subTest('same level twice'), self.assertRaises(ValueError):
            laser.rabi_frequency(g, g, 1)


class TestPhysicalDipoleCouplings(unittest.TestCase):
    """E1 couplings with tabulated reduced matrix elements <J||er||J'> (Steck convention), benchmarked against
    D. A. Steck, "Rubidium 87 D Line Data", rev. 1.6 (the reference of the 87Rb config file)."""
    TPI = 2. * np.pi
    Z = np.array([0., 0., 1.])
    E_A0_OVER_HBAR = const.e * const.value('Bohr radius') / const.hbar

    def test_rb87_d2_cycling_saturation_intensity(self):
        """sigma+ |F=2, mF=2> -> |F'=3, mF'=3> at I_sat = 1.669 mW/cm^2 (Steck Table 7) has Omega = Gamma/sqrt(2)
        (I/I_sat = 2 Omega^2/Gamma^2, Gamma = 2pi x 6.065 MHz, Table 3), and effective dipole 2.989 e a0 (Table 7)."""
        rb = AtomicStructure.from_species(species='87Rb', term_symbols=['S1/2', 'P3/2'], level_names=['S1/2,2,2', 'P3/2,3,3'])
        g, e = rb.energy_levels
        laser = Laser.from_frequency(e.energy - g.energy, self.Z, 0., Polarization.circular(self.Z, '+'), PlaneWave(1.669e-3 / 1e-4), None)
        rabi = abs(laser.rabi_frequency(g, e, 1))
        self.assertAlmostEqual(rabi / (self.TPI * 6.065e6 / np.sqrt(2.)), 1., delta=2e-3)    # Steck values: ~0.15% uncertainty
        dipole = rabi / laser.peak_electric_field_magnitude / self.E_A0_OVER_HBAR             # |<e|r.eps|g>| / a0
        self.assertAlmostEqual(dipole / 2.989, 1., delta=1e-3)

    def test_reduced_elements_reproduce_config_lifetimes(self):
        """1/tau = omega_0^3 / (3 pi eps0 hbar c^3) (2J+1)/(2J'+1) |<J||er||J'>|^2 (Steck Eq. 38) for D1 and D2, with the
        config's reduced elements, fine-structure energies and lifetimes."""
        rb = AtomicStructure.from_species(species='87Rb', term_symbols=['S1/2', 'P1/2', 'P3/2'])
        manifold = {level.term_symbol: level for level in rb.energy_levels}
        ground = manifold['S1/2']
        for upper in ('P1/2', 'P3/2'):
            with self.subTest(upper=upper):
                excited = manifold[upper]
                reduced = reduced_dipole_matrix_element(ground, excited) * const.e * const.value('Bohr radius')
                omega_0 = excited.fine_energy - ground.fine_energy
                rate = (omega_0**3 / (3. * np.pi * const.epsilon_0 * const.hbar * const.c**3)
                        * (2. * ground.j + 1.) / (2. * excited.j + 1.) * reduced**2)
                self.assertAlmostEqual(rate * excited.lifetime, 1., delta=5e-3)

    def test_lookup_and_fallback(self):
        rb = AtomicStructure.from_species(species='87Rb', term_symbols=['S1/2', 'P3/2', '6 P3/2'],
                                          level_names=['S1/2,2,2', 'P3/2,3,3', '6 P3/2,3,3'])
        g, e_5p, e_6p = rb.energy_levels
        components = Polarization.circular(self.Z, '+').dipole_components()
        def expected(excited, in_orbital_units, reduced=1.):
            return reduced * self.E_A0_OVER_HBAR * sum(
                components[q] * (-1)**q * compute_multipole_amplitude(g, excited, 1, -q, include_J_J_prime_matrix_element=in_orbital_units)
                for q in (-1, 0, 1))
        with self.subTest('tabulated <J||er||J\'> (5S -> 5P3/2): J-unit angular factor times the value'):
            self.assertEqual(reduced_dipole_matrix_element(g, e_5p), 4.227)
            coupling = compute_coupling_amplitude_between_atomic_levels(g, e_5p, components, [g, e_5p], 1)
            self.assertAlmostEqual(abs(coupling / expected(e_5p, False, 4.227) - 1.), 0., places=12)
        with self.subTest('no entry (5S -> 6P3/2): orbital units, <L||r||L\'> = 1 a0'):
            self.assertIsNone(reduced_dipole_matrix_element(g, e_6p))
            coupling = compute_coupling_amplitude_between_atomic_levels(g, e_6p, components, [g, e_6p], 1)
            self.assertAlmostEqual(abs(coupling / expected(e_6p, True) - 1.), 0., places=12)
        with self.subTest('stored like branching_ratios: on the upper manifold, keyed by the lower term symbol'):
            self.assertEqual(e_5p.reduced_dipole_matrix_elements, {'S1/2': 4.227})
            self.assertIsNone(g.reduced_dipole_matrix_elements)
        with self.subTest('entry stored on the lower manifold raises'), self.assertRaises(IonSimError):
            g_wrong = dataclasses.replace(g, reduced_dipole_matrix_elements={'P3/2': 4.227})
            e_bare = dataclasses.replace(e_5p, reduced_dipole_matrix_elements=None)
            compute_coupling_amplitude_between_atomic_levels(g_wrong, e_bare, components, [g_wrong, e_bare], 1)

    def test_e2_normalization_unchanged(self):
        """E2 keeps its previous absolute normalization: 2 x angular x (e a0/hbar) x i k a0 sqrt(2/3), in orbital units."""
        ca = AtomicStructure.from_species(species='40Ca+', term_symbols=['S1/2', 'D5/2'],
                                          quantum_numbers=[{'n': 4, 'l': 0, 'mj': 0.5}, {'n': 3, 'l': 2, 'j': 2.5, 'mj': 1.5}])
        g, e = ca.energy_levels
        x = np.array([1., 0., 0.])
        components = Polarization(np.array([0., 0., 1.]), x).quadrupole_components()
        k = 2. * np.pi / 729e-9
        angular = sum(components[q] * (-1)**q * compute_multipole_amplitude(g, e, 2, -q, include_J_J_prime_matrix_element=True)
                      for q in range(-2, 3))
        previous = 2. * angular * self.E_A0_OVER_HBAR * 1j * k * const.value('Bohr radius') * np.sqrt(2. / 3.)
        coupling = compute_coupling_amplitude_between_atomic_levels(g, e, components, [g, e], 2, wavenumber=k)
        self.assertAlmostEqual(abs(coupling / previous - 1.), 0., places=12)


if __name__ == '__main__':
    unittest.main()
