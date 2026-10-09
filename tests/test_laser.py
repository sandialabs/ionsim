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
from ionsim.degree_of_freedom import AtomicStructure
from ionsim.basis import StandardBasis
from ionsim.laser import Laser, Polarization, GaussianBeam
from ionsim.hamiltonian import Hamiltonian
from ionsim.atomic_internal_energy_level import LSFineLevel, compute_coupling_amplitude_between_atomic_levels
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


 #    def test_polarization(self):
 #        """ Test polarization vector functionality """ 
 #        laser_polarization = self.laser.polarization


    def test_laser_coupling_builder(self):
        """Test the process fidelity of the extra noisy gate."""
        # Test building coupling operators from laser and polarization information  
        ground_levels = [self.atom_a.energy_levels[0]] 
        excited_levels = [*self.atom_a.energy_levels[1:]] 
        atom_a_coupling_operators = self.laser.build_individual_atom_laser_coupling_operators(self.basis, self.atom_a, ground_levels, excited_levels, 1) 
        # These are not dipole allowed: should be zero operators 
        self.assertEqual(len(atom_a_coupling_operators), 0)

        ground_levels = [self.atom_b.energy_levels[0]] 
        excited_levels = [*self.atom_b.energy_levels[1:]] 
        # Addresses atom B in a 2-atom basis 
        atom_b_coupling_operators = self.laser.build_individual_atom_laser_coupling_operators(self.basis, self.atom_b, ground_levels, excited_levels, 1) 
        # Expect one transition |s1/2, 1,0> --> |P1/2,1,+1> transition, and 4 couplings since atom A has 4 levels, so the 2-qubit basis has 4 couplings  
        self.assertEqual(len(atom_b_coupling_operators), 1)
        # For 2-qubits and a + transition on qubit 2 and 4 levels on qubit 1, there are 4 unique couplings and 8 total couplings (h.c.) 
        self.assertEqual(len(atom_b_coupling_operators[0].couplings), 8)

        # Check that the transition is with P1/2,1,+1;
        excited_level_name = atom_b_coupling_operators[0].couplings[0].column_state.name[-8:]
        expected_name = 'P1/2,1,1'
        self.assertEqual(expected_name, excited_level_name)
 #        for op in atom_b_coupling_operators:
 #            print(f"Coupling operator contains {len(op.couplings)} couplings.")
 #            for coupling in op.couplings:
 #                print(f"Coupling between {coupling.row_state.name} and {coupling.column_state.name}.")
 #                print(f"Strength: {coupling.strength}\n")
 #            print()


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
        atom_c_coupling_operators = self.laser_c.build_individual_atom_laser_coupling_operators(self.neutral_basis, self.atom_c, ground_levels, excited_levels, 1) 
        assert len(atom_c_coupling_operators) == 1

        # We expect a Rabi frequency of about ~5 MHz, but this is not precise  
        dipole_factor = 0.5398 # Table 10 of reference 
        rabi_frequency = dipole_factor * atom_c_coupling_operators[0].couplings[0].strength/TPI/1E6
        self.assertAlmostEqual(np.abs(rabi_frequency), 3.905827783900461, places = 8)


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
        operators = laser.build_individual_atom_laser_coupling_operators(basis, ca, ground, excited, 2)
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
        operators = laser.build_individual_atom_laser_coupling_operators(basis, atom, [g], [e], 1, frequency_shift=frequency_shift)
        frame = [0.] * len(basis.states) if frame is None else frame
        return np.asarray(Hamiltonian(basis, operators, frame).hamiltonian_function(t))[0, 1]

    def test_resonant_laser_is_static_in_level_frame(self):
        for ground_first in (True, False):
            with self.subTest(ground_first=ground_first):
                atom, basis, g, e = self.build(ground_first)
                n_hat = np.array([0., 0., 1.])
                laser = Laser.gaussian_from_frequency(e.energy - g.energy, 1e-3, 20e-6, n_hat, Polarization.circular(n_hat, '+'), 0.)
                frame = [state.energy for state in basis.states]
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


if __name__ == '__main__':
    unittest.main()
