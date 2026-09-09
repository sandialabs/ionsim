#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

import unittest
import numpy as np
from ionsim.degree_of_freedom import AtomicStructure, MotionalMode
from ionsim.energy_level import EnergyEigenstate
from ionsim.basis import StandardBasis, ZPauliBasis, XPauliBasis
from ionsim.testing import assert_array_close

class TestBasis(unittest.TestCase):

    def setUp(self):
        """Set up the necessary objects for testing."""
        self.spin_a = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'], level_aliases=['0', '1'])
        self.spin_b = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'], level_aliases=['0', '1'])
        self.spin_c = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2','P1/2'], level_names=['S1/2,0,0', 'S1/2,1,0', 'P1/2,1,0'], level_aliases=['0', '1', 'R'])
        self.spin_d = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2','P1/2'], level_names=['S1/2,0,0', 'S1/2,1,0', 'P1/2,1,0'], level_aliases=['0', '1', 'R'])
        self.mode_0 = MotionalMode.from_frequency(frequency=3e6*2*np.pi, fock_dimension=3, level_aliases = ['Mode 0, n = ' + str(n) for n in range(3)])
        self.mode_1 = MotionalMode.from_frequency(frequency=4e6*2*np.pi, fock_dimension=2, level_aliases = ['Mode 1, n = ' + str(n) for n in range(2)])

    def test_spin_basis_states(self):
        """Test the states of the spin basis."""
        spin_basis = StandardBasis([self.spin_a, self.spin_b])
        expected_states = [
            'S1/2,0,0 : S1/2,0,0',
            'S1/2,0,0 : S1/2,1,0',
            'S1/2,1,0 : S1/2,0,0',
            'S1/2,1,0 : S1/2,1,0'
        ]
        actual_states = [state.name for state in spin_basis.states]
        self.assertEqual(actual_states, expected_states)

        expected_aliases = ['00', '01', '10', '11']
        actual_aliases = [state.alias() for state in spin_basis.states]

        # Test aliases for spin basis states 
        self.assertEqual(expected_aliases, actual_aliases)

    def test_motional_basis_states(self):
        """Test the states of the motional basis."""
        motional_basis = StandardBasis([self.mode_0, self.mode_1])
        expected_states = [
            '0 : 0',
            '0 : 1',
            '1 : 0',
            '1 : 1',
            '2 : 0',
            '2 : 1'
        ]
        actual_states = [state.name for state in motional_basis.states]
        self.assertEqual(actual_states, expected_states)

        # Test alias for motional basis state 
        self.assertEqual(motional_basis.states[4].alias('; '), 'Mode 0, n = 2; Mode 1, n = 0') 

    def test_full_basis_states(self):
        """Test the states of the full basis."""
        full_basis = StandardBasis([self.spin_a, self.spin_b, self.mode_0, self.mode_1])
        expected_states = [
            'S1/2,0,0 : S1/2,0,0 : 0 : 0',
            'S1/2,0,0 : S1/2,0,0 : 0 : 1',
            'S1/2,0,0 : S1/2,0,0 : 1 : 0',
            'S1/2,0,0 : S1/2,0,0 : 1 : 1',
            'S1/2,0,0 : S1/2,0,0 : 2 : 0',
            'S1/2,0,0 : S1/2,0,0 : 2 : 1',
            'S1/2,0,0 : S1/2,1,0 : 0 : 0',
            'S1/2,0,0 : S1/2,1,0 : 0 : 1',
            'S1/2,0,0 : S1/2,1,0 : 1 : 0',
            'S1/2,0,0 : S1/2,1,0 : 1 : 1',
            'S1/2,0,0 : S1/2,1,0 : 2 : 0',
            'S1/2,0,0 : S1/2,1,0 : 2 : 1',
            'S1/2,1,0 : S1/2,0,0 : 0 : 0',
            'S1/2,1,0 : S1/2,0,0 : 0 : 1',
            'S1/2,1,0 : S1/2,0,0 : 1 : 0',
            'S1/2,1,0 : S1/2,0,0 : 1 : 1',
            'S1/2,1,0 : S1/2,0,0 : 2 : 0',
            'S1/2,1,0 : S1/2,0,0 : 2 : 1',
            'S1/2,1,0 : S1/2,1,0 : 0 : 0',
            'S1/2,1,0 : S1/2,1,0 : 0 : 1',
            'S1/2,1,0 : S1/2,1,0 : 1 : 0',
            'S1/2,1,0 : S1/2,1,0 : 1 : 1',
            'S1/2,1,0 : S1/2,1,0 : 2 : 0',
            'S1/2,1,0 : S1/2,1,0 : 2 : 1'
        ]
        actual_states = [state.name for state in full_basis.states]
        self.assertEqual(actual_states, expected_states)
        # Test alias for full basis state 
        self.assertEqual(full_basis.states[3].alias('; '), '0; 0; Mode 0, n = 1; Mode 1, n = 1') 

    def test_z_pauli_basis_vectors(self):
        """Test the vectors of the Z Pauli basis."""
        basis_z = ZPauliBasis([self.spin_a])
        expected_vectors = [
            np.array([1., 0.]),
            np.array([0., 1.])
        ]
        actual_vectors = basis_z.vectors
        for expected, actual in zip(expected_vectors, actual_vectors):
            with self.subTest(expected=expected, actual=actual):
                assert_array_close(expected, actual)

    def test_x_pauli_basis_vectors(self):
        """Test the vectors of the X Pauli basis."""
        basis_x = XPauliBasis([self.spin_a])
        expected_vectors = [
            np.array([0.70710678, 0.70710678]),
            np.array([0.70710678, -0.70710678])
        ]
        actual_vectors = basis_x.vectors
        for expected, actual in zip(expected_vectors, actual_vectors):
            with self.subTest(expected=expected, actual=actual):
                assert_array_close(expected, actual)

    def test_xx_pauli_basis_vectors(self):
        """Test the vectors of the XX Pauli basis."""
        basis_xx = XPauliBasis([self.spin_a, self.spin_b])
        expected_vectors = [
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([0.5, -0.5, 0.5, -0.5]),
            np.array([0.5, 0.5, -0.5, -0.5]),
            np.array([0.5, -0.5, -0.5, 0.5])
        ]
        actual_vectors = basis_xx.vectors
        for expected, actual in zip(expected_vectors, actual_vectors):
            with self.subTest(expected=expected, actual=actual):
                assert_array_close(expected, actual)

    def test_subspace_bases(self):
        """ Test projection from 3-level ions to 2-level ions via 2 methods """ 
        full_basis = StandardBasis([self.spin_c, self.spin_d])

        # Test projection by specifying levels to project out. 
        undesired_levels = [self.spin_c.energy_levels[2], self.spin_d.energy_levels[2]] # P1/2 level to project out        
        reduced_basis = full_basis.build_subspace_basis_from_levels_to_project(undesired_levels)

        # Test projection by specifying states          
        states_to_project = []
        for s in full_basis.states:
            for l in undesired_levels:
                if l in s.components:
                    states_to_project.append(s)

        reduced_basis_v2 = full_basis.build_subspace_basis_from_states_to_project(states_to_project)
        for state1, state2 in zip(reduced_basis.states, reduced_basis_v2.states):
            self.assertAlmostEqual(state1.energy, state2.energy, places=10) 
            self.assertEqual(state1.name, state2.name)


if __name__ == '__main__':
    unittest.main()
