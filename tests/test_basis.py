import unittest
import numpy as np
from ionsim.degree_of_freedom import AtomicStructure, MotionalMode
from ionsim.energy_level import EnergyEigenstate
from ionsim.basis import StandardBasis, ZPauliBasis, XPauliBasis
from ionsim.testing import assert_array_close

class TestBasis(unittest.TestCase):

    def setUp(self):
        """Set up the necessary objects for testing."""
        self.spin_a = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.spin_b = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.mode_0 = MotionalMode.from_frequency(frequency=3e6*2*np.pi, fock_dimension=3)
        self.mode_1 = MotionalMode.from_frequency(frequency=4e6*2*np.pi, fock_dimension=2)

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

if __name__ == '__main__':
    unittest.main()
