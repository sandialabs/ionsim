import unittest

import numpy as np

from ionsim.state import State
from ionsim.degree_of_freedom import AtomicSpin, MotionalMode
from ionsim.basis import StandardBasis, XPauliBasis
from ionsim.named_operators import Pauli
from ionsim.testing import assert_array_close

class TestState(unittest.TestCase):

    def setUp(self):
        """Set up the necessary objects for testing."""
        self.spin_a = AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.spin_b = AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.basis = StandardBasis([self.spin_a, self.spin_b])
        self.basis_x = XPauliBasis([self.spin_a, self.spin_b])
        self.eig_x = np.array([[1, 0], [0, -1]])
        self.state = State.from_density_matrix(self.basis_x, np.kron(self.eig_x, Pauli.I))

        # Set up spin-motional state with 2 spins and 1 motional mode  
        self.mode = MotionalMode.from_frequency(frequency=2*np.pi * 3e6, fock_dimension=5)
        self.spin_motion_basis = StandardBasis([self.spin_a, self.spin_b, self.mode])
        state_coefficients = np.zeros(len(self.spin_motion_basis.states))
        state_coefficients[0] = 1.
        self.spin_motion_state = State.from_coefficients(self.spin_motion_basis, list(state_coefficients))

    def test_density_matrix_in_new_basis(self):
        """Test the density matrix in the new basis."""
        rho_p = State.from_state(self.basis, self.state).density_matrix
        expected_rho_p = np.kron(Pauli.X, Pauli.I)

        # Check that the density matrix matches the expected value
        assert_array_close(rho_p, expected_rho_p)

    def test_wigner_function(self):
        """ Test Wigner function computation from a motional density matrix in the Fock state basis""" 
        domain_limit = 2.
        x_grid = np.linspace(-domain_limit, domain_limit, 25)
        p_grid = np.linspace(-domain_limit, domain_limit, 25)
        W_distribution = self.spin_motion_state.compute_wigner_distribution(x_grid, p_grid)[0]        

        # Test just one slice of the array and half of its contents (U(1) symmetric about origin)) 
        slice_indx = 12
        W_expected_at_slice = np.array([0.00583005, 0.0110443, 0.0197914, 0.03354962, 0.05379861, 0.08160694, 0.11709966, 0.15894861, 0.20409406, 0.24789999, 0.2848362, 0.30958962, 0.31830989]) 
        norm = np.trapezoid(np.trapezoid(W_distribution, x_grid, axis=0), p_grid)
        expected_norm = 0.9902872798196112
        self.assertAlmostEqual(norm, expected_norm, places=7)
        expected_W_max = 0.31830988618379075
        W_max = np.max(W_distribution)
        self.assertAlmostEqual(W_max, expected_W_max, places=7)
        assert_array_close(W_distribution[slice_indx, 0:13], W_expected_at_slice, rtol=1e-04, atol=1E-7) 


if __name__ == '__main__':
    unittest.main()
