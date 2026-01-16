import unittest
import numpy as np
from ionsim.degree_of_freedom import AtomicSpin
from ionsim.energy_level import EnergyEigenstate
from ionsim.basis import StandardBasis
from ionsim.named_operators import Pauli
from ionsim.operator import CouplingOperator
from ionsim.testing import assert_array_close

class TestCouplingOperator(unittest.TestCase):

    def setUp(self):
        """Set up the necessary objects for testing."""
        self.spin_a = AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.spin_b = AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.basis = StandardBasis([self.spin_a, self.spin_b])  # 00, 01, 10, 11

        self.rabi_rate = 100e3 * 2 * np.pi  # rad/s
        self.omega = self.spin_a.energy_levels[1].energy - self.spin_a.energy_levels[0].energy
        self.raising_operator = self.rabi_rate / 2 * Pauli.plus
        self.lower_operator = self.rabi_rate / 2 * Pauli.minus

        self.spin_X_operator = Pauli.X 
        self.spin_Y_operator = Pauli.Y         

        # Create the CouplingOperator instances
        self.raise_spin_a = CouplingOperator.from_matrix(self.basis, self.raising_operator, self.omega, [self.spin_a])
        self.lower_spin_a = CouplingOperator.from_matrix(self.basis, self.lower_operator, -self.omega, [self.spin_a])
        self.X_spin_a = CouplingOperator.from_matrix(self.basis, self.spin_X_operator, 0., [self.spin_a])
        self.Y_spin_b = CouplingOperator.from_matrix(self.basis, self.spin_Y_operator, 0., [self.spin_b])


    def test_static_matrix(self):
        """Test the static matrix of the CouplingOperator."""
        big_matrix = self.raise_spin_a.static_matrix.toarray()
        big_lowering_matrix = self.lower_spin_a.static_matrix.toarray()
        expected_static_matrix = np.kron(self.raising_operator, np.eye(2))
        expected_static_lowering_matrix = np.kron(self.lower_operator, np.eye(2))
        
        # Assert that the static matrix matches the expected matrix
        assert_array_close(big_matrix, expected_static_matrix)
        assert_array_close(big_lowering_matrix, expected_static_lowering_matrix)

        # Check that lowering operator is h.c. of raising operator 
        assert_array_close(big_matrix - np.transpose(np.conjugate(big_lowering_matrix)), np.zeros_like(big_matrix))


    def test_spin_operators(self):
        """ Test the handling of spin X and Y (off-diagonal) operators with CouplingOperator """ 
        big_X_matrix = self.X_spin_a.static_matrix.toarray() 
      
        # When creating a CouplingOperator, the upper-right half of the Pauli X matrix will be seen first to create unique couplings:  
        expected_big_X_matrix = np.zeros_like(big_X_matrix) # np.kron(Pauli.X, np.eye(2))
        expected_big_X_matrix[0,2] = 1.
        expected_big_X_matrix[1,3] = 1.

        assert_array_close(big_X_matrix, expected_big_X_matrix)
        
        # When creating a CouplingOperator, the upper-right half of the Pauli Y matrix will be seen first to create unique couplings:  
        big_Y_matrix = self.Y_spin_b.static_matrix.toarray() 
        expected_big_Y_matrix = np.zeros_like(big_Y_matrix) 
        expected_big_Y_matrix[0,1] = -1j 
        expected_big_Y_matrix[2,3] = -1j 
        assert_array_close(big_Y_matrix, expected_big_Y_matrix)

        # Test that raising and lowering operators have correct relationship to Pauli X: X = Raise + lower  
        big_Pauli_X = np.kron(Pauli.X, np.eye(2))
        assert_array_close((self.raise_spin_a.static_matrix.toarray() + self.lower_spin_a.static_matrix.toarray())*2./self.rabi_rate, big_Pauli_X)


    def test_rate_matrix(self):
        """Test the rate matrix of the CouplingOperator."""
        raising_rate_matrix = self.raise_spin_a.rate_matrix.toarray()
        expected_raising_rate_matrix = np.where(self.raise_spin_a.static_matrix.toarray() != 0, self.omega, 0)
        
        lowering_rate_matrix = self.lower_spin_a.rate_matrix.toarray()
        expected_lowering_rate_matrix = np.where(self.lower_spin_a.static_matrix.toarray() != 0, -self.omega, 0)
        
        # Assert that the rate matrices match the expected matrices
        assert_array_close(raising_rate_matrix, expected_raising_rate_matrix)
        assert_array_close(lowering_rate_matrix, expected_lowering_rate_matrix)

        
if __name__ == '__main__':
    unittest.main()
