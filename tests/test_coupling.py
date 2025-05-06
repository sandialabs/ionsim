import unittest
import numpy as np
from ionsim.degree_of_freedom import AtomicSpin
from ionsim.energy_level import EnergyEigenstate
from ionsim.basis import StandardBasis
from ionsim.named_operators import Pauli
from ionsim.coupling import CouplingOperator
from ionsim.testing import assert_array_close

class TestCouplingOperator(unittest.TestCase):

    def setUp(self):
        """Set up the necessary objects for testing."""
        self.spin_a = AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.spin_b = AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.basis = StandardBasis([self.spin_a, self.spin_b])  # 00, 01, 10, 11

        self.rabi_rate = 100e3 * 2 * np.pi  # rad/s
        self.omega = self.spin_a.energy_levels[1].energy - self.spin_a.energy_levels[0].energy
        self.operator = self.rabi_rate / 2 * Pauli.plus

        # Create the CouplingOperator instance
        self.raise_spin_a = CouplingOperator.from_matrix(self.basis, self.operator, self.omega, [self.spin_a])

    def test_static_matrix(self):
        """Test the static matrix of the CouplingOperator."""
        big_matrix = self.raise_spin_a.static_matrix.toarray()
        expected_static_matrix = np.kron(self.operator, np.eye(2))
        
        # Assert that the static matrix matches the expected matrix
        assert_array_close(big_matrix, expected_static_matrix)

    def test_rate_matrix(self):
        """Test the rate matrix of the CouplingOperator."""
        rate_matrix = self.raise_spin_a.rate_matrix.toarray()
        expected_rate_matrix = np.where(self.raise_spin_a.static_matrix.toarray() != 0, self.omega, 0)
        
        # Assert that the rate matrix matches the expected matrix
        assert_array_close(rate_matrix, expected_rate_matrix)

if __name__ == '__main__':
    unittest.main()
