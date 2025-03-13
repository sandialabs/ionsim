import unittest
import numpy as np
from scipy.linalg import expm
from ionsim.named_operators import Pauli, Unitary

class TestNamedOperators(unittest.TestCase):

    def test_unitary_y(self):
        """Test that Unitary.Y is equal to Pauli.Y."""
        np.testing.assert_array_almost_equal(Unitary.Y, Pauli.Y, decimal=14)

    def test_expm_equivalence(self):
        """Test the equivalence of the matrix exponential and the expected result."""
        phi, theta = np.pi / 8, -2 * np.pi / 3
        sig_phi = np.cos(phi) * Pauli.X + np.sin(phi) * Pauli.Y
        expected_result = (
            np.cos(theta / 2) * np.kron(Pauli.I, Pauli.I) - 
            1j * np.sin(theta / 2) * np.kron(sig_phi, sig_phi)
        )
        
        result = expm(-1j * theta / 2 * np.kron(sig_phi, sig_phi))
        np.testing.assert_array_almost_equal(result, expected_result, decimal=14)

if __name__ == '__main__':
    unittest.main()
