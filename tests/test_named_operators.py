import unittest
import numpy as np
from scipy.linalg import expm
from ionsim.named_operators import Pauli, Unitary
from ionsim.testing import assert_array_close

class TestNamedOperators(unittest.TestCase):

    def test_unitary_y(self):
        """Test that Unitary.Y is equal to Pauli.Y."""
        assert_array_close(Unitary.Y, Pauli.Y)

    def test_expm_equivalence(self):
        """Test the equivalence of the matrix exponential and the expected result."""
        phi, theta = np.pi / 8, -2 * np.pi / 3
        sig_phi = np.cos(phi) * Pauli.X + np.sin(phi) * Pauli.Y
        expected_result = (
            np.cos(theta / 2) * np.kron(Pauli.I, Pauli.I) - 
            1j * np.sin(theta / 2) * np.kron(sig_phi, sig_phi)
        )
        
        result = expm(-1j * theta / 2 * np.kron(sig_phi, sig_phi))
        assert_array_close(result, expected_result)

    def test_Bloch_rotation_unitary(self):
        """Test that Unitary.Bloch(pi/2/2, 0, 0) is equal to SQRT_X gate."""
        # In named operators, the SQRT_X and X gates are equivalent up to single-qubit phases defined there: 
        assert_array_close(np.exp(1j*np.pi/2./2.)*Unitary.R_bloch([np.pi/2./2., 0, 0]), Unitary.sqrtX)
        assert_array_close(1j*Unitary.R_bloch([np.pi/2., 0, 0]), Pauli.X)

if __name__ == '__main__':
    unittest.main()
