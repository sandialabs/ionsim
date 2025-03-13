import unittest
import numpy as np
from ionsim.custom_math import slow_trapz_for_matrix, trapz_for_matrix

class TestCustomMath(unittest.TestCase):

    def setUp(self):
        """Set up the necessary objects for testing."""
        self.xs = np.linspace(-10, 10, 101)

        def f(x):
            return 1 / np.sqrt(2 * np.pi) * np.exp(-x**2 / 2)

        self.ys = [np.array([[0, f(x)], [2 * f(x), (3j + 1) * f(x)]]) for x in self.xs]

    def test_slow_trapz_for_matrix(self):
        """Test the slow_trapz_for_matrix function."""
        result = slow_trapz_for_matrix(self.ys, self.xs)
        expected_result = np.array([[0, 1], [2, (3j + 1)]])
        
        # Assert that the result matches the expected result with a precision of 14 decimal places
        np.testing.assert_array_almost_equal(result, expected_result, decimal=14)

    def test_trapz_for_matrix(self):
        """Test the trapz_for_matrix function."""
        result = trapz_for_matrix(self.ys, self.xs)
        expected_result = np.array([[0, 1], [2, (3j + 1)]])
        
        # Assert that the result matches the expected result with a precision of 14 decimal places
        np.testing.assert_array_almost_equal(result, expected_result, decimal=14)

if __name__ == '__main__':
    unittest.main()
