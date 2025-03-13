import unittest
from inspect import getfullargspec

import numpy as np

from ionsim.noise import Noise

class TestNoise(unittest.TestCase):

    def setUp(self):
        """Set up the necessary objects for testing."""
        self.dzs = np.linspace(-10, 10, 101)
        self.noise = Noise.from_named_pdf('z', 'gaussian', {'standard_deviation': 1, 'mean': 0}, self.dzs)

    def test_noise_creation(self):
        """Test the creation of noise from a named probability density function."""
        self.assertEqual(self.noise.parameter_name, 'z')
        self.assertEqual(self.noise.probability_density_function.__name__, 'pdf')
        self.assertTrue(np.allclose(self.noise.domain_arguments, self.dzs))

    def test_add_noise_to_matrix_function(self):
        """Test the addition of noise to a matrix function."""
        def f(z):
            return np.array([[1, 1], [1, 1]])

        # Get the parameter index for 'z'
        parameter_index = getfullargspec(f)[0].index('z')
        noisy_f = self.noise.add_noise_to_matrix_function(f, parameter_index)

        # Test the noisy function at specific points
        zs = np.linspace(-1, 1, 3)

        # Check that the noisy function returns the same result as f(z)
        for z in zs:
            np.testing.assert_array_almost_equal(noisy_f(z), f(z), decimal=14)

if __name__ == '__main__':
    unittest.main()
