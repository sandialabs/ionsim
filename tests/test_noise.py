#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

import unittest
from inspect import getfullargspec

import numpy as np

from ionsim.noise import Noise
from ionsim.testing import assert_array_close

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
            assert_array_close(noisy_f(z), f(z))

if __name__ == '__main__':
    unittest.main()
