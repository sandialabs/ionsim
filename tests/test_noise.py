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

    def test_pdf_parameters(self):
        """ The distribution's parameters are stored and can be varied """
        self.assertEqual(self.noise.pdf_parameters, {'standard_deviation': 1, 'mean': 0})
        self.assertEqual(self.noise.variable_parameters, {'z_noise_standard_deviation': 1, 'z_noise_mean': 0})
        # density with a different standard deviation
        sigma = 2.
        expected = np.exp(-0.5**2/(2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)
        self.assertAlmostEqual(self.noise.density(0.5, standard_deviation=sigma), expected, places=14)
        self.assertEqual(self.noise.density(0.5), self.noise.probability_density_function(0.5))
        with self.assertRaises(ValueError):
            self.noise.density(0.5, width=2.)

    def test_average(self):
        """ Noise average by the trapezoid rule on the (optionally scaled) grid """
        # <x^2> of a Gaussian is the variance
        self.assertAlmostEqual(self.noise.average(lambda x: np.array([[x**2]]))[0, 0], 1., places=6)
        self.assertAlmostEqual(self.noise.average(lambda x: np.array([[x**2]]), standard_deviation=1.5)[0, 0], 1.5**2, places=4)

        # A grid in units of the standard deviation follows the distribution's width
        scaled = Noise.from_named_pdf('z', 'gaussian', {'standard_deviation': 0.01}, np.linspace(-8, 8, 161),
                                      domain_scale_parameter='standard_deviation')
        np.testing.assert_allclose(scaled.displacements(), 0.01*np.linspace(-8, 8, 161))
        for sigma in [0.01, 1e-4, 3.]:
            self.assertAlmostEqual(scaled.average(lambda x: np.array([[1.]]), standard_deviation=sigma)[0, 0], 1., places=10)
            self.assertAlmostEqual(scaled.average(lambda x: np.array([[x**2]]), standard_deviation=sigma)[0, 0] / sigma**2, 1., places=10)
        with self.assertRaises(ValueError):
            Noise.from_named_pdf('z', 'gaussian', {'standard_deviation': 1.}, self.dzs, domain_scale_parameter='width')

    def test_unparameterized_pdf(self):
        """ Noise built from a plain pdf(x) has no variable distribution parameters """
        noise = Noise('z', lambda x: np.exp(-x**2/2)/np.sqrt(2*np.pi), self.dzs)
        self.assertEqual(noise.variable_parameters, {})
        with self.assertRaises(ValueError):
            noise.density(0., standard_deviation=2.)

if __name__ == '__main__':
    unittest.main()
