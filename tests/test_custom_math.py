#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

import unittest
import warnings
import numpy as np
from ionsim.custom_math import finite_difference_derivatives, slow_trapz_for_matrix, trapz_for_matrix
from ionsim.testing import assert_array_close

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
        
        assert_array_close(result, expected_result)

    def test_trapz_for_matrix(self):
        """Test the trapz_for_matrix function."""
        result = trapz_for_matrix(self.ys, self.xs)
        expected_result = np.array([[0, 1], [2, (3j + 1)]])
        
        assert_array_close(result, expected_result)

    def test_compare_slow_fast_trapz(self):
        """Test trapz_for_matrix using slow_trapz_for_matrix as an
        oracle."""
        for _ in range(16):
            ys = np.random.uniform(size=(len(self.xs), 2, 2), low=-1.0) + np.random.uniform(size=(len(self.xs), 2, 2), low=-1.0) * 1j
            exp = slow_trapz_for_matrix(ys, self.xs)
            act = trapz_for_matrix(ys, self.xs)
            assert_array_close(act, exp)

    def test_finite_difference_derivatives(self):
        """ Central finite differences for a vector-valued function """
        f = lambda x: np.array([np.sin(x[0]) * np.exp(x[1]), x[0]**2 * x[1]**3])
        x0 = np.array([0.3, -0.7])
        value, jacobian, hessian = finite_difference_derivatives(f, x0, order=2)
        a, b = x0
        np.testing.assert_allclose(value, f(x0))
        np.testing.assert_allclose(jacobian, [[np.cos(a)*np.exp(b), 2*a*b**3], [np.sin(a)*np.exp(b), 3*a**2*b**2]], atol=1e-10)
        np.testing.assert_allclose(hessian[0, 0], [-np.sin(a)*np.exp(b), 2*b**3], atol=1e-6)
        np.testing.assert_allclose(hessian[1, 1], [np.sin(a)*np.exp(b), 6*a**2*b], atol=1e-6)
        np.testing.assert_allclose(hessian[0, 1], [np.cos(a)*np.exp(b), 6*a*b**2], atol=1e-6)
        np.testing.assert_allclose(hessian[0, 1], hessian[1, 0])
        _, _, no_hessian = finite_difference_derivatives(f, x0, order=1)
        self.assertIsNone(no_hessian)

    def test_finite_difference_derivatives_at_boundary(self):
        """ One-sided differences at a bound and where the function is undefined on one side """
        # A rate parameter at zero: sqrt(rate) is NaN for rate < 0, but the function is smooth for rate >= 0.
        # The engine probes the invalid side, detects the NaN, and uses a one-sided stencil without emitting warnings.
        f = lambda x: np.sqrt(x[0])**2 * np.exp(-x[0]) + x[1]
        _, jacobian, hessian = finite_difference_derivatives(f, np.array([0., 1.]), order=2)
        self.assertAlmostEqual(jacobian[0], 1., places=8)
        self.assertAlmostEqual(hessian[0, 0], -2., places=5)
        self.assertAlmostEqual(jacobian[1], 1., places=8)

        # Explicit bounds force one-sided stencils
        g = lambda x: x[0]**3
        _, jacobian, hessian = finite_difference_derivatives(g, np.array([1.]), order=2, bounds=[(None, 1.)])
        self.assertAlmostEqual(jacobian[0], 3., places=8)
        self.assertAlmostEqual(hessian[0, 0], 6., places=5)


if __name__ == '__main__':
    unittest.main()
