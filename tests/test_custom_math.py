#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

import unittest
import numpy as np
from ionsim.custom_math import slow_trapz_for_matrix, trapz_for_matrix, solve_time_evolution_equation
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

    def test_solver_tolerances(self):
        """Test that tolerances passed to solve_ivp and zvode reach the solver."""
        times = np.linspace(0, 10, 5)
        for ode_solver in ['solve_ivp', 'zvode']:
            with self.subTest(ode_solver=ode_solver):
                _, states = solve_time_evolution_equation(lambda t: np.array([[0, 1], [1, 0]]), np.array([1, 0], dtype=complex), times[-1], times,
                                                          ode_solver=ode_solver, rtol=1e-12, atol=1e-12)
                assert_array_close(states, np.array([np.cos(times), -1j * np.sin(times)]).T, atol=1e-9)

if __name__ == '__main__':
    unittest.main()
