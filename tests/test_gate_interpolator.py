#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ionsim.basis import StandardBasis
from ionsim.degree_of_freedom import AtomicStructure
from ionsim.gate_interpolator import GateInterpolator
from ionsim.named_operators import Unitary
from ionsim.process import Gate


def rotation_process_matrix(theta: float, phi: float):
    """ Process matrix of a single-qubit rotation R(phi, theta) """
    U = Unitary.R(phi, theta)
    return np.kron(U, U.conj())


class TestGateInterpolator(unittest.TestCase):

    def setUp(self):
        qubit = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.basis = StandardBasis([qubit])
        # Uneven axes in both directions
        self.grid_axes = {'theta': np.array([0., 0.2, 0.5, 0.9, 1.4, 2.]), 'phi': np.linspace(-0.5, 0.5, 7)}
        grid = GateInterpolator.build_grid(self.grid_axes)
        gates = [Gate(basis=self.basis, process_matrix=rotation_process_matrix(*point)) for point in grid]
        self.interpolator = GateInterpolator(self.grid_axes, 'R', grid, self.basis, gates)

    def test_process_matrix_interpolator_matches_elementwise_splines(self):
        """ The process matrix interpolation equals a cubic spline through each process matrix element """
        spline_reals, spline_imags = self.interpolator.construct_spline_for_gate(complex_data=True)
        elementwise = self.interpolator.make_interpolated_property_from_splines([spline_reals, spline_imags], 'process matrix')
        rng = np.random.default_rng(0)
        points = [(rng.uniform(0., 2.), rng.uniform(-0.5, 0.5)) for _ in range(20)] + list(self.interpolator.grid[:5])
        for point in points:
            np.testing.assert_allclose(self.interpolator.process_matrix_interpolator(*point), elementwise(*point), atol=1e-14)
        # Reproduces the gates on the grid, and approximates them between grid points
        theta, phi = self.interpolator.grid[8]
        np.testing.assert_allclose(self.interpolator.process_matrix_interpolator(theta, phi), rotation_process_matrix(theta, phi), atol=1e-13)
        np.testing.assert_allclose(self.interpolator.process_matrix_interpolator(1.1, 0.13), rotation_process_matrix(1.1, 0.13), atol=5e-3)

    def test_call_forms(self):
        """ Positional, keyword, and tuple arguments give the same interpolation; a Gate can be built from it """
        f = self.interpolator.process_matrix_interpolator
        np.testing.assert_array_equal(f(1.1, 0.13), f(phi=0.13, theta=1.1))
        np.testing.assert_array_equal(f(1.1, 0.13), f((1.1, 0.13)))
        gate = self.interpolator.make_interpolated_gate_over_grid(1.1, 0.13)
        np.testing.assert_array_equal(gate.process_matrix, f(1.1, 0.13))

    def test_file_round_trip(self):
        """ An interpolator written to a file and read back interpolates identically """
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'R.hdf5'
            self.interpolator.write_to_file(filename, {'gate_name': 'R'})
            loaded = GateInterpolator.from_file_and_basis(filename, self.basis)
        self.assertEqual(loaded.parameter_list, ['theta', 'phi'])
        np.testing.assert_array_equal(loaded.process_matrix_interpolator(1.1, 0.13), self.interpolator.process_matrix_interpolator(1.1, 0.13))


if __name__ == '__main__':
    unittest.main()
