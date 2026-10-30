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

from ionsim.degree_of_freedom import AtomicStructure
from ionsim.energy_level import EnergyEigenstate
from ionsim.basis import StandardBasis
from ionsim.operator import CouplingOperator, EnergyShiftOperator, GeneralOperator
from ionsim.testing import assert_array_close
from ionsim.state import State
from ionsim.hamiltonian import Hamiltonian 

class TestOperator(unittest.TestCase):

    def setUp(self):
        """Set up the necessary objects for testing."""
        self.spin_a = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.spin_b = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.basis = StandardBasis([self.spin_a, self.spin_b])  # 00, 01, 10, 11

        self.static_matrix_input = np.zeros((2,2))
        self.static_matrix_input[1,1] += 2.
        self.static_matrix_input[0,0] -= 2.
        self.static_matrix_input[0,1] = np.pi

        self.general_op = GeneralOperator.from_matrix(self.basis, self.static_matrix_input, current_dofs = [self.spin_a])
        self.energy_shift_op = EnergyShiftOperator.from_matrix(self.basis, np.diag(np.diag(self.static_matrix_input)), current_dofs = [self.spin_a]) 
        self.coupling_op = CouplingOperator.from_matrix(self.basis, self.static_matrix_input - np.diag(np.diag(self.static_matrix_input)), 0., current_dofs = [self.spin_a])

    def test_static_matrix(self):
        """Test the static matrix of the GeneralOperator and EnergyShift Operator."""

        expected_general_matrix = np.kron(self.static_matrix_input, np.eye(2))
        expected_shift_matrix = np.kron(np.diag(np.diag(self.static_matrix_input)), np.eye(2))
        expected_coupling_matrix = np.kron(self.static_matrix_input - np.diag(np.diag(self.static_matrix_input)), np.eye(2))
        # Assert that the static matrix matches the expected matrix
        assert_array_close(self.coupling_op.static_matrix.toarray(), expected_coupling_matrix)
        assert_array_close(self.general_op.static_matrix.toarray(), expected_general_matrix)
        assert_array_close(self.energy_shift_op.static_matrix.toarray(), expected_shift_matrix)

    def test_component_extraction(self):
        """ Test extracting a coupling and energy shift operator contribution from General Operator object""" 
        big_general_matrix = np.kron(self.static_matrix_input, np.eye(2))

        energy_shift_contribution = self.general_op.energy_shift_operator_contribution
        coupling_op_contribution = self.general_op.coupling_operator_contribution

        assert_array_close(energy_shift_contribution.static_matrix.toarray(), np.diag(np.diag(big_general_matrix)))
        assert_array_close(coupling_op_contribution.static_matrix.toarray(), big_general_matrix - np.diag(np.diag(big_general_matrix)) )

    def test_time_evolution(self):
        """ Test Hamiltonian creation and time evolution of a general operator """ 
        state_coeffs = [1., 0., 0., 0.]
        psi_init = State.from_coefficients(self.basis, state_coeffs)

        # Create the Hamiltonian from the general operator 
        frame_energies = [-state.energy for state in self.basis.states]
        test_hamiltonian = Hamiltonian(self.basis, [self.general_op], frame_energies, sparse = False) 

        # Test time propagation of the initial state: 
        duration = 1.0E-5
        psi_t = psi_init.propagate_using_schrodinger_equation(test_hamiltonian, duration)

        probabilities = psi_t.compute_basis_state_probabilities()
        expected_00_probability = 1. 
        self.assertAlmostEqual(probabilities[0], expected_00_probability, places=6) 


class TestSmallAndZeroElements(unittest.TestCase):
    """ Generic operators keep every nonzero element, however small (no energy threshold: e.g. POVM effects), drop exact
        zeros, and an operator with no elements is the zero matrix. """

    def setUp(self):
        atom = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.basis = StandardBasis([atom])

    def test_parametrized_povm_effect(self):
        for p in (1e-9, 1e-30):
            for cls in (EnergyShiftOperator, GeneralOperator):
                with self.subTest(p=p, cls=cls.__name__):
                    effect = cls.from_matrix(self.basis, np.diag([1. - p, p]))
                    self.assertEqual(len(effect.elements), 2)
                    np.testing.assert_array_equal(effect.static_matrix.toarray(), np.diag([1. - p, p]))
                    np.testing.assert_array_equal(effect.superbra, [1. - p, 0., 0., p])

    def test_small_coupling_kept_and_exact_zeros_dropped(self):
        tiny = np.array([[0., 1e-12], [1e-12, 0.]])
        self.assertEqual(len(CouplingOperator.from_matrix(self.basis, tiny, 0.).couplings), 2)
        self.assertEqual(len(GeneralOperator.from_matrix(self.basis, np.diag([0.7, 0.])).elements), 1)

    def test_all_zero_operators(self):
        zero = np.zeros((2, 2))
        for operator in (EnergyShiftOperator.from_matrix(self.basis, zero), GeneralOperator.from_matrix(self.basis, zero),
                         CouplingOperator.from_matrix(self.basis, zero, 0.)):
            with self.subTest(cls=type(operator).__name__):
                self.assertEqual(len(operator.elements), 0)
                np.testing.assert_array_equal(operator.static_matrix.toarray(), zero)
                np.testing.assert_array_equal(operator.superbra, np.zeros(4))
        np.testing.assert_array_equal(CouplingOperator.from_matrix(self.basis, zero, 0.).rate_matrix.toarray(), zero)


if __name__ == '__main__':
    unittest.main()
