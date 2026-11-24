#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

""" MultiOperator rate groups: sum_k O_k(t) = sum_g f_g(t) e^{-i r_g t} A_g, and get_matrix evaluated from them. """

import unittest

import numpy as np
from scipy.sparse import issparse

from ionsim.basis import StandardBasis
from ionsim.degree_of_freedom import AtomicStructure, MotionalMode
from ionsim.hamiltonian import Hamiltonian
from ionsim.ionsim_error import IonSimError
from ionsim.multi_operator import MultiOperator
from ionsim.operators import Coupling, CouplingOperator, EnergyShift, EnergyShiftOperator, GeneralOperator, adjoint

TPI = 2. * np.pi


def dense(matrix):
    return matrix.toarray() if issparse(matrix) else np.asarray(matrix)


def elementwise_reference(multi_operator: MultiOperator, t: float) -> np.ndarray:
    """ prefactor * sum_k f_k(t) [D_k + C_k * exp(-i R_k t)], evaluated part by part with every element's own frame-shifted
        rate (the evaluation MultiOperator used before the rate-group decomposition). """
    total = np.zeros((multi_operator.size, multi_operator.size), dtype=complex)
    for operator in multi_operator.operators:
        parts = [operator]
        if isinstance(operator, GeneralOperator):
            parts = [part for part in (operator.energy_shift_operator_contribution if operator.energy_shifts else None,
                                       operator.coupling_operator_contribution if operator.couplings else None) if part is not None]
        for part in parts:
            f = 1. if part.modulation_function is None else part.modulation_function(t)
            if isinstance(part, EnergyShiftOperator):
                total += dense(part.static_matrix) * f
            else:
                matrix, rate = multi_operator._frame_shifted_coupling_matrix_and_rate_from_operator(part)
                total += dense(matrix) * np.exp(-1j * dense(rate) * t) * f
    return multi_operator.prefactor * total


def atom_and_mode():
    """ 171Yb+ S1/2 (0,0), S1/2 (1,0), P1/2 (1,1) x a 2 MHz mode (3 levels), in the interaction picture (frame -E). """
    atom = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2'], level_names=['S1/2,0,0', 'S1/2,1,0', 'P1/2,1,1'])
    mode = MotionalMode.from_frequency(TPI * 2e6, 3)
    basis = StandardBasis([atom, mode])
    return atom, mode, basis, [-state.energy for state in basis.states]


class TestRateGroups(unittest.TestCase):

    TIMES = (0., 3.1e-8, 2.7e-7, 1.3e-6)

    def setUp(self):
        self.atom, self.mode, self.basis, self.frame = atom_and_mode()
        self.states = self.basis.states
        self.index = {state.name: i for i, state in enumerate(self.states)}

    def state(self, atomic, n):
        return self.states[self.index[f'{atomic} : {n}']]

    def test_one_operator_split_and_merged(self):
        """ A decay path |S, n><P, n| for every n: one operator, one group (equal rates E_P - E_S up to rounding).
            A coupling with elements at different frame-energy differences: one operator, several groups. """
        decay = MultiOperator(self.basis, [CouplingOperator(self.basis, [
            Coupling(row_state=self.state('S1/2,0,0', n), column_state=self.state('P1/2,1,1', n), strength=1e4, oscillation_rate=0.)
            for n in range(3)])], self.frame)
        self.assertEqual(len(decay.rate_groups), 1)
        self.assertEqual(decay.rate_groups[0].matrix.nnz, 3)

        split = MultiOperator(self.basis, [CouplingOperator(self.basis, [
            Coupling(row_state=self.state('S1/2,0,0', 0), column_state=self.state('P1/2,1,1', 0), strength=1e4, oscillation_rate=0.),
            Coupling(row_state=self.state('S1/2,1,0', 0), column_state=self.state('P1/2,1,1', 0), strength=2e4, oscillation_rate=0.)])],
            self.frame)
        rates = sorted(group.rate for group in split.rate_groups)
        self.assertEqual(len(rates), 2)
        hyperfine = self.atom.energy_levels[1].energy - self.atom.energy_levels[0].energy
        self.assertAlmostEqual((rates[1] - rates[0]) / hyperfine, 1., places=6)

    def test_groups_across_operators_and_modulations(self):
        """ Elements of different operators share a group when modulation and rate agree; different modulation functions
            (e.g. C and adjoint(C)) give separate groups even at the same rate. """
        s = self.state
        coupling_a = CouplingOperator(self.basis, [Coupling(row_state=s('S1/2,0,0', 0), column_state=s('S1/2,0,0', 1), strength=1., oscillation_rate=0.)])
        coupling_b = CouplingOperator(self.basis, [Coupling(row_state=s('S1/2,1,0', 1), column_state=s('S1/2,1,0', 2), strength=2., oscillation_rate=0.)])
        shift = EnergyShiftOperator(self.basis, [EnergyShift(row_state=s('S1/2,0,0', 0), column_state=s('S1/2,0,0', 0), strength=3.)])
        merged = MultiOperator(self.basis, [coupling_a, coupling_b, shift], self.frame)
        self.assertEqual(sorted((round(g.rate / TPI / 1e6, 6), g.matrix.nnz) for g in merged.rate_groups), [(0., 1), (2., 2)])

        f = lambda t: np.exp(1j * TPI * 1e5 * t)
        resonant = CouplingOperator(self.basis, [Coupling(row_state=s('S1/2,1,0', 0), column_state=s('S1/2,0,0', 0), strength=1.,
                                                          oscillation_rate=self.atom.energy_levels[1].energy - self.atom.energy_levels[0].energy)],
                                    modulation_function=f)
        drive = MultiOperator(self.basis, [resonant, adjoint(resonant)], self.frame)
        self.assertEqual(len(drive.rate_groups), 2)
        self.assertIs(drive.rate_groups[0].modulation_function, f)
        self.assertIsNot(drive.rate_groups[1].modulation_function, f)
        for group in drive.rate_groups:
            self.assertLess(abs(group.rate), 1e3)  # resonant: static in the frame up to rounding

    def test_matrix_and_coefficients_match_elementwise_reference(self):
        s = self.state
        f = lambda t: (1. + 0.3 * np.sin(TPI * 0.5e6 * t)) * np.exp(1j * 0.7 * np.sin(TPI * 0.8e6 * t))
        g = lambda t: np.cos(TPI * 0.3e6 * t)
        raising = np.zeros((3, 3)); raising[1, 0] = TPI * 1e6  # |S1/2,1,0><S1/2,0,0|, 0.2 MHz from resonance
        drive = CouplingOperator.from_matrix(self.basis, self.basis.enlarge_matrix(raising, [self.atom]),
                                             self.atom.energy_levels[1].energy - self.atom.energy_levels[0].energy + TPI * 0.2e6,
                                             modulation_function=f)
        general = GeneralOperator(self.basis, [EnergyShift(row_state=s('S1/2,0,0', 0), column_state=s('S1/2,0,0', 0), strength=0.5 + 0.1j),
                                               Coupling(row_state=s('S1/2,0,0', 0), column_state=s('S1/2,0,0', 1), strength=1.5, oscillation_rate=0.),
                                               Coupling(row_state=s('P1/2,1,1', 1), column_state=s('P1/2,1,1', 2), strength=0.7, oscillation_rate=0.)])
        decay = CouplingOperator.from_matrix(self.basis, self.basis.enlarge_matrix(np.array([[0, 0, 1e4], [0, 0, 0], [0, 0, 0]]), [self.atom]), 0.,
                                             modulation_function=g)
        for sparse in (False, True):
            operator = MultiOperator(self.basis, [drive, adjoint(drive), general, decay], self.frame, sparse, prefactor=0.3 - 0.2j)
            for t in self.TIMES:
                with self.subTest(sparse=sparse, t=t):
                    M = operator.get_matrix(t)
                    self.assertEqual(issparse(M), sparse)
                    reference = elementwise_reference(operator, t)
                    # Optical-rate elements (the decay, ~5e15 rad/s) share one rounded rate: phases agree to ~(1 rad/s) t.
                    np.testing.assert_allclose(dense(M), reference, rtol=1e-5, atol=1e-9 * np.max(np.abs(reference)))
                    from_coefficients = sum(c * dense(group.matrix) for c, group in zip(operator.group_coefficients(t), operator.rate_groups))
                    np.testing.assert_allclose(from_coefficients, dense(M), rtol=1e-12, atol=1e-12 * np.max(np.abs(reference)))

    def test_empty_operator(self):
        for sparse in (False, True):
            operator = MultiOperator(self.basis, [EnergyShiftOperator.from_matrix(self.basis, np.zeros((9, 9)))], self.frame, sparse)
            self.assertEqual(operator.rate_groups, ())
            np.testing.assert_array_equal(dense(operator.get_matrix(1e-7)), np.zeros((9, 9)))

    def test_coupling_state_outside_the_basis(self):
        """ A coupling whose states belong to another basis raises IonSimError (previously a NameError from the handler). """
        _, _, other_basis, _ = atom_and_mode()
        stray = CouplingOperator(other_basis, [Coupling(row_state=other_basis.states[0], column_state=other_basis.states[1],
                                                        strength=1., oscillation_rate=0.)])
        with self.assertRaises(IonSimError):
            MultiOperator(self.basis, [stray], self.frame).rate_groups


if __name__ == '__main__':
    unittest.main()
