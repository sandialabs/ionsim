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
import yaml

from ionsim.gst_circuit_planner import (GSTCircuitPlanner, information_dependences, information_shares,
                                         information_weights)

X, Y, IDLE = 'Gxpi2:0', 'Gypi2:0', 'idle'


def germ_circuits(planner):
    """ {germ key: sorted germ powers} of the planner's long-sequence circuits """
    found = {}
    for circuit in planner._long_gst_circuits():
        found.setdefault(GSTCircuitPlanner._germ_key(circuit.germ_gates), set()).add(circuit.germ_power)
    return {key: sorted(powers) for key, powers in found.items()}


class TestGermPowers(unittest.TestCase):

    def ramsey_rabi(self, **kwargs):
        return GSTCircuitPlanner([X, IDLE, Y], [0], prep_fiducials=[[X]], measure_fiducials=[[X], [Y]], **kwargs)

    def test_same_powers_for_every_germ(self):
        """ A list of powers applies to every germ """
        planner = self.ramsey_rabi(germs=[[IDLE], [X], [Y]], germ_powers=[1, 2, 4])
        self.assertEqual(planner.germ_powers, [1, 2, 4])
        self.assertEqual(planner.germ_powers_by_germ, [[1, 2, 4]] * 3)
        self.assertEqual(germ_circuits(planner), {IDLE: [1, 2, 4], X: [1, 2, 4], Y: [1, 2, 4]})

    def test_powers_per_germ(self):
        """ A dictionary gives each germ its own powers; without germs, its keys are the germs """
        powers = {IDLE: list(range(1, 65)), X: [1, 2, 4, 8, 16], Y: [2, 8, 32]}
        planner = self.ramsey_rabi(germ_powers=powers)
        self.assertEqual([GSTCircuitPlanner._germ_key(g) for g in planner.germs], [IDLE, X, Y])
        self.assertEqual(planner.germ_powers, powers)
        self.assertEqual(germ_circuits(planner), {IDLE: list(range(1, 65)), X: [1, 2, 4, 8, 16], Y: [2, 8, 32]})
        # Each germ power with every prep / measure fiducial pair: 64 + 5 + 3 powers, 2 measure fiducials
        long_circuits = [c for c in planner._long_gst_circuits()]
        self.assertEqual(len(long_circuits), (64 + 5 + 3) * 2)

        # The same with germs given explicitly (in a different order than the dictionary)
        explicit = self.ramsey_rabi(germs=[[Y], [IDLE], [X]], germ_powers=powers)
        self.assertEqual(explicit.germ_powers_by_germ, [[2, 8, 32], list(range(1, 65)), [1, 2, 4, 8, 16]])

    def test_multi_gate_germs_and_idle_alias(self):
        """ Multi-gate germs are keyed by tuples of labels; '[]' and 'idle' are the same germ """
        planner = GSTCircuitPlanner(['Gxpi2:0', 'Gxpi2:1', 'Gypi2:0', 'Gypi2:1', 'MS:0:1'], [0, 1],
                                    germ_powers={'MS:0:1': [1, 2, 4, 8], ('Gxpi2:1', 'MS:0:1', 'Gxpi2:0'): [1, 2]})
        self.assertEqual(planner.germ_powers, {'MS:0:1': [1, 2, 4, 8], ('Gxpi2:1', 'MS:0:1', 'Gxpi2:0'): [1, 2]})
        self.assertEqual(len(planner.germs[1]), 3)
        ramsey = self.ramsey_rabi(germs=[['[]'], [X]], germ_powers={'idle': [1, 3], X: [2]})
        self.assertEqual(ramsey.germ_powers_by_germ, [[1, 3], [2]])

    def test_validation(self):
        """ Helpful errors for germ powers that do not match the germs or are not positive integers """
        with self.assertRaisesRegex(ValueError, "Germs without powers: \\['Gypi2:0'\\]"):
            self.ramsey_rabi(germs=[[IDLE], [Y]], germ_powers={IDLE: [1, 2]})
        with self.assertRaisesRegex(ValueError, "powers for germs not in germs: \\['Gxpi2:0'\\]"):
            self.ramsey_rabi(germs=[[IDLE]], germ_powers={IDLE: [1], X: [1]})
        with self.assertRaisesRegex(ValueError, "appears more than once"):
            self.ramsey_rabi(germ_powers={'idle': [1], '[]': [2]})
        with self.assertRaisesRegex(ValueError, "positive integers"):
            self.ramsey_rabi(germs=[[X]], germ_powers=[1, 0, 2])
        with self.assertRaisesRegex(ValueError, "positive integers"):
            self.ramsey_rabi(germ_powers={X: [1.5]})
        with self.assertRaisesRegex(ValueError, "Repeated germ power"):
            self.ramsey_rabi(germs=[[X]], germ_powers=[1, 2, 2])
        with self.assertRaisesRegex(ValueError, "empty"):
            self.ramsey_rabi(germ_powers={X: []})
        with self.assertRaisesRegex(TypeError, "tuple of gate labels"):
            self.ramsey_rabi(germ_powers={(): [1]})
        with self.assertRaisesRegex(ValueError, "not in the planner's default gate set"):
            self.ramsey_rabi(germ_powers={'Gzpi2:0': [1]})

class TestDuplicateCircuits(unittest.TestCase):

    def test_no_duplicate_gate_sequences(self):
        """ Circuits applying the same gate sequence are generated once; linear-GST circuits are kept with their structure """
        planner = GSTCircuitPlanner([X, Y, IDLE], [0])
        circuits = planner.generate_gst_circuits()
        sequences = [tuple(c.expanded_gates) for c in circuits]
        self.assertEqual(len(sequences), len(set(sequences)))
        self.assertEqual(len(circuits), 331)

        # Every linear-GST gate sequence is present, as the first (linear-GST) circuit generated with that sequence
        first_linear = {}
        for circuit in planner._linear_gst_circuits():
            first_linear.setdefault(tuple(circuit.expanded_gates), circuit)
        kept = {tuple(c.expanded_gates): c for c in circuits}
        for sequence, circuit in first_linear.items():
            self.assertIn(sequence, kept)
            self.assertEqual(kept[sequence].build_circuit_string(), circuit.build_circuit_string())

        # e.g. X(pi/2) X(pi/2) arises as fiducial pairs, fiducial + gate, and germ X(pi/2)^2, but is generated once
        self.assertEqual(sequences.count(tuple(planner.to_gst_sequence([X, X]))), 1)


if __name__ == '__main__':
    unittest.main()


class TestInformationAttribution(unittest.TestCase):
    """ information_shares, information_weights, information_dependences (module functions of gst_circuit_planner) """

    def setUp(self):
        rng = np.random.default_rng(3)
        self.groups = []
        for _ in range(4):                                  # four groups, each informing random combinations of 3 parameters
            A = rng.standard_normal((2, 3))
            self.groups.append(A.T @ A)
        self.theta = np.array([0.5, 2., -1.])

    def test_weights_partition_the_variance(self):
        """ w >= 0, sum_g w_ig = 1, and sum_g v^T I_g v = [I^-1]_ii """
        w = information_weights(self.groups, self.theta)
        self.assertTrue(np.all(w >= -1e-12))
        np.testing.assert_allclose(w.sum(axis=1), 1., rtol=1e-10)
        C = np.linalg.inv(sum(self.groups))
        np.testing.assert_allclose(w[:, 1], np.diag(C @ self.groups[1] @ C) / np.diag(C), rtol=1e-8)

    def test_weights_do_not_depend_on_the_other_parameters_definition(self):
        """ reparameterizing the other parameters (keeping theta_0) leaves w_0g unchanged; the diagonal share changes """
        T = np.array([[1., 0., 0.], [0.7, 1., 0.3], [-0.4, 0.2, 1.]])    # theta = T phi with phi_0 = theta_0 (first row e_0)
        transformed = [T.T @ f @ T for f in self.groups]                 # Fisher information about phi
        phi = np.linalg.solve(T, self.theta)
        np.testing.assert_allclose(information_weights(transformed, phi)[0], information_weights(self.groups, self.theta)[0],
                                   rtol=1e-8)
        self.assertFalse(np.allclose(information_shares(transformed)[0], information_shares(self.groups)[0]))

    def test_uncorrelated_parameters(self):
        """ with diagonal information the weight equals the share """
        diagonal = [np.diag(np.diag(f)) for f in self.groups]
        np.testing.assert_allclose(information_weights(diagonal, self.theta), information_shares(diagonal), rtol=1e-10)

    def test_dependence(self):
        """ D >= 1; D = inf for a group without which a parameter is not determined; nan for undetermined parameters """
        D = information_dependences(self.groups, self.theta)
        self.assertTrue(np.all(D >= 1. - 1e-10))
        a = np.outer([1., 1., 0.], [1., 1., 0.])                         # group a: theta_0 + theta_1
        b = np.diag([0., 1., 0.])                                         # group b: theta_1 alone; theta_2: never
        D = information_dependences([a, b], self.theta)
        self.assertTrue(np.isinf(D[0, 1]) and np.isinf(D[1, 1]) and np.isinf(D[0, 0]))
        self.assertTrue(np.all(np.isnan(D[2])))
        self.assertTrue(np.all(np.isnan(information_weights([a, b], self.theta)[2])))
        # theta_0 = (theta_0 + theta_1) - theta_1: equal unit information in a and b, so half the variance from each
        np.testing.assert_allclose(information_weights([a, b], self.theta)[0], [0.5, 0.5], rtol=1e-10)
