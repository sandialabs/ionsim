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

import yaml

from ionsim.gst_circuit_planner import GSTCircuitPlanner

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

if __name__ == '__main__':
    unittest.main()
