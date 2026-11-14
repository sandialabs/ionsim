#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************
""" The evaluation methods of the circuit outcome probabilities (gate_set_tomography.GERM_POWER_METHOD) agree """
import unittest

import numpy as np
from scipy.linalg import expm

import ionsim as sm
import ionsim.gate_set_tomography as gst
from ionsim.gst_circuit_planner import GSTCircuitPlanner

X, Y, IDLE = 'Gxpi2:0', 'Gypi2:0', 'idle'
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.diag([1., -1.]).astype(complex)


def _superoperator(H, L, duration):
    """ column-stacked exp(duration (-i[H, .] + D[L])) """
    I = np.eye(2)
    generator = -1j * (np.kron(I, H) - np.kron(H.T, I))
    LdL = L.conj().T @ L
    generator += np.kron(L.conj(), L) - 0.5 * (np.kron(I, LdL) + np.kron(LdL.T, I))
    return expm(duration * generator)


def rotation(axis):
    def gate(over_rotation, dephasing):
        return _superoperator((np.pi / 4) * (1. + over_rotation) * axis, np.sqrt(dephasing) * PAULI_Z, 1.)
    return gate


def idle(detuning, dephasing):
    return _superoperator(0.5 * detuning * PAULI_Z, np.sqrt(dephasing) * PAULI_Z, 1.)


def prep(SPAM_error):
    return np.diag([1. - SPAM_error, SPAM_error]).astype(complex).reshape(-1, order='F')


def POVM(SPAM_error):
    return {'0': np.diag([1. - SPAM_error, SPAM_error]).astype(complex).reshape(-1, order='F').conj(),
            '1': np.diag([SPAM_error, 1. - SPAM_error]).astype(complex).reshape(-1, order='F').conj()}


def build_model(circuit_noise=False):
    model = sm.GateSetModel(prep, POVM, {X: rotation(PAULI_X), Y: rotation(PAULI_Y), IDLE: idle})
    model.set_parameter('over_rotation', model=[X, Y], guess=0.01, bounds=(-0.5, 0.5))
    model.set_parameter('dephasing', model='shared', guess=0.002, bounds=(0., 1.))
    model.set_parameter('detuning', model='shared', guess=0.03, bounds=(-1., 1.))
    model.set_parameter('SPAM_error', model='shared', guess=0.01, bounds=(0., 0.5))
    if circuit_noise:
        pdf = lambda x, standard_deviation: np.exp(-x**2 / (2. * standard_deviation**2)) / np.sqrt(2. * np.pi) / standard_deviation
        noise = sm.Noise.from_pdf('over_rotation', pdf, {'standard_deviation': 0.02}, np.linspace(-4., 4., 9),
                                  domain_scale_parameter='standard_deviation')
        model.set_circuit_noise(noise, models=[X, Y], guess=0.02, bounds=(1e-4, 0.2))
    return model


def circuits(max_power=64):
    powers = [1, 2, 3, 4, 8, 16, 32, max_power]
    planner = GSTCircuitPlanner([X, Y, IDLE], [0], germs=[[X], [Y], [IDLE], [X, Y], [X, X, Y]], germ_powers=powers)
    return planner.generate_gst_circuits()


class TestPredictionMethods(unittest.TestCase):

    def check(self, model):
        design = circuits()
        theta = model.parse_theta(None) * 1.3
        original = gst.GERM_POWER_METHOD
        results = {}
        try:
            for method in ('composition', 'squaring', 'propagation', 'auto'):
                gst.GERM_POWER_METHOD = method
                predicted = gst.predict_circuit_probabilities(design, model, theta)
                results[method] = np.array([[p[o] for o in ('0', '1')] for p in predicted])
        finally:
            gst.GERM_POWER_METHOD = original
        for method in ('squaring', 'propagation', 'auto'):
            np.testing.assert_allclose(results[method], results['composition'], atol=1e-12, rtol=0, err_msg=method)
        # and against the circuit's map multiplied gate by gate (independent of the module's composition code)
        gate_set = model(theta)
        for circuit, p in zip(design[::7], results['auto'][::7]):
            if model.circuit_noise_gates and set(g.label for g in circuit.expanded_gates) & {X, Y}:
                continue
            v = np.asarray(gate_set['prep'])
            for gate in circuit.expanded_gates:
                v = np.asarray(gate_set[gate.label]) @ v
            reference = [np.real(np.asarray(gate_set['POVM'][o]) @ v) for o in ('0', '1')]
            np.testing.assert_allclose(np.clip(reference, 1e-12, 1.), p, atol=1e-12)

    def test_markovian(self):
        self.check(build_model())

    def test_circuit_noise(self):
        self.check(build_model(circuit_noise=True))

    def test_bad_method(self):
        original = gst.GERM_POWER_METHOD
        gst.GERM_POWER_METHOD = 'fastest'
        try:
            with self.assertRaises(ValueError):
                gst.predict_circuit_probabilities(circuits(8), build_model(), None)
        finally:
            gst.GERM_POWER_METHOD = original


if __name__ == '__main__':
    unittest.main()
