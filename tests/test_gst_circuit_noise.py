#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************
""" Tests of circuit-level (quasistatic) noise in GST: Noise realizations, GateSetModel.set_circuit_noise, the noise-averaged
    GST likelihood and simulated data, the circuit planner's Fisher information, and parameter_squared_errors """
import unittest
import warnings

import numpy as np
from scipy.linalg import block_diag

from ionsim.noise import Noise
from ionsim.process import Gate, Circuit
from ionsim.degree_of_freedom import AtomicStructure
from ionsim.basis import StandardBasis
from ionsim.named_operators import Unitary
from ionsim.operator import EnergyShiftOperator
from ionsim.state import State
from ionsim.gate_set_model import GateSetModel, CIRCUIT_NOISE
from ionsim.gst_circuit_planner import GSTCircuitPlanner
from ionsim.gst_circuit_parser import GstCircuit
from ionsim.gate_set_tomography import (_GstProblem, simulate_gst_data, staged_mle_solve_for_gate_parameters, log_likelihood,
                                        linear_solve_for_gate_parameters, parameter_squared_errors, gate_set_errors,
                                        parse_parameter_scalings)

X, Y, IDLE = 'Gxpi2:0', 'Gypi2:0', 'idle'
SIGMA = 'amplitude_offset_noise_standard_deviation'


def gaussian_noise(sigma=0.02, n_points=41, half_width=6.):
    return Noise.from_named_pdf('amplitude_offset', 'gaussian', {'standard_deviation': sigma},
                                np.linspace(-half_width, half_width, n_points), domain_scale_parameter='standard_deviation')


class TestNoiseRealizations(unittest.TestCase):

    def test_realizations_and_average(self):
        noise = gaussian_noise(0.1, 21, 5.)
        xs, weights = noise.realizations()
        np.testing.assert_allclose(xs, 0.1 * np.linspace(-5, 5, 21))
        self.assertAlmostEqual(weights.sum(), 1., places=14)
        np.testing.assert_allclose(noise.weights(), weights)
        # weights are independent of the width for a scaled grid; the average is sum_k w_k f(x_k)
        np.testing.assert_allclose(noise.weights(standard_deviation=0.3), weights, rtol=1e-12)
        f = lambda x: np.array([[np.cos(x), x], [x**2, 1.]])
        np.testing.assert_allclose(noise.average(f), sum(w * f(x) for x, w in zip(xs, weights)), atol=1e-15)
        self.assertAlmostEqual(noise.average(lambda x: x**2, standard_deviation=0.3) / 0.3**2, 1., places=4)   # +-5 sigma grid
        self.assertAlmostEqual(noise.average(np.cos), np.exp(-0.1**2 / 2), places=6)

    def test_domain_scale_function(self):
        """ A Gaussian parameterized by its variance, with a grid scaled by sqrt(variance), has the same realizations as one
            parameterized by its standard deviation """
        def gaussian_by_variance(x, variance):
            return np.exp(-x**2 / (2 * variance)) / np.sqrt(2 * np.pi * variance)
        grid = np.linspace(-5, 5, 21)
        by_variance = Noise.from_pdf('z', gaussian_by_variance, {'variance': 0.04}, grid, domain_scale_function=lambda variance: np.sqrt(variance))
        by_sigma = Noise.from_named_pdf('z', 'gaussian', {'standard_deviation': 0.2}, grid, domain_scale_parameter='standard_deviation')
        for a, b in zip(by_variance.realizations(), by_sigma.realizations()):
            np.testing.assert_allclose(a, b, rtol=1e-12)
        for a, b in zip(by_variance.realizations(variance=0.09), by_sigma.realizations(standard_deviation=0.3)):
            np.testing.assert_allclose(a, b, rtol=1e-12)
        self.assertAlmostEqual(by_variance.domain_scale(variance=0.09), 0.3, places=14)
        self.assertIsNone(Noise.from_named_pdf('z', 'gaussian', {'standard_deviation': 1.}, grid).domain_scale())

        with self.assertRaises(ValueError):         # both scale options
            Noise.from_pdf('z', gaussian_by_variance, {'variance': 0.04}, grid, domain_scale_parameter='variance',
                           domain_scale_function=np.sqrt)
        with self.assertRaises(ValueError):         # arguments that are not pdf parameters
            Noise.from_pdf('z', gaussian_by_variance, {'variance': 0.04}, grid, domain_scale_function=lambda sigma: sigma)
        with self.assertRaises(ValueError):         # zero width
            by_sigma.realizations(standard_deviation=0.)


class TestCircuitNoise(unittest.TestCase):

    def setUp(self):
        qubit = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.basis = basis = StandardBasis([qubit])

        def rotation(phi, angle):
            return Gate.from_unitary(basis, Unitary.R(phi, angle), [qubit]).process_matrix

        # X(pi/2) and Y(pi/2) with a common fractional amplitude error, and an idle with a detuning (no amplitude error)
        def X_pi_2(amplitude_offset: float):
            return rotation(0., np.pi / 2 * (1. + amplitude_offset))

        def Y_pi_2(amplitude_offset: float):
            return rotation(np.pi / 2, np.pi / 2 * (1. + amplitude_offset))

        def idle(detuning_phase: float):
            return Gate.from_unitary(basis, np.diag([1., np.exp(1j * detuning_phase)]), [qubit]).process_matrix

        def prep(SPAM_error_probability: float):
            return State.from_density_matrix(basis, np.diag([1. - SPAM_error_probability, SPAM_error_probability])).supervector

        def POVM(SPAM_error_probability: float):
            p = SPAM_error_probability
            return {'0': EnergyShiftOperator.from_matrix(basis, np.diag([1. - p, p])).superbra,
                    '1': EnergyShiftOperator.from_matrix(basis, np.diag([p, 1. - p])).superbra}

        self.models = dict(X=X_pi_2, Y=Y_pi_2, idle=idle, prep=prep, POVM=POVM)
        self.true_values = {'shared:SPAM_error_probability': 0.002, 'shared:amplitude_offset': 0.003, f'shared:{SIGMA}': 0.01,
                            'idle.detuning_phase': 0.05}

    def gate_set_model(self, noise=None, **kwargs):
        model = GateSetModel(self.models['prep'], self.models['POVM'], {X: self.models['X'], Y: self.models['Y'], IDLE: self.models['idle']})
        model.set_parameter('SPAM_error_probability', model='shared', guess=1e-3, bounds=(0., 0.2))
        model.set_parameter('amplitude_offset', model=[X, Y], guess=0., bounds=(-0.2, 0.2))
        model.set_parameter('detuning_phase', model=IDLE, guess=0.02, bounds=(-1., 1.))
        if noise is not None:
            model.set_circuit_noise(noise, **kwargs)
        return model

    def circuits(self, sequences):
        planner = GSTCircuitPlanner([X, Y, IDLE], [0], prep_fiducials=[[]], measure_fiducials=[[]], germs=[[X]], germ_powers=[1])
        return [GstCircuit._from_gst_gates([], [planner.to_gst_gate(g) for g in seq], 1, [], [0]) for seq in sequences]

    def probabilities(self, gate_set_model, sequences, values):
        circuits = self.circuits(sequences)
        problem = _GstProblem(circuits, gate_set_model, exact_models=True)
        data = problem.tabulate(circuits, require_data=False)
        p = problem.predict_sequence_probabilities(gate_set_model.parse_theta(values), data.sequences)
        return np.array([p[k] for k in data.sequence_index])

    ### GateSetModel ###
    def test_set_circuit_noise(self):
        model = self.gate_set_model(gaussian_noise(), guess=0.02, bounds=(1e-4, 1.))
        self.assertEqual(model.parameter_names,
                         ['shared:SPAM_error_probability', 'shared:amplitude_offset', f'shared:{SIGMA}', 'idle.detuning_phase'])
        index = model.parameter_names.index(f'shared:{SIGMA}')
        self.assertEqual(model.initial_guess[index], 0.02)
        self.assertEqual(model.layout['bounds'][index], (1e-4, 1.))
        self.assertEqual([g.label for g in model.circuit_noise_gates], [X, Y])       # default: every gate with the argument
        theta = model.parse_theta(self.true_values)
        self.assertEqual(model.circuit_noise_parameters(theta), {'standard_deviation': 0.01})
        np.testing.assert_allclose(model.circuit_noise_realizations(theta)[0], 0.01 * np.linspace(-6, 6, 41))
        # nominal gates by default, displaced on request
        np.testing.assert_allclose(model.gate_process_matrix(X, theta), self.models['X'](0.003), atol=1e-15)
        np.testing.assert_allclose(model.gate_process_matrix(X, theta, displacement=0.01), self.models['X'](0.013), atol=1e-15)
        with self.assertRaises(ValueError):
            model.gate_process_matrix(IDLE, theta, displacement=0.01)
        # the guess defaults to the noise's value; the parameter can be re-specified like any shared parameter
        other = self.gate_set_model(gaussian_noise(0.05), models=[X])
        self.assertEqual(other.initial_guess[other.parameter_names.index(f'shared:{SIGMA}')], 0.05)
        self.assertEqual([g.label for g in other.circuit_noise_gates], [X])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')        # re-organizing the layout after it was built
            other.set_parameter(SIGMA, model='shared', bounds=(1e-3, 0.5))
        self.assertEqual(other.layout['bounds'][other.parameter_names.index(f'shared:{SIGMA}')], (1e-3, 0.5))

    def test_set_circuit_noise_errors(self):
        model = self.gate_set_model(gaussian_noise())
        with self.assertRaises(ValueError):                       # one circuit noise only
            model.set_circuit_noise(gaussian_noise())
        with self.assertRaises(ValueError):                       # the idle has no amplitude_offset argument
            self.gate_set_model(gaussian_noise(), models=[X, IDLE])
        with self.assertRaises(ValueError):                       # no gate has the argument
            self.gate_set_model(Noise.from_named_pdf('rabi_rate', 'gaussian', {'standard_deviation': 1.}, np.linspace(-5, 5, 11)))
        with self.assertRaises(TypeError):
            self.gate_set_model('not a noise')
        with self.assertRaises(ValueError):                       # unknown pdf parameter
            self.gate_set_model(gaussian_noise(), guess={'width': 0.1})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.remove_circuit_noise()
        self.assertIsNone(model.circuit_noise)
        self.assertNotIn(f'shared:{SIGMA}', model.parameter_names)

    ### Likelihood ###
    def test_probabilities_match_analytic_average(self):
        """ X(pi/2)^p from |0> with Gaussian amplitude error: p(1) = (1 - cos(p pi/2 (1 + mu)) exp(-(p pi sigma / 2)^2 / 2)) / 2 """
        model = self.gate_set_model(gaussian_noise(n_points=61, half_width=7.))
        values = dict(self.true_values, **{'shared:SPAM_error_probability': 0.})
        powers = [1, 3, 8, 21, 64]
        p1 = self.probabilities(model, [[X] * p for p in powers], values)[:, 1]
        mu, sigma = values['shared:amplitude_offset'], values[f'shared:{SIGMA}']
        expected = [(1 - np.cos(p * np.pi / 2 * (1 + mu)) * np.exp(-(p * np.pi * sigma / 2)**2 / 2)) / 2 for p in powers]
        np.testing.assert_allclose(p1, expected, atol=1e-9)
        # Not the Markovian average of independently averaged gates
        markovian = (1 - np.cos(64 * np.pi / 2 * (1 + mu)) * np.exp(-(np.pi * sigma / 2)**2 / 2)**64) / 2
        self.assertGreater(abs(p1[-1] - markovian), 0.05)

    def test_probabilities_match_block_diagonal_reference(self):
        """ The noise-averaged likelihood equals an embedding of the gate set in one block per realization (blockdiag gates,
            weighted prep copies, repeated effects), and the circuit-level average of Circuit.from_gates """
        noise = gaussian_noise(n_points=15, half_width=5.)
        model = self.gate_set_model(noise)
        theta = model.parse_theta(self.true_values)
        xs, weights = model.circuit_noise_realizations(theta)
        mu, SPAM, detuning = 0.003, 0.002, 0.05

        def X_block(amplitude_offset: float):
            return block_diag(*[self.models['X'](amplitude_offset + x) for x in xs])
        def Y_block(amplitude_offset: float):
            return block_diag(*[self.models['Y'](amplitude_offset + x) for x in xs])
        def idle_block(detuning_phase: float):
            return block_diag(*[self.models['idle'](detuning_phase)] * len(xs))
        def prep_block(SPAM_error_probability: float):
            return np.concatenate([w * self.models['prep'](SPAM_error_probability) for w in weights])
        def POVM_block(SPAM_error_probability: float):
            return {k: np.concatenate([v] * len(xs)) for k, v in self.models['POVM'](SPAM_error_probability).items()}

        sequences = [[X, Y, X, X, IDLE, Y] * 5, [X, X, Y] * 20, [IDLE, IDLE], [Y] * 7 + [X]]
        p = self.probabilities(model, sequences, self.true_values)
        # block-diagonal reference: probabilities only (its prep / POVM are not d x d, so it bypasses GateSetModel's checks)
        rho, effects = prep_block(SPAM), POVM_block(SPAM)
        blocks = {X: X_block(mu), Y: Y_block(mu), IDLE: idle_block(detuning)}
        for k, seq in enumerate(sequences):
            state = rho
            for g in seq:
                state = blocks[g] @ state
            np.testing.assert_allclose(p[k], np.real([effects['0'] @ state, effects['1'] @ state]), atol=1e-13)

        # Circuit.from_gates with the same circuit-level Noise (gates as ionsim Gates with process matrix functions)
        gates = {X: Gate.from_process_matrix_function(self.basis, self.models['X'], {'amplitude_offset': mu}),
                 Y: Gate.from_process_matrix_function(self.basis, self.models['Y'], {'amplitude_offset': mu}),
                 IDLE: Gate.from_process_matrix_function(self.basis, self.models['idle'], {'detuning_phase': detuning})}
        seq = sequences[0]
        circuit = Circuit.from_gates([gates[g] for g in seq], noise=gaussian_noise(0.01, n_points=15, half_width=5.))
        state = circuit.process_matrix @ self.models['prep'](SPAM)
        np.testing.assert_allclose(p[0], np.real([self.models['POVM'](SPAM)[o] @ state for o in '01']), atol=1e-13)

    def test_only_listed_gates_are_displaced(self):
        """ With models=[X], Y gates take the nominal amplitude, so Y^p circuits are noiseless """
        model = self.gate_set_model(gaussian_noise(), models=[X])
        values = dict(self.true_values, **{'shared:SPAM_error_probability': 0., 'shared:amplitude_offset': 0.})
        p = self.probabilities(model, [[Y] * 8, [X] * 8], values)[:, 1]
        self.assertLess(p[0], 1e-10)
        self.assertGreater(p[1], 1e-3)

    ### Fisher information and fits ###
    def test_planner_fisher_information_matches_likelihood(self):
        """ The planner's Fisher information (Circuit_Process_Matrix_Function with the noise) equals the Fisher
            information of the GST likelihood's probabilities (finite differences), including for the noise width """
        model = self.gate_set_model(gaussian_noise(n_points=21, half_width=5.), bounds=(1e-4, 1.))
        planner = GSTCircuitPlanner([X, Y, IDLE], [0], prep_fiducials=[[]], measure_fiducials=[[]], germs=[[X]], germ_powers=[1],
                                    gate_set_model=model)
        theta = model.parse_theta(self.true_values)
        for seq in [[X] * 16, [X, IDLE, Y, X, X] * 3, [IDLE] * 4]:
            circuit = self.circuits([seq])[0]
            _, fisher = planner.compute_circuit_fisher_information(circuit, theta, N_shots=1000)
            jacobian = []
            for i in range(len(theta)):
                h = 1e-6 * max(abs(theta[i]), 1e-2)
                up, down = theta.copy(), theta.copy()
                up[i] += h
                down[i] -= h
                jacobian.append((self.probabilities(model, [seq], up)[0] - self.probabilities(model, [seq], down)[0]) / (2 * h))
            jacobian = np.array(jacobian)
            p = self.probabilities(model, [seq], theta)[0]
            expected = 1000 * np.einsum('ik,jk->ij', jacobian / p, jacobian)
            np.testing.assert_allclose(fisher, expected, rtol=1e-4, atol=1e-6 * np.abs(expected).max())
            if seq == [IDLE] * 4:            # no noisy gate: no information about sigma
                self.assertEqual(fisher[model.parameter_names.index(f'shared:{SIGMA}')].max(), 0.)

    def test_mle_recovers_noise_width(self):
        """ Staged MLE on simulated data recovers sigma within a few Cramer-Rao bounds, and the squared parameter errors are
            reported per parameter """
        model = self.gate_set_model(gaussian_noise(n_points=15, half_width=5.), guess=0.02, bounds=(1e-4, 0.2))
        planner = GSTCircuitPlanner([X, Y, IDLE], [0], germs=[[X], [X, Y], [IDLE]], germ_powers=[1, 4, 16, 32], gate_set_model=model)
        circuits = planner.generate_gst_circuits()
        data = simulate_gst_data(circuits, model, self.true_values, 2000, np.random.default_rng(7))
        fit = staged_mle_solve_for_gate_parameters(data, model, dimensionless_fit=True)
        _, fisher = planner.compute_design_fisher_information(circuits, self.true_values, N_shots=2000)
        bounds = np.sqrt(np.diag(np.linalg.inv(sum(fisher.values()))))
        truth = model.parse_theta(self.true_values)
        self.assertTrue(np.all(np.abs(fit.theta - truth) < 4 * bounds), f"{fit.theta} vs {truth} +- {bounds}")
        self.assertGreaterEqual(log_likelihood(data, model, fit.theta), log_likelihood(data, model, truth))

        squared = parameter_squared_errors(model, fit.theta, self.true_values)
        self.assertEqual(list(squared), model.parameter_names)
        np.testing.assert_allclose(list(squared.values()), (fit.theta - truth)**2)
        relative_errs = parameter_squared_errors(model, fit.theta, self.true_values, relative_error=True)
        np.testing.assert_allclose(list(relative_errs.values()), ((fit.theta - truth) / truth)**2)
        # gate set errors compare the nominal gates
        errors = gate_set_errors(model, fit.theta, reference=self.true_values)
        self.assertAlmostEqual(errors[X], np.linalg.norm(model.gate_process_matrix(X, fit.theta) - model.gate_process_matrix(X, truth)), places=14)

    def test_parameter_scales(self):
        """ Initial scales for dimensionless fits: |initial guess| by default, user scales by name or vector, and a warned
            fallback (bound width, or 1) for parameters whose guess is 0 """
        model = self.gate_set_model()         # amplitude_offset has guess 0 and bounds (-0.2, 0.2)
        names = model.parameter_names
        offset = names.index('shared:amplitude_offset')
        expected = np.abs(np.asarray(model.initial_guess, dtype=float))
        expected[offset] = 0.4

        with self.assertRaises(ValueError):   # unknown name
            parse_parameter_scalings(model, {'not_a_parameter': 1.})
        with self.assertRaises(ValueError):   # wrong length
            parse_parameter_scalings(model, [1.])
        with self.assertRaises(ValueError):   # non-positive
            parse_parameter_scalings(model, {'shared:amplitude_offset': 0.})
        circuits = self.circuits([[X]])
        with self.assertRaises(ValueError):   # scales without dimensionless_fit
            _GstProblem(circuits, model, parameter_scales={'shared:amplitude_offset': 1e-3})

    def test_linear_gst_is_not_supported(self):
        model = self.gate_set_model(gaussian_noise())
        planner = GSTCircuitPlanner([X, Y, IDLE], [0], gate_set_model=model)
        with self.assertRaises(NotImplementedError):
            linear_solve_for_gate_parameters(planner.generate_gst_circuits(), model, planner, target=self.true_values)


class TestParameterSquaredErrors(unittest.TestCase):

    def test_values(self):
        def prep(p: float):
            return np.array([1. - p, 0., 0., p])
        def POVM(p: float):
            return {'0': np.array([1., 0., 0., 0.]), '1': np.array([0., 0., 0., 1.])}
        def gate(a: float, b: float):
            return np.eye(4)
        model = GateSetModel(prep, POVM, {'Gxpi2:0': gate})
        model.set_parameter('p', model='shared')
        squared = parameter_squared_errors(model, [0.1, 1., 2.], {'shared:p': 0.2, 'Gxpi2:0.a': 0.5, 'Gxpi2:0.b': 0.})
        self.assertEqual(list(squared), ['shared:p', 'Gxpi2:0.a', 'Gxpi2:0.b'])
        np.testing.assert_allclose(list(squared.values()), [0.01, 0.25, 4.])
        relative = parameter_squared_errors(model, [0.1, 1., 2.], [0.2, 0.5, 0.], relative_error=True)
        np.testing.assert_allclose([relative['shared:p'], relative['Gxpi2:0.a']], [0.25, 1.])
        self.assertTrue(np.isnan(relative['Gxpi2:0.b']))


if __name__ == '__main__':
    unittest.main()
