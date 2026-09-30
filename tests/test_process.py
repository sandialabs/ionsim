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

from ionsim.process import Gate, Circuit, Circuit_Process_Matrix_Function_Helper, GateProcessMatrixCache
from ionsim.degree_of_freedom import AtomicStructure
from ionsim.basis import StandardBasis
from ionsim.named_operators import Unitary, Pauli
from ionsim.noise import Noise
from ionsim.operator import EnergyShiftOperator
from ionsim.state import State 

class TestProcess(unittest.TestCase):

    def setUp(self):
        """Set up the necessary objects for testing."""
        self.spin_a = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.spin_b = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.basis = StandardBasis([self.spin_a, self.spin_b])

        self.Sx = Gate.from_unitary(self.basis, Unitary.sqrtX, [self.spin_a])

        xs = np.linspace(-np.pi, np.pi, 21)
        self.phi_noise = Noise.from_named_pdf('phi', 'gaussian', {'standard_deviation': np.pi/10}, xs)
        self.noisy_phi_gate = Gate.from_unitary_function(
            self.basis, Unitary.R, {'phi': 0, 'theta': np.pi/2}, [self.spin_a], self.phi_noise,
        )

        self.theta_noise = Noise.from_named_pdf('theta', 'gaussian', {'standard_deviation': np.pi/10}, xs)
        self.noisy_theta_gate = Gate.from_unitary_function(
            self.basis, Unitary.R, {'phi': 0, 'theta': np.pi/2}, [self.spin_a], self.theta_noise,
        )

    def test_noisy_phi_gate_process_fidelity(self):
        """Test the process fidelity of the noisy phi gate."""
        fidelity = self.noisy_phi_gate.compute_process_fidelity(self.Sx.process_matrix)
        self.assertAlmostEqual(fidelity, 0.9535335189419549, places=14)

    def test_noisy_theta_gate_process_fidelity(self):
        """Test the process fidelity of the noisy theta gate."""
        fidelity = self.noisy_theta_gate.compute_process_fidelity(self.Sx.process_matrix)
        self.assertAlmostEqual(fidelity, 0.9759249157026244, places=14)

    def test_extra_noisy_gate_process_fidelity(self):
        """Test the process fidelity of the extra noisy gate."""
            #self.basis, self.noisy_phi_gate.process_matrix_function, {'phi': 0, 'theta': np.pi/2}, [self.spin_a], self.theta_noise,
        extra_noisy_gate = Gate.from_process_matrix_function(
            self.basis, self.noisy_phi_gate.process_matrix_function, {'phi': 0, 'theta': np.pi/2}, self.theta_noise,
        )
        fidelity = extra_noisy_gate.compute_process_fidelity(self.Sx.process_matrix)
        self.assertAlmostEqual(fidelity, 0.9306176541502549, places=14)

    def test_ramsey_circuit_process_fidelity(self):
        """Test the process fidelity of the Ramsey circuit."""
        ramsey = Circuit.from_gates(
            [
                Gate.from_unitary(self.basis, Unitary.sqrtX, [self.spin_a]),
                Gate.from_unitary_function(self.basis, Unitary.R, {'phi': 0, 'theta': np.pi/2}, [self.spin_a], self.phi_noise),
            ],
            self.theta_noise,
        )
        fidelity = ramsey.compute_process_fidelity(Gate.from_unitary(self.basis, Unitary.X, [self.spin_a]).process_matrix)
        self.assertAlmostEqual(fidelity, 0.9306176541502548, places=14)

        # Test computing outcome probabilities 
        outcome_operator = EnergyShiftOperator.from_matrix(self.basis, np.kron(Pauli.projector_1, Pauli.projector_0)) 
        initial_state = State.from_coefficients(self.basis, [1., 0., 0., 0.]) 

        outcome_probability = ramsey.predict_outcome_probabilities(initial_state, [outcome_operator]) 
        self.assertAlmostEqual(outcome_probability[0], 0.9530090510307307, places = 10)

    def test_circuit_process_matrix_functions(self):
        """ Test the process matrix function of a circuit and derivatives of probability outcomes """ 
        # Circuit-level noise is covered in test_circuit_function_with_circuit_noise
        noisy_R_gate = Gate.from_unitary_function(self.basis, Unitary.R, {'phi': 0, 'theta': np.pi/2}, [self.spin_a], self.phi_noise)

        ramsey_circuit = Circuit.from_gates([noisy_R_gate, noisy_R_gate])

        ## Fixed a bug where a noisy process matrix function would not work with kwargs 
        circuit_pm_function = ramsey_circuit.process_matrix_function 

        # Test outcome probability function  
        outcome_operator = EnergyShiftOperator.from_matrix(self.basis, np.kron(Pauli.projector_1, Pauli.projector_0)) 
        initial_state = State.from_coefficients(self.basis, [1., 0., 0., 0.]) 

        prob_function = ramsey_circuit.build_outcome_probability_function(initial_state, outcome_operator)
        circuit_parameters = {'R__phi' : 0., 'R__theta' : np.pi/2}
        outcome_prob = prob_function(**circuit_parameters)
        self.assertAlmostEqual(outcome_prob, 0.9530090510307307, places = 10)

        # Compute outcome probability using probability function: 
        prob, prob_gradients = circuit_pm_function.gradient(prob_function, wrt = ["R__phi", "R__theta"], **circuit_parameters) 

        ### Test Jacobian functionality: Compute Jacobian when considering more than 1 outcome: 
        outcome_operator2 = EnergyShiftOperator.from_matrix(self.basis, np.kron(Pauli.projector_0, Pauli.projector_0)) 

        probs_function = ramsey_circuit.build_outcome_probabilities_function(initial_state, [outcome_operator, outcome_operator2])

        probs, jacobian = circuit_pm_function.jacobian(probs_function, wrt = ["R__phi", "R__theta"], **circuit_parameters)
        #print(f"Jacobian: \n{jacobian}")


    def test_circuit_function_with_circuit_noise(self):
        """ The circuit process matrix function reproduces the circuit's process matrix with gate- and circuit-level noise """
        # Gate-level phi noise (varies gate-to-gate) and circuit-level theta noise (constant within the circuit)
        circuit = Circuit.from_gates([self.noisy_phi_gate, self.noisy_phi_gate], self.theta_noise)
        function_matrix = circuit.process_matrix_function(R__phi = 0., R__theta = np.pi/2)
        np.testing.assert_allclose(function_matrix, circuit.process_matrix, atol=1e-13)

        # Circuit-level theta noise (same displacement for both gates) differs from gate-level theta noise (independent per gate)
        R_gate = Gate.from_unitary_function(self.basis, Unitary.R, {'phi': 0, 'theta': np.pi/2}, [self.spin_a])
        correlated = Circuit.from_gates([R_gate, R_gate], self.theta_noise)
        independent = Circuit.from_gates([self.noisy_theta_gate, self.noisy_theta_gate])
        np.testing.assert_allclose(correlated.process_matrix_function(R__phi = 0., R__theta = np.pi/2), correlated.process_matrix, atol=1e-13)
        self.assertGreater(np.linalg.norm(correlated.process_matrix - independent.process_matrix), 1e-3)

    def test_outcome_probability_derivatives(self):
        """ Finite-difference derivatives of an outcome probability match analytic values """
        R_gate = Gate.from_unitary_function(self.basis, Unitary.R, {'phi': 0.3, 'theta': np.pi/3}, [self.spin_a])
        circuit = Circuit.from_gates([R_gate])
        outcome_operator = EnergyShiftOperator.from_matrix(self.basis, np.kron(Pauli.projector_1, Pauli.projector_0))
        initial_state = State.from_coefficients(self.basis, [1., 0., 0., 0.])
        prob_function = circuit.build_outcome_probability_function(initial_state, outcome_operator)

        # P(spin a in |1>) = sin^2(theta/2), independent of phi
        theta = np.pi/3
        parameters = {'R__phi': 0.3, 'R__theta': theta}
        prob, jacobian, hessian = circuit.process_matrix_function.derivatives(prob_function, wrt=['R__phi', 'R__theta'], order=2, **parameters)
        self.assertAlmostEqual(prob, np.sin(theta/2)**2, places=12)
        self.assertAlmostEqual(jacobian['R__theta'], np.sin(theta)/2., places=9)
        self.assertAlmostEqual(jacobian['R__phi'], 0., places=9)
        self.assertAlmostEqual(hessian['R__theta']['R__theta'], np.cos(theta)/2., places=6)
        self.assertAlmostEqual(hessian['R__phi']['R__theta'], 0., places=6)
        self.assertAlmostEqual(hessian['R__phi']['R__phi'], 0., places=6)

        # gradient / hessian wrappers agree with derivatives()
        _, gradients = circuit.process_matrix_function.gradient(prob_function, wrt=['R__theta'], **parameters)
        self.assertAlmostEqual(gradients['R__theta'], np.sin(theta)/2., places=9)

    def test_gate_cache(self):
        """ Caching gate process matrices does not change results and avoids repeated gate evaluations """
        # Two distinct gate functions: perturbing a parameter of one should reuse the cached matrix of the other
        def Rz(angle: float):
            return np.diag([np.exp(-0.5j*angle), np.exp(0.5j*angle)])
        z_gate = Gate.from_unitary_function(self.basis, Rz, {'angle': 0.2}, [self.spin_a])
        ramsey_circuit = Circuit.from_gates([self.noisy_phi_gate, z_gate, self.noisy_phi_gate])
        function = ramsey_circuit.process_matrix_function
        outcome_operators = [EnergyShiftOperator.from_matrix(self.basis, np.kron(Pauli.projector_1, Pauli.projector_0)),
                             EnergyShiftOperator.from_matrix(self.basis, np.kron(Pauli.projector_0, Pauli.projector_0))]
        initial_state = State.from_coefficients(self.basis, [1., 0., 0., 0.])
        probs_function = ramsey_circuit.build_outcome_probabilities_function(initial_state, outcome_operators)
        parameters = {'R__phi': 0.1, 'R__theta': np.pi/2, 'Rz__angle': 0.2}
        wrt = ['R__phi', 'R__theta', 'Rz__angle']

        _, jac_cached, hess_cached = function.derivatives(probs_function, wrt=wrt, **parameters)
        self.assertGreater(function.cache_hits, 0)

        function.cache_size = 0
        function.clear_cache()
        _, jac, hess = function.derivatives(probs_function, wrt=wrt, **parameters)
        self.assertEqual(function.cache_hits, 0)
        for name in jac:
            np.testing.assert_array_equal(jac[name], jac_cached[name])
            for other in jac:
                np.testing.assert_array_equal(hess[name][other], hess_cached[name][other])


    def test_circuit_function_labels_and_shared_parameters(self):
        """ Gate labels give independent parameters to one function; parameter_names shares parameters, including non-identifier names """
        def rotation(theta: float):
            return np.kron(Unitary.R(0., theta), Unitary.R(0., theta).conj())

        # The same function under two labels: independent parameters
        independent = Circuit_Process_Matrix_Function_Helper([rotation, rotation], separator='.', gate_labels=['Rx:0', 'Rx:1'])
        self.assertEqual(independent.parameter_names, ['Rx:0.theta', 'Rx:1.theta'])
        self.assertFalse(hasattr(independent, '__signature__'))   # names are not Python identifiers
        np.testing.assert_allclose(independent(**{'Rx:0.theta': 0.3, 'Rx:1.theta': 0.5}), rotation(0.3) @ rotation(0.5))

        # Shared: both arguments fed by one parameter
        shared = Circuit_Process_Matrix_Function_Helper([rotation, rotation], separator='.', gate_labels=['Rx:0', 'Rx:1'],
                        parameter_names={'Rx:0.theta': 'shared:theta', 'Rx:1.theta': 'shared:theta'})
        self.assertEqual(shared.parameter_names, ['shared:theta'])
        np.testing.assert_allclose(shared(**{'shared:theta': 0.4}), rotation(0.4) @ rotation(0.4))

        # Derivative with respect to the shared parameter is the sum of the partials
        element = lambda **kw: np.real(shared(**kw)[0, 0])
        element_independent = lambda **kw: np.real(independent(**kw)[0, 0])
        _, d_shared = shared.gradient(element, wrt=['shared:theta'], **{'shared:theta': 0.4})
        _, d_indep = independent.gradient(element_independent, wrt=['Rx:0.theta', 'Rx:1.theta'], **{'Rx:0.theta': 0.4, 'Rx:1.theta': 0.4})
        self.assertAlmostEqual(d_shared['shared:theta'], d_indep['Rx:0.theta'] + d_indep['Rx:1.theta'], places=9)

        with self.assertRaises(ValueError):
            Circuit_Process_Matrix_Function_Helper([rotation], gate_labels=['Rx:0'], parameter_names={'Rx:0__phi': 'x'})
        with self.assertRaises(TypeError):
            shared(**{'Rx:0.theta': 0.4})

    def test_shared_gate_cache(self):
        """ A cache shared by two circuit functions reuses gate evaluations across circuits """
        calls = []
        def rotation(theta: float):
            calls.append(theta)
            return np.kron(Unitary.R(0., theta), Unitary.R(0., theta).conj())
        cache = GateProcessMatrixCache()
        first = Circuit_Process_Matrix_Function_Helper([rotation], gate_cache=cache)
        second = Circuit_Process_Matrix_Function_Helper([rotation, rotation, rotation], gate_cache=cache)
        first(rotation__theta=0.2)
        second(rotation__theta=0.2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(cache.hits, 1)

if __name__ == '__main__':
    unittest.main()
