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

from ionsim.process import Gate, Circuit, Circuit_Process_Matrix_Function, GateProcessMatrixCache
from ionsim.degree_of_freedom import AtomicStructure
from ionsim.basis import StandardBasis, PauliProductBasis
from ionsim.named_operators import Unitary, Pauli
from ionsim.noise import Noise
from ionsim.operator import EnergyShiftOperator, CouplingOperator
from ionsim.hamiltonian import Hamiltonian
from ionsim.state import State 
from ionsim.lindbladian import Dissipator, Lindbladian
from ionsim.testing import assert_array_close

class TestProcess(unittest.TestCase):

    def setUp(self):
        """Set up the necessary objects for testing."""
        self.spin_a = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.spin_b = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.spin_c = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2','P1/2'], level_names=['S1/2,0,0', 'S1/2,1,0','P1/2,1,-1'])
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

    def test_R_hamiltonian_matches_unitary(self):
        """ Test that R hamiltonian produces a unitary evolution operator that matches expected value """ 
        basis = StandardBasis([self.spin_a])  # 00, 01, 10, 11 

        rabi_rate = 100e3 * 2 * np.pi  # rad./s
        omega = self.spin_a.energy_levels[1].energy - self.spin_a.energy_levels[0].energy
        static_operator = (rabi_rate / 2) * Pauli.plus
        theta = np.pi/16.
        duration = theta/rabi_rate

        operator_a = CouplingOperator.from_matrix(basis, static_operator, omega) 
        
        interaction_frame_energies = [-1 * state.energy for state in basis.states]
        hamiltonian = Hamiltonian(basis, [operator_a], interaction_frame_energies, sparse=False)
        X_pi16_gate = Gate.from_hamiltonian(basis, hamiltonian, duration)
        
        # Check R gate unitary matrix form  
        X_pi16_ref = Unitary.R(0., theta) 
        assert_array_close(X_pi16_ref, X_pi16_gate.unitary, rtol=1e-6, atol=1e-6) # we expect agreement up to numerical timestep error and precision  


    def test_noisy_phi_gate_process_fidelity(self):
        """Test the process fidelity of the noisy phi gate."""
        fidelity = self.noisy_phi_gate.compute_process_fidelity(self.Sx.process_matrix)
        self.assertAlmostEqual(fidelity, 0.9535335138400007, places=14)

    def test_noisy_theta_gate_process_fidelity(self):
        """Test the process fidelity of the noisy theta gate."""
        fidelity = self.noisy_theta_gate.compute_process_fidelity(self.Sx.process_matrix)
        # analytic (1 + exp(-sigma^2/2)) / 2 = 0.97592490368; normalized realization weights (Noise.realizations) are 6.8e-9 from it
        self.assertAlmostEqual(fidelity, 0.9759249104808633, places=14)

    def test_extra_noisy_gate_process_fidelity(self):
        """Test the process fidelity of the extra noisy gate."""
            #self.basis, self.noisy_phi_gate.process_matrix_function, {'phi': 0, 'theta': np.pi/2}, [self.spin_a], self.theta_noise,
        extra_noisy_gate = Gate.from_process_matrix_function(
            self.basis, self.noisy_phi_gate.process_matrix_function, {'phi': 0, 'theta': np.pi/2}, self.theta_noise,
        )
        fidelity = extra_noisy_gate.compute_process_fidelity(self.Sx.process_matrix)
        self.assertAlmostEqual(fidelity, 0.930617644191573, places=14)

    def test_gate_parameter_ordering(self):
        """Test removal of a bug where Gate creation depended on parameter ordering."""
        g0 = Gate.from_unitary_function(self.basis, Unitary.R, {'phi': 0, 'theta': np.pi/2}, [self.spin_a], self.phi_noise)
        g1 = Gate.from_unitary_function(self.basis, Unitary.R, {'theta': np.pi/2, 'phi': 0}, [self.spin_a], self.phi_noise)
        np.testing.assert_array_equal(g0.process_matrix, g1.process_matrix)

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
        self.assertAlmostEqual(fidelity, 0.930617644191573, places=14)

        # Test computing outcome probabilities 
        outcome_operator = EnergyShiftOperator.from_matrix(self.basis, np.kron(Pauli.projector_1, Pauli.projector_0)) 
        initial_state = State.from_coefficients(self.basis, [1., 0., 0., 0.]) 

        outcome_probability = ramsey.predict_outcome_probabilities(initial_state, [outcome_operator]) 
        self.assertAlmostEqual(outcome_probability[0], 0.9530090408324356, places = 10)

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
        self.assertAlmostEqual(outcome_prob, 0.9530090408324356, places = 10)

        # Compute outcome probability using probability function: 
        prob, prob_gradients = circuit_pm_function.gradient(prob_function, wrt = ["R__phi", "R__theta"], **circuit_parameters) 

        ### Test Jacobian functionality: Compute Jacobian when considering more than 1 outcome: 
        outcome_operator2 = EnergyShiftOperator.from_matrix(self.basis, np.kron(Pauli.projector_0, Pauli.projector_0)) 

        probs_function = ramsey_circuit.build_outcome_probabilities_function(initial_state, [outcome_operator, outcome_operator2])

        probs, jacobian = circuit_pm_function.jacobian(probs_function, wrt = ["R__phi", "R__theta"], **circuit_parameters)


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
        independent = Circuit_Process_Matrix_Function([rotation, rotation], separator='.', gate_labels=['Rx:0', 'Rx:1'])
        self.assertEqual(independent.parameter_names, ['Rx:0.theta', 'Rx:1.theta'])
        self.assertFalse(hasattr(independent, '__signature__'))   # names are not Python identifiers
        np.testing.assert_allclose(independent(**{'Rx:0.theta': 0.3, 'Rx:1.theta': 0.5}), rotation(0.3) @ rotation(0.5))

        # Shared: both arguments fed by one parameter
        shared = Circuit_Process_Matrix_Function([rotation, rotation], separator='.', gate_labels=['Rx:0', 'Rx:1'],
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
            Circuit_Process_Matrix_Function([rotation], gate_labels=['Rx:0'], parameter_names={'Rx:0__phi': 'x'})
        with self.assertRaises(TypeError):
            shared(**{'Rx:0.theta': 0.4})

    def test_shared_gate_cache(self):
        """ A cache shared by two circuit functions reuses gate evaluations across circuits """
        calls = []
        def rotation(theta: float):
            calls.append(theta)
            return np.kron(Unitary.R(0., theta), Unitary.R(0., theta).conj())
        cache = GateProcessMatrixCache()
        first = Circuit_Process_Matrix_Function([rotation], gate_cache=cache)
        second = Circuit_Process_Matrix_Function([rotation, rotation, rotation], gate_cache=cache)
        first(rotation__theta=0.2)
        second(rotation__theta=0.2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(cache.hits, 1)

    def test_circuit_noise_sensitivity(self):
        """ Sensitivities to the parameters of a circuit-level noise distribution, compared to analytic results """
        # R(theta) on spin a with Gaussian circuit-level noise on theta: P(1) = (1 - cos(theta) exp(-sigma^2/2)) / 2
        theta, sigma = 1.1, np.pi/10
        noise = Noise.from_named_pdf('theta', 'gaussian', {'standard_deviation': sigma}, np.linspace(-8, 8, 161),
                                     domain_scale_parameter='standard_deviation')
        circuit = Circuit.from_gates([Gate.from_unitary_function(self.basis, Unitary.R, {'phi': 0., 'theta': theta}, [self.spin_a])], noise)
        function = circuit.process_matrix_function
        name = 'theta_noise_standard_deviation'
        self.assertEqual(function.parameter_names, ['R__phi', 'R__theta', name])

        outcome_operator = EnergyShiftOperator.from_matrix(self.basis, np.kron(Pauli.projector_1, Pauli.projector_0))
        initial_state = State.from_coefficients(self.basis, [1., 0., 0., 0.])
        probability = circuit.build_outcome_probability_function(initial_state, outcome_operator)
        P, jacobian, hessian = function.derivatives(probability, wrt=['R__theta', name], **{'R__phi': 0., 'R__theta': theta, name: sigma})

        decay = np.exp(-sigma**2/2)
        self.assertAlmostEqual(P, (1 - np.cos(theta)*decay)/2, places=12)
        self.assertAlmostEqual(jacobian[name], sigma*np.cos(theta)*decay/2, places=9)
        self.assertAlmostEqual(jacobian['R__theta'], np.sin(theta)*decay/2, places=9)
        self.assertAlmostEqual(hessian[name][name], np.cos(theta)*decay*(1 - sigma**2)/2, places=6)
        self.assertAlmostEqual(hessian['R__theta'][name], -sigma*np.sin(theta)*decay/2, places=6)

        # Defaults reproduce the circuit's process matrix; the noise parameter can be renamed (e.g. GST-style names)
        np.testing.assert_allclose(function(R__phi=0., R__theta=theta), circuit.process_matrix, atol=1e-14)
        renamed = Circuit.from_gates([Gate.from_unitary_function(self.basis, Unitary.R, {'phi': 0., 'theta': theta}, [self.spin_a])], noise,
                                     parameter_names={name: 'shared:theta_noise_strength'})
        self.assertIn('shared:theta_noise_strength', renamed.process_matrix_function.parameter_names)

    def test_ramsey_circuit_process_matrix_function(self):
        """ A circuit with a fixed gate, gate-level phi noise, and circuit-level theta noise has a process matrix function whose
            noise sensitivity agrees with rebuilding the circuit at a different noise strength """
        xs = np.linspace(-np.pi, np.pi, 21)
        def ramsey(theta_noise):
            return Circuit.from_gates([Gate.from_unitary(self.basis, Unitary.sqrtX, [self.spin_a]),
                                       Gate.from_unitary_function(self.basis, Unitary.R, {'phi': 0, 'theta': np.pi/2}, [self.spin_a], self.phi_noise)],
                                      theta_noise)
        circuit = ramsey(self.theta_noise)
        function = circuit.process_matrix_function
        name = 'theta_noise_standard_deviation'
        self.assertEqual(function.parameter_names, ['R__phi', 'R__theta', name])   # the fixed sqrtX gate has no parameters
        np.testing.assert_allclose(function(R__phi=0., R__theta=np.pi/2), circuit.process_matrix, atol=1e-14)

        outcome_operator = EnergyShiftOperator.from_matrix(self.basis, np.kron(Pauli.projector_1, Pauli.projector_0))
        initial_state = State.from_coefficients(self.basis, [1., 0., 0., 0.])
        probability = circuit.build_outcome_probability_function(initial_state, outcome_operator)
        _, gradient = function.gradient(probability, wrt=[name], R__phi=0., R__theta=np.pi/2, **{name: np.pi/10})

        h = 1e-5
        def rebuilt(sigma):
            noise = Noise.from_named_pdf('theta', 'gaussian', {'standard_deviation': sigma}, xs)
            return ramsey(noise).predict_outcome_probabilities(initial_state, [outcome_operator])[0]
        self.assertAlmostEqual(gradient[name], (rebuilt(np.pi/10 + h) - rebuilt(np.pi/10 - h)) / (2*h), places=8)

    def test_gates_from_hamiltonian_and_lindbladian_functions(self):
        """ Gates built from Hamiltonian / Lindbladian functions, without and with gate-level noise """
        spin = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        basis = StandardBasis([spin])
        omega = spin.energy_levels[1].energy - spin.energy_levels[0].energy
        def hamiltonian(rabi_rate: float):
            return Hamiltonian(basis, [CouplingOperator.from_matrix(basis, rabi_rate/2*Pauli.plus, omega, None)], [-state.energy for state in basis.states])
        def lindbladian(rabi_rate: float):
            return Lindbladian(hamiltonian(rabi_rate), Dissipator(basis, [CouplingOperator.from_matrix(basis, 0.*Pauli.X, 0)], [0, 0]))
        duration, rabi_rate = 1e-6, 2*np.pi*1e5
        noise = Noise.from_named_pdf('rabi_rate', 'gaussian', {'standard_deviation': 2*np.pi*1e3}, np.linspace(-4, 4, 41),
                                     domain_scale_parameter='standard_deviation')

        gate = Gate.from_lindbladian_function(basis, lindbladian, duration, {'rabi_rate': rabi_rate}, lindbladian_time_independent=True)
        reference = Gate.from_lindbladian(basis, lindbladian(rabi_rate), duration, lindbladian_time_independent=True)
        np.testing.assert_allclose(gate.process_matrix, reference.process_matrix, atol=1e-12)

        noisy = Gate.from_lindbladian_function(basis, lindbladian, duration, {'rabi_rate': rabi_rate}, noise=noise, lindbladian_time_independent=True)
        averaged = noise.average(lambda x: Gate.from_lindbladian(basis, lindbladian(rabi_rate + x), duration, lindbladian_time_independent=True).process_matrix)
        np.testing.assert_allclose(noisy.process_matrix, averaged, atol=1e-12)
        # rotation-angle noise sigma_theta = 2 pi 1e3 * 1e-6 = 6.3e-3 rad dephases by ~sigma_theta^2 / 2 ~ 2e-5 (normalized weights;
        # unnormalized trapezoid weights on +-4 sigma also lost 6e-5 of the weight)
        self.assertGreater(np.linalg.norm(noisy.process_matrix - gate.process_matrix), 1e-5)

        h_gate = Gate.from_hamiltonian_function(basis, hamiltonian, duration, {'rabi_rate': rabi_rate}, [spin])
        h_reference = Gate.from_hamiltonian(basis, hamiltonian(rabi_rate), duration)
        np.testing.assert_allclose(h_gate.process_matrix, h_reference.process_matrix, atol=1e-10)
        np.testing.assert_allclose(h_gate.process_matrix, gate.process_matrix, atol=1e-6)   # same unitary dynamics

    def test_pauli_transfer_matrix_computations(self):
        noisy_gate = self.noisy_phi_gate
        noisy_gate_pauli = noisy_gate.convert_to_pauli_basis()
        Sx_pauli = self.Sx.convert_to_pauli_basis()

        fidelity = noisy_gate_pauli.compute_process_fidelity(Sx_pauli.process_matrix)
        # same gate as test_noisy_phi_gate_process_fidelity (normalized realization weights)
        self.assertAlmostEqual(fidelity, 0.9535335138400007, places=14)

    def test_Raman_gate(self):
        full_basis = StandardBasis([self.spin_c])
        target_rabi_rate = 10 * 2 * np.pi * 1E3 # rad/s 
        detuning = -2. * 2. * np.pi * 1E9
        rabi_rate = np.sqrt(np.abs(detuning) * 2. * target_rabi_rate) 

        omega1 = self.spin_c.energy_levels[2].energy - self.spin_c.energy_levels[0].energy + detuning
        omega2 = self.spin_c.energy_levels[2].energy - self.spin_c.energy_levels[1].energy + detuning
        omega_qubit = self.spin_c.energy_levels[1].energy - self.spin_c.energy_levels[0].energy
        assert (np.abs(omega1 - omega2) - omega_qubit)/(2*np.pi)/1E3 < 1.E-2, 'Raman resonance condition is violated.'

        size = len(full_basis.states)
        static_operator1 = np.zeros((size, size))
        static_operator1[2, 0] = rabi_rate/2. 
        static_operator2 = np.zeros((size, size))
        static_operator2[2, 1] = -rabi_rate/2.  # phase of pi

        raising_op1 = CouplingOperator.from_matrix(full_basis, static_operator1, omega1, [self.spin_c])
        raising_op2 = CouplingOperator.from_matrix(full_basis, static_operator2, omega2, [self.spin_c])
        
        interaction_frame_energies = [-1 * state.energy for state in full_basis.states]
        hamiltonian = Hamiltonian(full_basis, [raising_op1, raising_op2], interaction_frame_energies, sparse=False)

        theta = np.pi/16.
        duration = (np.pi/16.)/target_rabi_rate
        undesired_levels = [self.spin_c.energy_levels[2]] # P1/2 level to project out        
        #projection_input = {'levels' : undesired_levels}
        dt = 0.002 * 1E-6 
        times = np.arange(0, duration + dt, dt)
        X_pi16_Raman_gate = Gate.from_hamiltonian(full_basis, hamiltonian, duration, levels_to_project = undesired_levels, ode_solver = 'odeintz', time_evals = times) 
        reduced_basis = X_pi16_Raman_gate.basis

        # Process fidelity 
        X_pi16_ref = Unitary.R(0., theta) 

        X_pi16_ref = Gate.from_unitary(reduced_basis, X_pi16_ref, [self.spin_c])

        process_fidelity = X_pi16_Raman_gate.compute_process_fidelity(X_pi16_ref.process_matrix)
        self.assertAlmostEqual(process_fidelity, 0.999998720951, places=8)

        # Test pauli transfer matrix and pauli error rate calculations 
        import scipy
        error_channel_as_gate = Gate(reduced_basis, X_pi16_Raman_gate.process_matrix @ scipy.linalg.inv(X_pi16_ref.process_matrix))   
        error_rates = error_channel_as_gate.compute_pauli_error_rates()        
        # I <==> gate occurs as intended (no error); should be equal to process fidelity 
        self.assertAlmostEqual(0.9999987209516402, error_rates["I"].real, places=10)
        self.assertAlmostEqual(process_fidelity, error_rates["I"].real, places=10)

    def test_pauli_transfer_matrix(self):
        standard_2Q_basis = StandardBasis([self.spin_a, self.spin_b])
        Pauli_2Q_basis = PauliProductBasis([self.spin_a, self.spin_b]) 
        CZ_unitary = np.diag([1., 1., 1., -1.])

        CZ_gate_standard = Gate.from_unitary(standard_2Q_basis, CZ_unitary, [self.spin_a, self.spin_b])

        CZ_pauli_basis_gate = CZ_gate_standard.convert_to_pauli_basis()
        CZ_ptm = CZ_pauli_basis_gate.process_matrix

        test_diag_elements = ["II", "IZ", "ZI", "ZZ"] 
        expected_diag_values = [1., 1., 1., 1.]

        for i, diag_element in enumerate(test_diag_elements):
            indx = Pauli_2Q_basis.index_of_label(diag_element)
            self.assertEqual(CZ_ptm[indx, indx], expected_diag_values[i]) 

        test_offdiag_elements = [("IX", "II"), ("ZX", "IX"), ("IX", "ZX"), ("YX", "XY"), ("XZ", "XI"), ("YY", "XX"), ("YI", "XZ")]
        expected_offdiag_values = [0., 1., 1., -1., 1., 1., 0.]
        for k, offdiag_element in enumerate(test_offdiag_elements):
            i, j = Pauli_2Q_basis.indices_of_pauli_transfer_matrix_from_labels(*offdiag_element)
            self.assertEqual(CZ_ptm[i, j], expected_offdiag_values[k]) 


if __name__ == '__main__':
    unittest.main()
