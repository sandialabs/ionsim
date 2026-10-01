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

from ionsim.process import Gate, Circuit
from ionsim.degree_of_freedom import AtomicStructure
from ionsim.basis import StandardBasis
from ionsim.named_operators import Unitary, Pauli
from ionsim.operator import EnergyShiftOperator, CouplingOperator
from ionsim.state import State 
from ionsim.hamiltonian import Hamiltonian
from ionsim.lindbladian import Dissipator, Lindbladian
from ionsim.gst_circuit_planner import GSTCircuitPlanner
from ionsim.gst_circuit_parser import CircuitData 
from ionsim.gate_set_tomography import (linear_solve_for_gate_parameters, mle_solve_for_gate_parameters, gate_set_errors,
                                        log_likelihood, chi_squared, simulate_gst_data, evaluate_gate_set)
from ionsim.gate_set_model import GateSetModel


def E0_1Q(prob_false_bright:float, prob_false_dark: float):
    M = np.zeros((2,2))
    M[0,0] = (1. - prob_false_bright)
    M[1,1] = prob_false_dark
    return M

def E1_1Q(prob_false_bright:float, prob_false_dark: float):
    M = np.zeros((2,2))
    M[0,0] = prob_false_bright
    M[1,1] = (1. - prob_false_dark)
    return M


class TestGST(unittest.TestCase):

    def setUp(self):
        """Set up the gst circuits for analysis. This set up tests the circuit planner and parsing."""
        self.qubit = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.basis = StandardBasis([self.qubit])

        def prep_state_function(SPAM_error_probability: float): 
            """ Model of the prep state as a function of parameters (a vector with d^2 - 1 entries), returns a constrained supervector """ 
            p = SPAM_error_probability 
            rho_q1 = np.zeros((2,2))
            rho_q1[0,0] = (1. - p)
            rho_q1[1,1] = p 
            state = State.from_density_matrix(self.basis, rho_q1)
            return state.supervector 
        
        def POVM_models(SPAM_error_probability: float): 
            """ Dictionary of POVMs evaluated at the function parameters: """ 
            POVMs = {}
            p = SPAM_error_probability 

            # 0
            M0 = E0_1Q(p, 0.)
            operator = EnergyShiftOperator.from_matrix(self.basis, M0)
            POVMs["0"] = operator.superbra 
        
            # 1
            M1 = E1_1Q(p, 0.)
            operator = EnergyShiftOperator.from_matrix(self.basis, M1) 
            POVMs["1"] = operator.superbra 
            return POVMs ## POVMs["00"] -> row vector 

        def X_pi_2_co_prop_simple(amplitude_noise_strength: float): 
            """ Single parameter process matrix model for an X(pi/8) rotation subject to white amplitude noise"""
            # Set up Hamiltonian and sigma_X dissipator: 
            rotation_angle = np.pi/2.
            phi = 0.
            omega = self.qubit.energy_levels[1].energy - self.qubit.energy_levels[0].energy
            rabi_rate = 100e3 * 2*np.pi # rad./s
            pi_time = abs(np.pi)/rabi_rate
            prefactor = np.exp(1j*phi) * rabi_rate/2.
            ham_operators = [CouplingOperator.from_matrix(self.basis, prefactor * Pauli.plus, omega, None)]
            interaction_frame_energies = [-state.energy for state in self.basis.states] # implement arbitrary hamiltonian (with time-dependence? need an adiabatic intertwiner)
            ham = Hamiltonian(self.basis, ham_operators, interaction_frame_energies)

            spin_flip_x_rate = (amplitude_noise_strength * 1E-6 * rabi_rate**2 )/ 4.
            spin_flipper_x = np.sqrt(spin_flip_x_rate) * Pauli.X 
            diss_operators = [CouplingOperator.from_matrix(self.basis, spin_flipper_x, 0)]
            diss_interaction_frame_energies = [0 for state in self.basis.states] # implement arbitrary hamiltonian (with time-dependence? need an adiabatic intertwiner)
            dissipator = Dissipator(self.basis, diss_operators, diss_interaction_frame_energies)
            rabi_lindbladian = Lindbladian(ham, dissipator) 
        
            duration = rotation_angle/rabi_rate 
            gate = Gate.from_lindbladian(self.basis, rabi_lindbladian, duration, lindbladian_time_independent=True)
            return gate.process_matrix 

        def Y_pi_2_co_prop_simple(amplitude_noise_strength: float): 
            """ Single parameter process matrix model for an X(pi/8) rotation subject to white amplitude noise"""
            # Set up Hamiltonian and sigma_X dissipator: 
            rotation_angle = np.pi/2.
            phi = np.pi/2.
            omega = self.qubit.energy_levels[1].energy - self.qubit.energy_levels[0].energy
            rabi_rate = 100e3 * 2*np.pi # rad./s
            pi_time = abs(np.pi)/rabi_rate
            prefactor = np.exp(1j*phi) * rabi_rate/2.
            ham_operators = [CouplingOperator.from_matrix(self.basis, prefactor * Pauli.plus, omega, None)]
            interaction_frame_energies = [-state.energy for state in self.basis.states] # implement arbitrary hamiltonian (with time-dependence? need an adiabatic intertwiner)
            ham = Hamiltonian(self.basis, ham_operators, interaction_frame_energies)

            spin_flip_y_rate = (amplitude_noise_strength * 1E-6 * rabi_rate**2 )/ 4.
            spin_flipper_y = np.sqrt(spin_flip_y_rate) * Pauli.Y 
            diss_operators = [CouplingOperator.from_matrix(self.basis, spin_flipper_y, 0)]
            diss_interaction_frame_energies = [0 for state in self.basis.states] # implement arbitrary hamiltonian (with time-dependence? need an adiabatic intertwiner)
            dissipator = Dissipator(self.basis, diss_operators, diss_interaction_frame_energies)
            rabi_lindbladian = Lindbladian(ham, dissipator) 
        
            duration = rotation_angle/rabi_rate 
            gate = Gate.from_lindbladian(self.basis, rabi_lindbladian, duration, lindbladian_time_independent=True)
            return gate.process_matrix 


        self.prep_state_model = prep_state_function 
        self.POVM_models = POVM_models

        # Gates are specified by string label with their qubit argument(s)
        gate_names = ['Gxpi2:0', 'Gypi2:0'] 
        qubit_indices = [0] 
        amplitude_noise_strength = 0.125 # S0 in rad^2/MHz 
        self.gate_models = {'Gxpi2:0' : X_pi_2_co_prop_simple, 'Gypi2:0' : Y_pi_2_co_prop_simple} 
        self.evaluated_gate_models = {'Gxpi2:0' : X_pi_2_co_prop_simple(amplitude_noise_strength), 
            'Gypi2:0' : Y_pi_2_co_prop_simple(amplitude_noise_strength) 
        } 

        num_qubits = len(qubit_indices)
        powers = [1, 2, 4]
        self.gst_circuit_planner = GSTCircuitPlanner(gate_names, qubit_indices, germ_powers = powers) 

        self.gst_circuits = self.gst_circuit_planner.generate_gst_circuits()

        # Construct initial state 
        SPAM_error_prob = 0.0025
        self.rho_0 = State.from_supervector(self.basis, self.prep_state_model(SPAM_error_prob))
    
        self.outcome_labels = ['0', '1']
        self.outcome_matrix = np.vstack(np.array([outcome_vector for outcome_vector in self.POVM_models(SPAM_error_prob).values()])) 

        # Run method to generate and populate circuit outcomes and test circuit planning 
        self.test_circuit_simulations_and_outcomes()
    
        ## Models and parameter information: initial guesses, bounds, and sharing among models.
        # model = "shared" ties together every model with an argument of that name:
        #   SPAM_error_probability -> prep & POVM models;  amplitude_noise_strength -> Gxpi2:0 & Gypi2:0 models
        self.gate_set_model = GateSetModel(self.prep_state_model, self.POVM_models, self.gate_models)
        self.gate_set_model.specify_parameter("SPAM_error_probability", model = "shared", guess = 1e-4, bounds = (0., 1.))
        self.gate_set_model.specify_parameter("amplitude_noise_strength", model = "shared", guess = 0.5, bounds = (0.0001, 10.0))

        # True parameter values used to simulate the data (the reference for gate set errors), and ideal values (the gauge
        # target for linear GST: noise-free gates, perfect state preparation and measurement)
        self.true_values = {'shared:SPAM_error_probability': SPAM_error_prob, 'shared:amplitude_noise_strength': amplitude_noise_strength}
        self.ideal_values = {'shared:SPAM_error_probability': 0., 'shared:amplitude_noise_strength': 0.}


    def test_circuit_simulations_and_outcomes(self):
        """ Test the generating of circuit outcomes from simulations and writing outcomes """
        _rng = np.random.default_rng(1) # explicit seed 
        N_shots = 5000
        for i, circuit in enumerate(self.gst_circuits):
            # Reinitialize the state: 
            rho = self.rho_0 # cp 
    
            # For each gate in the simulator, evolve the state forward according to the gate dynamics         
            for gate_label in circuit.expanded_gate_labels:
                # Run IonSim simulation of the gate 
                rho = rho.propagate_using_process_matrix(self.evaluated_gate_models[gate_label])
    
            # Estimate and record circuit outcomes in a dictionary to create GstCircuit object: 
            outcome_probabilities = rho.compute_basis_state_probabilities_from_effect_matrix(self.outcome_matrix) 
            estimated_outcome_counts = _rng.multinomial(N_shots, [*outcome_probabilities])
            outcome_info = {}
            for label, counts in zip(self.outcome_labels, estimated_outcome_counts):
                outcome_info[label] = counts
    
            # Update the circuit's attribute directly with the "measurement" outcome information as a CircuitData object  
            circuit_data = CircuitData.from_counts(outcome_info)
            circuit.measurement_data = circuit_data

        self.gst_circuits = self.gst_circuits

    def test_linear_gst_analysis(self):
        """ Test linear GST (LGST) starting from the ideal gate set as the gauge reference, with re-gauging """
        result = linear_solve_for_gate_parameters(self.gst_circuits, self.gate_set_model, self.gst_circuit_planner, target = self.ideal_values)
        gate_set_error = gate_set_errors(self.gate_set_model, result.theta, reference = self.true_values)
        X_pi2_error = gate_set_error['Gxpi2:0']
        Y_pi2_error = gate_set_error['Gypi2:0']
        SPAM_error = gate_set_error["prep"]
        SPAM_error += gate_set_error["POVM"]
        self.assertAlmostEqual(X_pi2_error, 0.0004938334895727403, places=5)
        self.assertAlmostEqual(Y_pi2_error, 0.0004938334895728776, places=5)
        self.assertAlmostEqual(SPAM_error, 2.039563696979629e-05, places=5)
        self.assertEqual(result.method, 'linear')
        self.assertTrue(result.success)
        self.assertGreater(result.gauge_iterations, 0)
        self.assertEqual(sorted(result.lgst_estimates['gate_estimates']), ['Gxpi2:0', 'Gypi2:0'])

        # The gauge target can also be given explicitly, e.g. an ideal prep |0><0|, ideal measurement projectors, and ideal rotations
        ideal_gate_set = {'prep': State.from_density_matrix(self.basis, Pauli.projector_0),
                          'POVM': {'0': EnergyShiftOperator.from_matrix(self.basis, Pauli.projector_0),
                                   '1': EnergyShiftOperator.from_matrix(self.basis, Pauli.projector_1)},
                          'Gxpi2:0': self.gate_models['Gxpi2:0'](0.), 'Gypi2:0': self.gate_models['Gypi2:0'](0.)}
        explicit = linear_solve_for_gate_parameters(self.gst_circuits, self.gate_set_model, self.gst_circuit_planner, target_gate_set = ideal_gate_set)
        np.testing.assert_array_equal(explicit.theta, result.theta)

    def test_linear_gst_regauging(self):
        """ Re-gauging removes the bias of fitting models to LGST estimates in the gauge of an ideal (not the true) gate set """
        # Noise-free data: outcome frequencies equal to the true probabilities
        exact_data = simulate_gst_data(self.gst_circuits, self.gate_set_model, self.true_values, 10**15, rng = 0)
        true_theta = self.gate_set_model.parse_theta(self.true_values)

        regauged = linear_solve_for_gate_parameters(exact_data, self.gate_set_model, self.gst_circuit_planner, target = self.ideal_values)
        np.testing.assert_allclose(regauged.theta, true_theta, rtol=1e-5)
        self.assertTrue(regauged.success)

        # Without re-gauging, the ideal gauge reference biases the fit, whatever the number of shots
        ideal_gauge_only = linear_solve_for_gate_parameters(exact_data, self.gate_set_model, self.gst_circuit_planner,
                                                            target = self.ideal_values, max_gauge_iterations = 0)
        self.assertGreater(abs(ideal_gauge_only.theta[1] / true_theta[1] - 1.), 0.01)
        self.assertEqual(ideal_gauge_only.gauge_iterations, 0)

        # Too few passes: reported as not converged, with a warning
        with self.assertWarnsRegex(UserWarning, "did not converge"):
            unconverged = linear_solve_for_gate_parameters(exact_data, self.gate_set_model, self.gst_circuit_planner,
                                                           target = self.ideal_values, max_gauge_iterations = 1, gauge_tolerance = 1e-14)
        self.assertFalse(unconverged.success)

        # Independent models (no shared parameters): each model is fit on its own parameters, also re-gauged
        independent = GateSetModel(self.prep_state_model, self.POVM_models, self.gate_models)
        ideal_independent = {'Gxpi2:0.amplitude_noise_strength': 0., 'Gypi2:0.amplitude_noise_strength': 0.,
                             'prep.SPAM_error_probability': 0., 'POVM.SPAM_error_probability': 0.}
        result = linear_solve_for_gate_parameters(exact_data, independent, self.gst_circuit_planner, target = ideal_independent)
        values = result.parameter_values
        self.assertAlmostEqual(values['Gxpi2:0.amplitude_noise_strength'], 0.125, places=5)
        self.assertAlmostEqual(values['Gypi2:0.amplitude_noise_strength'], 0.125, places=5)
        # Independent prep and measurement errors are gauge-equivalent here: the data determine the SPAM outcome probabilities
        # (no gates applied), not how the error is split between preparation and measurement
        def spam_probabilities(gate_set_model, theta):
            gate_set = evaluate_gate_set(gate_set_model, theta)
            return np.real([np.asarray(gate_set['POVM'][outcome]) @ gate_set['prep'] for outcome in ['0', '1']])
        np.testing.assert_allclose(spam_probabilities(independent, result.theta), spam_probabilities(self.gate_set_model, true_theta), atol=1e-7)

    def test_mle_seeded_by_linear_gst(self):
        """ MLE started from the linear GST estimate converges to the same estimate as from the specified guesses """
        seed = linear_solve_for_gate_parameters(self.gst_circuits, self.gate_set_model, self.gst_circuit_planner, target = self.ideal_values)
        seeded = mle_solve_for_gate_parameters(self.gst_circuits, self.gate_set_model, initial_guess = seed.theta)
        plain = mle_solve_for_gate_parameters(self.gst_circuits, self.gate_set_model)
        np.testing.assert_array_equal(seeded.initial_guess, seed.theta)
        # Same optimum within the optimizer's tolerance
        np.testing.assert_allclose(seeded.theta, plain.theta, atol=5e-5)
        self.assertAlmostEqual(seeded.log_likelihood, plain.log_likelihood, delta=1e-9*abs(plain.log_likelihood))

    def test_mle_gst_analysis(self):
        """ Test GST via maximum likelihood estimation (MLE)""" 
        result = mle_solve_for_gate_parameters(self.gst_circuits, self.gate_set_model)
        gate_set_error = gate_set_errors(self.gate_set_model, result.theta, reference = self.true_values)

        X_pi2_error = gate_set_error['Gxpi2:0']
        Y_pi2_error = gate_set_error['Gypi2:0']
        SPAM_error = gate_set_error["prep"]
        SPAM_error += gate_set_error["POVM"]
        self.assertAlmostEqual(X_pi2_error, 0.0006672864884890055, places=5)
        self.assertAlmostEqual(Y_pi2_error, 0.0006672864884889269, places=5)
        self.assertAlmostEqual(SPAM_error, 0.0010575712223980156, places=5)

        # Errors against the true values equal errors against the explicit true gate set
        true_gate_set = evaluate_gate_set(self.gate_set_model, self.true_values)
        errors_explicit = gate_set_errors(self.gate_set_model, result.theta, reference_gate_set = true_gate_set)
        for key in gate_set_error:
            self.assertAlmostEqual(errors_explicit[key], gate_set_error[key], places=14)

    def test_shared_parameter_sensitivity(self):
        """ Sensitivity to a parameter shared by two gates equals the sum of the sensitivities to independent copies """
        circuit = [c for c in self.gst_circuits if {'Gxpi2:0', 'Gypi2:0'} <= set(c.expanded_gate_labels)][-1]

        shared = GateSetModel(self.prep_state_model, self.POVM_models, self.gate_models)
        shared.specify_parameter("amplitude_noise_strength", model = "shared")
        independent = GateSetModel(self.prep_state_model, self.POVM_models, self.gate_models)

        self.gst_circuit_planner.gate_set_model = shared
        S = self.gst_circuit_planner.compute_circuit_sensitivity(circuit, {'shared:amplitude_noise_strength': 0.125,
                            'prep.SPAM_error_probability': 0.0025, 'POVM.SPAM_error_probability': 0.0025})
        self.gst_circuit_planner.gate_set_model = independent
        I = self.gst_circuit_planner.compute_circuit_sensitivity(circuit, {'Gxpi2:0.amplitude_noise_strength': 0.125,
                            'Gypi2:0.amplitude_noise_strength': 0.125, 'prep.SPAM_error_probability': 0.0025, 'POVM.SPAM_error_probability': 0.0025})

        self.assertIn('shared:amplitude_noise_strength', S)
        for outcome in ['0', '1']:
            self.assertAlmostEqual(S['shared:amplitude_noise_strength'][outcome],
                I['Gxpi2:0.amplitude_noise_strength'][outcome] + I['Gypi2:0.amplitude_noise_strength'][outcome], places=7)
            self.assertAlmostEqual(S['prep.SPAM_error_probability'][outcome], I['prep.SPAM_error_probability'][outcome], places=7)

    def test_analysis_reads_current_data(self):
        """ Every analysis reads the circuits' current data: no caches to clear after replacing or editing data """
        LL = log_likelihood(self.gst_circuits, self.gate_set_model, self.true_values)

        # Order of outcomes in the count data does not matter
        for circ in self.gst_circuits:
            counts = circ.measurement_data.counts
            circ.measurement_data = CircuitData.from_counts({'1': counts['1'], '0': counts['0']})
        self.assertAlmostEqual(log_likelihood(self.gst_circuits, self.gate_set_model, self.true_values), LL, places=6)

        # Counts edited in place are picked up (the log-likelihood is linear in the counts)
        for circ in self.gst_circuits:
            for outcome in circ.measurement_data.counts:
                circ.measurement_data.counts[outcome] *= 2
        self.assertAlmostEqual(log_likelihood(self.gst_circuits, self.gate_set_model, self.true_values) / LL, 2., places=12)

        # Replaced data is used by linear GST: the Gram matrix entry for the empty circuit is its new observed frequency
        empty_circuit = [c for c in self.gst_circuits if c.depth == 0][0]
        empty_circuit.measurement_data = CircuitData.from_counts({'0': 9900, '1': 100})
        result = linear_solve_for_gate_parameters(self.gst_circuits, self.gate_set_model, self.gst_circuit_planner, target = self.ideal_values)
        self.assertAlmostEqual(result.lgst_estimates['gram_matrix'][0, 0], 0.99, places=12)

    def test_likelihood_of_circuit_subsets(self):
        """ log_likelihood and chi_squared take the circuit list explicitly; both are sums over circuits """
        first, second = self.gst_circuits[::2], self.gst_circuits[1::2]
        for objective in (log_likelihood, chi_squared):
            total = objective(self.gst_circuits, self.gate_set_model, self.true_values)
            parts = objective(first, self.gate_set_model, self.true_values) + objective(second, self.gate_set_model, self.true_values)
            self.assertAlmostEqual(parts, total, delta=1e-12*abs(total))

    def test_simulate_gst_data(self):
        """ Simulated data: new circuits with sampled counts, reproducible by seed; the original circuits are unchanged """
        data_before = [circ.measurement_data for circ in self.gst_circuits]
        simulated = simulate_gst_data(self.gst_circuits, self.gate_set_model, self.true_values, 1000, rng = 11)
        again = simulate_gst_data(self.gst_circuits, self.gate_set_model, self.true_values, 1000, rng = 11)

        self.assertEqual(len(simulated), len(self.gst_circuits))
        self.assertTrue(all(circ.measurement_data is data for circ, data in zip(self.gst_circuits, data_before)))
        for sim, rep, orig in zip(simulated, again, self.gst_circuits):
            self.assertIsNot(sim, orig)
            self.assertEqual(sim.expanded_gates, orig.expanded_gates)
            self.assertEqual(sum(sim.measurement_data.counts.values()), 1000)
            self.assertEqual(sim.measurement_data.counts, rep.measurement_data.counts)

        # Planned circuits (no data) can be simulated too, and the simulated data are analyzable
        planned = self.gst_circuit_planner.generate_gst_circuits()
        simulated = simulate_gst_data(planned, self.gate_set_model, self.true_values, 20000, rng = np.random.default_rng(3))
        result = mle_solve_for_gate_parameters(simulated, self.gate_set_model)
        np.testing.assert_allclose(result.theta, [0.0025, 0.125], rtol=0.1)

    def test_input_validation(self):
        """ Helpful errors for inputs that are no longer supported or are ambiguous """
        with self.assertRaisesRegex(ValueError, "linear_solve_for_gate_parameters"):
            mle_solve_for_gate_parameters(self.gst_circuits, self.gate_set_model, initial_guess = 'lgst')
        with self.assertRaisesRegex(ValueError, "either parameter values"):
            linear_solve_for_gate_parameters(self.gst_circuits, self.gate_set_model, self.gst_circuit_planner)
        with self.assertRaisesRegex(ValueError, "either parameter values"):
            gate_set_errors(self.gate_set_model, self.true_values, reference = self.true_values,
                            reference_gate_set = evaluate_gate_set(self.gate_set_model, self.true_values))
        with self.assertRaises(TypeError):
            mle_solve_for_gate_parameters(self.gate_set_model, self.gst_circuits)

if __name__ == '__main__':
    unittest.main()
