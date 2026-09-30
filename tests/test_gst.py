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
from ionsim.gate_set_tomography import GateSetTomography
from ionsim.gst_parameters import GstModelParameters


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

        def X_pi_8_co_prop_simple(amplitude_noise_strength: float): 
            """ Single parameter process matrix model for an X(pi/8) rotation subject to white amplitude noise"""
            # Set up Hamiltonian and sigma_X dissipator: 
            rotation_angle = np.pi/8.
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
        
            # rotation angle = pi/8 = Omega*duration
            duration = rotation_angle/rabi_rate 
            gate = Gate.from_lindbladian(self.basis, rabi_lindbladian, duration, lindbladian_time_independent=True)
            return gate.process_matrix 


        self.prep_state_model = prep_state_function 
        self.POVM_models = POVM_models

        # Gates are specified by string label with their qubit argument(s)
        gate_names = ['Gxpi8:0'] 
        qubit_indices = [0] 
        self.amplitude_noise_strength = 0.125 # S0 in rad^2/MHz 
        self.gate_models = {'Gxpi8:0' : X_pi_8_co_prop_simple} 
        self.evaluated_gate_models = {'Gxpi8:0' : X_pi_8_co_prop_simple(self.amplitude_noise_strength)} 

        ## Models and parameter information (initial guesses, bounds, and sharing among models), specified once and used by
        # both the circuit planner (Fisher information) and the GST analysis.
        # SPAM_error_probability is an argument of both the prep and POVM models, so model = "shared" ties them together.
        self.model_parameters = GstModelParameters(self.prep_state_model, self.POVM_models, self.gate_models)
        self.model_parameters.specify_parameter("SPAM_error_probability", model = "shared", guess = 1e-4, bounds = (0., 1.))
        self.model_parameters.specify_parameter("amplitude_noise_strength", model = "Gxpi8:0", guess = 0.5, bounds = (0.0001, 10.0))

        num_qubits = len(qubit_indices)
        powers = [1, 2, 4, 8, 16, 32, 64, 128]
        fiducials = [[]]
        germs = [['Gxpi8:0']] 
        self.gst_circuit_planner = GSTCircuitPlanner(gate_names, qubit_indices, prep_fiducials = fiducials, measure_fiducials = fiducials, germ_powers = powers, 
                                                        germs = germs, parameters = self.model_parameters) 

        self.gst_circuits = self.gst_circuit_planner.generate_gst_circuits()

        # Construct initial state 
        SPAM_error_prob = 0.0025
        self.rho_0 = State.from_supervector(self.basis, self.prep_state_model(SPAM_error_prob))
    
        self.outcome_labels = ['0', '1']
        self.outcome_matrix = np.vstack(np.array([outcome_vector for outcome_vector in self.POVM_models(SPAM_error_prob).values()])) 

        # Run method to generate and populate circuit outcomes and test circuit planning 
        self.test_circuit_simulations_and_outcomes()
    
        self.true_POVM_effects = {} 
        self.true_POVM_effects['0'] = EnergyShiftOperator.from_matrix(self.basis, self.POVM_models(SPAM_error_prob)['0'].reshape(2,2))
        self.true_POVM_effects['1'] = EnergyShiftOperator.from_matrix(self.basis, self.POVM_models(SPAM_error_prob)['1'].reshape(2,2))

        self.true_gate_set = {}
        self.true_gate_set['prep'] = self.rho_0 
        self.true_gate_set['POVM'] = self.true_POVM_effects 
        self.true_gate_set['Gxpi8:0'] =  X_pi_8_co_prop_simple(self.amplitude_noise_strength)

        self.GST_analyzer = GateSetTomography.from_model_parameters(self.basis, self.parsed_circuits, self.model_parameters, 
                                    circuit_design = self.gst_circuit_planner, ideal_gate_set = self.true_gate_set, verbose = False)


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

        self.parsed_circuits = self.gst_circuits

    def test_mle_gst_analysis(self):
        """ Test GST via maximum likelihood estimation (MLE)""" 
        solver_results = self.GST_analyzer.mle_solve_for_gate_parameters() 
        gate_set_error = self.GST_analyzer.compute_gate_set_error_by_element(solver_results.x, self.true_gate_set)
        X_pi8_error = gate_set_error['Gxpi8:0']
        SPAM_error = gate_set_error["prep"]
        SPAM_error += gate_set_error["POVM"]
        self.assertAlmostEqual(X_pi8_error, 0.00027408522081525397, places=5)
        self.assertAlmostEqual(SPAM_error, 0.00011231981002808708, places=5)

    def test_staged_mle_gst_analysis(self):
        """ Test staged MLE: agrees with MLE, forwards scipy options to every stage, and restores the circuit list """
        circuits_before = self.GST_analyzer.parsed_circuits
        mle_result = self.GST_analyzer.mle_solve_for_gate_parameters()
        staged_result = self.GST_analyzer.staged_mle_solve_for_gate_parameters(options = {'maxiter': 500})

        np.testing.assert_allclose(staged_result.x, mle_result.x, rtol=1e-3)
        np.testing.assert_array_equal(self.GST_analyzer.gst_parameters, staged_result.x)
        self.assertIs(self.GST_analyzer.parsed_circuits, circuits_before)
        # One estimate per stage (germ power), the last equal to the final result
        self.assertEqual(sorted(self.GST_analyzer.results_by_stage), [1, 2, 4, 8, 16, 32, 64, 128])
        np.testing.assert_array_equal(self.GST_analyzer.results_by_stage[128], staged_result.x)

    def test_fisher_info(self):
        """ Test calculation of Fisher information, including the SPAM models and the shared SPAM parameter """
        # Parameters are named as in the GST analysis; unlisted parameters take their specified initial guesses
        parameter_values = {'shared:SPAM_error_probability' : 0.0025, 'Gxpi8:0.amplitude_noise_strength' : self.amplitude_noise_strength}
        # Returns (Fisher information dictionaries, Fisher information matrices), each keyed by the circuit's expanded gate sequence
        FI_dicts, FI_matrices = self.gst_circuit_planner.compute_design_fisher_information(self.parsed_circuits, parameter_values)

        # Matrices use the same parameter order as the GST analysis
        self.assertEqual(self.GST_analyzer.parameter_names, self.model_parameters.parameter_names)

        # Independent reference for the deepest circuit: I = N sum_k (dp_k)(dp_k)^T / p_k with derivatives from Richardson-extrapolated
        # central differences of directly computed probabilities (prep and POVM models share the SPAM parameter)
        last_circuit = self.parsed_circuits[-1]
        gate_model = self.gate_models['Gxpi8:0']
        def probabilities(x):
            rho = self.prep_state_model(x[0])
            effects = self.POVM_models(x[0])
            circuit_map = np.linalg.matrix_power(gate_model(x[1]), len(last_circuit.expanded_gates))
            return np.real(np.array([effects[label] @ circuit_map @ rho for label in ['0', '1']]))
        x0 = np.array([0.0025, self.amplitude_noise_strength])
        central = lambda h: np.array([(probabilities(x0 + h*e) - probabilities(x0 - h*e)) / (2*h) for e in np.eye(2)])
        gradients = (4*central(5e-5) - central(1e-4)) / 3.
        FI_reference = last_circuit.total_counts * np.einsum('ik,jk->ij', gradients / probabilities(x0), gradients)

        FI_last_circuit = FI_matrices[tuple(last_circuit.expanded_gates)]
        np.testing.assert_allclose(FI_last_circuit, FI_reference, rtol=1e-6)
        amplitude = 'Gxpi8:0.amplitude_noise_strength'
        self.assertAlmostEqual(FI_dicts[tuple(last_circuit.expanded_gates)][(amplitude, amplitude)], FI_reference[1, 1], delta=1e-6*FI_reference[1, 1])

        # The empty circuit informs only the SPAM parameter
        FI_empty = FI_matrices[()]
        self.assertGreater(FI_empty[0, 0], 0.)
        np.testing.assert_array_equal(FI_empty[1, :], 0.)

        # Accumulated over the design, the Fisher information is symmetric and positive definite (both parameters determined)
        eigenvalues, inverse_FI = self.gst_circuit_planner.compute_design_eigenvalues_and_inverse_fisher_matrix(FI_matrices)
        final = tuple(last_circuit.expanded_gates)
        self.assertGreater(eigenvalues[final][0], 0.)
        np.testing.assert_allclose(inverse_FI[final] @ sum(FI_matrices.values()), np.eye(2), atol=1e-8)

        # Including the Hessian term changes nothing for a complete POVM (sum_k p_k = 1)
        _, FI_with_hessian = self.gst_circuit_planner.compute_circuit_fisher_information(last_circuit, parameter_values, include_hessian=True)
        np.testing.assert_allclose(FI_with_hessian, FI_last_circuit, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
