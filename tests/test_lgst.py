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
    
        self.true_POVM_effects = {} 
        self.true_POVM_effects['0'] = EnergyShiftOperator.from_matrix(self.basis, self.POVM_models(SPAM_error_prob)['0'].reshape(2,2))
        self.true_POVM_effects['1'] = EnergyShiftOperator.from_matrix(self.basis, self.POVM_models(SPAM_error_prob)['1'].reshape(2,2))

        self.true_gate_set = {}
        self.true_gate_set['prep'] = self.rho_0 
        self.true_gate_set['POVM'] = self.true_POVM_effects 
        self.true_gate_set['Gxpi2:0'] =  X_pi_2_co_prop_simple(amplitude_noise_strength)
        self.true_gate_set['Gypi2:0'] =  Y_pi_2_co_prop_simple(amplitude_noise_strength)

        self.GST_analyzer = GateSetTomography(self.basis, self.prep_state_model, self.POVM_models, self.parsed_circuits, self.gate_models, 
                                    circuit_design = self.gst_circuit_planner, ideal_gate_set = self.true_gate_set, verbose = False)

        ## Parameter information: initial guesses, bounds, and sharing among models.
        # model = "shared" ties together every model with an argument of that name:
        #   SPAM_error_probability -> prep & POVM models;  amplitude_noise_strength -> Gxpi2:0 & Gypi2:0 models
        self.GST_analyzer.specify_parameter("SPAM_error_probability", model = "shared", guess = 1e-4, bounds = (0., 1.))
        self.GST_analyzer.specify_parameter("amplitude_noise_strength", model = "shared", guess = 0.5, bounds = (0.0001, 10.0))


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

    def test_linear_gst_analysis(self):
        """ Test linear GST (LGST) """ 
        solver_results = self.GST_analyzer.solve_for_gate_parameters('linear') 
        gate_set_error = self.GST_analyzer.compute_gate_set_error_by_element(solver_results, self.true_gate_set)
        X_pi2_error = gate_set_error['Gxpi2:0']
        Y_pi2_error = gate_set_error['Gypi2:0']
        SPAM_error = gate_set_error["prep"]
        SPAM_error += gate_set_error["POVM"]
        self.assertAlmostEqual(X_pi2_error, 0.0004924175661171493, places=5)
        self.assertAlmostEqual(Y_pi2_error, 0.0004924175661169726, places=5)
        self.assertAlmostEqual(SPAM_error, 2.1712445132231874e-05, places=5)

    def test_mle_gst_analysis(self):
        """ Test GST via maximum likelihood estimation (MLE)""" 
        solver_results = self.GST_analyzer.solve_for_gate_parameters('MLE') 
        gate_set_error = self.GST_analyzer.compute_gate_set_error_by_element(solver_results.x, self.true_gate_set)

        X_pi2_error = gate_set_error['Gxpi2:0']
        Y_pi2_error = gate_set_error['Gypi2:0']
        SPAM_error = gate_set_error["prep"]
        SPAM_error += gate_set_error["POVM"]
        self.assertAlmostEqual(X_pi2_error, 0.0006672864884890055, places=5)
        self.assertAlmostEqual(Y_pi2_error, 0.0006672864884889269, places=5)
        self.assertAlmostEqual(SPAM_error, 0.0010575712223980156, places=5)

    def test_shared_parameter_sensitivity(self):
        """ Sensitivity to a parameter shared by two gates equals the sum of the sensitivities to independent copies """
        circuit = [c for c in self.parsed_circuits if {'Gxpi2:0', 'Gypi2:0'} <= set(c.expanded_gate_labels)][-1]

        shared = GstModelParameters(self.prep_state_model, self.POVM_models, self.gate_models)
        shared.specify_parameter("amplitude_noise_strength", model = "shared")
        independent = GstModelParameters(self.prep_state_model, self.POVM_models, self.gate_models)

        self.gst_circuit_planner.parameters = shared
        S = self.gst_circuit_planner.compute_circuit_sensitivity(circuit, {'shared:amplitude_noise_strength': 0.125,
                            'prep.SPAM_error_probability': 0.0025, 'POVM.SPAM_error_probability': 0.0025})
        self.gst_circuit_planner.parameters = independent
        I = self.gst_circuit_planner.compute_circuit_sensitivity(circuit, {'Gxpi2:0.amplitude_noise_strength': 0.125,
                            'Gypi2:0.amplitude_noise_strength': 0.125, 'prep.SPAM_error_probability': 0.0025, 'POVM.SPAM_error_probability': 0.0025})

        self.assertIn('shared:amplitude_noise_strength', S)
        for outcome in ['0', '1']:
            self.assertAlmostEqual(S['shared:amplitude_noise_strength'][outcome],
                I['Gxpi2:0.amplitude_noise_strength'][outcome] + I['Gypi2:0.amplitude_noise_strength'][outcome], places=7)
            self.assertAlmostEqual(S['prep.SPAM_error_probability'][outcome], I['prep.SPAM_error_probability'][outcome], places=7)

if __name__ == '__main__':
    unittest.main()
