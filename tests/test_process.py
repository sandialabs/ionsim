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
        # TODO: This currently works without circuit noise only; we need to fix this to work with Noise objects 
        noisy_R_gate = Gate.from_unitary_function(self.basis, Unitary.R, {'phi': 0, 'theta': np.pi/2}, [self.spin_a], self.phi_noise)

        ramsey_circuit = Circuit.from_gates([noisy_R_gate, noisy_R_gate])

        #ramsey_circuit = Circuit.from_gates([noisy_R_gate, noisy_R_gate], self.theta_noise) # functions but not accurate  
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


if __name__ == '__main__':
    unittest.main()
