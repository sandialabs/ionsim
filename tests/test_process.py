import unittest

import numpy as np

from ionsim.process import Gate, Circuit
from ionsim.degree_of_freedom import AtomicSpin
from ionsim.basis import StandardBasis
from ionsim.named_operators import Unitary, Pauli
from ionsim.noise import Noise
from ionsim.operator import EnergyShiftOperator
from ionsim.state import State 

class TestProcess(unittest.TestCase):

    def setUp(self):
        """Set up the necessary objects for testing."""
        self.spin_a = AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.spin_b = AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
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
        #noisy_R_gate = Gate.from_unitary_function(self.basis, Unitary.R, {'phi': 0, 'theta': np.pi/2}, [self.spin_a], None)
        noisy_R_gate = Gate.from_unitary_function(self.basis, Unitary.R, {'phi': 0, 'theta': np.pi/2}, [self.spin_a], self.phi_noise)

        # TODO: This currently works without circuit noise only; we need to fix this to work with Noise objects 
        ramsey_circuit = Circuit.from_gates([noisy_R_gate, noisy_R_gate])
        #ramsey_circuit = Circuit.from_gates([noisy_R_gate, noisy_R_gate], self.theta_noise) # functions but not accurate  

        ## Fixed a bug where a noisy process matrix function would not work with kwargs 
        # test pm function with noise and kwargs 
        #tmp_pm_fxn = noisy_R_gate.process_matrix_function
        #params = {'phi' : 0., 'theta' : np.pi/2.} 

        circuit_pm_function = ramsey_circuit.process_matrix_function 
        #print(circuit_pm_function(**circuit_parameters))

        # Test outcome probability function  
        outcome_operator = EnergyShiftOperator.from_matrix(self.basis, np.kron(Pauli.projector_1, Pauli.projector_0)) 
        initial_state = State.from_coefficients(self.basis, [1., 0., 0., 0.]) 

        prob_function = ramsey_circuit.build_outcome_probability_function(initial_state, outcome_operator)
        #import inspect
        #sig = inspect.signature(prob_function)
        #parameter_names = list(sig.parameters.keys())  
        #print(f"Parameters: {parameter_names}")
        #print(f"Parameter dict: {sig.parameters}")
        circuit_parameters = {'R__phi' : 0., 'R__theta' : np.pi/2}
        outcome_prob = prob_function(**circuit_parameters)
        #print(outcome_prob)
        #self.assertAlmostEqual(outcome_prob, 0.9530090510307307, places = 10)

        # Compute outcome probability using probability function: 
        #prob_gradient_wrt_theta = prob_function.gradient("R__theta") 
        #prob_gradient_wrt_theta = circuit_pm_function.gradient(prob_function, ["R__theta", "R__phi"], R__theta = np.pi/2., R__phi = 0.) 

        #prob, prob_gradients = circuit_pm_function.gradient(prob_function, wrt = ["R__phi", "R__theta"], **circuit_parameters) 
        #print(prob)
        #print(prob_gradients)
        



if __name__ == '__main__':
    unittest.main()
