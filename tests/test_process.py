import unittest

import numpy as np

from ionsim.process import Gate, Circuit
from ionsim.degree_of_freedom import AtomicSpin
from ionsim.basis import StandardBasis
from ionsim.named_operators import Unitary
from ionsim.noise import Noise

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
        extra_noisy_gate = Gate.from_process_matrix_function(
            self.basis, self.noisy_phi_gate.process_matrix_function, {'phi': 0, 'theta': np.pi/2}, [self.spin_a], self.theta_noise,
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

if __name__ == '__main__':
    unittest.main()
