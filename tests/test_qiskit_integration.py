import unittest

import numpy as np

from ionsim.process import Gate, Circuit
from ionsim.degree_of_freedom import AtomicStructure
from ionsim.basis import StandardBasis
from ionsim.named_operators import Unitary, Pauli
from ionsim.noise import Noise
from ionsim.circuit_transpiler import CircuitTranspiler, HAS_QISKIT

if HAS_QISKIT:
    import qiskit


@unittest.skipIf(not HAS_QISKIT, "QISKIT not available")
class QiskitIntegrationTester(unittest.TestCase):

    def setUp(self):
        """Set up the necessary objects for testing."""
        self.spin_a = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.spin_b = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.spin_c = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2','P1/2'], level_names=['S1/2,0,0', 'S1/2,1,0','P1/2,1,-1'])
        self.basis = StandardBasis([self.spin_a, self.spin_b])

        # self.Sx = Gate.from_unitary(self.basis, Unitary.sqrtX, [self.spin_a])

        xs = np.linspace(-np.pi, np.pi, 21)
        self.phi_noise = Noise.from_named_pdf('phi', 'gaussian', {'standard_deviation': np.pi/10}, xs)
        # self.noisy_phi_gate = Gate.from_unitary_function(
        #     self.basis, Unitary.R, {'phi': 0, 'theta': np.pi/2}, [self.spin_a], self.phi_noise,
        # )

        self.theta_noise = Noise.from_named_pdf('theta', 'gaussian', {'standard_deviation': np.pi/10}, xs)
        # self.noisy_theta_gate = Gate.from_unitary_function(
        #     self.basis, Unitary.R, {'phi': 0, 'theta': np.pi/2}, [self.spin_a], self.theta_noise,
        # )

    def test_circuit(self):
        # Create a qiskit circuit
        qc = qiskit.QuantumCircuit(1)
        qc.sx(0)
        qc.r(theta=np.pi/2, phi=0, qubit=0)

        # Convert to ionsim Circuit
        tp = CircuitTranspiler.from_qiskit(qc)
        tp.set_basis(self.basis)
        tp.set_dofs([self.spin_a])
        tp.set_gate_noise('R', self.phi_noise)

        ramsey = tp.transpiler(noise=self.theta_noise)

        # Compute process fidelity
        fidelity = ramsey.compute_process_fidelity(Gate.from_unitary(self.basis, Unitary.X, [self.spin_a]).process_matrix)

        # Compare
        self.assertAlmostEqual(fidelity, 0.9306176541502548, places=14)
