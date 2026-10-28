import unittest

import numpy as np

from ionsim.process import Gate, Circuit
from ionsim.named_operators import Unitary
from ionsim.circuit_builder import CircuitBuilder
from ionsim.degree_of_freedom import AtomicStructure
from ionsim.basis import StandardBasis


class TestCircuitBuilder(unittest.TestCase):

    def setUp(self):
        self.spin_a = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.spin_b = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.spin_c = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2','P1/2'], level_names=['S1/2,0,0', 'S1/2,1,0','P1/2,1,-1'])
        self.dofs = [self.spin_a, self.spin_b]
        self.basis = StandardBasis(self.dofs)

    def _iter_qubits(self, count):
        if count == 1:
            return [[0], [1]]
        elif count == 2:
            return [[0, 1], [1, 0]]
        else:
            raise ValueError(f"Unsupported count {count}")

    def test_gates(self):
        """Test the gates produced by the circuit builder match their Unitaries."""

        params = {'phi': 0, 'theta': np.pi/2}

        cases = [
            ('I', [0], {}, Gate.from_unitary(self.basis, Unitary.I, [self.dofs[0]])),
            ('X', [0], {}, Gate.from_unitary(self.basis, Unitary.X, [self.dofs[0]])),
            ('sqrtX', [0], {}, Gate.from_unitary(self.basis, Unitary.sqrtX, [self.dofs[0]])),
            ('Y', [0], {}, Gate.from_unitary(self.basis, Unitary.Y, [self.dofs[0]])),
            ('Z', [0], {}, Gate.from_unitary(self.basis, Unitary.Z, [self.dofs[0]])),
            ('H', [0], {}, Gate.from_unitary(self.basis, Unitary.H, [self.dofs[0]])),
            ('R', [0], params, Gate.from_unitary_function(self.basis, Unitary.R, params, [self.dofs[0]])),
            ('CNOT', [0, 1], {}, Gate.from_unitary(self.basis, Unitary.CNOT, [self.dofs[0], self.dofs[1]])),
            ('MS', [0, 1], params, Gate.from_unitary_function(self.basis, Unitary.MS, params, [self.dofs[0], self.dofs[1]])),
        ]

        for name, qubits, params, exp_gate in cases:
            with self.subTest(gate=name):
                builder = CircuitBuilder()
                builder.set_basis(self.basis)
                getattr(builder, name)(qubits, **params)
                circuit = builder.build()
                act_gate = circuit.gates[0]
                np.testing.assert_almost_equal(exp_gate.process_matrix, act_gate.process_matrix)
