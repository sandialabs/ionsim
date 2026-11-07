import unittest

import numpy as np

from ionsim.ionsim_error import IonSimError
from ionsim.circuit_builder import CircuitBuilder
from ionsim.jaqalpaq_interface import HAS_JAQALPAQ
from ionsim.degree_of_freedom import AtomicStructure
from ionsim.basis import StandardBasis
from ionsim.named_operators import Unitary, Pauli
from ionsim.process import Gate

if HAS_JAQALPAQ:
    from jaqalpaq.qsyntax import circuit
    from jaqalpaq.parser import parse_jaqal_string


@unittest.skipIf(not HAS_JAQALPAQ, "Jaqalpaq not available")
class JaqalpaqIntegrationTester(unittest.TestCase):

    def setUp(self):
        self.spin_a = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.spin_b = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.basis = StandardBasis([self.spin_a, self.spin_b])

    def test_simple_circuit(self):
        """Test creating a simple circuit with Jaqalpaq."""
        @circuit
        def jaqal(Q):
            q = Q.register(2)
            Q.Px(q[0])

        builder = CircuitBuilder.from_jaqalpaq(jaqal(), ignore_spam=True)
        builder.set_basis(self.basis)
        circ = builder.build()
        exp_gate = Gate.from_unitary(self.basis, Unitary.X, [self.spin_a])
        act_gate = circ.gates[0]

        np.testing.assert_almost_equal(exp_gate.process_matrix, act_gate.process_matrix)

    def test_macro(self):
        """Test a circuit with a macro."""
        # Qsyntax does not allow macros
        text = """
        register q[2]

        macro foo a {
          Px a
          Sx a
        }

        foo q[0]
        """
        # Do not attempt to load pulse definitions when parsing
        jaqal = parse_jaqal_string(text, autoload_pulses=False)
        builder = CircuitBuilder.from_jaqalpaq(jaqal, ignore_spam=True)
        builder.set_basis(self.basis)
        circ = builder.build()

        exp_gate = Gate.from_unitary(self.basis, Unitary.X, [self.spin_a])
        act_gate = circ.gates[0]
        np.testing.assert_almost_equal(exp_gate.process_matrix, act_gate.process_matrix)

        exp_gate = Gate.from_unitary(self.basis, Unitary.sqrtX, [self.spin_a])
        act_gate = circ.gates[1]
        np.testing.assert_almost_equal(exp_gate.process_matrix, act_gate.process_matrix)

    def test_let_statement(self):
        """Test a circuit with a let statement."""
        @circuit
        def jaqal(Q):
            q = Q.register(2)
            idx = Q.let(0)
            Q.Px(q[idx])

        builder = CircuitBuilder.from_jaqalpaq(jaqal(), ignore_spam=True)
        builder.set_basis(self.basis)
        circ = builder.build()
        exp_gate = Gate.from_unitary(self.basis, Unitary.X, [self.spin_a])
        act_gate = circ.gates[0]

        np.testing.assert_almost_equal(exp_gate.process_matrix, act_gate.process_matrix)

    def test_map_statement(self):
        """Test qubits that are remapped."""
        @circuit
        def jaqal(Q):
            q = Q.register(2)
            idx = Q.let(0)
            Q.Px(q[idx])

        builder = CircuitBuilder.from_jaqalpaq(jaqal(), ignore_spam=True)
        builder.set_basis(self.basis)
        circ = builder.build()
        exp_gate = Gate.from_unitary(self.basis, Unitary.X, [self.spin_a])
        act_gate = circ.gates[0]

        np.testing.assert_almost_equal(exp_gate.process_matrix, act_gate.process_matrix)

    def test_gate_with_parameters(self):
        """Test a gate with parameters."""
        @circuit
        def jaqal(Q):
            q = Q.register(2)
            Q.R(q[0], 0, np.pi/2)

        builder = CircuitBuilder.from_jaqalpaq(jaqal(), ignore_spam=True)
        builder.set_basis(self.basis)
        circ = builder.build()
        params = {
            'phi': 0,
            'theta': np.pi/2,
        }
        exp_gate = Gate.from_unitary_function(self.basis, Unitary.R, params, [self.spin_a])
        act_gate = circ.gates[0]

        np.testing.assert_almost_equal(exp_gate.process_matrix, act_gate.process_matrix)

    def test_subcircuit_block(self):
        """Test a circuit with subcircuits."""
        @circuit
        def jaqal(Q):
            with Q.subcircuit:
                q = Q.register(2)
                Q.Px(q[0])

        builder = CircuitBuilder.from_jaqalpaq(jaqal(), ignore_spam=True)
        builder.set_basis(self.basis)
        circ = builder.build()
        exp_gate = Gate.from_unitary(self.basis, Unitary.X, [self.spin_a])
        act_gate = circ.gates[0]

        np.testing.assert_almost_equal(exp_gate.process_matrix, act_gate.process_matrix)

    def test_unknown_gate(self):
        """Test a circuit with a gate without a standard Ionsim mapping."""
        @circuit
        def jaqal(Q):
            with Q.subcircuit:
                q = Q.register(2)
                Q.Unknown(q[0])

        def unknown(qubits, params, basis, noise):
            unitary = np.eye(2**len(qubits))
            target_dofs = [basis.degrees_of_freedom[i] for i in qubits]
            return Gate.from_unitary(basis, unitary, target_dofs)

        builder = CircuitBuilder.from_jaqalpaq(jaqal(), ignore_spam=True)
        builder.set_basis(self.basis)
        builder.set_gate_function('Unknown', unknown)
        circ = builder.build()
        exp_gate = Gate.from_unitary(self.basis, Unitary.I, [self.spin_a])
        act_gate = circ.gates[0]

        np.testing.assert_almost_equal(exp_gate.process_matrix, act_gate.process_matrix)

    def test_loop(self):
        """Test that loops are unrolled."""
        @circuit
        def jaqal(Q):
            q = Q.register(2)
            with Q.loop(4):
                Q.Px(q[0])

        builder = CircuitBuilder.from_jaqalpaq(jaqal(), ignore_spam=True)
        builder.set_basis(self.basis)
        circ = builder.build()
        exp_gate = Gate.from_unitary(self.basis, Unitary.X, [self.spin_a])
        act_gate = circ.gates[0]

        np.testing.assert_almost_equal(exp_gate.process_matrix, act_gate.process_matrix)
        self.assertEqual(4, len(circ.gates))
        

    def test_linearize_parallel_block(self):
        """Test linearizing the different paths in a parallel block."""
        @circuit
        def jaqal(Q):
            q = Q.register(2)
            with Q.parallel:
                Q.Px(q[0])
                Q.Px(q[1])

        # Make sure we fail if we don't linearize the parallel blocks
        with self.assertRaises(IonSimError):
            builder = CircuitBuilder.from_jaqalpaq(jaqal(), ignore_spam=True)

        builder = CircuitBuilder.from_jaqalpaq(jaqal(), ignore_spam=True, linearize_parallel_blocks=True)
        builder.set_basis(self.basis)
        circ = builder.build()
        exp_gate = Gate.from_unitary(self.basis, Unitary.X, [self.spin_a])
        act_gate = circ.gates[0]

        np.testing.assert_almost_equal(exp_gate.process_matrix, act_gate.process_matrix)
        self.assertEqual(2, len(circ.gates))
