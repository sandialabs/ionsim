from __future__ import annotations  # Unknown types will not be errors in type declarations

from ionsim.ionsim_error import IonSimError

try:
    import jaqalpaq
    HAS_JAQALPAQ = True
except ImportError:
    HAS_JAQALPAQ = False


if HAS_JAQALPAQ:
    from jaqalpaq.core import Circuit, NamedQubit
    from jaqalpaq.core.algorithm.visitor import Visitor
    from jaqalpaq.core.algorithm import expand_macros, expand_subcircuits, fill_in_let
else:
    class Visitor:
        pass


class JaqalpaqCircuitBuilder:
    """Mix-in to CircuitBuilder to add Jaqalpaq functionality."""

    @classmethod
    def from_jaqalpaq(cls, circ: Circuit, linearize_parallel_blocks=False, ignore_spam=False):
        """Lazily copy gates from a Qiskit circuit into this circuit builder.

        Args:
            circ (Circuit): The circuit object to copy gates from
        """

        if not HAS_JAQALPAQ:
            raise IonSimError("Jaqalpaq must be installed to read Jaqal circuits")

        gates = linearize_gates(circ, linearize_parallel_blocks=linearize_parallel_blocks,
                                ignore_spam=ignore_spam)
        return cls(gates)


def linearize_gates(circuit, *, linearize_parallel_blocks, ignore_spam):
    circuit = expand_macros(circuit)
    circuit = expand_subcircuits(circuit)
    circuit = fill_in_let(circuit)
    visitor = JaqalpaqCircuitIteratorVisitor(linearize_parallel_blocks=linearize_parallel_blocks,
                                             ignore_spam=ignore_spam)
    return visitor.visit(circuit)


class JaqalpaqCircuitIteratorVisitor(Visitor):

    def __init__(self, linearize_parallel_blocks=False, ignore_spam=False):
        super().__init__()
        self._linearize_parallel_blocks = linearize_parallel_blocks
        self._ignore_spam = ignore_spam
        self._register = None

    def visit_Circuit(self, circuit):
        self._register = self._select_register(circuit.registers)
        return self.visit(circuit.body)

    def _select_register(self, registers: dict):
        pass

    def visit_GateStatement(self, gate):
        """Convert a Jaqal gate into a (name, qubits, params) tuple"""

        if self._ignore_spam and gate.name in ('prepare_all', 'measure_all'):
            return []

        info = jaqalpaq_gate_table.get(gate.name)
        if info is not None:
            ionsim_name, param_map = info
        else:
            ionsim_name = gate.name
            param_map = None

        # While the internal representation can track names for the
        # parameters, it requires definitions for those gates which
        # the user may not have, so working by position is more
        # reliable.
        qubits = []
        parameters = {} if param_map is not None else []
        index = 0
        for param in gate.parameters_linear:
            if isinstance(param, NamedQubit) or param is all:
                qubits.extend(self._visit_qubit(param))
            else:
                if param_map is not None:
                    parameters[param_map[index]] = param
                else:
                    parameters.append(param)
                index += 1

        return [(ionsim_name, qubits, parameters)]

    def _visit_qubit(self, qubit):
        if qubit is all:
            return list(range(int(self._register.size)))
        _, index = qubit.resolve_qubit()
        return [index]

    def visit_BlockStatement(self, block):
        if block.parallel and not self._linearize_parallel_blocks:
            # This isn't intrinsically limiting, but solving it
            # requires figuring out how the user wants crosstalk
            # represented.
            raise IonSimError("Parallel blocks in Jaqal must be linearized")
        gates = []
        for gate in block.statements:
            gates.extend(self.visit(gate))
        return gates

    def visit_LoopStatement(self, loop):
        gates = []
        for _ in range(loop.iterations):
            gates.extend(self.visit(loop.statements))
        return gates


if HAS_JAQALPAQ:
    # Since Jaqalpaq has no native gate set, these are drawn from the
    # qscout native gates package.
    jaqalpaq_gate_table = {
        'R': ('R', ('phi', 'theta')),
        'Px': ('X', ()),
        'Py': ('Y', ()),
        'Pz': ('Z', ()),
        'Sx': ('sqrtX', ()),
        'MS': ('MS', ('phi', 'theta')),
        # Instead of one idle gate, Jaqal has an idle matched to the
        # length of each other gate.
        'I_R': ('I', ()),
        'I_Px': ('I', ()),
        'I_Py': ('I', ()),
        'I_Pz': ('I', ()),
        'I_Sx': ('I', ()),
        'I_MS': ('I', ()),
        # Only native ion gates are defined, so no H, CNOT, and CZ,
        # among others.
    }
