import numpy as np

from ionsim.named_operators import Unitary
from ionsim.process import Gate

try:
    import qiskit
    HAS_QISKIT = True
    QiskitCircuit = qiskit.QuantumCircuit
except ImportError:
    HAS_QISKIT = False


class QiskitCircuitBuilder:
    """Mix-in to CircuitBuilder to add Qiskit functionality."""

    @classmethod
    def from_qiskit(cls, qc: QiskitCircuit, gate_map: dict[str, str] | None = None):
        gates = []
        for ci in qc.data:
            name = ci.name
            info = qiskit_unitary_table.get(name)
            if info is None:
                raise IonSimError(f"Could not look up unitary for Qiskit gate {name}")
            ionsim_name, unitary, args = info
            qubits = [q._index for q in ci.qubits]
            params = {name: ci.params[i] for i, name in enumerate(args)}
            if isinstance(unitary, np.ndarray):
                lazy_gate = QiskitGateFromUnitary(unitary, Gate.from_unitary)
            elif callable(unitary):
                lazy_gate = QiskitGateFromUnitaryFunction(
                    unitary, Gate.from_unitary_function)
            else:
                raise IonSimError(f"Unknown unitary type {unitary}")
            gates.append((lazy_gate, ionsim_name, qubits, params))
        return cls(gates)


class QiskitGateFromUnitary:

    def __init__(self, unitary, func):
        self.unitary = unitary
        self.func = func

    def __call__(self, qubits, params, basis, noise):
        target_dofs = [basis.degrees_of_freedom[i] for i in qubits]
        return self.func(basis, self.unitary, target_dofs)


class QiskitGateFromUnitaryFunction:

    def __init__(self, unitary, func):
        self.unitary = unitary
        self.func = func

    def __call__(self, qubits, params, basis, noise):
        dofs = [basis.degrees_of_freedom[i] for i in qubits]
        return self.func(basis, self.unitary, params, dofs, noise)


if HAS_QISKIT:
    # TODO: Fill out this table
    # MS in an example
    # Hadamard
    qiskit_unitary_table = {
        'id': ('I', Unitary.I, ()),
        'x': ('X', Unitary.X, ()),
        'y': ('Y', Unitary.Y, ()),
        'z': ('Z', Unitary.Z, ()),
        'sx': ('sqrtX', Unitary.sqrtX, ()),
        'r': ('R', Unitary.R, ('theta', 'phi')),
        'cx': ('CNOT', Unitary.CNOT, ()),
    }
