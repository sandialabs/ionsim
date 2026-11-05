import numpy as np

from ionsim.named_operators import Unitary
from ionsim.process import Gate
from ionsim._circuit_builder_gates import GateFromUnitary, GateFromUnitaryFunction
from ionsim.ionsim_error import IonSimError

try:
    import qiskit
    HAS_QISKIT = True
    QiskitCircuit = qiskit.QuantumCircuit
except ImportError:
    HAS_QISKIT = False


class QiskitCircuitBuilder:
    """Mix-in to CircuitBuilder to add Qiskit functionality."""

    @classmethod
    def from_qiskit(cls, qc: QiskitCircuit, gate_map: dict[str, tuple[str, list]]|None = None):
        """Lazily copy gates from a Qiskit circuit into this circuit builder.

        Args:
            qc (QiskitCircuit): The circuit object to copy gates from
            gate_map (dict): A mapping of qiskit names to how to
                interpret the gates in ionsim. Each value is a (name,
                param_list) tuple where name is the ionsim name and
                param_list is a list of parameter names for any gate
                parameters in the order they appear in the qiskit gate.
        """
        gate_map = gate_map or {}
        gates = []
        for ci in qc.data:
            name = ci.name
            # Check if the user provides a mapping from this qiskit
            # gate to an Ionsim gate
            remap_info = gate_map.get(name)
            if remap_info is None:
                # See if we have a default mapping from a qiskit gate
                # to an Ionsim unitary
                info = qiskit_unitary_table.get(name)
                if info is None:
                    # The user must totally define this gate with
                    # set_gate_function. The circuit will fail to
                    # build otherwise.
                    ionsim_name = name
                    unitary = None
                    args = ()
                else:
                    ionsim_name, unitary, args = info
            else:
                ionsim_name, args = remap_info
                unitary = getattr(Unitary, ionsim_name, None)
            qubits = [q._index for q in ci.qubits]
            params = {name: ci.params[i] for i, name in enumerate(args)}
            if isinstance(unitary, np.ndarray):
                lazy_gate = GateFromUnitary(unitary)
            elif callable(unitary):
                lazy_gate = GateFromUnitaryFunction(unitary)
            elif unitary is None:
                # The user must provide a definition with set_gate_function
                lazy_gate = None
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
    qiskit_unitary_table = {
        'id': ('I', Unitary.I, ()),
        'x': ('X', Unitary.X, ()),
        'y': ('Y', Unitary.Y, ()),
        'z': ('Z', Unitary.Z, ()),
        'sx': ('sqrtX', Unitary.sqrtX, ()),
        'r': ('R', Unitary.R, ('theta', 'phi')),
        'cx': ('CNOT', Unitary.CNOT, ()),
        'h': ('H', Unitary.H, ()),
    }
