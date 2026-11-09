import numpy as np

from ionsim.named_operators import Unitary
from ionsim.process import Gate
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
    def from_qiskit(cls, qc: QiskitCircuit):
        """Lazily copy gates from a Qiskit circuit into this circuit builder.

        Args:
            qc (QiskitCircuit): The circuit object to copy gates from
        """
        gates = []
        for ci in qc.data:
            # See if we have a default mapping from a qiskit gate
            # to an Ionsim unitary
            info = qiskit_gate_table.get(ci.name)
            if info is None:
                # The user must totally define this gate with
                # set_gate_function. The circuit will fail to
                # build otherwise.
                ionsim_name = ci.name
                args = ()
                # The user will have to unpack these either with an alias or gate function
                params = list(ci.params) if ci.params is not None else []
            else:
                ionsim_name, args = info
                params = {param_name: ci.params[i] for i, param_name in enumerate(args)}

            qubits = [q._index for q in ci.qubits]
            gates.append((ionsim_name, qubits, params))
        return cls(gates)


if HAS_QISKIT:
    qiskit_gate_table = {
        'id': ('I', ()),
        'x': ('X', ()),
        'y': ('Y', ()),
        'z': ('Z', ()),
        'sx': ('sqrtX', ()),
        'r': ('R', ('theta', 'phi')),
        'cx': ('CNOT', ()),
        'cz': ('CZ', ()),
        'h': ('H', ()),
    }
