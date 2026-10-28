from __future__ import annotations  # Unknown types will not be errors in type declarations

from typing import Callable

import numpy as np

from ionsim.process import Circuit, Gate
from ionsim.named_operators import Unitary, Pauli
from ionsim.basis import StandardBasis, Basis
from ionsim.custom_types import Matrix


try:
    import qiskit
    HAS_QISKIT = True
    QiskitCircuit = qiskit.QuantumCircuit
except ImportError:
    HAS_QISKIT = False


class CircuitTranspiler:

    @classmethod
    def from_qiskit(cls, qc: QiskitCircuit, gate_map: dict[str, str] | None = None):
        gates = []
        for ci in qc.data:
            name = ci.name
            info = qiskit_unitary_table.get(name)
            if info is None:
                raise ValueError(f"Could not look up unitary for Qiskit gate {name}")
            ionsim_name, unitary, args = info
            params = {name: ci.params[i] for i, name in enumerate(args)}
            if isinstance(unitary, np.ndarray):
                lazy_gate = QiskitGateFromUnitary(unitary, ci.qubits, Gate.from_unitary)
            elif callable(unitary):
                lazy_gate = QiskitGateFromUnitaryFunction(
                    unitary, ci.qubits, Gate.from_unitary_function)
            else:
                raise TypeError(f"Unknown unitary type {unitary}")
            gates.append((lazy_gate, ionsim_name, params))
        return cls(gates)

    def __init__(self, lazy_gates: list[tuple[Callable, str, dict]]):
        """Create a circuit from gate drawn from some external
        source. Call one of the from_* class methods instead of the
        constructor directly."""
        # A list of the gates to be converted into a Circuit after the
        # user has finished specifying basis, degrees of freedom,
        # noise, and any overriding conversion functions.
        self._gates = lazy_gates

        self._basis = None
        self._dofs = None
        self._circuit_noise = None

        # A lookup table for functions producing Gate objects provided
        # by the user. These override the default which is to create a
        # Gate from a unitary using standard IonSim methods.
        self._gate_functions = {}

        # The noise to apply to a particular gate, if any.
        self._gate_noise = {}

    def transpile(self) -> Circuit:
        """Transpile from the input to an ionsim.Circuit using the
        basis, degrees of freedom, and noise set in this transpiler
        object."""
        gates = [
            self._gate_functions.get(name, func)(
                params, self._get_basis(), self._get_dofs(), self._gate_noise.get(name))
            for func, name, params in self._gates
        ]
        return Circuit.from_gates(gates, self._circuit_noise)

    def set_gate_function(self, gate: str, func: Callable):
        """Set a function to create an ionsim Gate object from
        parameters. The function must take a parameter dictionary,
        basis object, degrees of freedom, and noise object, and then
        return a Circuit.
        """
        self._gate_functions[gate] = func

    def set_basis(self, basis):
        """Create a basis from the argument. If the argument is an
        IonSim basis type, use it directly, otherwise attempt to
        create a StandardBasis from a list of degrees of freedom."""
        if isinstance(basis, Basis):
            self._basis = basis
        elif isinstance(basis, (list, tuple)):
            self._basis = StandardBasis(basis)
        else:
            raise ValueError(f"Unsupported basis {basis}")

    def set_dofs(self, dofs):
        """Set the degrees of freedom corresponding to the input qubits."""
        self._dofs = dofs

    def set_circuit_noise(self, noise):
        """Set the noise for the circuit as a whole"""
        self._circuit_noise = noise

    def set_gate_noise(self, gate: str, noise):
        """Set noise for a specific gate."""
        self._gate_noise[gate] = noise

    def _get_basis(self):
        if self._basis is not None:
            return self._basis
        if self._dofs is not None:
            return StandardBasis(self._dofs)
        raise RuntimeError("No basis or degrees of freedom were given")

    def _get_dofs(self):
        if self._dofs is not None:
            return self._dofs
        if self._basis is not None:
            return self._basis.degrees_of_freedom
        raise RuntimeError("No basis or degrees of freedom were given")


class QiskitGateFromUnitary:

    def __init__(self, unitary, qubits, func):
        self.unitary = unitary
        self.qubits = [q._index for q in qubits]
        self.func = func

    def __call__(self, params, basis, dofs, noise):
        dofs = [dofs[i] for i in self.qubits]
        return self.func(basis, self.unitary, dofs)


class QiskitGateFromUnitaryFunction:

    def __init__(self, unitary, qubits, func):
        self.unitary = unitary
        self.qubits = [q._index for q in qubits]
        self.func = func

    def __call__(self, params, basis, dofs, noise):
        dofs = [dofs[i] for i in self.qubits]
        return self.func(basis, self.unitary, params, dofs, noise)


if HAS_QISKIT:
    # TODO: Fill out this table
    qiskit_unitary_table = {
        'id': ('I', Unitary.I, ()),
        'x': ('X', Unitary.X, ()),
        'y': ('Y', Unitary.Y, ()),
        'z': ('Z', Unitary.Z, ()),
        'sx': ('sqrtX', Unitary.sqrtX, ()),
        'r': ('R', Unitary.R, ('theta', 'phi')),
        'cx': ('CNOT', Unitary.CNOT, ()),
    }
