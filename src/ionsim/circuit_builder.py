from __future__ import annotations  # Unknown types will not be errors in type declarations

import numpy as np

from ionsim.process import Circuit, Gate
from ionsim.named_operators import Unitary, Pauli
from ionsim.basis import StandardBasis, PauliProductBasis
from ionsim.custom_types import Matrix


try:
    import qiskit
    HAS_QISKIT = True
    QiskitCircuit = qiskit.QuantumCircuit
except ImportError:
    HAS_QISKIT = False


class CircuitBuilder:

    @classmethod
    def from_qiskit(cls, qc: QiskitCircuit):
        gates = []
        for ci in qc.data:
            name = ci.name
            pair = qiskit_unitary_table.get(name)
            if pair is None:
                raise ValueError(f"Could not look up unitary for Qiskit gate {name}")
            ionsim_name, unitary = pair
            if name == 'r':
                # TODO: Don't hard code
                params = {'phi': ci.params[1], 'theta': ci.params[0]}
            else:
                params = {}
            if isinstance(unitary, np.ndarray):
                lazy_gate = QiskitGateFromUnitary(ionsim_name, unitary, ci.qubits, Gate.from_unitary)
            elif callable(unitary):
                lazy_gate = QiskitGateFromUnitaryFunction(
                    ionsim_name, unitary, params, ci.qubits, Gate.from_unitary_function)
            else:
                raise TypeError(f"Unknown unitary type {unitary}")
            gates.append(lazy_gate)
        return cls(gates)

    def __init__(self, lazy_gates: list[tuple[str, Matrix, dict, list[int]]]):
        """Create a circuit from gate drawn from some external
        source. Call one of the from_* class methods instead of the
        constructor directly."""
        self._gates = lazy_gates
        self._basis = None
        self._dofs = None
        self._noise = None
        self._gate_noise = {}

    def build(self, noise=None) -> Circuit:
        noise = noise if noise is not None else self._noise
        gates = [
            gate(self._get_basis(), self._get_dofs(), self._gate_noise.get(gate.name))
            for gate in self._gates
        ]
        return Circuit.from_gates(gates, noise)

    def set_basis(self, basis):
        """Create a basis from the argument. If the argument is an
        IonSim basis type, use it directly, otherwise attempt to
        create a StandardBasis from a list of degrees of freedom."""
        if isinstance(basis, (StandardBasis, PauliProductBasis)):
            self._basis = basis
        elif isinstance(basis, (list, tuple)):
            self._basis = StandardBasis(basis)
        else:
            # TODO: Support other Ionsim bases
            raise ValueError(f"Invalid basis {basis}")

    def set_dofs(self, dofs):
        """Set the degrees of freedom corresponding to the input qubits."""
        self._dofs = dofs

    def set_noise(self, noise):
        """Set some sort of overall noise model"""
        # TODO: better docstring
        self._noise = noise

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

    def __init__(self, name, unitary, qubits, func):
        self.name = name
        self.unitary = unitary
        self.qubits = [q._index for q in qubits]
        self.func = func

    def __call__(self, basis, dofs, noise):
        dofs = [dofs[i] for i in self.qubits]
        return self.func(basis, self.unitary, dofs)


class QiskitGateFromUnitaryFunction:

    def __init__(self, name, unitary, params, qubits, func):
        self.name = name
        self.unitary = unitary
        self.params = params
        self.qubits = [q._index for q in qubits]
        self.func = func

    def __call__(self, basis, dofs, noise):
        dofs = [dofs[i] for i in self.qubits]
        print(self.params)
        return self.func(basis, self.unitary, self.params, dofs, noise)


if HAS_QISKIT:
    # TODO: Fill out this table
    qiskit_unitary_table = {
        'sx': ('sqrtX', Unitary.sqrtX),
        'r': ('R', Unitary.R),
    }
