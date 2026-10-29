from __future__ import annotations  # Unknown types will not be errors in type declarations

from typing import Callable

import numpy as np

from ionsim.process import Circuit
from ionsim.named_operators import Unitary, Pauli
from ionsim.basis import StandardBasis, Basis
from ionsim.custom_types import Matrix
from ionsim.ionsim_error import IonSimError
from ionsim.qiskit_interface import QiskitCircuitBuilder
from ionsim._circuit_builder_gates import GateFromUnitary, GateFromUnitaryFunction


class CircuitBuilderBase:

    def __init__(self, lazy_gates: list[tuple[Callable, str, list, dict]] | None = None):
        """Create a circuit from gate drawn from some external
        source. Call one of the from_* class methods instead of the
        constructor directly."""
        # A list of the gates to be converted into a Circuit after the
        # user has finished specifying basis, degrees of freedom,
        # noise, and any overriding conversion functions.
        self._gates = lazy_gates or []

        self._basis = None
        self._circuit_noise = None

        # A lookup table for functions producing Gate objects provided
        # by the user. These override the default which is to create a
        # Gate from a unitary using standard IonSim methods.
        self._gate_functions = {}

        # The noise to apply to a particular gate, if any.
        self._gate_noise = {}

    def build(self, *, empty_ok=False) -> Circuit:
        """Create an ionsim.Circuit using the basis, and noise set in
        this builder."""
        if self._basis is None:
            raise IonSimError("Cannot create a circuit without a basis")
        if not self._gates and not empty_ok:
            raise IonSimError("Creating empty circuit. Use CircuitBuilder.build(empty_ok=True) if you really want this")

        def make_gate(gate_tuple):
            func, name, qubits, params = gate_tuple
            gate_func = self._gate_functions.get(name, func)
            if gate_func is None:
                raise IonSimError(f"Not gate {name} defined and no default available")
            return gate_func(qubits, params, self._basis, self._gate_noise.get(name))

        gates = [make_gate(gate_tuple) for gate_tuple in self._gates]
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
        """Set the degrees of freedom corresponding to the input
        qubits. Only one of basis or degrees of freedom must be set."""
        self._basis = StandardBasis(dofs)

    def set_circuit_noise(self, noise):
        """Set the noise for the circuit as a whole"""
        self._circuit_noise = noise

    def set_gate_noise(self, gate: str, noise):
        """Set noise for a specific gate."""
        self._gate_noise[gate] = noise


class IonSimCircuitBuilder:
    """Mix-in to allow the CircuitBuilder to separate out IonSim gates
    from the rest of the CircuitBuilder machinery."""

    def gate(self, name, qubits, **kwargs):
        obj = getattr(Unitary, name, None)
        if obj is None:
            gate = (None, name, qubits, kwargs)
        else:
            obj = getattr(Unitary, name)
            if callable(obj):
                wrapped = GateFromUnitaryFunction(obj)
            else:
                wrapped = GateFromUnitary(obj)
            gate = (wrapped, name, qubits, kwargs)
        self._gates.append(gate)

    def I(self, qubits):
        self.gate('I', qubits)

    def X(self, qubits):
        self.gate('X', qubits)

    def sqrtX(self, qubits):
        self.gate('sqrtX', qubits)

    def Y(self, qubits):
        self.gate('Y', qubits)

    def Z(self, qubits):
        self.gate('Z', qubits)

    def H(self, qubits):
        self.gate('H', qubits)

    def R(self, qubits, *, phi, theta):
        self.gate('R', qubits, phi=phi, theta=theta)

    def CNOT(self, qubits):
        self.gate('CNOT', qubits)

    def MS(self, qubits, *, phi, theta):
        self.gate('MS', qubits, phi=phi, theta=theta)


class CircuitBuilder(CircuitBuilderBase, IonSimCircuitBuilder, QiskitCircuitBuilder):
    pass
