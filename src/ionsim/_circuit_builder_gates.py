from ionsim.process import Gate


class GateFromUnitary:

    def __init__(self, unitary):
        self.unitary = unitary
        self.func = Gate.from_unitary

    def __call__(self, qubits, params, basis, noise):
        target_dofs = [basis.degrees_of_freedom[i] for i in qubits]
        return self.func(basis, self.unitary, target_dofs)


class GateFromUnitaryFunction:

    def __init__(self, unitary):
        self.unitary = unitary
        self.func = Gate.from_unitary_function

    def __call__(self, qubits, params, basis, noise):
        dofs = [basis.degrees_of_freedom[i] for i in qubits]
        return self.func(basis, self.unitary, params, dofs, noise)
