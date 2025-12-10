from ionsim.ionsim_error import IonSimError
from ionsim.basis import StandardBasis
from ionsim.energy_level import EnergyEigenstate
from ionsim.degree_of_freedom import DegreeOfFreedom
from ionsim.custom_types import Matrix

from dataclasses import dataclass
from scipy.sparse import csr_matrix
import numpy as np
from typing import Callable

from icecream import ic


@dataclass(frozen=True, eq=False)
class EnergyShift:
    """A matrix element of a diagonal Hamiltonian in the basis of energy eigenstates."""
    basis: StandardBasis
    state: EnergyEigenstate # corresponds to the diagonal state
    strength: float


@dataclass(frozen=True, eq=False)   
class EnergyShiftOperator:
    """A diagonal quantum operator in a basis of energy eigenstates, representing energy shifts."""
    basis: StandardBasis
    elements: list[EnergyShift]
    modulation_function: Callable | None=None # Provides functionality for time-dependent diagonal matrix elements

    @classmethod
    def from_vector(cls, basis: StandardBasis, diagonal_elements: Vector, 
        current_dofs: list[DegreeOfFreedom] | None = None, modulation_function: list[Callable] | None = None):
        """Build a diagonal operator from an array of energy shifts representation a diagonal operator acting on some DoFs in the basis."""
        # Loop through the diagonal matrix elements and build the Diagonal Operator from matrix elements
        matrix_elements = []
        for i, value in enumerate(diagonal_elements):
            state = basis.states[i]
            matrix_element = EnergyShift(basis, state, strength = value)
            matrix_elements.append(matrix_element)
        return cls(basis, matrix_elements, modulation_function)
    

    @classmethod
    def from_matrix(cls, basis: StandardBasis, static_matrix: Matrix, #TODO: potentially rename -> static_matrix
            current_dofs: list[DegreeOfFreedom] | None = None, modulation_function: list[Callable] | None = None):
        """Build a diagonal operator from the matrix representation of an operator acting on some DoFs in the basis."""
        operator = csr_matrix(static_matrix) 
        if current_dofs is not None:
            operator = basis.enlarge_matrix(operator, current_dofs)
        # Get elements that are non-zero and checks that the user passed in a diagonal operator 
        rows, columns = operator.nonzero()
        if not all(rows == cols):
            raise IonSimError('Error, cannot create off-diagonal element in diagonal operator class. Off-diagonal elements are handled in CouplingOperator class.')
        
        return self.from_vector(basis, operator.diagonal(), current_dofs, modulation_function)

    @property
    def static_matrix(self):
        """The sparse-matrix representation of the diagonal operator"""
        size = len(self.basis.vectors)
        matrices = []
        for i, element in enumerate(self.diagonal_elements):
            state = self.basis.states.index(element.state)
            matrices.append(csr_matrix(([element.strength], ([i], [i])), shape=(size, size)))
        return np.sum(matrices)
