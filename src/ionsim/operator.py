from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy.sparse import csr_matrix, diags
import numpy as np
from typing import Callable

from ionsim.ionsim_error import IonSimError
from ionsim.basis import StandardBasis
from ionsim.energy_level import EnergyEigenstate
from ionsim.degree_of_freedom import DegreeOfFreedom
from ionsim.custom_types import Matrix, Vector
from ionsim.config import SMALLEST_ENERGY_SCALE


# ------- Contains classes for Operators and Operator Elements -------- 

# ---- Classes for operator elements ----  
@dataclass(frozen=True, eq=False)
class OperatorElement(ABC): 
    """An abstract element of a generalized operator in a basis of energy eigenstates. 

        An operator element lives in a matrix corresponding to a basis of energy eigenstates. 
        The "ith,jth element of operator O" O_ij propto |i><j| where |i> represents the row state, and <j| the column state, assuming a convention of kets as column vectors.  

        e.g. the raising operator has a lower state as the column state, and upper state as row state. 

    """
    # This abstract class serves to factor common attributes from Coupling and EnergyShift 
    column_state: EnergyEigenstate # e.g. corresponds to the column index (lower state) of the non-zero element of a raising operator
    row_state: EnergyEigenstate # e.g. corresponds to the row index (upper state) of the non-zero element of a raising operator
    strength: float

    def __post_init__(self):
        if np.abs(self.strength) < SMALLEST_ENERGY_SCALE : 
            raise IonSimError("Invalid matrix element. Element must contain a non-zero strength value.")


@dataclass(frozen=True, eq=False)
class Coupling(OperatorElement):
    """A time-dependent coupling (off-diagonal element) between two energy eigenstates of the non-interacting Hamiltonian."""
    oscillation_rate: float # provides w in the phase factor exp[-i w t]

    def __post_init__(self):
        super().__post_init__()
        if self.row_state.name == self.column_state.name: # or check equality between states? not currently allowed 
          raise IonSimError('Coupling must entail different row and column state to represent an off-diagonal element.')

    @classmethod
    def from_energy_ordered_states(basis, lower_state: EnergyEigenstate, upper_state: EnergyEigenstate,
                                    strength: float, rate: float):
        """ Class method to construct a coupling based on state input. 
            lower_state corresponds to an energy eigenstate that is lower in energy than upper_state. """
        # Check that user input lower and upper states have the expected energy relationship 
        lower_state_index = basis.states.index(lower_state)
        upper_state_index = basis.states.index(upper_state)
        if basis.states[lower_state_index].energy > basis.states[upper_state_index].energy:
            IonSimError("Error: Lower state should be lower or equal in energy compared to upper state.")
        return cls(row_state = upper_state, column_state = lower_state, strength = strength, oscillation_rate = rate)
        

@dataclass(frozen=True, eq=False)
class EnergyShift(OperatorElement):
    """A matrix element of a diagonal Hamiltonian operator in the basis of energy eigenstates."""

    @classmethod
    def from_state(cls, state: EnergyEigenstate, shift_strength: float):
        """ Creates a diagonal element from a state """
        return cls(row_state = state, column_state = state, strength = shift_strength) 

    def __post_init__(self):
        super().__post_init__()
        # Checks that user is using this functionality for diagonal elements only  
        if self.row_state.name != self.column_state.name: # or check equality between state objects? not currently allowed 
          raise IonSimError('Energy Shift must use same row and column state to represent a diagonal element.')


# ---- Classes for operators ----  
@dataclass(frozen=True, eq=False)
class Operator(ABC):
    """A parent class for matrix operators needed to construct Hamiltonian functions in the basis of energy eigenstates."""
    basis: StandardBasis
    elements: list[OperatorElement]
    modulation_function: Callable | None=None

    def __post_init__(self):
        # Checks that all elements have non-zero strength compared to the smallest energy scale  
        if any(np.abs(element.strength) < SMALLEST_ENERGY_SCALE for element in self.elements):
            raise IonSimError("Invalid matrix element. Element must contain a non-zero strength value.")

    @classmethod
    @abstractmethod
    def from_matrix(cls, basis: StandardBasis, static_matrix: Matrix, oscillation_rate: float=0., 
            current_dofs: list[DegreeOfFreedom] | None = None, modulation_function: list[Callable] | None = None):
        """Build an operator from the matrix representation of a static operator acting on some DoFs in the basis."""
         
    @property
    @abstractmethod
    def static_matrix(self):
        """The sparse-matrix representation of the operator. If purely offdiagonal, the time-dependent phase factor is set equal to one."""


    @property
    def superbra(self):
        """ Flattened representation of a static operator (often a measurement (POVM)) as a row vector """ 
        return (np.conj(self.static_matrix.toarray())).flatten() # TODO: add warning / fail for non-static operators? 

    @staticmethod
    def _create_sparse_static_coupling_matrix_and_rate_matrix(static_matrix: Matrix, oscillation_rate: float):
        """Create sparse matrices for the static operator matrix and the osciallation rate matrix."""
        static_coupling_matrix = csr_matrix(static_matrix)
        rows, cols = static_coupling_matrix.nonzero()
        nonzero_diagonal = any(rows == cols)
        nonzero_offdiagonal = not all(rows == cols) 
        # Check that input static_matrix operator is purely off-diagonal
        assert not nonzero_diagonal, 'Error: Matrix input should be purely off-diagonal to correspond with oscillation rate input. Dense matrix input with oscillation rate is ambiguous.'
        
        rate = csr_matrix(([oscillation_rate for _ in rows], (rows, cols)), shape=static_coupling_matrix.shape)
        return static_coupling_matrix, rate

    @staticmethod
    def _couplings_from_coupling_matrix(basis: StandardBasis, coupling_matrix: csr_matrix, rate: csr_matrix):
        """ Builds and return list of unique Couplings from the matrix representation of a static operator acting on some DoFs in the basis.
            Assumes matrix input is of the "Coupling" form (purely off-diagonal) and Hermitian."""
        rows, columns = coupling_matrix.nonzero()
        # For Hermitian inputs, the upper-right portion of the matrix is handled first by convention of the nonzero() operation output for a csr matrix 
        indices_list = [(row, column) for row, column in zip(rows, columns)]
        included_indices = []
        couplings = []
        for row, column in indices_list:
            assert row != column, 'Input error: Input should be a purely off-diagonal matrix.' 
            row_state, column_state = basis.states[row], basis.states[column]
            if (column,row) not in included_indices:
                coupling = Coupling(row_state = row_state, column_state = column_state, strength = coupling_matrix[row, column], oscillation_rate = rate[row, column])
                couplings.append(coupling)
                included_indices.append((row, column))
            elif column == row:
                raise IonSimError('Diagonal elements of oscillating (coupling) operators are not currently allowed.')

        return couplings

    def _check_for_one_oscillation_rate(self):
        """ A unique off-diagonal operator is defind with one oscillation rate.""" 
        if all([coupling.oscillation_rate == self.couplings[0].oscillation_rate for coupling in self.couplings]):
            return
        raise IonSimError('All couplings in the operator must have the same oscillation rate.')

    @staticmethod
    def _energy_shifts_from_vector(basis: StandardBasis, diagonal_elements: Vector): 
        """Return a list of energy shift elements from the matrix representation of a static diagonal operator acting on some DoFs in the basis."""
        # Helper function for building EnergyShifts from a list of diagonal elements. 
        # Assumes diagonal elements sorted order matches the sorted order of basis states in basis. 
        matrix_elements = []
        for i, value in enumerate(diagonal_elements):
            if np.abs(value) > SMALLEST_ENERGY_SCALE : 
                matrix_element = EnergyShift.from_state(basis.states[i], value)
                matrix_elements.append(matrix_element)
              
        return matrix_elements 
 
    @classmethod
    def from_vector(cls, basis: StandardBasis, diagonal_elements: Vector, 
        current_dofs: list[DegreeOfFreedom] | None = None, modulation_function: list[Callable] | None = None):
        """Build an EnergyShift operator from an array of energy shifts representation a diagonal operator acting on some DoFs in the basis."""
        energy_shift_elements = cls._energy_shifts_from_vector(basis, diagonal_elements)
        return cls(basis, energy_shift_elements, modulation_function)


# ---- Operator subclasses ---- 
@dataclass(frozen=True, eq=False)   
class EnergyShiftOperator(Operator):
    """A diagonal quantum operator in a basis of energy eigenstates, representing energy shifts."""

    def __post_init__(self):
        super().__post_init__()
        self._check_all_elements_are_energy_shifts()

    def _check_all_elements_are_energy_shifts(self):
        # Check that no couplings are in the list of elements 
        if all(isinstance(element, EnergyShift) for element in self.elements):
            return 
        raise IonSimError('All elements of EnergyShiftOperator must be Energy Shifts.')

    @property
    def energy_shifts(self):
        return self.elements 

    @classmethod
    def from_matrix(cls, basis: StandardBasis, static_matrix: Matrix, current_dofs: list[DegreeOfFreedom] | None = None, 
                     modulation_function: list[Callable] | None = None):
        """Build a diagonal operator from the matrix representation of a static operator acting on some DoFs in the basis."""
        matrix = csr_matrix(static_matrix) 
        if current_dofs is not None:
            matrix = basis.enlarge_matrix(matrix, current_dofs)

        # Get elements that are non-zero and check that the user passed in a diagonal operator 
        rows, cols = matrix.nonzero()
        if not all(rows == cols):
            raise IonSimError('Error, cannot create Coupling (off-diagonal) element in EnergyShiftOperator class.') 

        return cls.from_vector(basis, matrix.diagonal(), modulation_function)

    @property
    def static_matrix(self):
        """The sparse-matrix representation of the energy shift (diagonal) operator."""
        # Use vector representation of diagonals to create and return a sparse matrix 
        return diags(diagonals = [self.static_vector], offsets = [0], format = 'csr')

    @property
    def static_vector(self):
        """The vector representation of a static energy shift (diagonal) operator."""
        v = np.zeros(len(self.basis.states))
        for energy_shift in self.energy_shifts:
            state_indx = self.basis.states.index(energy_shift.row_state)
            # Insert strength at appropriate index to match basis.states sorted order  
            v[state_indx] = energy_shift.strength
        return v 


@dataclass(frozen=True, eq=False)   
class CouplingOperator(Operator):
    """An off-diagonal quantum operator in a basis of energy eigenstates."""

    def __post_init__(self):
        super().__post_init__()
        self._check_for_one_oscillation_rate() # should use inherited method 

    def _check_all_elements_are_couplings(self):
        # Check all elements are couplings
        if all([isinstance(element, Coupling) for element in self.elements]):  
            pass 
        raise IonSimError('All elements of CouplingOperator must be Couplings.')

    @property
    def couplings(self):
        return self.elements 

    @classmethod
    def from_matrix(cls, basis: StandardBasis, static_matrix: Matrix, oscillation_rate: float,
            current_dofs: list[DegreeOfFreedom] | None = None, modulation_function: list[Callable] | None = None):
        """Build a coupling operator from the matrix representation of an operator acting on some DoFs in the basis."""
        coupling_matrix, rate = cls._create_sparse_static_coupling_matrix_and_rate_matrix(static_matrix, oscillation_rate)
        if current_dofs is not None:
            coupling_matrix, rate = basis.enlarge_matrix(coupling_matrix, current_dofs), basis.enlarge_matrix(rate, current_dofs)

        # Retrieve unique coupling matrix elements
        couplings = cls._couplings_from_coupling_matrix(basis, coupling_matrix, rate)
        return cls(basis, couplings, modulation_function)

    @property
    def rate_matrix(self):
        """The sparse-matrix representation of the oscillation rate matrix."""
        size = len(self.basis.vectors)
        matrices = []
        for coupling in self.couplings:
            row = self.basis.states.index(coupling.row_state)
            column = self.basis.states.index(coupling.column_state)
            matrices.append(csr_matrix(([coupling.oscillation_rate], ([row], [column])), shape=(size, size)))
        return np.sum(matrices)

    @property
    def static_matrix(self):
        """The sparse-matrix representation of the coupling operator with its time-dependent phase factor set equal to one."""
        size = len(self.basis.vectors)
        matrices = []
        # This loop works correctly if the coupling matrix is filled with unique coupling elements. 
        for coupling in self.couplings:
            row = self.basis.states.index(coupling.row_state)
            column = self.basis.states.index(coupling.column_state)
            matrices.append(csr_matrix(([coupling.strength], ([row], [column])), shape=(size, size)))
        return np.sum(matrices)


@dataclass(frozen=True, eq=False)   
class GeneralOperator(Operator): # is there a better name? We avoid "DenseOperator" to avoid clashing with sparse vs. dense functionality later on 
    """ A general quantum operator in the basis of energy eigenstates. This operator can be dense, containing both diagonal (energyShifts) and 
        non-diagonal (Coupling) OperatorElements. For dense operators, we use a convention where no oscillation rate is specified for off-diagonal elements.  

        The Hamiltonian class will require separating a general operator into CouplingOperator and EnergyShiftOperator contributions for efficiency. 
    """
    def __post_init__(self):
        super().__post_init__()

    @property
    def couplings(self):
        return [element for element in self.elements if isinstance(element, Coupling)]

    @property
    def energy_shifts(self):
        return [element for element in self.elements if isinstance(element, EnergyShift)]

    # --- Methods to retrieve coupling operator and energy shift operator contribution to a general dense operator ---  
    @property
    def coupling_operator_contribution(self) -> CouplingOperator | None: 
        """ Returns the coupling operator that contributes to the general operator.
            Any matrix can be decomposed into diagonal (energy-shift) and off-diagonal (coupling) contributions. """ 
        if not self.couplings:
            IonSimError("Error: General Operator does not contain couplings.")
            return None 
        else:
            return CouplingOperator(basis = self.basis, elements = self.couplings, modulation_function = self.modulation_function) 
        
    @property
    def energy_shift_operator_contribution(self) -> EnergyShiftOperator | None : 
        """ Returns the energy shift operator that contributes to the general operator.
            Any matrix can be decomposed into diagonal (energy-shift) and off-diagonal (coupling) contributions. """ 
        if not self.energy_shifts:
            IonSimError("Error: General Operator does not contain energy shifts.")
            return None 
        else:
            return EnergyShiftOperator(basis = self.basis, elements = self.energy_shifts, modulation_function = self.modulation_function) 

    @classmethod
    def from_matrix(cls, basis: StandardBasis, static_matrix: Matrix, oscillation_rate: float=0., 
            current_dofs: list[DegreeOfFreedom] | None = None, modulation_function: list[Callable] | None = None):
        """ Build a general operator from the matrix representation of a static operator acting on some DoFs in the basis. 
            Fills an operator object with a list of OperatorElements. This parses a matrix input for diagonal (energy shifts) and 
            off-diagonal (coupling) elements. Modulation function is applied uniformly to all matrix elements. """

        input_operator = csr_matrix(static_matrix)
        if current_dofs is not None:
            input_operator = basis.enlarge_matrix(input_operator, current_dofs)

        # Check whether user inputs a dense matrix 
        rows, cols = input_operator.nonzero()

        # Define booleans for checking whether input matrix has diagonal elements or off-diagonal elements. 
        nonzero_diagonal = any(rows == cols) # True if there are some non-zero diagonal entries 
        nonzero_offdiagonal = not all(rows == cols) # True if there are some non-zero off-diagonal entries  
        matrix_is_diagonal = not nonzero_offdiagonal  

        # Input matrix is dense if both nonzero_diagonal and nonzero_offdiagonal are true 
        dense_input_matrix = (nonzero_diagonal and nonzero_offdiagonal)

        # Check if oscillation rate is non-zero and raise an error if the matrix is also dense or purely diagonal  
        if oscillation_rate != 0. and (dense_input_matrix or matrix_is_diagonal): 
          raise IonSimError("Oscillation rate functionality is ambiguous for dense matrix. Separate matrix operator into off-diagonal (coupling) and diagonal (energy shift) operator objects and use CouplingOperator for oscillating off-diagonal components.")

        # Extract couplings
        coupling_matrix, rate_matrix = cls._create_sparse_static_coupling_matrix_and_rate_matrix(
                                     input_operator - np.diag(input_operator.diagonal()), oscillation_rate)

        couplings = cls._couplings_from_coupling_matrix(basis, coupling_matrix, rate_matrix)

        # Extract diagonal energy shifts 
        energy_shift_elements = []
        energy_shift_elements = cls._energy_shifts_from_vector(basis, input_operator.diagonal()) 

        # Combine EnergyShift and Coupling Elements 
        matrix_elements = [] 
        matrix_elements = energy_shift_elements + couplings

        return cls(basis, matrix_elements, modulation_function)

    @property
    def static_matrix(self):
        """The sparse-matrix representation of the operator. If purely offdiagonal, the time-dependent phase factor is set equal to one."""
        # Convention is to return a matrix with only unique off-diagonal elements (don't include h.c.)
        size = len(self.basis.vectors)
        matrices = []
        for element in self.elements:
            row = self.basis.states.index(element.row_state)
            column = self.basis.states.index(element.column_state)
            matrices.append(csr_matrix(([element.strength], ([row], [column])), shape=(size, size)))
        return np.sum(matrices)
