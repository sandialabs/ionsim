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
class Coupling:
    """A time-dependent coupling between two energy eigenstates of the non-interacting Hamiltonian."""
    basis: StandardBasis
    lower_state: EnergyEigenstate # corresponds to the column index of the non-zero element of a raising operator
    upper_state: EnergyEigenstate # corresponds to the row index of the non-zero element of a raising operator
    strength: float
    oscillation_rate: float # provides w in the phase factor exp[-i w t]

@dataclass(frozen=True, eq=False)   
class CouplingOperator:
    """An off-diagonal quantum operator in a basis of energy eigenstates."""
    basis: StandardBasis
    couplings: list[Coupling]
    modulation_function: Callable | None = None
    # TODO: add static_operator and oscillation rate as attributes; make couplings a cached property
    # Then, build Hamiltonian from the static_operators and rates directly, instead of the from the couplings.

    def __post_init__(self):
        self._check_for_one_oscillation_rate()

    def _check_for_one_oscillation_rate(self):
        if all([coupling.oscillation_rate == self.couplings[0].oscillation_rate for coupling in self.couplings]):
            return
        raise IonSimError('All couplings in the operator must have the same oscillation rate.')

    # TODO: change Matrix type to "NDArray | CSRMatrix" if necessary
    @classmethod
    def from_matrix(cls, basis: StandardBasis, static_operator: Matrix, oscillation_rate: float,
            current_dofs: list[DegreeOfFreedom] | None = None, modulation_function: list[Callable] | None = None):
        """Build a coupling operator from the matrix representation of an operator acting on some DoFs in the basis."""
        operator, rate = cls._create_sparse_static_matrix_and_rate_matrix(static_operator, oscillation_rate)
        if current_dofs is not None:
            operator, rate = basis.enlarge_matrix(operator, current_dofs), basis.enlarge_matrix(rate, current_dofs)
        rows, columns = operator.nonzero()
        indices_list = [(row, column) for row, column in zip(rows, columns)]
        couplings = []
        included_indices = []
        for row, column in indices_list:
            row_state, column_state = basis.states[row], basis.states[column]
            if row > column and (column, row) not in included_indices:
                coupling = Coupling(basis, column_state, row_state, operator[row, column], rate[row, column])
                couplings.append(coupling)
                included_indices.append((row, column))
            elif column > row and (column, row) not in included_indices:
                coupling = Coupling(basis, row_state, column_state, operator[row, column], -1*rate[row, column])
                couplings.append(coupling)
                included_indices.append((row, column))
            elif column == row:
                raise IonSimError('Diagonal elements of oscillating operators are not currently allowed.')
        return cls(basis, couplings, modulation_function)

    @staticmethod
    def _create_sparse_static_matrix_and_rate_matrix(static_operator: Matrix, oscillation_rate: float):
        """Create sparse matrices for the static operator matrix and the osciallation rate matrix."""
        operator = csr_matrix(static_operator)
        nzrs, nzcs = operator.nonzero()
        rate = csr_matrix(([oscillation_rate for _ in nzrs], (nzrs, nzcs)), shape=operator.shape)
        return operator, rate

    @property
    def static_matrix(self):
        """The sparse-matrix representation of the coupling operator with its time-dependent phase factor set equal to one."""
        size = len(self.basis.vectors)
        matrices = []
        for coupling in self.couplings:
            row = self.basis.states.index(coupling.upper_state)
            column = self.basis.states.index(coupling.lower_state)
            matrices.append(csr_matrix(([coupling.strength], ([row], [column])), shape=(size, size)))
        return np.sum(matrices)

    @property
    def rate_matrix(self):
        """The sparse-matrix representation of the oscillation rate matrix."""
        size = len(self.basis.vectors)
        matrices = []
        for coupling in self.couplings:
            row = self.basis.states.index(coupling.upper_state)
            column = self.basis.states.index(coupling.lower_state)
            matrices.append(csr_matrix(([coupling.oscillation_rate], ([row], [column])), shape=(size, size)))
        return np.sum(matrices)
