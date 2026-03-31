import numpy as np
from dataclasses import dataclass
from scipy.sparse import csr_matrix
from functools import cached_property
from abc import ABC 

from ionsim.basis import StandardBasis
from ionsim.operator import Operator, Coupling, EnergyShift, GeneralOperator, EnergyShiftOperator, CouplingOperator
from ionsim.custom_types import Vector, Matrix, SparseMatrix, AnyMatrix, as_dense_matrix
from ionsim.ionsim_error import IonSimError
from ionsim.config import SMALLEST_ENERGY_SCALE

@dataclass(frozen=True, eq=False)
class CompositeOperator(ABC):
    """Abstract class for a composite operator object, which represents an operator that contains multiple bare operator contributions"""
    
    basis: StandardBasis
    operators: list[Operator] 
    rotating_frame_energies: list[float]
    sparse: bool = False

    def __post_init__(self):
        """Check that all operators belong to the same basis and have same dimensionality.""" 
        size = len(self.basis.states) # cheaper than lindblad_operators[0].static_matrix.shape() 
        shape = (size, size) 
        for operator in self.operators:
            if len(operator.basis.states) != size and operator.static_matrix.shape() != shape:
                raise IonSimError('All operators must be the same size and shape and live in the specified basis.')

    @property
    def size(self):
        return len(self.basis.states)

    @cached_property
    def coupling_operators(self):
        """ Returns a list of all CouplingOperators in the operator list """
        coupling_ops = []
        for operator in self.operators:
            if isinstance(operator, GeneralOperator):
                if operator.couplings:
                    coupling_ops.append(operator.coupling_operator_contribution) 
            elif isinstance(operator, CouplingOperator):
                coupling_ops.append(operator)
        return coupling_ops            

    @cached_property
    def energy_shift_operators(self):
        """ Returns a list of all EnergyShiftOperators in the operator list """
        energy_shift_ops = []
        for operator in self.operators:
            if isinstance(operator, GeneralOperator):
                if operator.energy_shifts:
                    energy_shift_ops.append(operator.energy_shift_operator_contribution) 
            elif isinstance(operator, EnergyShiftOperator):
                energy_shift_ops.append(operator)
        return energy_shift_ops

    def _frame_shifted_coupling_matrix_and_rate_from_operator(self, operator: CouplingOperator | GeneralOperator):
        """Extracts the offdiagonal (coupling/interaction) matrix and oscillation rate matrix (Rate) from an operator 
            while incorporating necessary reference frame shifts. 

            Transform operator (O) to the interaction picture by a diagonal unitary operator U representing a frame shift: 
            O' = U O U^† 

            O can be decomposed into diagonal and non-diagonal components. 
            For diagonal unitaries such as the interaction picture, the diagonal portion of O commutes: U O_diag U^† = O_diag 
            The off-diagonal (coupling) portion of O does not commute with a diagonal unitary except in the special case where all diagonal elements of U are the same.

            The off-diagonal elements of O' are |i><j| Omega_{ij} exp(-i t [omega_jj - omega_ii - omega_O,[ij] )
            where Omega_ij represents a coupling strength between state i and j, omega_aa represents the frame shift energy for state "a",
            and omega_L,[ij] is an oscillation rate for the original O operator for coupling between states i and j.  
        """
        if isinstance(operator, EnergyShiftOperator):
            ValueError('Error: Operator input should not be an EnergyShift (purely diagonal) operator.')

        op_ints = [csr_matrix(([0], ([0], [0])), shape=(self.size, self.size))]
        op_Rates = [csr_matrix(([0], ([0], [0])), shape=(self.size, self.size))]
        # Loop over all oscillating elements: off-diagonal Couplings and diagonal OscillatingEnergyShifts
        oscillating = operator.oscillating_elements if hasattr(operator, 'oscillating_elements') else operator.couplings
        for coupling in oscillating:
            assert(np.abs(coupling.strength) >= 0) 
            if np.abs(coupling.strength) < SMALLEST_ENERGY_SCALE: continue
            for row, row_state in enumerate(self.basis.states):
                for column, column_state in enumerate(self.basis.states):
                    if (row_state, column_state) == (coupling.row_state, coupling.column_state):
                        op_ints.append(csr_matrix(([coupling.strength], ([row], [column])), shape=(self.size, self.size)))
                        total_rate = (
                            + coupling.oscillation_rate
                            + self.rotating_frame_energies[row]
                            - self.rotating_frame_energies[column]
                            )
                        total_rate = total_rate if abs(total_rate) > SMALLEST_ENERGY_SCALE else 0
                        op_Rates.append(csr_matrix(([total_rate], ([row], [column])), shape=(self.size, self.size)))
                        ### [row, column] corresponds to phase factor next to raising operator: sigma^dagger exp[-i rate t]
        coupling_matrix = np.sum(op_ints, axis=0)
        Rate = np.sum(op_Rates, axis=0)
        return coupling_matrix, Rate  


