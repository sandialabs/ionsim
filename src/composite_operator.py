#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

import numpy as np
from dataclasses import dataclass
from scipy.sparse import csr_matrix
from functools import cached_property
from abc import ABC
from typing import Callable, ClassVar

from ionsim.basis import StandardBasis
from ionsim.operator import Operator, Coupling, EnergyShift, GeneralOperator, EnergyShiftOperator, CouplingOperator
from ionsim.custom_types import Vector, Matrix, SparseMatrix, AnyMatrix, as_dense_matrix
from ionsim.ionsim_error import IonSimError
from ionsim.config import SMALLEST_ENERGY_SCALE, NUMERICAL_EQUIVALENCE_THRESHOLD

@dataclass(frozen=True, eq=False)
class CompositeOperator(ABC):
    """Abstract class for a composite operator object, which represents an operator that contains multiple bare operator contributions"""
    
    basis: StandardBasis
    operators: list[Operator] 
    rotating_frame_energies: list[float]
    sparse: bool = False

    # How operators with hermitian=None (unflagged) are evaluated in this composite (see Operator.hermitian): True adds the
    # Hermitian conjugate (Hamiltonian), False uses the operator as stored (Lindblad operators). Not a dataclass field.
    UNFLAGGED_HERMITIAN: ClassVar[bool] = False

    def __post_init__(self):
        """ Check that all operators belong to the same basis and have same dimensionality.""" 
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

    @cached_property
    def _state_index(self) -> dict:
        """ Index of each basis state in self.basis.states (built once; used to place coupling elements). """
        return {state: index for index, state in enumerate(self.basis.states)}

    def _frame_shifted_coupling_matrix_and_rate_from_operator(self, operator: CouplingOperator | GeneralOperator):
        """Extracts the offdiagonal (coupling/interaction) matrix and oscillation rate matrix (Rate) from an operator 
            while incorporating necessary reference frame shifts, for evaluation as
                coupling matrix * exp(-1j Rate t) * f(t)   [+ its Hermitian conjugate if hermitian].

            Transform operator (O) to the interaction picture by a diagonal unitary operator U representing a frame shift: 
            O' = U O U^† 

            O can be decomposed into diagonal and non-diagonal components. 
            For diagonal unitaries such as the interaction picture, the diagonal portion of O commutes: U O_diag U^† = O_diag 
            The off-diagonal (coupling) portion of O does not commute with a diagonal unitary except in the special case where all diagonal elements of U are the same.

            The off-diagonal elements of O' are |i><j| Omega_{ij} exp(-i t [omega_O,[ij] + omega_ii - omega_jj])
            where Omega_ij represents a coupling strength between state i and j, omega_aa represents the frame shift energy for
            state "a" (self.rotating_frame_energies[a]), and omega_O,[ij] is the oscillation rate of the original O operator for
            the coupling between states i and j (Coupling.oscillation_rate). The returned Rate matrix holds
            omega_O,[ij] + omega_ii - omega_jj at [i, j]. E.g. with rotating_frame_energies = -E (bare energies), a coupling |e><g|
            with oscillation_rate = laser frequency omega_L has Rate = omega_L - (E_e - E_g), the detuning: static on resonance.

            Which couplings are used follows the operator's hermitian flag, resolved by _is_hermitian:
                not Hermitian: every stored coupling.
                Hermitian: uses only one coupling from the pair (i, j), (j, i) 
                    The caller adds the Hermitian conjugate. If the operator is explicitly
                    flagged hermitian=True and stores both elements of a pair, they must be complex conjugates.

            Returns (coupling matrix, Rate) as sparse (csr) matrices.
        """
        if isinstance(operator, EnergyShiftOperator):
            raise IonSimError('Operator input should not be an EnergyShift (purely diagonal) operator.')
        if isinstance(operator, GeneralOperator):
            operator = operator.coupling_operator_contribution
            if operator is None:   # a GeneralOperator with only diagonal elements
                return csr_matrix((self.size, self.size), dtype=complex), csr_matrix((self.size, self.size), dtype=float)

        couplings = operator.couplings
        if self._is_hermitian(operator):
            static_matrix = csr_matrix(operator.static_matrix)
            # Check that an operator marked Hermitian with doubly-stored coupling pairs) is consistent
            if operator.hermitian:   
                rows, columns = static_matrix.nonzero()
                for row, column in zip(rows, columns):
                    hc_partner = static_matrix[column, row]
                    if np.abs(hc_partner > NUMERICAL_EQUIVALENCE_THRESHOLD) and abs(hc_partner - np.conj(static_matrix[row, column])) > NUMERICAL_EQUIVALENCE_THRESHOLD: #or SMALLEST_ENERGY_SCALE?
                        raise IonSimError(f"Operator flagged hermitian=True stores both elements ({row}, {column}) and ({column}, {row}), "
                                          f"but they are not complex conjugates: {static_matrix[row, column]} and {partner}.")
            couplings = operator._unique_couplings_from_hermitian_coupling_matrix(operator.basis, static_matrix, operator.rate_matrix)

        rows, columns, strengths, rates = [], [], [], []
        for coupling in couplings:   # off-diagonal elements only
            if np.abs(coupling.strength) < SMALLEST_ENERGY_SCALE:
                continue
            try:
                row, column = self._state_index[coupling.row_state], self._state_index[coupling.column_state]
            except KeyError:
                raise IonSimError(f"A coupling of {type(operator).__name__} refers to a state that is not in the basis of "
                                  f"{type(self).__name__}.") from None
            ### [row, column] corresponds to phase factor next to raising operator: sigma^dagger exp[-i rate t]
            total_rate = coupling.oscillation_rate + self.rotating_frame_energies[row] - self.rotating_frame_energies[column]
            rows.append(row)
            columns.append(column)
            strengths.append(coupling.strength)
            rates.append(total_rate if abs(total_rate) > SMALLEST_ENERGY_SCALE else 0.)
        shape = (self.size, self.size)
        coupling_matrix = csr_matrix((np.array(strengths, dtype=complex), (rows, columns)), shape=shape)
        Rate = csr_matrix((np.array(rates, dtype=float), (rows, columns)), shape=shape)
        return coupling_matrix, Rate

    # ---- Evaluation of the operator sum, shared by Hamiltonian and LindbladOperator ----
    def _is_hermitian(self, operator: Operator) -> bool:
        """ The operator's hermitian flag, or this composite's convention (UNFLAGGED_HERMITIAN) if it is unflagged (None). """
        return self.UNFLAGGED_HERMITIAN if operator.hermitian is None else bool(operator.hermitian)

    def build_operator_sum_function(self) -> Callable:
        """ Function of time returning sum_k O_k(t), the operators of this composite operator in its rotating frame.
            Bare-energy contributions from the basis are not included. 

            Each operator is evaluated according to its hermitian flag (see Operator); operators with hermitian=None follow
            this composite's UNFLAGGED_HERMITIAN (True for a Hamiltonian, False for a Lindblad operator):

            Scheme for combining operator contributions with modulation functions f(t) and g(t) respectively: 
                energy shifts: static diagonal * f(t), once.
                couplings: C * exp(-1j Rate t) * g(t), plus the Hermitian conjugate if hermitian.

            Returns a function that returns the dense or sparse matrix representation. """
        diagonal_terms, coupling_terms = [], []   # (matrix, f), (matrix, Rate, f, hermitian)
        for operator in self.operators:
            parts = [operator]
            if isinstance(operator, GeneralOperator):
                parts = [part for part in (operator.energy_shift_operator_contribution if operator.energy_shifts else None,
                                           operator.coupling_operator_contribution if operator.couplings else None) if part is not None]
            for part in parts:
                if isinstance(part, EnergyShiftOperator):
                    diagonal_terms.append((csr_matrix(part.static_matrix, dtype=complex), part.modulation_function))
                elif isinstance(part, CouplingOperator):
                    matrix, rate = self._frame_shifted_coupling_matrix_and_rate_from_operator(part)
                    coupling_terms.append((csr_matrix(matrix, dtype=complex), csr_matrix(rate), part.modulation_function,
                                           self._is_hermitian(part)))
                else:
                    raise IonSimError(f"Unsupported operator type {type(part).__name__} in {type(self).__name__}.")

        if not self.sparse:
            diagonal_terms = [(matrix.toarray(), f) for matrix, f in diagonal_terms]
            coupling_terms = [(matrix.toarray(), rate.toarray(), f, hermitian) for matrix, rate, f, hermitian in coupling_terms]

        def _operator_sum_function(t: float):
            if self.sparse:
                total = csr_matrix((self.size, self.size), dtype=complex)
            else:
                total = np.zeros((self.size, self.size), dtype=complex)
            for matrix, f in diagonal_terms:
                total = total + (matrix if f is None else matrix * f(t))
            for matrix, rate, f, hermitian in coupling_terms:
                if self.sparse:
                    term = matrix + matrix.multiply(rate.multiply(-1j * t).expm1())   # matrix * exp(-1j Rate t), sparsity kept
                else:
                    term = matrix * np.exp(-1j * rate * t)
                if f is not None:
                    term = term * f(t)
                total = total + term
                if hermitian:
                    total = total + term.conj().T
            return total

        return _operator_sum_function
