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
from typing import Callable
from abc import ABC, abstractmethod

from ionsim.basis import StandardBasis
from ionsim.operators import Operator, Coupling, EnergyShift, GeneralOperator, EnergyShiftOperator, CouplingOperator
from ionsim.custom_types import Vector, Matrix, SparseMatrix, AnyMatrix
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

    @abstractmethod
    def get_matrix(self, t: float) -> AnyMatrix:
        """ A function of time that returns the matrix representation of the operator. """ 

    def _frame_shifted_coupling_matrix_and_rate_from_operator(self, operator: CouplingOperator | GeneralOperator):
        """Extracts the offdiagonal (coupling/interaction) matrix and oscillation rate matrix from an operator 
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
            raise IonSimError('Operator input should not be an EnergyShift (purely diagonal) operator.')
        if isinstance(operator, GeneralOperator):
            if operator.coupling_operator_contribution is None:   # a GeneralOperator with only diagonal elements
                return csr_matrix((self.size, self.size), dtype=complex), csr_matrix((self.size, self.size), dtype=float)

        rows, columns, strengths, rates = [], [], [], []
        for coupling in operator.couplings:   # off-diagonal elements only
            if np.abs(coupling.strength) < SMALLEST_ENERGY_SCALE:
                continue
            try:
                row, column = self.basis.state_indices[coupling.row_state], self.basis.state_indices[coupling.column_state]
            except KeyError:
                msg = []
                if coupling.row_state not in self.basis.state_indices:
                    msg.append(f"{coupling.row_state=}")
                if coupling.column_state not in self.basis.state_indices:
                    msg.append(f"{coupling.column_state=}")
                msg = ", ".join(msg)
                raise IonSimError(f"A coupling of {type(operator).__name__} has {msg} not in the basis {self.basis}"
                                  f" type={type(self).__name__}.") from exc

            total_rate = coupling.oscillation_rate + self.rotating_frame_energies[row] - self.rotating_frame_energies[column]
            ### [row, column] corresponds to phase factor next to raising operator: sigma^dagger exp[-i rate t]
            rows.append(row)
            columns.append(column)
            strengths.append(coupling.strength)
            rates.append(total_rate if abs(total_rate) > SMALLEST_ENERGY_SCALE else 0.)
        shape = (self.size, self.size)
        coupling_matrix = csr_matrix((np.array(strengths, dtype=complex), (rows, columns)), shape=shape)
        rate_matrix = csr_matrix((np.array(rates, dtype=float), (rows, columns)), shape=shape)
        return coupling_matrix, rate_matrix 

    # ---- Evaluation of the operator sum as stored ----
    @cached_property
    def build_matrix_function(self) -> Callable:
        """ Function of time returning (dense or sparse) matrix as an operator sum: sum_k O_k(t), the operators of this composite in its rotating frame. 

            Every operator is used as stored, (with no Hermitian conjugate added; pass C and adjoint(C) directly to construct Hermitian operator). 
                energy shifts: static diagonal * f(t).
                couplings: C * exp(-1j rate t) * f(t).
        """
        diagonal_terms, coupling_terms = [], []   # (matrix, f), (matrix, rate, f)
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
                    coupling_terms.append((csr_matrix(matrix, dtype=complex), csr_matrix(rate), part.modulation_function))
                else:
                    raise IonSimError(f"Unsupported operator type {type(part).__name__} in {type(self).__name__}.")

        if not self.sparse:
            diagonal_terms = [(matrix.toarray(), f) for matrix, f in diagonal_terms]
            coupling_terms = [(matrix.toarray(), rate.toarray(), f) for matrix, rate, f in coupling_terms]

        def _operator_sum_function(t: float):
            if self.sparse:
                total = csr_matrix((self.size, self.size), dtype=complex)
            else:
                total = np.zeros((self.size, self.size), dtype=complex)
            for matrix, f in diagonal_terms:
                total = total + (matrix if f is None else matrix * f(t))
            for matrix, rate, f in coupling_terms:
                if self.sparse:
                    term = matrix + matrix.multiply(rate.multiply(-1j * t).expm1())   # matrix * exp(-1j rate t), sparsity kept
                else:
                    term = matrix * np.exp(-1j * rate * t)
                if f is not None:
                    term = term * f(t)
                total = total + term
            return total

        return _operator_sum_function
