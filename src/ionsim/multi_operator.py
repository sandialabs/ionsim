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
from typing import Callable, NamedTuple

from ionsim.basis import StandardBasis
from ionsim.operators import Operator, Coupling, EnergyShift, GeneralOperator, EnergyShiftOperator, CouplingOperator
from ionsim.custom_types import Vector, Matrix, SparseMatrix, AnyMatrix
from ionsim.ionsim_error import IonSimError
from ionsim.config import SMALLEST_ENERGY_SCALE, NUMERICAL_EQUIVALENCE_THRESHOLD

class RateGroup(NamedTuple):
    """ One term f_g(t) e^{-i r_g t} A_g of a MultiOperator's rate-group decomposition (see MultiOperator.rate_groups). """
    modulation_function: Callable | None
    rate: float
    matrix: csr_matrix


def _rate_tolerance(rates: list[float], rotating_frame_energies: list[float]) -> float:
    """ Tolerance for treating two frame-shifted rates as equal. A rate is oscillation_rate + E_row - E_column with frame
        energies E that can be optical (~1e15 rad/s), so its floating-point error scales with the frame energies, not with
        the (possibly much smaller) rate. """
    rate_scale = max([1.] + [abs(rate) for rate in rates])
    frame_scale = max([0.] + [abs(energy) for energy in rotating_frame_energies])
    return max(NUMERICAL_EQUIVALENCE_THRESHOLD * rate_scale, 64. * np.finfo(float).eps * frame_scale)


@dataclass(frozen=True, eq=False)
class MultiOperator():
    """Class for a multi-operator (composite operator) object, which represents an operator that contains multiple bare operator contributions"""
    
    basis: StandardBasis
    operators: list[Operator] 
    rotating_frame_energies: list[float]
    sparse: bool = False
    prefactor: complex = 1.

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

    def get_matrix(self, t: float) -> AnyMatrix:
        """ A function of time that returns the matrix representation of the operator. """ 
        return self.build_matrix_function(t) * self.prefactor

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
            except KeyError as exc:
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

    # ---- Rate-group decomposition and evaluation of the operator sum as stored ----
    @cached_property
    def rate_groups(self) -> tuple[RateGroup, ...]:
        """ The operator sum regrouped by modulation function and frame-shifted rate:

                sum_k O_k(t) = sum_g f_g(t) e^{-i r_g t} A_g   (get_matrix(t) multiplies this by the prefactor)

            Every nonzero element e = (part k, row i, column j) has a value, the part's modulation function f_k (None means
            f = 1) and a frame-shifted rate r = oscillation_rate_k + E_i - E_j (0 for diagonal elements), with E the
            rotating_frame_energies. A rate group is the set of elements with the same modulation function (the same
            callable object) and the same rate (equal up to floating-point rounding, see _rate_tolerance); A_g is the
            static (sparse) matrix of its elements. This is the smallest decomposition into static matrices that each
            multiply a single scalar function of time. One operator can split into several groups (elements at different
            frame-energy differences), and elements of several operators can share a group.
        """
        elements = []  # (modulation function, rate, row, column, value)
        for operator in self.operators:
            parts = [operator]
            if isinstance(operator, GeneralOperator):
                parts = [part for part in (operator.energy_shift_operator_contribution if operator.energy_shifts else None,
                                           operator.coupling_operator_contribution if operator.couplings else None) if part is not None]
            for part in parts:
                if isinstance(part, EnergyShiftOperator):
                    matrix = csr_matrix(part.static_matrix, dtype=complex).tocoo()
                    elements.extend((part.modulation_function, 0., row, column, value)
                                    for row, column, value in zip(matrix.row, matrix.col, matrix.data))
                elif isinstance(part, CouplingOperator):
                    matrix, rate = self._frame_shifted_coupling_matrix_and_rate_from_operator(part)
                    matrix, rate = csr_matrix(matrix, dtype=complex).tocoo(), csr_matrix(rate)
                    elements.extend((part.modulation_function, float(rate[row, column]), row, column, value)
                                    for row, column, value in zip(matrix.row, matrix.col, matrix.data))
                else:
                    raise IonSimError(f"Unsupported operator type {type(part).__name__} in {type(self).__name__}.")

        tolerance = _rate_tolerance([element[1] for element in elements], self.rotating_frame_energies)
        groups = []  # [modulation function, rate, rows, columns, values]
        for modulation, rate, row, column, value in elements:
            if value == 0:
                continue
            for group in groups:
                if group[0] is modulation and abs(group[1] - rate) <= tolerance:
                    group[2].append(row); group[3].append(column); group[4].append(value)
                    break
            else:
                groups.append([modulation, rate, [row], [column], [value]])
        shape = (self.size, self.size)
        return tuple(RateGroup(modulation, rate, csr_matrix((values, (rows, columns)), shape=shape, dtype=complex))
                     for modulation, rate, rows, columns, values in groups)

    def group_time_factors(self, t: float) -> list[complex]:
        """ f_g(t) e^{-i r_g t} for every rate group (each modulation function is evaluated once). """
        values = {}
        factors = []
        for group in self.rate_groups:
            f = group.modulation_function
            if f is None:
                value = 1.
            else:
                if id(f) not in values:
                    values[id(f)] = f(t)
                value = values[id(f)]
            factors.append(value * np.exp(-1j * group.rate * t))
        return factors

    def group_coefficients(self, t: float) -> list[complex]:
        """ prefactor f_g(t) e^{-i r_g t} for every rate group: get_matrix(t) = sum_g coefficient_g A_g. """
        return [self.prefactor * factor for factor in self.group_time_factors(t)]

    @cached_property
    def build_matrix_function(self) -> Callable:
        """ Function of time returning (dense or sparse) matrix as an operator sum: sum_k O_k(t), the operators of this composite
            in its rotating frame, evaluated from the rate groups as sum_g f_g(t) e^{-i r_g t} A_g (see rate_groups).

            Every operator is used as stored, (with no Hermitian conjugate added; pass C and adjoint(C) directly to construct Hermitian operator).
                energy shifts: static diagonal * f(t).
                couplings: C * exp(-1j rate t) * f(t), with the frame-shifted rate of each element.
            Elements whose rates are equal up to floating-point rounding share one rate, so phases at optical rates agree with
            an element-by-element evaluation to about (rounding of the rate) * t.
        """
        shape = (self.size, self.size)
        coordinates = []  # per group: (rows, columns, values) with unique (row, column)
        for group in self.rate_groups:
            matrix = group.matrix.tocoo()
            matrix.sum_duplicates()
            coordinates.append((matrix.row, matrix.col, matrix.data))

        def _operator_sum_function(t: float):
            factors = self.group_time_factors(t)
            if self.sparse:
                if not coordinates:
                    return csr_matrix(shape, dtype=complex)
                rows = np.concatenate([rows for rows, _, _ in coordinates])
                columns = np.concatenate([columns for _, columns, _ in coordinates])
                data = np.concatenate([values * factor for (_, _, values), factor in zip(coordinates, factors)])
                return csr_matrix((data, (rows, columns)), shape=shape, dtype=complex)
            total = np.zeros(shape, dtype=complex)
            for (rows, columns, values), factor in zip(coordinates, factors):
                total[rows, columns] += values * factor
            return total

        return _operator_sum_function


def to_MultiOperator(input_operator: Operator, rotating_frame_energies: list[float] | None=None, sparse: bool=False):
    """ Converts an Operator object to a MultiOperator object """ 
    _rotating_frame_energies = rotating_frame_energies if rotating_frame_energies else [0.]*len(input_operator.basis.states) 
    return MultiOperator(input_operator.basis, [input_operator], _rotating_frame_energies, sparse)
