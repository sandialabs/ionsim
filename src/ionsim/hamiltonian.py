#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from typing import Callable
from functools import cached_property
from icecream import ic

from ionsim.basis import StandardBasis
from ionsim.operator import Operator, Coupling, EnergyShift, GeneralOperator, EnergyShiftOperator, CouplingOperator
from ionsim.custom_types import Vector, Matrix, SparseMatrix, AnyMatrix, as_dense_matrix
from ionsim.config import NUMERICAL_EQUIVALENCE_THRESHOLD, SMALLEST_ENERGY_SCALE
from ionsim.custom_math import solve_time_evolution_equation
from ionsim.composite_operator import CompositeOperator
from ionsim.ionsim_error import IonSimError

def all_none(mod_functions: list):
    return all(modulation_function is None for modulation_function in mod_functions)

def all_same(mod_functions: list):
    if all(mod is mod_functions[0] for mod in mod_functions):
        return True
    return False


@dataclass(frozen=True, eq=False)
class Hamiltonian(CompositeOperator):

    def __post_init__(self):
        super().__post_init__()

    @property
    def energies(self):
        return [state.energy + energy for state, energy in zip(self.basis.states, self.rotating_frame_energies)]

    @property
    def coupling_modulation_functions(self):
        return [operator.modulation_function for operator in self.coupling_operators]

    @property
    def energy_shift_modulation_functions(self):
        return [operator.modulation_function for operator in self.energy_shift_operators]

    @property
    def modulation_functions(self):
        mod_functions = self.coupling_modulation_functions
        mod_functions.extend(self.energy_shift_modulation_functions)
        return mod_functions

    @property
    def all_rates_are_zero(self):
        sparse_H0, sparse_H0_shifts, sparse_Hints, sparse_Rates = self.H0_H0Shifts_Hints_and_Rates
        if all(Rate.getnnz() == 0 for Rate in sparse_Rates):
            return True
        return False

    @property
    def all_ints_are_isolated(self):
        sparse_H0, sparse_H0_shifts, sparse_Hints, sparse_Rates = self.H0_H0Shifts_Hints_and_Rates
        for Hint in sparse_Hints:
            rows, cols = Hint.nonzero()
            for Hint_p in sparse_Hints:
                if Hint_p is not Hint:
                    rows_p, cols_p = Hint_p.nonzero()
                    if any((row, col) in zip(rows_p, cols_p) for row, col in zip(rows, cols)):
                        return False
        return True

    @cached_property
    def H0_H0Shifts_Hints_and_Rates(self):
        """
            Function to compute contributions to the Hamiltonian from the user-specified operators list. 
            Returns: 
                - The non-interacting Hamiltonian (H0), 
                - energy shift Hamiltonians (H0_shifts) and, 
                - the interacting Hamiltonian (Hint) for each coupling operator 
                - Hint's corresponding oscillation rate matrix (Rate).
            In the constructor, GeneralOperators are decomposed into EnergyShiftOperator (diagonal) 
                and CouplingOperator (off-diagonal) contributions.
        """
        # H0 is bare Hamiltonian that accounts for interaction frame shifts 
        H0 = csr_matrix(np.diag([energy if abs(energy) > SMALLEST_ENERGY_SCALE else 0 for energy in self.energies]))
        H0_shifts = []  
        Hints = []
        Rates = []

        # Extract all hamiltonian contributions from each operator 
        # Coupling operators:  
        for operator in self.coupling_operators:
            # Extract offdiagonal elements --> Hint and Oscillation rate 
            Hint, Rate = self._frame_shifted_coupling_matrix_and_rate_from_operator(operator)
            Hints.append(Hint)
            Rates.append(Rate)

        # Energy shift operators:  
        for operator in self.energy_shift_operators:
            H0_shifts.append(operator.static_matrix) 

        return H0, H0_shifts, Hints, Rates

    @cached_property
    def hamiltonian_function(self) -> Callable: # TODO: perhaps deprecate in favor of "build_hamiltonian_function"
        """A function that computes the Hamiltonian at a specified time."""

        sparse_H0, sparse_H0_shifts, sparse_Hints, sparse_Rates = self.H0_H0Shifts_Hints_and_Rates

        if self.sparse:
            H0 = sparse_H0
            H0_shifts = sparse_H0_shifts
            Hints = sparse_Hints
            Rates = sparse_Rates
        else:
            H0 = sparse_H0.toarray()
            H0_shifts = []
            for H_shift in sparse_H0_shifts:
                H0_shifts.append(H_shift.toarray())

            Hints, Rates = [], []
            for Hint, Rate in zip(sparse_Hints, sparse_Rates):
                Hints.append(Hint.toarray())
                Rates.append(Rate.toarray())

        if self.all_rates_are_zero:
            if self.sparse:
                def _hamiltonian_function(t: float):
                    if not all_none(self.coupling_modulation_functions):
                        Hint = csr_matrix(([0], ([0], [0])), shape=(self.size, self.size), dtype='complex')
                        if all_same(self.coupling_modulation_functions) and self.coupling_modulation_functions[0] is not None:
                            Hint += np.sum(Hints, axis=0).multiply(self.coupling_modulation_functions[0](t))
                        else:
                            for ham, mod in zip(Hints, self.coupling_modulation_functions):
                                if mod is None:
                                    Hint += ham
                                else:
                                    Hint += ham.multiply(mod(t))
                    else:
                        Hint = np.sum(Hints, axis=0)
                    Hint += Hint.conj().transpose()
                        
                    if not all_none(self.energy_shift_modulation_functions):
                        H0_shift = csr_matrix(([0], ([0], [0])), shape=(self.size, self.size), dtype='complex')
                        if all_same(self.energy_shift_modulation_functions) and self.energy_shift_modulation_functions[0] is not None:
                            H0_shift += np.sum(H0_shifts, axis=0).multiply(self.energy_shift_modulation_functions[0](t))
                        else:
                            for ham, mod in zip(H0_shifts, self.energy_shift_modulation_functions):    
                                if mod is None:
                                    H0_shift += ham
                                else:
                                    H0_shift += ham.multiply(mod(t))
                    else:
                        H0_shift = np.sum(H0_shifts, axis=0)

                    return H0 + H0_shift + Hint
            else:
                def _hamiltonian_function(t: float):
                    if not all_none(self.coupling_modulation_functions):
                        Hint = np.zeros((self.size, self.size), dtype='complex')
                        if all_same(self.coupling_modulation_functions) and self.coupling_modulation_functions[0] is not None:
                            Hint += np.sum(Hints, axis=0) * self.coupling_modulation_functions[0](t)
                        else:
                            for Ham, mod in zip(Hints, self.coupling_modulation_functions):
                                if mod is None:
                                    Hint += Ham 
                                else:
                                    Hint += Ham * mod(t)
                    else:
                        Hint = np.sum(Hints, axis=0); 
                    Hint += Hint.conj().T 
                            
                    if not all_none(self.energy_shift_modulation_functions):
                        H0_shift = np.zeros((self.size, self.size), dtype='complex')
                        if all_same(self.energy_shift_modulation_functions) and self.energy_shift_modulation_functions[0] is not None:
                            H0_shift += np.sum(H0_shifts, axis=0) * self.energy_shift_modulation_functions[0](t)
                        else:
                            for Ham, mod in zip(H0_shifts, self.energy_shift_modulation_functions):
                                if mod is None:
                                    H0_shift += Ham
                                else:
                                    H0_shift += Ham * mod(t)
                    else:
                        H0_shift = np.sum(H0_shifts, axis=0)

                    return H0 + H0_shift + Hint
        else:
            if self.sparse:
                def _hamiltonian_function(t: float):
                    H0_shift = csr_matrix(([0], ([0], [0])), shape=(self.size, self.size), dtype='complex') 
                    Hint = csr_matrix(([0], ([0], [0])), shape=(self.size, self.size), dtype='complex') 
                    for Ham, Rate, mod in zip(Hints, Rates, self.coupling_modulation_functions):
                        phase_factor_minus_one = Rate.multiply(-1j*t).expm1() # equivalent to exp(-1j * Rate * t) - 1
                        Htemp = Ham + Ham.multiply(phase_factor_minus_one)
                        if mod is not None:
                            Hint += Htemp.multiply(mod(t))
                        else:
                            Hint += Htemp

                    Hint += Hint.conj().transpose()

                    for Ham, mod in zip(H0_shifts, self.energy_shift_modulation_functions):
                        if mod is not None:
                            H0_shift += Ham.multiply(mod(t))
                        else:
                            H0_shift += Ham

                    return H0 + H0_shift + Hint
            else:
                if all_none(self.modulation_functions):
                    if self.all_ints_are_isolated: # TODO: apply this check and simplification for other cases with nonzero Rates.
                        Ham = np.sum(Hints, axis=0)
                        Rate = np.sum(Rates, axis=0)
                        def _hamiltonian_function(t: float):
                            Hint = Ham * np.exp(-1j * Rate * t)
                            Hint += Hint.conj().T
                            return H0 + np.sum(H0_shifts, axis=0) + Hint
                    else:
                        def _hamiltonian_function(t: float):
                            Hint = np.zeros((self.size, self.size), dtype='complex')
                            for Ham, Rate in zip(Hints, Rates):
                                Hint += Ham * np.exp(-1j * Rate * t)
                            Hint += Hint.conj().T
                            return H0 + np.sum(H0_shifts, axis=0) + Hint
                else:
                    if self.all_ints_are_isolated:
                        def _hamiltonian_function(t: float):
                            if all_same(self.coupling_modulation_functions):
                                Ham = np.sum(Hints, axis=0)
                                Rate = np.sum(Rates, axis=0)
                                Hint = Ham * np.exp(-1j * Rate * t) * self.coupling_modulation_functions[0](t)
                            else:
                                Hint = np.zeros((self.size, self.size), dtype='complex')
                                for Ham, Rate, mod in zip(Hints, Rates, self.coupling_modulation_functions):
                                    Hint += Ham * np.exp(-1j * Rate * t) * mod(t)
                            Hint += Hint.conj().T

                            H0_shift = np.zeros_like(Hint)
                            if all_same(self.energy_shift_modulation_functions):
                                H0_shift = np.sum(H0_shifts, axis=0)
                                if self.energy_shift_modulation_functions and self.energy_shift_modulation_functions[0] is not None: 
                                    H0_shift *= self.energy_shift_modulation_functions[0](t)
                            else:
                                for Ham, mod in zip(H0_shifts, self.energy_shift_modulation_functions):
                                    if mod is not None:
                                        H0_shift += Ham * mod(t)
                                    else:
                                        H0_shift += Ham
                            
                            return H0 + H0_shift + Hint
                    else:
                        def _hamiltonian_function(t: float):
                            Hint = np.zeros((self.size, self.size), dtype='complex')
                            for Ham, Rate, mod in zip(Hints, Rates, self.coupling_modulation_functions):
                                Hint += Ham * np.exp(-1j * Rate * t) * mod(t)
                            Hint += Hint.conj().T

                            H0_shift = np.zeros_like(Hint)
                            for Ham, mod in zip(H0_shifts, self.energy_shift_modulation_functions):
                                if mod is not None:
                                    H0_shift += Ham * mod(t)
                                else:
                                    H0_shift += Ham

                            return H0 + H0_shift + Hint

        return _hamiltonian_function

    def evolve_wavefunction(self, initial_wavefunction: Vector, duration: float, time_evals: Vector | None = None, **kwargs):
        """Evolve a wavefunction by solving the time-dependent Schrodinger equation."""
        assert(self.size == len(initial_wavefunction))
        import time
        from icecream import ic
        start = time.perf_counter()
        result = solve_time_evolution_equation(self.hamiltonian_function, initial_wavefunction, duration, time_evals, **kwargs)
        end = time.perf_counter()
        ic(f'Evolving wavefunction took {end-start} seconds.')
        return result


    ## Adiabatic elimination methodology 
    @classmethod
    def adiabatic_elimination(cls, self, states_to_eliminate: list[EnergyEigenstate]):
        """ Adiabatic elimination method, returns a Hamiltonian in a reduced basis """

        states_to_keep = [state for state in self.basis.states if state not in states_to_eliminate]    

        # Build projectors into the appropriate subspaces 

        # Define reduced basis:
        reduced_basis = self.basis 
        # Could try a basis method (from reduced set of states or something) 


        # Build new operators in the reduced basis 
        operators = []


 
        return cls(reduced_basis, operators, self.rotating_frame_energies, self.sparse) 
