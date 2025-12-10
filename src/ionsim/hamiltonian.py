from ionsim.basis import StandardBasis
from ionsim.coupling import Coupling, CouplingOperator
from ionsim.custom_types import Vector, Matrix, SparseMatrix, AnyMatrix, as_dense_matrix
from ionsim.config import NUMERICAL_EQUIVALENCE_THRESHOLD, SMALLEST_ENERGY_SCALE
from ionsim.custom_math import solve_time_evolution_equation
from ionsim.ionsim_error import IonSimError
from ionsim.energy_shift_operator import EnergyShiftOperator, EnergyShift

from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from typing import Callable
from functools import cached_property

from icecream import ic

@dataclass(frozen=True, eq=False)
class Hamiltonian:
    """ Hamiltonian object with arbitrary time-dependence """
    basis: StandardBasis
    coupling_operators: list[CouplingOperator]
    energy_shift_operators: list[EnergyShiftOperator] 
    rotating_frame_energies: list[float] 
    sparse: bool = False
    #operators = list[Operator] # list of general operator objects 

    # @property
    # def energy_shift_operators() -> list[EnergyShiftOperator] :
    #     return [op for op in operators if isinstance(op, EnergyShiftOperator)]

    # @property
    # def coupling_operators() -> list[CouplingOperator] :
    #     return [op for op in operators if isinstance(op, CouplingOperator)]

    # @property
    # def operators() -> list[Operator] : 
    #     return operators


    # operators: list[Operator]
    # TODO: consider getters for EnergyShiftOperators and Coupling as well as all the operators 
    # H = H_0 + \delta H_0 (energy shift operators) + H_int 

    @property
    def energies(self):
        ''' These energies account for eigenstate energies and energy shifts from the rotating frame.''' 
        return [state.energy + energy for state, energy in zip(self.basis.states, self.rotating_frame_energies)]

    @property
    def size(self):
        return len(self.basis.states)

    @property
    def modulation_functions(self):
        mod_functions = self.coupling_modulation_functions
        mod_functions.extend(self.energy_shift_modulation_functions)
        return mod_functions
        #return [operator.modulation_function for operator in self.coupling_operators]

    @property
    def energy_shift_modulation_functions(self):
        # potentially [operator.modulation_function for operator in self.energy_shift_operators]
        mod_functions = [operator.modulation_function for operator in self.energy_shift_operators]
        return mod_functions

    @property
    def coupling_modulation_functions(self):
        return [operator.modulation_function for operator in self.coupling_operators]

    @property
    def all_energy_shift_mods_are_none(self):
        return all(mod is None for mod in self.energy_shift_modulation_functions)

    def all_coupling_mods_are_none(self):
        return all(mod is None for mod in self.coupling_modulation_functions)

    @property
    def all_mods_are_none(self):
        return all(mod is None for mod in self.modulation_functions)

    @property
    def all_mods_are_equal(self):
        if all(mod is self.modulation_functions[0] for mod in self.modulation_functions):
            return True
        else:
            return False

    @property
    def all_coupling_mods_are_equal(self):
        if all(mod is self.coupling_modulation_functions[0] for mod in self.coupling_modulation_functions):
            return True
        else:
            return False

    @property
    def all_energy_shift_mods_are_equal(self):
        if all(mod is self.energy_shift_modulation_functions[0] for mod in self.energy_shift_modulation_functions):
            return True
        else:
            return False

    @property
    def all_rates_are_zero(self):
        sparse_H0, sparse_H0_shifts, sparse_Hints, sparse_Rates = self.H0_H0Shifts_Hints_and_Rates
        if all(Rate.getnnz() == 0 for Rate in sparse_Rates):
            return True
        else:
            return False

    @property
    def all_ints_are_isolated(self):
        sparse_H0, sparse_H0_shifts, sparse_Hints, sparse_Rates = self.H0_H0Shifts_Hints_and_Rates
        result = True
        for Hint in sparse_Hints:
            rows, cols = Hint.nonzero()
            for Hint_p in sparse_Hints:
                if Hint_p is not Hint:
                    rows_p, cols_p = Hint_p.nonzero()
                    if any((row, col) in zip(rows_p, cols_p) for row, col in zip(rows, cols)):
                        result = False
        return result


    # TODO: Consider adding "H_misc" for miscellanous, general (dense) operator objects? 
    @cached_property
    def H0_H0Shifts_Hints_and_Rates(self):
        """
            The non-interacting Hamiltonians (H0_shifts) and, for each coupling operator, the interacting Hamiltonian (Hint),
            and its corresponding oscillation rate matrix (Rate).
        """
        # TODO: move conditions for eliminating couplings into the creation of the coupling_operators.
        H0 = csr_matrix(np.diag([energy if abs(energy) > SMALLEST_ENERGY_SCALE else 0 for energy in self.energies]))
        
        H0_shifts = []
        Hints = []
        Rates = []
        # Fill Hints with Coulping Operators 
        for operator in self.coupling_operators:
            op_Hints = [csr_matrix(([0], ([0], [0])), shape=(self.size, self.size))]
            op_Rates = [csr_matrix(([0], ([0], [0])), shape=(self.size, self.size))]
            for coupling in operator.couplings:
                assert(np.abs(coupling.strength) >= 0) # TODO: consider raising an IonSimError
                if np.abs(coupling.strength) < SMALLEST_ENERGY_SCALE: continue
                for row, row_state in enumerate(self.basis.states):
                    for column, column_state in enumerate(self.basis.states):
                        if (row_state, column_state) == (coupling.upper_state, coupling.lower_state):
                            op_Hints.append(csr_matrix(([coupling.strength], ([row], [column])), shape=(self.size, self.size)))
                            total_rate = (
                                + coupling.oscillation_rate
                                + self.rotating_frame_energies[row]
                                - self.rotating_frame_energies[column]
                            )
                            total_rate = total_rate if abs(total_rate) > SMALLEST_ENERGY_SCALE else 0
                            op_Rates.append(csr_matrix(([total_rate], ([row], [column])), shape=(self.size, self.size)))
                            ### [row, column] corresponds to phase factor next to raising operator: sigma^dagger exp[-i rate t]
            Hints.append(np.sum(op_Hints, axis=0))
            Rates.append(np.sum(op_Rates, axis=0))
        
        # Fill H0 shifts from EnergyShift Operators 
        # Ultimately, we aim to combine H0 with Hint (off-diagonal), so we format H0's as matrices. 
        # TODO: Consider cache'ing H0 and H0_shifts as vectors (arrays) instead of matrices. 
        #   This would save on computation and storage.
        for operator in self.energy_shift_operators:
            op_H0_shifts = [csr_matrix(([0], ([0], [0])), shape=(self.size, self.size))]
            for element in operator.elements:
                assert(np.abs(element.strength) >= 0)
                if np.abs(element.strength) < SMALLEST_ENERGY_SCALE: continue
                for diagonal, diagonal_state in enumerate(self.basis.states):
                    if diagonal_state == element.state : 
                        op_H0_shifts.append(csr_matrix(([element.strength], ([diagonal], [diagonal])), shape=(self.size, self.size)))
            H0_shifts.append(np.sum(op_H0_shifts, axis=0))
        return H0, H0_shifts, Hints, Rates

    @cached_property
    def hamiltonian_function(self): # TODO: perhaps deprecate in favor of "build_hamiltonian_function"
        """A function that computes the Hamiltonian at a specified time."""

        import time
        from icecream import ic

        start = time.perf_counter()

        # TODO: Make sure we don't recompute matrices if the time-dependence is separable 
        sparse_H0, sparse_H0_shifts, sparse_Hints, sparse_Rates = self.H0_H0Shifts_Hints_and_Rates

        if self.sparse:
            H0 = sparse_H0
            H0_shifts = sparse_H0_shifts
            Hints = sparse_Hints
            Rates = sparse_Rates
            # ic([Rate.data/(2*np.pi*1e3) for Rate in Rates])
        else:
            H0 = sparse_H0.toarray()
            H0_shifts = []
            for H_shift in sparse_H0_shifts:
                H0_shifts.append(H_shift.toarray())

            Hints, Rates = [], []
            for Hint, Rate in zip(sparse_Hints, sparse_Rates):
                Hints.append(Hint.toarray())
                Rates.append(Rate.toarray())
            # ic([Rate/(2*np.pi*1e3) for Rate in Rates])

        # TODO: Try to consolidate the following code blocks; e.g. make a function for mostly common code 
        # TODO: Consider wrapper functions for consolidating code with "sparse" vs. array ops 
        #  - e.g. matrix operations 
        #  - e.g. matrix exponentiation 
        #  - add issue to Gitlab for wrapper functions for sparse vs. dense matrix operations 
        #  - separating logic based on sparse vs. dense should be handled via wrappers rather than if/else 
        #  - need to make sure that sparse matrix/operator handling is cleaned up 
        if self.all_rates_are_zero:
            if self.sparse:
                def _hamiltonian_function(t: float):
                    if not self.all_coupling_mods_are_none:
                        Hint = csr_matrix(([0], ([0], [0])), shape=(self.size, self.size), dtype='complex')
                        for ham, mod in zip(Hints, self.coupling_modulation_functions):
                            Hint += ham.multiply(mod(t))
                    else:
                        Hint = np.sum(Hints, axis=0)
                    Hint += Hint.conj().transpose()
                        
                    if not self.all_energy_shift_mods_are_none:
                        H0_shift = csr_matrix(([0], ([0], [0])), shape=(self.size, self.size), dtype='complex')
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
                    if not self.all_coupling_mods_are_none:
                        Hint = np.zeros((self.size, self.size), dtype='complex')
                        for Ham, mod in zip(Hints, self.coupling_modulation_functions):
                            Hint += Ham * mod(t)
                    else:
                        Hint = np.sum(Hints, axis=0); 
                    Hint += Hint.conj().T 
                            
                    if not self.all_energy_shift_mods_are_none:
                        H0_shift = np.zeros((self.size, self.size), dtype='complex') # should this be real? 
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
                if self.all_mods_are_none:
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
                else: # TODO: Check if each modulation is the same function, and sum ints outside of H(t). 
                    if self.all_ints_are_isolated: #and self.all_mods_are_equal:
                        def _hamiltonian_function(t: float):
                            if self.all_coupling_mods_are_equal:
                                Ham = np.sum(Hints, axis=0)
                                Rate = np.sum(Rates, axis=0)
                                Hint = Ham * np.exp(-1j * Rate * t) * self.coupling_modulation_functions[0](t)
                            else:
                                Hint = np.zeros((self.size, self.size), dtype='complex')
                                for Ham, Rate, mod in zip(Hints, Rates, self.coupling_modulation_functions):
                                    Hint += Ham * np.exp(-1j * Rate * t) * mod(t)
                            Hint += Hint.conj().T

                            H0_shift = np.zeros_like(Hint)
                            if self.all_energy_shift_mods_are_equal: 
                                H0_shift = np.sum(H0_shifts, axis=0)
                                if self.energy_shift_modulation_functions[0] is not None :
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

        end = time.perf_counter()
        ic(f'Building Hamiltonian took {end-start} seconds.')

        return _hamiltonian_function

    # deprecated
    # @cached_property
    # def H0_Hints_Rates_Ones_and_mods(self):
    #     """
    #         The non-interacting Hamiltonian (H0) and, for each coupling, the interacting Hamiltonian (Hint),
    #         its corresponding oscillation rate matrix (Rate), its corresponding one matrix (One), and its corresponding
    #         modulation function.
    #     """
    #     H0 = csr_matrix(np.diag([energy if abs(energy) > SMALLEST_ENERGY_SCALE else 0 for energy in self.energies]))
    #     Hints = [csr_matrix(([0], ([0], [0])), shape=(self.size, self.size))]
    #     Rates = [csr_matrix(([0], ([0], [0])), shape=(self.size, self.size))]
    #     Ones = [csr_matrix(([0], ([0], [0])), shape=(self.size, self.size))]
    #     mods = []
    #     for coupling in self.couplings:
    #         assert(np.abs(coupling.strength) >= 0) # TODO: consider raising an IonSimError
    #         if np.abs(coupling.strength) < SMALLEST_ENERGY_SCALE: continue
    #         mods.append(coupling.modulation_function)
    #         for row, row_state in enumerate(self.basis.states):
    #             for column, column_state in enumerate(self.basis.states):
    #                 if (row_state, column_state) == (coupling.upper_state, coupling.lower_state):
    #                     Hints.append(csr_matrix(([coupling.strength], ([row], [column])), shape=(self.size, self.size)))
    #                     total_rate = (
    #                         + coupling.oscillation_rate
    #                         + self.rotating_frame_energies[row]
    #                         - self.rotating_frame_energies[column]
    #                     )
    #                     total_rate = total_rate if abs(total_rate) > SMALLEST_ENERGY_SCALE else 0
    #                     Rates.append(csr_matrix(([total_rate], ([row], [column])), shape=(self.size, self.size)))
    #                     Ones.append(csr_matrix(([1], ([row], [column])), shape=(self.size, self.size)))
    #                     ### [row, column] corresponds to phase factor next to raising operator: sigma^dagger exp[-i rate t]
    #     return H0, Hints, Rates, Ones, mods


    # deprecated
    # @cached_property
    # def hamiltonian_function(self): # TODO: perhaps deprecate in favor of "build_hamiltonian_function"
    #     """A function that computes the Hamiltonian at a specified time."""
    #     # TODO: the way time dependnce is added in here (in both Rates and mods) will not work if there's more than one 
    #     # coupling between the same two states (i.e. occupying the same matrix element, but with different time dependencies).
    #     H0, Hints, Rates, Ones = self.H0_Hints_Rates_and_Ones

    #     # ic([R.data/(2*np.pi*1e3) for R in Rates])

    #     Hint = np.sum(Hints, axis=0)
    #     Rate = np.sum(Rates, axis=0)

    #     Hint = Hint + Hint.conj().transpose()
    #     Rate = Rate - Rate.transpose()

    #     if Rate.getnnz() == 0:
    #         # H = H0 + Hint
    #         # if not self.sparse: H = H.toarray()
    #         if self.sparse:
    #             if all(mod is None for mod in self.modulation_functions):
    #                 def _hamiltonian_function(t: float):
    #                     return H0 + Hint
    #             else:
    #                 def _hamiltonian_function(t: float):
    #                     Mod = np.sum([One.multiply(mod(t)) for One, mod in zip(Ones, self.modulation_functions)], axis=0)
    #                     Mod = Mod + Mod.conj().transpose()
    #                     return H0 + Hint.multiply(Mod)
    #         else:
    #             H0, Hint = H0.toarray(), Hint.toarray()
    #             if all(mod is None for mod in self.modulation_functions):
    #                 def _hamiltonian_function(t: float):
    #                     return H0 + Hint
    #             else:
    #                 Ones = [One.toarray() for One in Ones]
    #                 def _hamiltonian_function(t: float):
    #                     Mod = np.sum([One * mod(t) for One, mod in zip(Ones, self.modulation_functions)], axis=0)
    #                     Mod += Mod.conj().T
    #                     return H0 + Hint * Mod

    #     else:
    #         if self.sparse:
    #             if all(mod is None for mod in self.modulation_functions):
    #                 def _hamiltonian_function(t: float):
    #                     phase_factor_minus_one = Rate.multiply(-1j*t).expm1() # equivalent to exp(-1j * Rate * t) - 1
    #                     return H0 + Hint.multiply(phase_factor_minus_one) + Hint
    #             else:
    #                 def _hamiltonian_function(t: float):
    #                     Mod = np.sum([One.multiply(mod(t)) for One, mod in zip(Ones, self.modulation_functions)], axis=0)
    #                     Mod += Mod.conj().transpose()
    #                     phase_factor_minus_one = Rate.multiply(-1j*t).expm1() # equivalent to exp(-1j * Rate * t) - 1
    #                     return H0 + (Hint.multiply(phase_factor_minus_one) + Hint).multiply(Mod)
    #         else:
    #             H0, Hint, Rate = H0.toarray(), Hint.toarray(), Rate.toarray()
    #             if all(mod is None for mod in self.modulation_functions):
    #                 def _hamiltonian_function(t: float):
    #                     return H0 + Hint * np.exp(-1j * Rate * t)
    #             else:
    #                 Ones = [One.toarray() for One in Ones]
    #                 def _hamiltonian_function(t: float):
    #                     # Mod = np.sum([One * mod(t) for One, mod in zip(Ones, self.modulation_functions)], axis=0)
    #                     Mod = Ones[0] # temp
    #                     Mod += Mod.conj().T
    #                     return H0 + Hint * np.exp(-1j * Rate * t) * Mod

    #     return _hamiltonian_function

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

    def evolve_supervector(self, initial_supervector: Vector, duration: float, time_evals: Vector | None = None,
        dissipation_matrix: AnyMatrix | None = None, **kwargs):
        """Evolve a supervector by solving the time-dependent Lindblad master equation."""
        # TODO: add suport for sparse matrices
        assert(self.size == np.sqrt(len(initial_supervector)))
        dissipation_matrix = as_dense_matrix(dissipation_matrix)
        super_ham = lambda t: (
            np.kron(np.eye(self.size), self.hamiltonian_function(t))
            - np.kron(self.hamiltonian_function(t).T, np.eye(self.size))
            )
        if dissipation_matrix is None:
            lindbladian_function = super_ham
        else:
            lindbladian_function = lambda t: super_ham(t) + 1j*dissipation_matrix # TODO: verify this line and decide how to define disspation matrix. perhaps we should use a disspator class?
        return solve_time_evolution_equation(lindbladian_function, initial_supervector, duration, time_evals, **kwargs)
