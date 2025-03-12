from ionsim.basis import StandardBasis
from ionsim.coupling import Coupling, CouplingOperator
from ionsim.custom_types import Vector, Matrix
from ionsim.config import NUMERICAL_EQUIVALENCE_THRESHOLD, SMALLEST_ENERGY_SCALE
from ionsim.custom_math import solve_time_evolution_equation
from ionsim.ionsim_error import IonSimError

from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from typing import Callable
from functools import cached_property

from icecream import ic

@dataclass(frozen=True, eq=False)
class Hamiltonian:
    basis: StandardBasis
    coupling_operators: list[CouplingOperator]
    rotating_frame_energies: list[float]
    sparse: bool = False

    @property
    def energies(self):
        return [state.energy + energy for state, energy in zip(self.basis.states, self.rotating_frame_energies)]

    @property
    def size(self):
        return len(self.basis.states)

    @property
    def modulation_functions(self):
        return [operator.modulation_function for operator in self.coupling_operators]

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
    def all_rates_are_zero(self):
        sparse_H0, sparse_Hints, sparse_Rates = self.H0_Hints_and_Rates
        if all(Rate.getnnz() == 0 for Rate in sparse_Rates):
            return True
        else:
            return False

    @property
    def all_ints_are_isolated(self):
        sparse_H0, sparse_Hints, sparse_Rates = self.H0_Hints_and_Rates
        result = True
        for Hint in sparse_Hints:
            rows, cols = Hint.nonzero()
            for Hint_p in sparse_Hints:
                if Hint_p is not Hint:
                    rows_p, cols_p = Hint_p.nonzero()
                    if any((row, col) in zip(rows_p, cols_p) for row, col in zip(rows, cols)):
                        result = False
        return result


    @cached_property
    def H0_Hints_and_Rates(self):
        """
            The non-interacting Hamiltonian (H0) and, for each coupling operator, the interacting Hamiltonian (Hint),
            and its corresponding oscillation rate matrix (Rate).
        """
        # TODO: move conditions for eliminating couplings into the creation of the coupling_operators.
        H0 = csr_matrix(np.diag([energy if abs(energy) > SMALLEST_ENERGY_SCALE else 0 for energy in self.energies]))
        Hints = []
        Rates = []
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
        return H0, Hints, Rates

    @cached_property
    def hamiltonian_function(self): # TODO: perhaps deprecate in favor of "build_hamiltonian_function"
        """A function that computes the Hamiltonian at a specified time."""

        import time
        from icecream import ic

        start = time.perf_counter()

        sparse_H0, sparse_Hints, sparse_Rates = self.H0_Hints_and_Rates

        if self.sparse:
            H0 = sparse_H0
            Hints = sparse_Hints
            Rates = sparse_Rates
            # ic([Rate.data/(2*np.pi*1e3) for Rate in Rates])
        else:
            H0 = sparse_H0.toarray()
            Hints, Rates = [], []
            for Hint, Rate in zip(sparse_Hints, sparse_Rates):
                Hints.append(Hint.toarray())
                Rates.append(Rate.toarray())
            # ic([Rate/(2*np.pi*1e3) for Rate in Rates])

        if self.all_rates_are_zero:
            if self.sparse:
                if self.all_mods_are_none:
                    Hint = np.sum(Hints, axis=0)
                    Hint += Hint.conj().transpose()
                    def _hamiltonian_function(t: float):
                        return H0 + Hint
                else:
                    def _hamiltonian_function(t: float):
                        Hint = csr_matrix(([0], ([0], [0])), shape=(self.size, self.size), dtype='complex')
                        for ham, mod in zip(Hints, self.modulation_functions):
                            Hint += ham.multiply(mod(t))
                        Hint += Hint.conj().transpose()
                        return H0 + Hint
            else:
                if self.all_mods_are_none:
                    Hint = np.sum(Hints, axis=0)
                    Hint += Hint.conj().T
                    def _hamiltonian_function(t: float):
                        return H0 + Hint
                else:
                    def _hamiltonian_function(t: float):
                        Hint = np.zeros((self.size, self.size), dtype='complex')
                        for Ham, mod in zip(Hints, self.modulation_functions):
                            Hint += Ham * mod(t)
                        Hint += Hint.conj().T
                        return H0 + Hint
        else:
            if self.sparse:
                if self.all_mods_are_none:
                    def _hamiltonian_function(t: float):
                        Hint = csr_matrix(([0], ([0], [0])), shape=(self.size, self.size), dtype='complex') 
                        for Ham, Rate in zip(Hints, Rates):
                            phase_factor_minus_one = Rate.multiply(-1j*t).expm1() # equivalent to exp(-1j * Rate * t) - 1
                            Htemp = Ham + Ham.multiply(phase_factor_minus_one)
                            Hint += Htemp
                        Hint += Hint.conj().transpose()
                        return H0 + Hint
                else:
                    def _hamiltonian_function(t: float):
                        Hint = csr_matrix(([0], ([0], [0])), shape=(self.size, self.size), dtype='complex')
                        for Ham, Rate, mod in zip(Hints, Rates, self.modulation_functions):
                            phase_factor_minus_one = Rate.multiply(-1j*t).expm1() # equivalent to exp(-1j * Rate * t) - 1
                            Htemp = Ham + Ham.multiply(phase_factor_minus_one)
                            Hint += Htemp.multiply(mod(t))   
                        Hint += Hint.conj().transpose()
                        return H0 + Hint
            else:
                if self.all_mods_are_none:
                    if self.all_ints_are_isolated: # TODO: apply this check and simplification for other cases with nonzero Rates.
                        Ham = np.sum(Hints, axis=0)
                        Rate = np.sum(Rates, axis=0)
                        def _hamiltonian_function(t: float):
                            Hint = Ham * np.exp(-1j * Rate * t)
                            Hint += Hint.conj().T
                            return H0 + Hint
                    else:
                        def _hamiltonian_function(t: float):
                            Hint = np.zeros((self.size, self.size), dtype='complex')
                            for Ham, Rate in zip(Hints, Rates):
                                Hint += Ham * np.exp(-1j * Rate * t)
                            Hint += Hint.conj().T
                            return H0 + Hint
                else: # TODO: Check if each modulation is the same function, and sum ints outside of H(t). 
                    if self.all_ints_are_isolated and self.all_mods_are_equal:
                        Ham = np.sum(Hints, axis=0)
                        Rate = np.sum(Rates, axis=0)
                        def _hamiltonian_function(t: float):
                            Hint = Ham * np.exp(-1j * Rate * t) * self.modulation_functions[0](t)
                            Hint += Hint.conj().T
                            return H0 + Hint
                    else:
                        def _hamiltonian_function(t: float):
                            Hint = np.zeros((self.size, self.size), dtype='complex')
                            for Ham, Rate, mod in zip(Hints, Rates, self.modulation_functions):
                                Hint += Ham * np.exp(-1j * Rate * t) * mod(t)
                            Hint += Hint.conj().T
                            return H0 + Hint

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
        dissipation_matrix: Matrix | None = None, **kwargs):
        """Evolve a supervector by solving the time-dependent Lindblad master equation."""
        # TODO: add suport for sparse matrices
        assert(self.size == np.sqrt(len(initial_supervector)))
        from scipy.sparse import issparse
        if issparse(dissipation_matrix):
            dissipation_matrix = dissipation_matrix.toarray()
        super_ham = lambda t: (
            np.kron(np.eye(self.size), self.hamiltonian_function(t))
            - np.kron(self.hamiltonian_function(t).T, np.eye(self.size))
            )
        if dissipation_matrix is None:
            lindbladian_function = super_ham
        else:
            lindbladian_function = lambda t: super_ham(t) + 1j*dissipation_matrix # TODO: verify this line and decide how to define disspation matrix. perhaps we should use a disspator class?
        return solve_time_evolution_equation(lindbladian_function, initial_supervector, duration, time_evals, **kwargs)

def main():
    """Script to execute if module is ran directly."""
    import time
    from matplotlib import pyplot as plt
    from ionsim.degree_of_freedom import AtomicSpin, MotionalMode
    from ionsim.named_operators import Pauli

    spin_a = AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
    spin_b = AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
    basis = StandardBasis([spin_a, spin_b]) # 00, 01, 10, 11 

    rabi_rate = 100e3 * 2*np.pi # rad./s
    omega = spin_a.energy_levels[1].energy - spin_a.energy_levels[0].energy

    static_operator = rabi_rate/2 * Pauli.plus

    operator_a = CouplingOperator.from_matrix(basis, static_operator, omega, [spin_a])
    operator_b = CouplingOperator.from_matrix(basis, static_operator, omega, [spin_b])
    
    interaction_frame_energies = [-1*state.energy for state in basis.states]
    hamiltonian = Hamiltonian.from_operators(basis, [operator_a, operator_b], interaction_frame_energies, sparse=False)

    start = time.perf_counter()
    ic(hamiltonian.hamiltonian_function(0))
    end = time.perf_counter()
    ic(f'Building Hamiltonian took {end - start} s.')

    duration = np.pi / rabi_rate
    wavefunction = np.array([1, 0, 0, 0])
    times = np.linspace(0, duration, 21)

    start = time.perf_counter()
    ts, ys = hamiltonian.evolve_wavefunction(wavefunction, duration, times, 'odeintz')
    end = time.perf_counter()
    ic(f'Simulating gate took {end - start} s.')
    ic(ys[-1])

    for i in range(len(wavefunction)):
        plt.plot(ts, [np.abs(y[i])**2 for y in ys], label=i)
    plt.ylabel('Probabilities')
    plt.xlabel('Gate Duration (s)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()



