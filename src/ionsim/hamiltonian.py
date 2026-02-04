from ionsim.basis import StandardBasis
from ionsim.coupling import Coupling, CouplingOperator
from ionsim.custom_types import Vector, Matrix, SparseMatrix, AnyMatrix, as_dense_matrix
from ionsim.config import NUMERICAL_EQUIVALENCE_THRESHOLD, SMALLEST_ENERGY_SCALE
from ionsim.custom_math import solve_time_evolution_equation
from ionsim.ionsim_error import IonSimError

from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from typing import Callable, Any
from functools import cached_property

from icecream import ic


@dataclass(frozen=True)
class StochasticHamiltonianComponentData:
    """Dense data needed to evaluate stochastic Hamiltonians without Python callbacks."""

    H0: np.ndarray
    deterministic_hints: np.ndarray
    deterministic_rates: np.ndarray
    deterministic_has_rate: np.ndarray
    stochastic_hints: np.ndarray
    stochastic_rates: np.ndarray
    stochastic_has_rate: np.ndarray
    noise_strengths: np.ndarray
    noise_offsets: np.ndarray
    noise_source_indices: np.ndarray

@dataclass(frozen=True, eq=False)
class Hamiltonian:
    basis: StandardBasis
    coupling_operators: list[CouplingOperator]
    rotating_frame_energies: list[float]
    sparse: bool = False

    @property
    def stochastic(self) -> bool:
        """
        Returns True if any coupling operator has non-empty stochastic_param_info.
        """
        return any(
            hasattr(op, 'stochastic_info') and op.stochastic_info not in (None, {})
            for op in getattr(self, 'coupling_operators', [])
        )

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

        def is_diagonal_hermitian(mat):
            return np.allclose(mat, mat.conj().T)

        if self.all_rates_are_zero:
            if self.sparse:
                if self.all_mods_are_none:
                    Hint = np.sum(Hints, axis=0)
                    # Only hermitianize if not diagonal hermitian
                    if not is_diagonal_hermitian(Hint.toarray()):
                        Hint += Hint.conj().transpose()
                    def _hamiltonian_function(t: float):
                        return H0 + Hint
                else:
                    def _hamiltonian_function(t: float):
                        Hint = csr_matrix(([0], ([0], [0])), shape=(self.size, self.size), dtype='complex')
                        for ham, mod in zip(Hints, self.modulation_functions):
                            Hint += ham.multiply(mod(t))
                        # Only hermitianize if not diagonal hermitian
                        if not is_diagonal_hermitian(Hint.toarray()):
                            Hint += Hint.conj().transpose()
                        return H0 + Hint
            else:
                if self.all_mods_are_none:
                    Hint = np.sum(Hints, axis=0)
                    # Only hermitianize if not diagonal hermitian
                    if not is_diagonal_hermitian(Hint):
                        Hint += Hint.conj().T
                    def _hamiltonian_function(t: float):
                        return H0 + Hint
                else:
                    def _hamiltonian_function(t: float):
                        Hint = np.zeros((self.size, self.size), dtype='complex')
                        for Ham, mod in zip(Hints, self.modulation_functions):
                            Hint += Ham * mod(t)
                        # Only hermitianize if not diagonal hermitian
                        if not is_diagonal_hermitian(Hint):
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
                        if not is_diagonal_hermitian(Hint.toarray()):
                            Hint += Hint.conj().transpose()
                        return H0 + Hint
                else:
                    def _hamiltonian_function(t: float):
                        Hint = csr_matrix(([0], ([0], [0])), shape=(self.size, self.size), dtype='complex')
                        for Ham, Rate, mod in zip(Hints, Rates, self.modulation_functions):
                            phase_factor_minus_one = Rate.multiply(-1j*t).expm1() # equivalent to exp(-1j * Rate * t) - 1
                            Htemp = Ham + Ham.multiply(phase_factor_minus_one)
                            Hint += Htemp.multiply(mod(t))   
                        if not is_diagonal_hermitian(Hint.toarray()):
                            Hint += Hint.conj().transpose()
                        return H0 + Hint
            else:
                if self.all_mods_are_none:
                    if self.all_ints_are_isolated: # TODO: apply this check and simplification for other cases with nonzero Rates.
                        Ham = np.sum(Hints, axis=0)
                        Rate = np.sum(Rates, axis=0)
                        def _hamiltonian_function(t: float):
                            Hint = Ham * np.exp(-1j * Rate * t)
                            if not is_diagonal_hermitian(Hint):
                                Hint += Hint.conj().T
                            return H0 + Hint
                    else:
                        def _hamiltonian_function(t: float):
                            Hint = np.zeros((self.size, self.size), dtype='complex')
                            for Ham, Rate in zip(Hints, Rates):
                                Hint += Ham * np.exp(-1j * Rate * t)
                            if not is_diagonal_hermitian(Hint):
                                Hint += Hint.conj().T
                            return H0 + Hint
                else: # TODO: Check if each modulation is the same function, and sum ints outside of H(t). 
                    if self.all_ints_are_isolated and self.all_mods_are_equal:
                        Ham = np.sum(Hints, axis=0)
                        Rate = np.sum(Rates, axis=0)
                        def _hamiltonian_function(t: float):
                            Hint = Ham * np.exp(-1j * Rate * t) * self.modulation_functions[0](t)
                            if not is_diagonal_hermitian(Hint):
                                Hint += Hint.conj().T
                            return H0 + Hint
                    else:
                        def _hamiltonian_function(t: float):
                            Hint = np.zeros((self.size, self.size), dtype='complex')
                            for Ham, Rate, mod in zip(Hints, Rates, self.modulation_functions):
                                Hint += Ham * np.exp(-1j * Rate * t) * mod(t)
                            if not is_diagonal_hermitian(Hint):
                                Hint += Hint.conj().T
                            return H0 + Hint

        end = time.perf_counter()
        ic(f'Building Hamiltonian took {end-start} seconds.')

        return _hamiltonian_function
    
    def _build_stochastic_component_data(self, n_noise_sources: int) -> StochasticHamiltonianComponentData:
        """Extract dense component arrays describing deterministic and stochastic couplings.
        Enables fast, vectorized evaluation of the Hamiltonian during stochastic propagation"""
        sparse_H0, sparse_Hints, sparse_Rates = self.H0_Hints_and_Rates
        H0_dense = np.array(as_dense_matrix(sparse_H0, warn=False), copy=True).astype(np.complex128, copy=False)

        det_hints: list[np.ndarray] = []
        det_rates: list[np.ndarray] = []
        det_has_rate: list[bool] = []

        stoch_hints: list[np.ndarray] = []
        stoch_rates: list[np.ndarray] = []
        stoch_has_rate: list[bool] = []
        noise_strengths: list[complex] = []
        noise_offsets: list[float] = []
        noise_source_indices: list[int] = []

        for operator, hint_matrix, rate_matrix in zip(self.coupling_operators, sparse_Hints, sparse_Rates):
            comp_hint = np.array(as_dense_matrix(hint_matrix, warn=False), copy=True).astype(np.complex128, copy=False)
            # Ensure hermiticity once, here
            if not np.allclose(comp_hint, comp_hint.conj().T, atol=1e-10):
                comp_hint = (comp_hint + comp_hint.conj().T) / 2
            comp_rate = np.array(as_dense_matrix(rate_matrix, warn=False), copy=True).astype(float, copy=False)
            has_rate = bool(np.any(np.abs(comp_rate) > 0))

            stochastic_info = getattr(operator, 'stochastic_info', None) or {}
            if stochastic_info:
                if n_noise_sources <= 0:
                    raise IonSimError('Stochastic evolution requested but no noise sources were provided.')
                has_explicit_source = ('noise_source' in stochastic_info) or ('noise_channel' in stochastic_info)
                default_noise_source = 0 if n_noise_sources == 1 else len(stoch_hints)
                noise_source = int(stochastic_info.get('noise_source', stochastic_info.get('noise_channel', default_noise_source)))
                if (not has_explicit_source) and n_noise_sources > 1 and default_noise_source >= n_noise_sources:
                    raise IonSimError(
                        'Not enough noise sources were provided for the number of stochastic coupling operators. '
                        f'Expected at least {default_noise_source + 1} noise sources but got {n_noise_sources}. '
                        'Either provide more noise sources (one per stochastic operator by default) '
                        'or set stochastic_info["noise_source"] explicitly on each stochastic operator.'
                    )
                if noise_source < 0 or noise_source >= n_noise_sources:
                    raise IonSimError(
                        f"Noise source {noise_source} not available for stochastic coupling (n_sources: {n_noise_sources})."
                    )
                strength = stochastic_info.get('strength', 1.0)
                if strength is None:
                    strength = 1.0
                offset = stochastic_info.get('offset', stochastic_info.get('bias', 0.0))
                if offset is None:
                    offset = 0.0

                stoch_hints.append(comp_hint)
                stoch_rates.append(comp_rate)
                stoch_has_rate.append(has_rate)
                noise_strengths.append(complex(strength))
                noise_offsets.append(float(offset))
                noise_source_indices.append(noise_source)
            else:
                det_hints.append(comp_hint)
                det_rates.append(comp_rate)
                det_has_rate.append(has_rate)

        dim = H0_dense.shape[0]

        def _stack_or_empty(items: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
            if not items:
                return np.zeros((0, dim, dim), dtype=dtype)
            return np.stack([np.array(item, copy=False) for item in items]).astype(dtype, copy=False)

        deterministic_hint_array = _stack_or_empty(det_hints, np.complex128)
        deterministic_rate_array = _stack_or_empty(det_rates, float)
        deterministic_has_rate_array = np.array(det_has_rate, dtype=np.uint8)

        stochastic_hint_array = _stack_or_empty(stoch_hints, np.complex128)
        stochastic_rate_array = _stack_or_empty(stoch_rates, float)
        stochastic_has_rate_array = np.array(stoch_has_rate, dtype=np.uint8)

        return StochasticHamiltonianComponentData(
            H0=H0_dense,
            deterministic_hints=deterministic_hint_array,
            deterministic_rates=deterministic_rate_array,
            deterministic_has_rate=deterministic_has_rate_array,
            stochastic_hints=stochastic_hint_array,
            stochastic_rates=stochastic_rate_array,
            stochastic_has_rate=stochastic_has_rate_array,
            noise_strengths=np.array(noise_strengths, dtype=np.complex128),
            noise_offsets=np.array(noise_offsets, dtype=float),
            noise_source_indices=np.array(noise_source_indices, dtype=np.int64),
        )

    def stochastic_hamiltonian_function(self, time_evals: Vector, trajectory_noise: Matrix | None = None, **kwargs):
        """Build a stochastic Hamiltonian callable for a single noise trajectory with multiple noise sources."""
        # Require a single, trajectory noise matrix; orchestration layer (solver) handles batch formatting.

        # Fast-fail if the Hamiltonian has no stochastic couplings configured
        if not self.stochastic:
            raise IonSimError('Hamiltonian is not set up for stochastic evolution (no stochastic coupling operators found).' )

        if trajectory_noise is None:
            raise IonSimError('trajectory_noise must be provided as a 2D array (n_sources, n_time).')
        trajectory_noise = np.asarray(trajectory_noise)

        # Expect solver to provide per-trajectory noise as 2D (n_channels, n_time) for the k_th trajectory
        if trajectory_noise.ndim != 2:
            raise IonSimError('trajectory_noise must be a 2D array of shape (n_sources, n_time).')
        trajectory_noise = np.asarray(trajectory_noise, dtype=float)

        # Enforce explicit time_evals; use it directly as the interpolation grid.
        if time_evals is None:
            raise IonSimError('time_evals must be provided for stochastic Hamiltonian construction.')
        time_evals = np.asarray(time_evals, dtype=float)

        if trajectory_noise.shape[1] != time_evals.shape[0]:
            raise IonSimError('Noise trajectory length does not match the supplied time grid.')

        return_component_data = kwargs.pop('return_component_data', False)

        component_data = self._build_stochastic_component_data(trajectory_noise.shape[0])

        if component_data.stochastic_hints.shape[0] == 0:
            raise IonSimError(
                'Stochastic evolution requested, but no stochastic components were constructed for this trajectory.\n'
                'Check coupling operator stochastic_info (channel mapping, strengths, thresholds).')

        if return_component_data:
            # Provide dense arrays for backends that can operate without Python callables.
            return component_data

        det_hints = component_data.deterministic_hints
        det_rates = component_data.deterministic_rates
        det_has_rate = component_data.deterministic_has_rate

        stoch_hints = component_data.stochastic_hints
        stoch_rates = component_data.stochastic_rates
        stoch_has_rate = component_data.stochastic_has_rate
        noise_strengths = component_data.noise_strengths
        noise_offsets = component_data.noise_offsets
        noise_sources = component_data.noise_source_indices

        def _apply_phase(hint: np.ndarray, rate: np.ndarray, has_rate: int, t: float) -> np.ndarray:
            if has_rate:
                return hint * np.exp(-1j * rate * t)
            return hint

        def _deterministic_matrix(t: float) -> np.ndarray:
            base = np.array(component_data.H0, copy=True)
            for idx in range(det_hints.shape[0]):
                base += _apply_phase(det_hints[idx], det_rates[idx], det_has_rate[idx], t)
            return base

        def interpolate_noise(noise_source: int, t: float) -> float:
            """ This helper uses linear interpolation to fetch the noise value for a coupling operator 
            at any time t, ensuring the value matches the ODE solver’s requested time even if it doesn’t 
            align with the original noise sample grid.

            Inputs:
            - noise_source: index of the noise source (0 .. n_sources-1)
            - t: continuous evaluation time in seconds

            Returns:
            - float value of the noise process for the selected source at time t.
            """
            return float(np.interp(float(t), time_evals, trajectory_noise[noise_source]))

        def _stochastic_hamiltonian(t: float):
            base_matrix = _deterministic_matrix(t)
            for idx in range(stoch_hints.shape[0]):
                template = _apply_phase(stoch_hints[idx], stoch_rates[idx], stoch_has_rate[idx], t)
                noise_value = interpolate_noise(int(noise_sources[idx]), t) + noise_offsets[idx]
                base_matrix += noise_strengths[idx] * noise_value * template
            if self.sparse:
                return csr_matrix(base_matrix)
            return base_matrix

        return _stochastic_hamiltonian

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
    
    def evolve_stochastic_wavefunction(self, initial_wavefunction: Vector, time_evals: Vector | None = None,
        noisy_trajectories: Any = None, return_density_average: bool = True, **kwargs):
        assert(self.size == len(initial_wavefunction))
        import time
        from icecream import ic

        start = time.perf_counter()

        if not self.stochastic:
            raise IonSimError('Hamiltonian is not set up for stochastic evolution (no stochastic coupling operators found).')
        if noisy_trajectories is None:
            raise IonSimError('No noisy trajectories provided for stochastic evolution.')
        if not self.stochastic:
            raise IonSimError(
                "Hamiltonian is not set up for stochastic evolution: no stochastic coupling operator found (missing or empty stochastic_info). ")
        
        # Validate time grid for solver
        if time_evals is None:
            raise IonSimError('time_evals must be provided for stochastic evolution.')
        time_evals = np.asarray(time_evals, dtype=float)
        if time_evals.ndim != 1:
            time_evals = time_evals.reshape(-1)

        trajectory_backend = kwargs.pop('trajectory_backend', 'python')
        base_solver = kwargs.pop('base_solver', 'odeintz')
        base_solver_kwargs = kwargs.pop('base_solver_kwargs', {})
        if kwargs:
            base_solver_kwargs = {**base_solver_kwargs, **kwargs}
        # Provide a duration consistent with the time grid (used only when a solver needs it)
        duration = float(time_evals[-1] - time_evals[0]) if len(time_evals) > 0 else 0.0
        times, trajectory_results = solve_time_evolution_equation(
            self.stochastic_hamiltonian_function,
            initial_wavefunction,
            duration,
            time_evals,
            ode_solver='stochastic',
            noisy_trajectories=noisy_trajectories,
            base_solver=base_solver,
            base_solver_kwargs=base_solver_kwargs,
            trajectory_backend=trajectory_backend)
        
        # trajectory_results: shape (n_traj, n_time, dim)
        if return_density_average:
            # Build ensemble-averaged density matrices ρ_avg(t) = E[ |ψ_k(t)><ψ_k(t)| ]
            traj, n_time, dim = trajectory_results.shape
            rhos_avg: list[np.ndarray] = []
            for ti in range(n_time):
                psi_trajs = trajectory_results[:, ti, :]
                # Normalize each trajectory defensively to avoid norm drift
                norms = np.sqrt((psi_trajs.conj() * psi_trajs).sum(axis=1).real)
                norms[norms == 0] = 1.0
                psi_norm = psi_trajs / norms[:, None]
                # Outer products per trajectory, then average
                # rhos shape: (n_traj, dim, dim)
                rhos = psi_norm[:, :, None] * psi_norm.conj()[:, None, :]
                rho_avg = rhos.mean(axis=0)
                rhos_avg.append(rho_avg)
            result = rhos_avg
        else:
            # Backward-compatible path: average wavefunctions directly (can hide zero-mean noise)
            ensemble_wavefunctions = trajectory_results.mean(axis=0)
            result = [ensemble_wavefunctions[i] for i in range(len(ensemble_wavefunctions))]
        end = time.perf_counter()
        ic(f'Evolving wavefunction took {end-start} seconds.')
        return times, result

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
