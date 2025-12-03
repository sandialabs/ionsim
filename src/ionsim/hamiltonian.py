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

        sparse_H0, sparse_Hints, sparse_Rates = self.H0_Hints_and_Rates
        H0_dense = np.array(as_dense_matrix(sparse_H0, warn=False), copy=True).astype(complex, copy=False)

        deterministic_components: list[dict[str, object]] = []
        stochastic_components: list[dict[str, object]] = []

        for operator, hint_matrix, rate_matrix in zip(self.coupling_operators, sparse_Hints, sparse_Rates):
            comp_hint = np.array(as_dense_matrix(hint_matrix, warn=False), copy=True).astype(complex, copy=False)
            comp_rate = np.array(as_dense_matrix(rate_matrix, warn=False), copy=True).astype(float, copy=False)
            has_rate = bool(np.count_nonzero(np.abs(comp_rate) > 0))

            component: dict[str, object] = {
                'hint': comp_hint,
                'rate': comp_rate,
                'has_rate': has_rate,
            }

            info = getattr(operator, 'stochastic_info', None) or {}
            if info:
                strength = info.get('strength', 1.0)
                if strength is None:
                    strength = 1.0
                strength = complex(strength)
                # Map operator to a noise source index (previously called "channel")
                default_noise_source = 0 if trajectory_noise.shape[0] == 1 else len(stochastic_components)
                noise_source_index = int(info.get('noise_source', info.get('noise_channel', default_noise_source)))
                if noise_source_index < 0 or noise_source_index >= trajectory_noise.shape[0]:
                    raise IonSimError(
                        f"Noise source {noise_source_index} not available for stochastic coupling (n_sources: {trajectory_noise.shape[0]})."
                    )
                # By default include the deterministic component at full strength
                deterministic_strength = info.get('deterministic_strength', info.get('mean_strength', 1.0))
                if deterministic_strength is None:
                    deterministic_strength = 1.0
                deterministic_strength = complex(deterministic_strength)
                offset = info.get('offset', info.get('bias', 0.0))
                if offset is None:
                    offset = 0.0
                offset = float(offset)
                # Optional: a bare operator to use for additive noise, excluding the deterministic prefactor.
                bare_op = info.get('bare_operator', None)
                bare_hint = None
                if bare_op is not None:
                    bare_hint = np.array(as_dense_matrix(bare_op, warn=False), copy=True).astype(complex, copy=False)
                component.update({
                    'strength': strength,
                    'noise_source': noise_source_index,
                    'deterministic_strength': deterministic_strength,
                    'offset': offset,
                    'bare_hint': bare_hint,
                })
                stochastic_components.append(component)
            else:
                deterministic_components.append(component)

        # At this point we already validated self.stochastic and trajectory_noise;
        # if no stochastic components were constructed, treat it as a misconfiguration.
        if not stochastic_components:
            raise IonSimError(
                'Stochastic evolution requested, but no stochastic components were constructed for this trajectory.\n'
                'Check coupling operator stochastic_info (channel mapping, strengths, thresholds).')

        def _evaluate_component(comp: dict[str, object], t: float) -> np.ndarray:
            hint_matrix = np.array(comp['hint'], copy=True)
            if comp['has_rate']:
                rate_matrix = comp['rate']
                hint_matrix *= np.exp(-1j * rate_matrix * t)
            return hint_matrix + hint_matrix.conj().T

        def _evaluate_with_hint(hint_like: np.ndarray, comp: dict[str, object], t: float) -> np.ndarray:
            mat = np.array(hint_like, copy=True)
            if comp['has_rate']:
                rate_matrix = comp['rate']
                mat *= np.exp(-1j * rate_matrix * t)
            return mat + mat.conj().T

        def _deterministic_matrix(t: float) -> np.ndarray:
            base = np.array(H0_dense, copy=True)
            for comp in deterministic_components:
                base += _evaluate_component(comp, t)
            for comp in stochastic_components:
                deterministic_strength = comp['deterministic_strength']
                if deterministic_strength:
                    base += deterministic_strength * _evaluate_component(comp, t)
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
            for comp in stochastic_components:
                # If a bare operator is provided, use it to build an additive-noise template
                # so that H_noise(t) = strength * noise(t) * (bare_template + h.c.).
                # Otherwise, default to scaling the full deterministic template (fractional amplitude noise).
                if comp.get('bare_hint') is not None:
                    template = _evaluate_with_hint(comp['bare_hint'], comp, t)
                else:
                    template = _evaluate_component(comp, t)
                noise_value = interpolate_noise(comp['noise_source'], t) + comp['offset']
                base_matrix += comp['strength'] * noise_value * template
            if self.sparse:
                return csr_matrix(base_matrix)
            return base_matrix

        return _stochastic_hamiltonian

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
        noisy_trajectories: Matrix | None = None, return_density_average: bool = True, **kwargs):
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
            base_solver_kwargs=base_solver_kwargs)

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
