from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from typing import Any, Callable
from scipy.integrate import trapezoid as trapz
import itertools as it
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from scipy.integrate import odeint, solve_ivp, ode
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse import kron as skron

try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _NUMBA_AVAILABLE = False

from icecream import ic

from ionsim.custom_types import Vector, AnyMatrix
from ionsim.ionsim_error import IonSimError

def matrix_AYB_multiply_to_superoperator(A: AnyMatrix | None, B: AnyMatrix | None=None) -> AnyMatrix:
    """Helper function to convert matrix multiplication to a superoperator form.
        Matrix Y can be flattened column-wise, mapping to a vector "y".

        Consider three-matrix product: A Y B ==> Oy

        A is a matrix multiplying a matrix of interest on the left.
        B is a matrix multiplying a matrix of interest on the right.

        A, Y, B are each N x N matrices,
            O is a N^2 x N^2 matrix,
            y is a column vector with N^2 entries.

        This function takes in A and B matrices and returns O.
        To compute O, the general formula is:
            A Y B --> (B^{T} kron A) y

    """
    #Note: np.kron silently fails for sparse matrix inputs; instead use kron from scipy.sparse
    if A is None and B is None:
        raise IonSimError('Input error: Specify either a left or right matrix A or B.')

    if A is not None:
        N = A.shape[0]

    if B is not None:
        N = B.shape[0]

    if A is not None and B is not None:
        assert N == A.shape[0]

    # Default behavior: If one matrix input is none, assume it is the identity.
    if A is None and B is not None:
        result = skron(B.T, np.eye(N))
    elif B is None and A is not None:
        result = skron(np.eye(N), A)
    elif A is not None and B is not None:
        result = skron(B.T, A)
    else:
        assert False, "A and B should not be None here"

    # If one or both matrices are sparse, return a sparse matrix
    if sparse.issparse(A) or sparse.issparse(B):
        return result
    else:
        return result.toarray()

def solve_time_evolution_equation(interaction_function: Callable, initial_state_vector: Vector, duration: float,
    time_evals: Vector | None = None, ode_solver: str = 'odeintz', **kwargs):
    """Solve the time-dependent Schrodinger equation or the vectorized Lindblad master equation."""
    print(f'Solving ODE with {ode_solver}.')
    if ode_solver == 'odeintz':
        return OdeIntz(interaction_function, initial_state_vector, duration, time_evals, **kwargs).solve()
    elif ode_solver == 'solve_ivp':
        return SolveIvp(interaction_function, initial_state_vector, duration, time_evals, **kwargs).solve()
    elif ode_solver == 'zvode':
        return ZVODE(interaction_function, initial_state_vector, duration, time_evals, **kwargs).solve()
    elif ode_solver == 'stochastic':
        return StochasticOdeSolver(interaction_function, initial_state_vector, duration, time_evals, **kwargs).solve()
    else:
        raise IonSimError(f'ODE solver {ode_solver} is not implemented.')

@dataclass(frozen=True, eq=False)
class OdeSolver(ABC):
    """A numerical routine to solve an ordinarty differential equation (ODE)."""
    # interaction_function: Callable
    interaction_function: Callable
    initial_vector: Vector
    duration: float
    time_evals: Vector | None

    @abstractmethod
    def solve(self):
        """Solves the ODE."""

@dataclass(frozen=True, eq=False)
class OdeIntz(OdeSolver):
    """A complex-valued version of Python's odeint routine."""
    def solve(self):
        """Solves the ODE."""
        def right_hand_side(t, y):
            return self.interaction_function(t).dot(-1j * y)
        def right_hand_side_flip_args(y, t):
            return right_hand_side(t, y)
        if self.time_evals is None:
            times = np.linspace(0, self.duration, 4)
        else:
            times = self.time_evals
        y0 = np.array(self.initial_vector, dtype='complex')
        result = odeintz(right_hand_side_flip_args, y0, times)
        return list(times), [y for y in result]

@dataclass(frozen=True, eq=False)
class SolveIvp(OdeSolver):
    """Python's solve_ivp routine."""
    def solve(self):
        """Solves the ODE."""
        def right_hand_side(t, y):
            return self.interaction_function(t).dot(-1j * y)
        y0 = np.array(self.initial_vector, dtype='complex')
        result = solve_ivp(right_hand_side, (0, self.duration), y0, t_eval=self.time_evals)
        return list(result['t']), [result['y'][:, i] for i in range(len(result['t']))]

@dataclass(frozen=True, eq=False)
class ZVODE(OdeSolver):
    """Python's zvode routine."""
    nsteps: float = 1e6

    def solve(self):
        """Solves the ODE."""
        if self.time_evals is None:
            num_steps = 3
        else:
            num_steps = len(time_evals)
            assert(time_evals[-1] == duration)

        ic(self.nsteps)

        n_states = len(self.initial_vector)
        hamiltonian = self.interaction_function
        t_final = self.duration
        initial_state = self.initial_vector

        if initial_state is None:
            initial_state = _np.zeros(n_states)
            initial_state[0] = 1.

        intermediate_states = [initial_state]
        intermediate_times = [0]
        def schrodinger(t, y):
            return  -1.0j * hamiltonian(t).dot(y)
        def jacobian(t, y):
            tempham = hamiltonian(t)
            if sparse.issparse(tempham):
                return -1.0j * tempham.todense()
            else:
                return -1.0j * tempham
        r = ode(schrodinger, jacobian)
        r.set_integrator('zvode', method='adams', with_jacobian=True, atol=1e-16, rtol=1e-14, nsteps=self.nsteps) # use method='bdf' for stiff ode
        r.set_initial_value(initial_state, 0)
        dt = t_final/float(num_steps)
        while r.successful() and r.t < t_final:
            r.integrate(r.t + dt)
            intermediate_states += [r.y]
            intermediate_times += [r.t]
        return intermediate_times, intermediate_states

@dataclass(frozen=True, eq=False)
class StochasticOdeSolver(OdeSolver):
    """A numerical routine to solve stochastic differential equations (SDEs) for wavefunction evolution."""
    noisy_trajectories: np.ndarray | None = None
    base_solver: str = 'odeintz'
    base_solver_kwargs: dict[str, Any] = field(default_factory=dict)
    trajectory_backend: str = 'python'
    # stochastic_params removed: all per-operator noise mapping comes from Hamiltonian.stochastic_info and per-trajectory noise inputs

    @staticmethod
    def _shape_noise_to_trajectories(noisy_trajectories: np.ndarray | None) -> np.ndarray:
        """
        Normalize input noise to a strict 3D layout (n_traj, n_sources, n_time).

        Accepted inputs:
        - 1D: (n_time) -> interpreted as a single trajectory with one noise source -> (1, 1, n_time)
        - 2D: (n_traj, n_time) -> single noise source per trajectory -> (n_traj, 1, n_time)
        - 3D: (n_traj, n_sources, n_time) -> used as-is

        This helper only reshapes; it does not validate semantic mapping between channels and operators.
        """
        if noisy_trajectories is None:
            raise IonSimError('No noise trajectories provided to the stochastic solver.')
        noise_array = np.asarray(noisy_trajectories)
        if noise_array.ndim == 1:
            noise_array = noise_array[np.newaxis, :]
        if noise_array.ndim == 2:
            noise_array = noise_array[:, np.newaxis, :]
        if noise_array.ndim != 3:
            raise IonSimError('noisy_trajectories must have shape (n_traj, n_time) or (n_traj, n_sources, n_time).')
        return noise_array

    def solve(self):
        """Solve the stochastic Schrödinger equation by evolving each trajectory independently."""
        noise_array = self._shape_noise_to_trajectories(self.noisy_trajectories)
        n_trajectories, _, n_time = noise_array.shape
        if n_trajectories == 0:
            raise IonSimError('noisy_trajectories must contain at least one trajectory.')

        if self.base_solver == 'stochastic':
            raise IonSimError('base_solver for StochasticOdeSolver cannot be "stochastic".')

        if self.time_evals is None:
            time_grid = np.linspace(0.0, self.duration, n_time)
        else:
            time_grid = np.asarray(self.time_evals, dtype=float)
            if time_grid.shape[0] != n_time:
                raise IonSimError('Length of time_evals must match the number of noise samples per trajectory.')

        backend = self.trajectory_backend.lower()
        if backend not in {"python", "numba_rk4", "numba_rk5", "numba_general_propagator"}:
            raise IonSimError(
                f'Unknown trajectory_backend "{self.trajectory_backend}"'
            )

        solver_kwargs = dict(self.base_solver_kwargs)

        component_data = self.interaction_function(
                initial_wavefunction=self.initial_vector,
                duration=self.duration,
                time_evals=time_grid,
                trajectory_noise=noise_array[0],
                noise_times=time_grid,
                return_component_data=True,
            )

        if backend == "numba_rk4":

            stacked_results = _run_stochastic_trajectories_numba(
                noise_array,
                np.asarray(self.initial_vector, dtype=np.complex128, order='C'),
                time_grid,
                component_data,
                method='RK4',
            )
        elif backend == "numba_rk5":

            stacked_results = _run_stochastic_trajectories_numba(
                noise_array,
                np.asarray(self.initial_vector, dtype=np.complex128, order='C'),
                time_grid,
                component_data,
                method='RK5',
            )
        elif backend == "numba_general_propagator":

            stacked_results = _run_stochastic_trajectories_numba(
                noise_array,
                np.asarray(self.initial_vector, dtype=np.complex128, order='C'),
                time_grid,
                component_data,
                method='general_propagator',
            )


        elif backend == 'python':
            stacked_results = parallel_trajectory_ode_solver(
                n_trajectories,
                solver_kwargs,
                time_grid,
                noise_array,
                interaction_function=self.interaction_function,
                initial_vector=self.initial_vector,
                duration=self.duration,
                base_solver=self.base_solver,
            )
        else:
            raise IonSimError(f'Unknown trajectory_backend "{self.trajectory_backend}".')

        return list(time_grid), stacked_results

def parallel_trajectory_ode_solver(
        n_trajectories: int,
        solver_kwargs: dict[str, Any],
        time_grid: np.ndarray,
        noise_array: np.ndarray,
        interaction_function: Callable,
        initial_vector: Vector,
        duration: float,
        base_solver: str,
    ) -> np.ndarray:
    """Solve each trajectory independently, optionally in parallel.

    Notes (Windows / notebooks): multiprocessing uses spawn, which requires the
    passed callables to be picklable. If pickling fails, we fall back to a
    serial loop to keep behavior correct.
    """
    if n_trajectories <= 0:
        raise IonSimError('n_trajectories must be a positive integer.')

    processes = os.cpu_count() or 1

    # Prepare per-trajectory work items.
    work_items = [
        (
            noise_array[trajectory_index],
            interaction_function,
            initial_vector,
            duration,
            time_grid,
            base_solver,
            solver_kwargs,
        )
        for trajectory_index in range(n_trajectories)
    ]

    try:
        with mp.Pool(processes=processes) as pool:
            results = pool.starmap(_solve_single_stochastic_trajectory, work_items)
        print("Using multiprocessing with", processes, "processes for stochastic trajectories.")
    except Exception as exc:
        # Most common cause: interaction_function (or one of its captured objects)
        # is not picklable under spawn. Keep correctness by falling back to serial.
        print(f"Multiprocessing failed ({type(exc).__name__}: {exc}); falling back to serial.")
        return _serial_trajectory_ode_solver(
            n_trajectories,
            solver_kwargs,
            time_grid,
            noise_array,
            interaction_function,
            initial_vector,
            duration,
            base_solver,
        )

    trajectory_vectors: list[np.ndarray] = []
    for times, state_vectors in results:
        if len(times) != len(time_grid) or not np.allclose(times, time_grid, atol=1e-12, rtol=1e-9):
            raise IonSimError('All stochastic trajectories must share the same integration time grid.')
        trajectory_vectors.append(state_vectors)
    return np.stack(trajectory_vectors, axis=0)


def _serial_trajectory_ode_solver(
    n_trajectories: int,
    solver_kwargs: dict[str, Any],
    time_grid: np.ndarray,
    noise_array: np.ndarray,
    interaction_function: Callable,
    initial_vector: Vector,
    duration: float,
    base_solver: str,
) -> np.ndarray:
    trajectory_vectors: list[np.ndarray] = []
    for trajectory_index in range(n_trajectories):
        trajectory_noise = noise_array[trajectory_index]
        times, state_vectors = _solve_single_stochastic_trajectory(
            trajectory_noise,
            interaction_function,
            initial_vector,
            duration,
            time_grid,
            base_solver,
            solver_kwargs,
        )
        if len(times) != len(time_grid) or not np.allclose(times, time_grid, atol=1e-12, rtol=1e-9):
            raise IonSimError('All stochastic trajectories must share the same integration time grid.')
        trajectory_vectors.append(state_vectors)
    return np.stack(trajectory_vectors, axis=0)


def _solve_single_stochastic_trajectory(
    trajectory_noise: np.ndarray,
    interaction_function: Callable,
    initial_vector: Vector,
    duration: float,
    time_grid: np.ndarray,
    base_solver: str,
    solver_kwargs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a single stochastic trajectory using the configured base solver."""
    stochastic_builder = interaction_function(
        initial_wavefunction=initial_vector,
        duration=duration,
        time_evals=time_grid,
        trajectory_noise=trajectory_noise,
        noise_times=time_grid,
    )
    inner_kwargs = dict(solver_kwargs)
    times, state_vectors = solve_time_evolution_equation(
        stochastic_builder,
        initial_vector,
        duration,
        time_grid,
        ode_solver=base_solver,
        **inner_kwargs,
    )
    times_array = np.asarray(times, dtype=float)
    if len(times_array) != len(time_grid) or not np.allclose(times_array, time_grid, atol=1e-12, rtol=1e-9):
        raise IonSimError('All stochastic trajectories must share the same integration time grid.')
    stacked_states = np.stack(state_vectors, axis=0)
    return times_array, stacked_states


def _run_stochastic_trajectories_numba(
    noise_array: np.ndarray,
    initial_vector: np.ndarray,
    time_grid: np.ndarray,
    component_data: Any,
    method: str = 'RK4',
) -> np.ndarray:
    """Execute stochastic trajectories using a fully nopython Runge-Kutta integrator."""
    if not _NUMBA_AVAILABLE:
        raise IonSimError('trajectory_backend="numba" requested but numba is not installed.')

    noise_array_f64 = np.ascontiguousarray(noise_array, dtype=np.float64)
    time_grid_f64 = np.ascontiguousarray(np.asarray(time_grid, dtype=np.float64))
    initial_vec_c = np.ascontiguousarray(initial_vector, dtype=np.complex128)

    H_det = np.ascontiguousarray(component_data.H_det, dtype=np.complex128)
    det_hints = np.ascontiguousarray(component_data.deterministic_hints, dtype=np.complex128)
    det_rates = np.ascontiguousarray(component_data.deterministic_rates, dtype=np.float64)
    det_has_rate = np.ascontiguousarray(component_data.deterministic_has_rate, dtype=np.uint8)

    stoch_hints = np.ascontiguousarray(component_data.stochastic_hints, dtype=np.complex128)
    stoch_rates = np.ascontiguousarray(component_data.stochastic_rates, dtype=np.float64)
    stoch_has_rate = np.ascontiguousarray(component_data.stochastic_has_rate, dtype=np.uint8)

    noise_strengths = np.ascontiguousarray(component_data.noise_strengths, dtype=np.complex128)
    noise_offsets = np.ascontiguousarray(component_data.noise_offsets, dtype=np.float64)
    noise_sources = np.ascontiguousarray(component_data.noise_source_indices, dtype=np.int64)

    noise_transformation_types = np.ascontiguousarray(component_data.noise_transformation_types, dtype=np.int32)
    noise_transformation_params = np.ascontiguousarray(component_data.noise_transformation_params, dtype=np.float64)
    all_rates_zero = np.uint8(np.all(det_has_rate == 0) and np.all(stoch_has_rate == 0))

    if method == 'RK4':
        return _run_stochastic_trajectories_numba_RK4(
            noise_array_f64,
            initial_vec_c,
            time_grid_f64,
            H_det,
            all_rates_zero,
            det_hints,
            det_rates,
            det_has_rate,
            stoch_hints,
            stoch_rates,
            stoch_has_rate,
            noise_strengths,
            noise_offsets,
            noise_sources,
            noise_transformation_types,
            noise_transformation_params,
        )
    elif method == 'RK5':
        return _run_stochastic_trajectories_numba_RK5(
            noise_array_f64,
            initial_vec_c,
            time_grid_f64,
            H_det,
            det_hints,
            det_rates,
            det_has_rate,
            stoch_hints,
            stoch_rates,
            stoch_has_rate,
            noise_strengths,
            noise_offsets,
            noise_sources,
            noise_transformation_types,
            noise_transformation_params,
        )
    elif method == 'general_propagator':
        return _run_stochastic_trajectories_numba_general_propagator(
            noise_array_f64,
            initial_vec_c,
            time_grid_f64,
            H_det,
            det_hints,
            det_rates,
            det_has_rate,
            stoch_hints,
            stoch_rates,
            stoch_has_rate,
            noise_strengths,
            noise_offsets,
            noise_sources,
            noise_transformation_types,
            noise_transformation_params,
        )
    else:
        raise IonSimError(f'Unknown numba integration method "{method}".')


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _numba_linear_interp(time_grid: np.ndarray, values: np.ndarray, t: float) -> float:
        if t <= time_grid[0]:
            return values[0]
        if t >= time_grid[-1]:
            return values[-1]
        idx = np.searchsorted(time_grid, t)
        if idx == 0:
            return values[0]
        t0 = time_grid[idx - 1]
        t1 = time_grid[idx]
        v0 = values[idx - 1]
        v1 = values[idx]
        if t1 == t0:
            return v1
        return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    
    @njit(cache=True)
    def _get_diag_matrix(H):
        n = H.shape[0]
        D = np.zeros_like(H)
        for i in range(n):
            D[i, i] = H[i, i]
        return D

    @njit(cache=True)
    def _numba_evaluate_hermitian(hint: np.ndarray, rate: np.ndarray, has_rate: int, t: float) -> np.ndarray:
        if has_rate != 0:
            mat = hint * np.exp(-1j * rate * t)
        else:
            mat = hint

        # Check for hermiticity
        if np.allclose(mat, mat.conj().T, atol=1e-10):
            return mat

        return mat + mat.conj().T

    @njit(cache=True)
    def _numba_build_hamiltonian(
        t: float,
        step: int,
        traj_idx: int,
        noise_array: np.ndarray,
        time_grid: np.ndarray,
        H_det: np.ndarray, #TODO name it H_det
        det_hints: np.ndarray,
        det_rates: np.ndarray,
        det_has_rate: np.ndarray,
        stoch_hints: np.ndarray,
        stoch_rates: np.ndarray,
        stoch_has_rate: np.ndarray,
        noise_strengths: np.ndarray,
        noise_offsets: np.ndarray,
        noise_source_indices: np.ndarray,
        noise_transformation_types: np.ndarray,
        noise_transformation_params: np.ndarray,
        interpolate: bool = True,
    ) -> np.ndarray:
        H = H_det.copy()
        for idx in range(det_hints.shape[0]):
            if det_has_rate[idx] != 0:
                herm = det_hints[idx] * np.exp(-1j * det_rates[idx] * t)
            else:
                herm = det_hints[idx]
            H += herm
        for idx in range(stoch_hints.shape[0]):
            if stoch_has_rate[idx] != 0:
                template = stoch_hints[idx] * np.exp(-1j * stoch_rates[idx] * t)
            else:
                template = stoch_hints[idx]
            source_index = int(noise_source_indices[idx])
            if interpolate:
                noise_values = noise_array[traj_idx, source_index]
                noise_val = _numba_linear_interp(time_grid, noise_values, t) + noise_offsets[idx]
            else:
                noise_val = noise_array[traj_idx, source_index, step] + noise_offsets[idx]
            
            # Apply noise transformation
            trans_type = noise_transformation_types[idx]
            if trans_type == 0:  # linear
                noise_factor = noise_val
            elif trans_type == 1:  # exponential
                noise_factor = np.exp(1j * noise_val * noise_transformation_params[idx])
            else:
                noise_factor = noise_val  # default to linear
            
            H += noise_strengths[idx] * noise_factor * template
        
        # TODO mixed Hermitian: ensure the Hamiltonian is Hermitian by adding its conjugate transpose and removing double-counted diagonal
        # if hermicity == 1: # non-Hermitian
        #     H = H + H.conj().T
        # elif hermicity == 2: # mixed Hermitian
        #     H = H + H.conj().T - _get_diag_matrix(H)
        # else:
        #     pass # hermicity == 0, assume H is already Hermitian and do nothing

        H = H + np.conj(H).T - _get_diag_matrix(H)
        
        return H

    @njit(cache=True)
    def _numba_rhs(
        t: float,
        psi: np.ndarray,
        step: int,
        traj_idx: int,
        noise_array: np.ndarray,
        time_grid: np.ndarray,
        H_det: np.ndarray,
        det_hints: np.ndarray,
        det_rates: np.ndarray,
        det_has_rate: np.ndarray,
        stoch_hints: np.ndarray,
        stoch_rates: np.ndarray,
        stoch_has_rate: np.ndarray,
        noise_strengths: np.ndarray,
        noise_offsets: np.ndarray,
        noise_source_indices: np.ndarray,
        noise_transformation_types: np.ndarray,
        noise_transformation_params: np.ndarray,
        interpolate: bool = True,
    ) -> np.ndarray:
        H = _numba_build_hamiltonian(
            t,
            step,
            traj_idx,
            noise_array,
            time_grid,
            H_det,
            det_hints,
            det_rates,
            det_has_rate,
            stoch_hints,
            stoch_rates,
            stoch_has_rate,
            noise_strengths,
            noise_offsets,
            noise_source_indices,
            noise_transformation_types,
            noise_transformation_params,
            interpolate,
        )
        return -1j * H.dot(psi)

    @njit(cache=True)
    def _numba_rhs_from_hamiltonian(
        H: np.ndarray,
        psi: np.ndarray,
    ) -> np.ndarray:
        return -1j * H.dot(psi)

    @njit(cache=True)
    def _numba_rk4_step(
        t0: float,
        dt: float,
        step: int,
        psi: np.ndarray,
        traj_idx: int,
        noise_array: np.ndarray,
        time_grid: np.ndarray,
        H_det: np.ndarray,
        all_rates_zero: np.uint8,
        det_hints: np.ndarray,
        det_rates: np.ndarray,
        det_has_rate: np.ndarray,
        stoch_hints: np.ndarray,
        stoch_rates: np.ndarray,
        stoch_has_rate: np.ndarray,
        noise_strengths: np.ndarray,
        noise_offsets: np.ndarray,
        noise_source_indices: np.ndarray,
        noise_transformation_types: np.ndarray,
        noise_transformation_params: np.ndarray,
    ) -> np.ndarray:
        if all_rates_zero != 0:
            H_step = _numba_build_hamiltonian(
                t0,
                step,
                traj_idx,
                noise_array,
                time_grid,
                H_det,
                det_hints,
                det_rates,
                det_has_rate,
                stoch_hints,
                stoch_rates,
                stoch_has_rate,
                noise_strengths,
                noise_offsets,
                noise_source_indices,
                noise_transformation_types,
                noise_transformation_params,
                interpolate=False,
            )
            k1 = _numba_rhs_from_hamiltonian(H_step, psi)
            k2 = _numba_rhs_from_hamiltonian(H_step, psi + 0.5 * dt * k1)
            k3 = _numba_rhs_from_hamiltonian(H_step, psi + 0.5 * dt * k2)
            k4 = _numba_rhs_from_hamiltonian(H_step, psi + dt * k3)
            return psi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        k1 = _numba_rhs(
            t0,
            psi,
            step,
            traj_idx,
            noise_array,
            time_grid,
            H_det,
            det_hints,
            det_rates,
            det_has_rate,
            stoch_hints,
            stoch_rates,
            stoch_has_rate,
            noise_strengths,
            noise_offsets,
            noise_source_indices,
            noise_transformation_types,
            noise_transformation_params,
            interpolate=False,
        )
        k2 = _numba_rhs(
            t0 + 0.5 * dt,
            psi + 0.5 * dt * k1,
            step,
            traj_idx,
            noise_array,
            time_grid,
            H_det,
            det_hints,
            det_rates,
            det_has_rate,
            stoch_hints,
            stoch_rates,
            stoch_has_rate,
            noise_strengths,
            noise_offsets,
            noise_source_indices,
            noise_transformation_types,
            noise_transformation_params,
            interpolate=False,
        )
        k3 = _numba_rhs(
            t0 + 0.5 * dt,
            psi + 0.5 * dt * k2,
            step,
            traj_idx,
            noise_array,
            time_grid,
            H_det,
            det_hints,
            det_rates,
            det_has_rate,
            stoch_hints,
            stoch_rates,
            stoch_has_rate,
            noise_strengths,
            noise_offsets,
            noise_source_indices,
            noise_transformation_types,
            noise_transformation_params,
            interpolate=False,
        )
        k4 = _numba_rhs(
            t0 + dt,
            psi + dt * k3,
            step,
            traj_idx,
            noise_array,
            time_grid,
            H_det,
            det_hints,
            det_rates,
            det_has_rate,
            stoch_hints,
            stoch_rates,
            stoch_has_rate,
            noise_strengths,
            noise_offsets,
            noise_source_indices,
            noise_transformation_types,
            noise_transformation_params,
            interpolate=False,
        )
        return psi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    @njit(parallel=True, cache=True)
    def _run_stochastic_trajectories_numba_RK4(
        noise_array: np.ndarray,
        initial_vector: np.ndarray,
        time_grid: np.ndarray,
        H_det: np.ndarray,
        all_rates_zero: np.uint8,
        det_hints: np.ndarray,
        det_rates: np.ndarray,
        det_has_rate: np.ndarray,
        stoch_hints: np.ndarray,
        stoch_rates: np.ndarray,
        stoch_has_rate: np.ndarray,
        noise_strengths: np.ndarray,
        noise_offsets: np.ndarray,
        noise_source_indices: np.ndarray,
        noise_transformation_types: np.ndarray,
        noise_transformation_params: np.ndarray,
    ) -> np.ndarray:
        n_traj = noise_array.shape[0]
        n_time = time_grid.shape[0]
        n_state = initial_vector.shape[0]
        result = np.empty((n_traj, n_time, n_state), dtype=np.complex128)

        for traj_idx in prange(n_traj):
            psi = initial_vector.copy()
            result[traj_idx, 0, :] = psi
            for step in range(n_time - 1):
                t0 = time_grid[step]
                t1 = time_grid[step + 1]
                dt = t1 - t0
                psi = _numba_rk4_step(
                    t0,
                    dt,
                    step,
                    psi,
                    traj_idx,
                    noise_array,
                    time_grid,
                    H_det,
                    all_rates_zero,
                    det_hints,
                    det_rates,
                    det_has_rate,
                    stoch_hints,
                    stoch_rates,
                    stoch_has_rate,
                    noise_strengths,
                    noise_offsets,
                    noise_source_indices,
                    noise_transformation_types,
                    noise_transformation_params,
                )
                result[traj_idx, step + 1, :] = psi
        return result
    # ============ End RK4 numba implementation ============

    # ============ RK5 numba implementation ============
    @njit(cache=True)
    def _numba_rk5_step(
        t0: float,
        dt: float,
        step: int,
        psi: np.ndarray,
        traj_idx: int,
        noise_array: np.ndarray,
        time_grid: np.ndarray,
        H_det: np.ndarray,
        det_hints: np.ndarray,
        det_rates: np.ndarray,
        det_has_rate: np.ndarray,
        stoch_hints: np.ndarray,
        stoch_rates: np.ndarray,
        stoch_has_rate: np.ndarray,
        noise_strengths: np.ndarray,
        noise_offsets: np.ndarray,
        noise_source_indices: np.ndarray,
        noise_transformation_types: np.ndarray,
        noise_transformation_params: np.ndarray,
    ) -> np.ndarray:
        k1 = _numba_rhs(
            t0,
            psi,
            step,
            traj_idx,
            noise_array,
            time_grid,
            H_det,
            det_hints,
            det_rates,
            det_has_rate,
            stoch_hints,
            stoch_rates,
            stoch_has_rate,
            noise_strengths,
            noise_offsets,
            noise_source_indices,
            noise_transformation_types,
            noise_transformation_params,
            interpolate=True,
        )
        k2 = _numba_rhs(
            t0 + dt / 5.0,
            psi + dt * (1.0/5.0) * k1,
            step,
            traj_idx,
            noise_array,
            time_grid,
            H_det,
            det_hints,
            det_rates,
            det_has_rate,
            stoch_hints,
            stoch_rates,
            stoch_has_rate,
            noise_strengths,
            noise_offsets,
            noise_source_indices,
            noise_transformation_types,
            noise_transformation_params,
            interpolate=True,
        )
        k3 = _numba_rhs(
            t0 + 3.0 * dt / 10.0,
            psi + dt * (3.0/40.0 * k1 + 9.0/40.0 * k2),
            step,
            traj_idx,
            noise_array,
            time_grid,
            H_det,
            det_hints,
            det_rates,
            det_has_rate,
            stoch_hints,
            stoch_rates,
            stoch_has_rate,
            noise_strengths,
            noise_offsets,
            noise_source_indices,
            noise_transformation_types,
            noise_transformation_params,
            interpolate=True,
        )
        k4 = _numba_rhs(
            t0 + 3.0 * dt / 5.0,
            psi + dt * (3.0/10.0 * k1 - 9.0/10.0 * k2 + 6.0/5.0 * k3),
            step,
            traj_idx,
            noise_array,
            time_grid,
            H_det,
            det_hints,
            det_rates,
            det_has_rate,
            stoch_hints,
            stoch_rates,
            stoch_has_rate,
            noise_strengths,
            noise_offsets,
            noise_source_indices,
            noise_transformation_types,
            noise_transformation_params,
            interpolate=True,
        )
        k5 = _numba_rhs(
            t0 + dt,
            psi + dt * (-11.0/54.0 * k1 + 5.0/2.0 * k2 - 70.0/27.0 * k3 + 35.0/27.0 * k4),
            step,
            traj_idx,
            noise_array,
            time_grid,
            H_det,
            det_hints,
            det_rates,
            det_has_rate,
            stoch_hints,
            stoch_rates,
            stoch_has_rate,
            noise_strengths,
            noise_offsets,
            noise_source_indices,
            noise_transformation_types,
            noise_transformation_params,
            interpolate=True,
        )
        k6 = _numba_rhs(
            t0 + 7.0 * dt / 8.0,
            psi + dt * (1631.0/55296.0 * k1 + 175.0/512.0 * k2 + 575.0/13824.0 * k3 + 44275.0/110592.0 * k4 + 253.0/4096.0 * k5),
            step,
            traj_idx,
            noise_array,
            time_grid,
            H_det,
            det_hints,
            det_rates,
            det_has_rate,
            stoch_hints,
            stoch_rates,
            stoch_has_rate,
            noise_strengths,
            noise_offsets,
            noise_source_indices,
            noise_transformation_types,
            noise_transformation_params,
            interpolate=True,
        )
        # Cash-Karp 5th order weights
        return psi + dt * (37.0/378.0 * k1 + 250.0/621.0 * k3 + 125.0/594.0 * k4 + 512.0/1771.0 * k6)

    @njit(parallel=True, cache=True)
    def _run_stochastic_trajectories_numba_RK5(
        noise_array: np.ndarray,
        initial_vector: np.ndarray,
        time_grid: np.ndarray,
        H_det: np.ndarray,
        det_hints: np.ndarray,
        det_rates: np.ndarray,
        det_has_rate: np.ndarray,
        stoch_hints: np.ndarray,
        stoch_rates: np.ndarray,
        stoch_has_rate: np.ndarray,
        noise_strengths: np.ndarray,
        noise_offsets: np.ndarray,
        noise_source_indices: np.ndarray,
        noise_transformation_types: np.ndarray,
        noise_transformation_params: np.ndarray,
    ) -> np.ndarray:
        n_traj = noise_array.shape[0]
        n_time = time_grid.shape[0]
        n_state = initial_vector.shape[0]
        result = np.empty((n_traj, n_time, n_state), dtype=np.complex128)

        for traj_idx in prange(n_traj):
            psi = initial_vector.copy()
            result[traj_idx, 0, :] = psi
            for step in range(n_time - 1):
                t0 = time_grid[step]
                t1 = time_grid[step + 1]
                dt = t1 - t0
                psi = _numba_rk5_step(
                    t0,
                    dt,
                    step,
                    psi,
                    traj_idx,
                    noise_array,
                    time_grid,
                    H_det,
                    det_hints,
                    det_rates,
                    det_has_rate,
                    stoch_hints,
                    stoch_rates,
                    stoch_has_rate,
                    noise_strengths,
                    noise_offsets,
                    noise_source_indices,
                    noise_transformation_types,
                    noise_transformation_params,
                )
                result[traj_idx, step + 1, :] = psi
        return result
    # ============ End RK5 numba implementation ============

    # ============ Single-qubit specialized evolution ============
    @njit(parallel=True, fastmath=True, cache=True)
    def evolve_batch_numba_general_single_qubit(noise_all, dt, base_pauli_coeffs, noise_pauli_coeffs, base_c, noise_c, psi_init, meas_obs):
        """
        Evolve a single qubit state under a stochastic Hamiltonian H = c I + (1/2) (c_x X + c_y Y + c_z Z),
        where c = base_c + noise * noise_c, and c_x = base_pauli_coeffs[0] + noise * noise_pauli_coeffs[0], etc.,
        and cos(1/2 Ω), i sin(1/2 Ω) are absorbed into the above coefficients.
        noise_all: (trajs, N) array of noise values
        dt: time step
        base_pauli_coeffs: (3,) array [c_x_base, c_y_base, c_z_base]
        noise_pauli_coeffs: (3,) array [c_x_noise, c_y_noise, c_z_noise]
        base_c: float, base coefficient for identity
        noise_c: float, noise coefficient for identity
        psi_init: (2,) complex array initial state
        meas_obs: (2,2) complex array observable to measure
        Returns: obs (trajs, N) expectation values

        Example usage:
        Ω = 2 * np.pi * 1e6
        φ = 0.0
        cφ, sφ = np.cos(φ), np.sin(φ)
        base_cx, base_cy, base_cz = (Ω/2) * cφ, (Ω/2) * sφ, 0.0
        noise_cx, noise_cy, noise_cz = (1/2) * cφ, (1/2) * sφ, 0.0
        base_c, noise_c = 0.0, 0.0
        base_pauli_coeffs = np.array([ base_cx, base_cy, base_cz ], dtype=np.float64)
        noise_pauli_coeffs = np.array([ noise_cx, noise_cy, noise_cz ], dtype=np.float64)
        meas_obs = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)  # Z observable
        psi0_init = 1.0 + 0.0j
        psi1_init = 0.0 + 0.0j
        psi_init = np.array([psi0_init, psi1_init], dtype=np.complex128)
        obs = evolve_batch_numba_general_single_qubit(noise_all, dt, base_pauli_coeffs, noise_pauli_coeffs, base_c, noise_c, psi_init, meas_obs)
        """

        trajs, N = noise_all.shape
        obs = np.empty((trajs, N), dtype=np.float64)

        for t_idx in prange(trajs):
            # initialize state |psi>
            psi0, psi1 = psi_init  # complex

            for n in range(N):
                dω = noise_all[t_idx, n]

                # Compute effective c_x, c_y, c_z and c
                cx = base_pauli_coeffs[0] + dω * noise_pauli_coeffs[0]
                cy = base_pauli_coeffs[1] + dω * noise_pauli_coeffs[1]
                cz = base_pauli_coeffs[2] + dω * noise_pauli_coeffs[2]
                c = base_c + dω * noise_c

                # Half-angles for the rotation
                ax = 0.5 * cx
                ay = 0.5 * cy
                az = 0.5 * cz
                a2 = ax*ax + ay*ay + az*az

                if a2 == 0.0:
                    U00 = 1.0 + 0.0j
                    U01 = 0.0 + 0.0j
                    U10 = 0.0 + 0.0j
                    U11 = 1.0 + 0.0j
                else:
                    a = np.sqrt(a2)
                    c_rot = np.cos(a * dt)
                    s_over_a = np.sin(a * dt) / a
                    A00 = az
                    A01 = ax - 1j*ay
                    A10 = ax + 1j*ay
                    A11 = -az

                    U00 = c_rot - 1j * s_over_a * A00
                    U01 = -1j * s_over_a * A01
                    U10 = -1j * s_over_a * A10
                    U11 = c_rot - 1j * s_over_a * A11

                # Apply the rotation
                new0 = U00 * psi0 + U01 * psi1
                new1 = U10 * psi0 + U11 * psi1

                # Apply the global phase from identity term
                phase = np.cos(c * dt) - 1j * np.sin(c * dt)  # exp(-1j * c * dt)
                psi0 = phase * new0
                psi1 = phase * new1

                # ⟨M⟩ = Re(ψ† M ψ)
                obs[t_idx, n] = (
                    np.conjugate(psi0) * (meas_obs[0,0]*psi0 + meas_obs[0,1]*psi1)
                    + np.conjugate(psi1) * (meas_obs[1,0]*psi0 + meas_obs[1,1]*psi1)
                ).real

        return obs

    # General propagator-based evolution for any Hilbert space dimension
    # @njit(parallel=True, fastmath=True, cache=True)
    # def evolve_numba_general_propagator(hamiltonian_func, dt, psi_init, N_steps, meas_obs=None):
    #     """
    #     Evolve a quantum state under a time-dependent Hamiltonian H(t) using step-by-step unitary evolution exp(-i H(t) dt).
    #     Assumes H(t) is Hermitian.
    #     hamiltonian_func: Numba-compiled function taking t (float) and returning H (n x n complex array)
    #     dt: time step
    #     n: dimension of Hilbert space
    #     psi_init: (n,) complex array initial state
    #     meas_obs: (n, n) complex array observable
    #     N_steps: number of time steps
    #     Returns: obs (N_steps,) expectation values <psi|meas_obs|psi>
    #     """
    #     obs_all = np.empty(N_steps, dtype=np.float128)
    #     psi_all = np.empty((N_steps, psi_init.shape[0]), dtype=np.complex128)

    #     psi = psi_init.copy()

    #     for step in range(N_steps):
    #         current_time = step * dt
    #         H = hamiltonian_func(current_time)
    #         e, V = np.linalg.eigh(H)
    #         exp_e = np.exp(-1j * e * dt)
    #         U = V @ np.diag(exp_e) @ V.conj().T
    #         psi = U @ psi
    #         # if meas_obs is not None:
    #         #     obs_all[step] = np.real(np.conj(psi).T @ meas_obs @ psi)
    #         psi_all[step, :] = psi

    #     return psi_all

    @njit(parallel=True, cache=True)
    def _run_stochastic_trajectories_numba_general_propagator(
        noise_array: np.ndarray,
        initial_vector: np.ndarray,
        time_grid: np.ndarray,
        H_det: np.ndarray,
        det_hints: np.ndarray,
        det_rates: np.ndarray,
        det_has_rate: np.ndarray,
        stoch_hints: np.ndarray,
        stoch_rates: np.ndarray,
        stoch_has_rate: np.ndarray,
        noise_strengths: np.ndarray,
        noise_offsets: np.ndarray,
        noise_source_indices: np.ndarray,
        noise_transformation_types: np.ndarray,
        noise_transformation_params: np.ndarray,
    ) -> np.ndarray:
        n_traj = noise_array.shape[0]
        n_time = time_grid.shape[0]
        n_state = initial_vector.shape[0]
        result = np.empty((n_traj, n_time, n_state), dtype=np.complex128)

        dt = time_grid[1] - time_grid[0]
        for traj_idx in prange(n_traj):
            psi = initial_vector.copy()
            result[traj_idx, 0, :] = psi

            for step in range(n_time - 1):
                t = time_grid[step]
                dt_step = time_grid[step + 1] - time_grid[step]

                H = _numba_build_hamiltonian(
                    t,
                    step,
                    traj_idx,
                    noise_array,
                    time_grid,
                    H_det,
                    det_hints,
                    det_rates,
                    det_has_rate,
                    stoch_hints,
                    stoch_rates,
                    stoch_has_rate,
                    noise_strengths,
                    noise_offsets,
                    noise_source_indices,
                    noise_transformation_types,
                    noise_transformation_params,
                    interpolate=False,
                )

                # Diagonalize Hamiltonian
                e, V = np.linalg.eigh(H)

                # Apply evolution: psi_new = V @ exp(-i e dt) @ V† @ psi
                # Compute V† @ psi first
                V_dag_psi = V.conj().T @ psi
                
                # Multiply by exp(-i e dt) element-wise
                exp_e = np.exp(-1j * e * dt_step)
                scaled = V_dag_psi * exp_e
                
                # Apply V
                psi = V @ scaled
                
                result[traj_idx, step + 1, :] = psi
        return result

else:  # pragma: no cover - fallback when numba is missing
    raise IonSimError(f'trajectory_backend with numba requested but Numba is not installed.')

# working version
# @dataclass(frozen=True, eq=False)
# class ZVODE(OdeSolver):
#     """Python's zvode routine."""
#     nsteps: float = 1e6

#     def solve(self):
#         """Solves the ODE."""
#         if self.time_evals is None:
#             num_steps = 3
#         else:
#             num_steps = len(time_evals)
#             assert(time_evals[-1] == duration)

#         # TODO: remove the "propgagte" method below and just solve the ODE within the "solve" method.
#         def propagate(n_states, hamiltonian, t_final, initial_state=None, initial_time=0., display_progress=False,
#             return_intermediate=False, verbose=False, atol=1e-16, rtol=1e-14, nsteps=self.nsteps, num_steps=3):
#             """Propagate the initial wavefunction."""

#             if initial_state is None:
#                 initial_state = _np.zeros(n_states)
#                 initial_state[0] = 1.
#             if return_intermediate:
#                 # intermediate_states = []
#                 # intermediate_times = []
#                 intermediate_states = [initial_state]
#                 intermediate_times = [initial_time]

#             # Define the Schrodinger equation and the Jacobian
#             def schrodinger(t, y):
#                 return  -1.0j * hamiltonian(t).dot(y)
#             def jacobian(t, y):
#                 tempham = hamiltonian(t)
#                 if sparse.issparse(tempham):
#                     return -1.0j * tempham.todense()
#                 else:
#                     return -1.0j * tempham
#             # Instantiate the integrator
#             r = ode(schrodinger, jacobian)
#             r.set_integrator('zvode', method='adams', with_jacobian=True, atol=atol, rtol=rtol, nsteps=nsteps) # use method='bdf' for stiff ode
#             r.set_initial_value(initial_state, initial_time)
#             if display_progress or return_intermediate:
#                 # Do the integral in peices and display progress
#                 # n_steps = 1000
#                 dt = t_final/float(num_steps)
#                 if display_progress:
#                     evaluation_times = []
#                     previous_time = time()
#                 while r.successful() and r.t < t_final:
#                     r.integrate(r.t+dt)
#                     # print ''
#                     # print datetime.datetime.today()
#                     # print "%g" % (r.t)
#                     if display_progress:
#                         current_time = time()
#                         evaluation_times += [current_time-previous_time]
#                         previous_time = current_time
#                         this_step = len(evaluation_times)
#                         print('Finished step {0} of {1} in time {2}s.  Expected time remaining: {3}s'.format(
#                             this_step, num_steps, evaluation_times[-1], mean(evaluation_times) * (num_steps - this_step)))
#                     if return_intermediate:
#                         intermediate_states += [r.y]
#                         intermediate_times += [r.t]
#             else:
#                 r.integrate(t_final)
#                 if verbose:
#                     print('')
#                     print(argmax(initial_state))
#                     print(initial_time, t_final)
#                     print(datetime.datetime.today())
#                     print("%g" % (r.t))
#             if return_intermediate:
#                 return intermediate_states, intermediate_times
#             else:
#                 return r.y

#         states, times = propagate(
#             len(self.initial_vector),
#             self.interaction_function,
#             self.duration,
#             initial_state = self.initial_vector,
#             return_intermediate = True,
#             num_steps = num_steps,
#             )
#         assert(len(times) == num_steps + 1)
#         return times, states

        

# original below
# class ZVODE(OdeSolver):
#     """Python's zvode routine."""

# def propagate(n_states, hamiltonian, t_final, initial_state = None, initial_time = 0., display_progress = False,
#                   return_intermediate=False, verbose=False, atol=1e-16, rtol=1e-14, nsteps=1e6):

#     if initial_state is None:
#         initial_state = _np.zeros(n_states)
#         initial_state[0] = 1.
#     if return_intermediate:
#         intermediate_states = []
#         intermediate_times = []

#     # Define the Schrodinger equation and the Jacobian
#     def schrodinger(t, y):
#         return  -1.0j * hamiltonian(t).dot(y)
#     def jacobian(t, y):
#         tempham = hamiltonian(t)
#         if sparse.issparse(tempham):
#             return -1.0j * tempham.todense()
#         else:
#             return -1.0j * tempham
#     # Instantiate the integrator
#     r = ode(schrodinger, jacobian)
#     r.set_integrator('zvode', method='adams', with_jacobian=True, atol=atol, rtol=rtol, nsteps=nsteps) # use method='bdf' for stiff ode
#     r.set_initial_value(initial_state, initial_time)
#     if display_progress or return_intermediate:
#         # Do the integral in peices and display progress
#         n_steps = 1000
#         dt = 1.*t_final/n_steps
#         evaluation_times = []
#         previous_time = time()
#         while r.successful() and r.t < t_final:
#             r.integrate(r.t+dt)
#             # print ''
#             # print datetime.datetime.today()
#             # print "%g" % (r.t)
#             current_time = time()
#             evaluation_times += [current_time-previous_time]
#             previous_time = current_time
#             this_step = len(evaluation_times)
#             if display_progress:
#                 print('Finished step {0} of {1} in time {2}s.  Expected time remaining: {3}s'.format(
#                     this_step, n_steps, evaluation_times[-1], mean(evaluation_times) * (n_steps - this_step)))
#             if return_intermediate:
#                 intermediate_states += [r.y]
#                 intermediate_times += [r.t]
#     else:
#         r.integrate(t_final)
#         if verbose:
#             print('')
#             print(argmax(initial_state))
#             print(initial_time, t_final)
#             print(datetime.datetime.today())
#             print("%g" % (r.t))

#     # r.integrate(t_final)
#     if return_intermediate:
#         return (array(intermediate_states),array(intermediate_times))
#     else:
#         return r.y

def odeintz(func, z0, t, **kwargs):
    """An odeint-like function for complex valued differential equations."""

    # Disallow Jacobian-related arguments.
    _unsupported_odeint_args = ['Dfun', 'col_deriv', 'ml', 'mu']
    bad_args = [arg for arg in kwargs if arg in _unsupported_odeint_args]
    if len(bad_args) > 0:
        raise ValueError("The odeint argument %r is not supported by "
                         "odeintz." % (bad_args[0],))

    # Make sure z0 is a numpy array of type np.complex128.
    z0 = np.array(z0, dtype=np.complex128, ndmin=1)

    def realfunc(x, t, *args):
        z = x.view(np.complex128)
        dzdt = func(z, t, *args)
        # func might return a python list, so convert its return
        # value to an array with type np.complex128, and then return
        # a np.float64 view of that array.
        return np.asarray(dzdt, dtype=np.complex128).view(np.float64)

    result = odeint(realfunc, z0.view(np.float64), t, **kwargs)

    if kwargs.get('full_output', False):
        z = result[0].view(np.complex128)
        infodict = result[1]
        return z, infodict
    else:
        z = result.view(np.complex128)
        return z

def slow_trapz_for_matrix(ys: Vector, xs: Vector, *args, **kwargs): 
    """Apply scipy.integrate.trapz to a matrix of integrands."""
    num_rows, num_columns = ys[0].shape
    integral = np.zeros((num_rows, num_columns), dtype='complex')
    for row in range(num_rows):
        for column in range(num_columns):
            integrands = np.array([y[row, column] for y in ys])
            integral[row, column] = trapz(integrands, xs, *args, **kwargs)
    return integral

def trapz_for_matrix(ys: Vector, xs: Vector, *args, **kwargs): 
    """Apply scipy.integrate.trapz to a matrix of integrands."""
    num_rows, num_columns = ys[0].shape
    prods = list(it.product(range(num_rows), range(num_columns)))
    index_map = {k: (row, column) for k, (row, column) in enumerate(prods)}
    integrands_list = [np.array([y[row, column] for y in ys]) for row, column in prods]
    function = lambda integs: trapz(integs, xs, *args, **kwargs)
    results = [function(integs) for integs in integrands_list]
    integral = np.zeros((num_rows, num_columns), dtype='complex')
    for k, result in enumerate(results):
        row, column = index_map[k]
        integral[row, column] = result
    return integral
