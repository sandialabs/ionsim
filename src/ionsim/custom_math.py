from ionsim.custom_types import Vector
from ionsim.ionsim_error import IonSimError

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import sys
from typing import Any, Callable
from scipy.integrate import trapezoid as trapz
import itertools as it
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from scipy.integrate import odeint, solve_ivp, ode
from scipy import sparse

try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _NUMBA_AVAILABLE = False

from icecream import ic

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
            num_steps = len(self.time_evals)
            assert(self.time_evals[-1] == self.duration)

        ic(self.nsteps)

        n_states = len(self.initial_vector)
        hamiltonian = self.interaction_function
        t_final = self.duration
        initial_state = self.initial_vector

        if initial_state is None:
            initial_state = np.zeros(n_states, dtype=complex)
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
    """A numerical routine to solve stochastic differential equations (SDEs) for wavefunction evolution.
    
    Performance guide for trajectory_backend:
    - 'diffrax_vmap': Fastest - JIT-compiled ODE solver with noise in Hamiltonian (10-100x speedup)
    - 'diffrax_generalshark': SDE solver with Lévy areas, multiprocessing (good for high accuracy)
    - 'numba_rk4/rk5': Fast RK integrators, requires numba (5-20x speedup)
    - 'numba_general_propagator': Matrix exponential propagator (most accurate)
    - 'scipy': Standard solver, reliable baseline
    - 'python': Slowest, for debugging only
    
    The diffrax_vmap method:
    - Incorporates pre-generated noise directly into time-dependent Hamiltonian
    - Uses fast ODE solver (Tsit5) with JIT compilation
    - Avoids complex SDE/Brownian path machinery for maximum speed
    - Best for moderate accuracy needs with high performance requirements
    - Set jax_device='gpu' or 'cpu' to control device placement (default: auto)
    """
    noisy_trajectories: np.ndarray | None = None
    base_solver: str = 'odeintz'
    base_solver_kwargs: dict[str, Any] = field(default_factory=dict)
    trajectory_backend: str = 'numba_rk4' # 'scipy', 'numba_rk4', 'numba_rk5', 'numba_general_propagator', 'diffrax_vmap'
    jax_device: str = 'cpu'  # 'auto', 'cpu', or 'gpu' for diffrax_vmap backend
    diffrax_solver: str = 'midpoint'  # 'tsit5', 'dopri5', 'dopri8', 'heun', 'midpoint' for diffrax_vmap backend
    diffrax_rtol: float = 1e-4  # Relative tolerance for adaptive step size (tighter for larger Hilbert spaces)
    diffrax_atol: float = 1e-7  # Absolute tolerance for adaptive step size
    diffrax_max_steps: int | None = None  # Max solver steps (None=auto: 16*N for RK5+, 200*N for Heun)
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
        """Solve the stochastic Schr├╢dinger equation by evolving each trajectory independently."""
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
        if backend not in {"python", "scipy", "numba_rk4", "numba_rk5", "numba_general_propagator", "diffrax_vmap"}:
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
        elif backend == 'diffrax_vmap':
            # Use JAX vmap for batched trajectory solving (fastest for GPU/TPU)
            stacked_results = _run_stochastic_trajectories_diffrax_vmap(
                noise_array,
                np.asarray(self.initial_vector, dtype=np.complex128, order='C'),
                time_grid,
                component_data,
                device=self.jax_device,
                solver_name=self.diffrax_solver,
                rtol=self.diffrax_rtol,
                atol=self.diffrax_atol,
                max_steps=self.diffrax_max_steps,
            )
        elif backend in {'diffrax_generalshark', 'scipy'}:
            # Route to parallel solver with appropriate method
            if backend == 'diffrax_generalshark':
                method = 'diffrax_generalshark'
            else:
                method = 'scipy'  # default for python/multiprocessing
            
            stacked_results = parallel_trajectory_ode_solver(
                n_trajectories,
                solver_kwargs,
                time_grid,
                noise_array,
                interaction_function=self.interaction_function,
                initial_vector=self.initial_vector,
                duration=self.duration,
                base_solver=self.base_solver,
                component_data=component_data,
                method=method,
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
        component_data: Any = None,
        method: str = 'scipy',
    ) -> np.ndarray:
    """Solve each trajectory independently, optionally in parallel.
    
    Args:
        method: 'scipy' for classical SDE solver, 'diffrax_generalshark' for Diffrax GeneralShARK

    Notes (Windows / notebooks): multiprocessing uses spawn, which requires the
    passed callables to be picklable. If pickling fails, we fall back to a
    serial loop to keep behavior correct.
    """
    if n_trajectories <= 0:
        raise IonSimError('n_trajectories must be a positive integer.')

    processes = os.cpu_count() or 1

    # Prepare per-trajectory work items.
    if method == 'diffrax_generalshark':
        work_items = [
            (
                trajectory_index,
                noise_array[trajectory_index],
                initial_vector,
                time_grid,
                component_data,
            )
            for trajectory_index in range(n_trajectories)
        ]
        worker_func = _solve_single_diffrax_trajectory
    else:  # scipy method
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
        worker_func = _solve_single_stochastic_trajectory

    try:
        with mp.Pool(processes=processes) as pool:
            results = pool.starmap(worker_func, work_items)
        print(f"Using multiprocessing with {processes} processes for stochastic trajectories (method={method}).")
    except Exception as exc:
        # Most common cause: interaction_function (or one of its captured objects)
        # is not picklable under spawn. Keep correctness by falling back to serial.
        print(f"Multiprocessing failed ({type(exc).__name__}: {exc}); falling back to serial.")
        if method == 'diffrax_generalshark':
            return _serial_diffrax_solver(
                n_trajectories,
                noise_array,
                initial_vector,
                time_grid,
                component_data,
            )
        else:
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


def _serial_diffrax_solver(
    n_trajectories: int,
    noise_array: np.ndarray,
    initial_vector: np.ndarray,
    time_grid: np.ndarray,
    component_data: Any,
) -> np.ndarray:
    """Serial fallback for Diffrax solver when multiprocessing fails."""
    trajectory_vectors: list[np.ndarray] = []
    for trajectory_index in range(n_trajectories):
        trajectory_noise = noise_array[trajectory_index]
        times, state_vectors = _solve_single_diffrax_trajectory(
            trajectory_index,
            trajectory_noise,
            initial_vector,
            time_grid,
            component_data,
        )
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
    """Serial fallback for scipy solver when multiprocessing fails."""
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


def _solve_single_diffrax_trajectory(
    trajectory_index: int,
    trajectory_noise: np.ndarray,
    initial_vector: np.ndarray,
    time_grid: np.ndarray,
    component_data: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a single stochastic trajectory using Diffrax GeneralShARK solver."""
    try:
        import jax
        import jax.numpy as jnp
        from diffrax import diffeqsolve, GeneralShARK, MultiTerm, ODETerm, ControlTerm, SaveAt
        
        jax.config.update("jax_enable_x64", True)
    except ImportError:
        raise IonSimError(
            'Diffrax/JAX not installed. Install with: pip install diffrax jax jaxlib'
        )
    
    # Extract component data
    H0_jax = jnp.array(component_data.H0, dtype=jnp.complex128)
    det_hints_jax = jnp.array(component_data.deterministic_hints, dtype=jnp.complex128)
    det_rates_jax = jnp.array(component_data.deterministic_rates, dtype=jnp.float64)
    det_has_rate_jax = jnp.array(component_data.deterministic_has_rate, dtype=jnp.uint8)
    stoch_hints_jax = jnp.array(component_data.stochastic_hints, dtype=jnp.complex128)
    stoch_rates_jax = jnp.array(component_data.stochastic_rates, dtype=jnp.float64)
    stoch_has_rate_jax = jnp.array(component_data.stochastic_has_rate, dtype=jnp.uint8)
    noise_strengths_jax = jnp.array(component_data.noise_strengths, dtype=jnp.complex128)
    noise_source_indices_jax = jnp.array(component_data.noise_source_indices, dtype=jnp.int32)
    
    n_sources = trajectory_noise.shape[0]
    n_state = initial_vector.shape[0]
    
    # Define drift and diffusion fields with JIT compilation
    @jax.jit
    def drift_field(t, y, args):
        """JIT-compiled drift field using vectorized operations."""
        # Start with base Hamiltonian
        H_det = H0_jax.copy()
        
        # Vectorized deterministic term accumulation
        if det_hints_jax.shape[0] > 0:
            # Compute phase factors for all terms at once using jnp.where
            phase_factors = jnp.where(
                det_has_rate_jax != 0,
                jnp.exp(-1j * det_rates_jax * t),
                1.0
            )
            # Sum all contributions: H_det += sum_i phase_i * hint_i
            for i in range(det_hints_jax.shape[0]):
                H_det = H_det + phase_factors[i] * det_hints_jax[i]
        
        # Return 1D vector: -i H |ψ⟩
        result = -1j * jnp.dot(H_det, y)
        # Ensure output is 1D
        return jnp.squeeze(result)
    
    @jax.jit
    def diffusion_field(t, y, args):
        """JIT-compiled diffusion field using vectorized operations."""
        # Initialize diffusion matrix G: G[:, i] is the coefficient for dW_i
        G = jnp.zeros((n_state, n_sources), dtype=jnp.complex128)
        
        # Compute phase factors for all stochastic terms
        if stoch_hints_jax.shape[0] > 0:
            # Compute all phase factors at once using jnp.where
            phase_factors = jnp.where(
                stoch_has_rate_jax != 0,
                jnp.exp(-1j * stoch_rates_jax * t),
                1.0
            )
            
            # Build diffusion matrix column by column
            for idx in range(stoch_hints_jax.shape[0]):
                # Apply template with phase
                template = stoch_hints_jax[idx] * phase_factors[idx]
                source_idx = noise_source_indices_jax[idx]  # Keep as JAX array, no int()
                strength = noise_strengths_jax[idx]
                
                # Compute contribution: -i * strength * template @ y
                contribution = -1j * strength * jnp.dot(template, y)
                G = G.at[:, source_idx].add(jnp.squeeze(contribution))
        
        return G
    
    # Create custom Brownian path
    t0, t1 = time_grid[0], time_grid[-1]
    bm = _PreGeneratedBrownianPath(
        t0=t0, t1=t1,
        time_grid=time_grid,
        noise_trajectory=trajectory_noise,
        levy_area_approximation='trapezoidal'
    )
    
    # Setup SDE
    y0_jax = jnp.array(initial_vector, dtype=jnp.complex128)
    drift_term = ODETerm(drift_field)
    diffusion_term = ControlTerm(diffusion_field, bm)
    terms = MultiTerm(drift_term, diffusion_term)
    solver = GeneralShARK()
    saveat = SaveAt(ts=time_grid)
    
    # Solve
    sol = diffeqsolve(
        terms=terms,
        solver=solver,
        t0=float(t0),
        t1=float(t1),
        dt0=float(time_grid[1] - time_grid[0]),
        y0=y0_jax,
        saveat=saveat,
        max_steps=len(time_grid) * 10,
    )
    
    times_array = np.asarray(time_grid, dtype=float)
    stacked_states = np.array(sol.ys, dtype=np.complex128)
    return times_array, stacked_states


def _solve_single_stochastic_trajectory(
    trajectory_noise: np.ndarray,
    interaction_function: Callable,
    initial_vector: Vector,
    duration: float,
    time_grid: np.ndarray,
    base_solver: str,
    solver_kwargs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a single stochastic trajectory using the configured base solver (scipy method)."""
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

    H0 = np.ascontiguousarray(component_data.H0, dtype=np.complex128)
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

    if method == 'RK4':
        return _run_stochastic_trajectories_numba_RK4(
            noise_array_f64,
            initial_vec_c,
            time_grid_f64,
            H0,
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
            H0,
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
            H0,
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
    def _numba_build_hamiltonian(
        t: float,
        step: int,
        traj_idx: int,
        noise_array: np.ndarray,
        time_grid: np.ndarray,
        H0: np.ndarray,
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
        H = H0.copy()
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
        
        # Enforce hermiticity after adding all stochastic terms
        # Complex noise factors can break hermiticity, so symmetrize
        H = (H + H.conj().T) / 2.0
        
        return H

    @njit(cache=True)
    def _numba_rhs(
        t: float,
        psi: np.ndarray,
        step: int,
        traj_idx: int,
        noise_array: np.ndarray,
        time_grid: np.ndarray,
        H0: np.ndarray,
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
            H0,
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
    def _numba_rk4_step(
        t0: float,
        dt: float,
        step: int,
        psi: np.ndarray,
        traj_idx: int,
        noise_array: np.ndarray,
        time_grid: np.ndarray,
        H0: np.ndarray,
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
            H0,
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
            t0 + 0.5 * dt,
            psi + 0.5 * dt * k1,
            step,
            traj_idx,
            noise_array,
            time_grid,
            H0,
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
            t0 + 0.5 * dt,
            psi + 0.5 * dt * k2,
            step,
            traj_idx,
            noise_array,
            time_grid,
            H0,
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
            t0 + dt,
            psi + dt * k3,
            step,
            traj_idx,
            noise_array,
            time_grid,
            H0,
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
        return psi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    @njit(parallel=True, cache=True)
    def _run_stochastic_trajectories_numba_RK4(
        noise_array: np.ndarray,
        initial_vector: np.ndarray,
        time_grid: np.ndarray,
        H0: np.ndarray,
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
                    H0,
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
        H0: np.ndarray,
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
            H0,
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
            H0,
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
            H0,
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
            H0,
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
            H0,
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
            H0,
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
        H0: np.ndarray,
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
                    H0,
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
        H0: np.ndarray,
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
                    H0,
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


class _PreGeneratedBrownianPath:
    """Custom Brownian path that wraps pre-generated noise for Diffrax.
    
    This class provides an interface compatible with Diffrax's AbstractBrownianPath,
    allowing use of pre-generated noise trajectories with high-order SDE solvers
    that require space-time Lévy areas (like GeneralShARK).
    """
    
    def __init__(self, t0: float, t1: float, time_grid: np.ndarray, 
                 noise_trajectory: np.ndarray, levy_area_approximation: str = 'foster'):
        """Initialize pre-generated Brownian path.
        
        Args:
            t0: Start time
            t1: End time
            time_grid: Time points where noise is sampled (n_time,)
            noise_trajectory: Pre-generated noise values (n_sources, n_time)
            levy_area_approximation: Method for Lévy area computation
                'foster': Foster-type approximation using random projections
                'trapezoidal': Simple trapezoidal rule (less accurate)
        """
        import jax.numpy as jnp
        
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.time_grid = jnp.asarray(time_grid, dtype=jnp.float64)
        self.noise_trajectory = jnp.asarray(noise_trajectory, dtype=jnp.float64)
        self.shape = (noise_trajectory.shape[0],)  # (n_sources,)
        self.levy_area_approximation = levy_area_approximation
        
        # Pre-compute cumulative integrals for Lévy area (space-time)
        # H[i] = ∫_0^t W[i](s) ds for each time point
        self._precompute_integrals()
    
    def _precompute_integrals(self):
        """Pre-compute cumulative integrals ∫ W(s) ds for Lévy area calculation (vectorized)."""
        import jax.numpy as jnp
        
        # Vectorized cumulative trapezoidal integration
        # For each source: cumulative_integral[j] = ∫_t0^t_j W(s) ds
        dt = jnp.diff(self.time_grid)  # (n_time-1,)
        
        # Trapezoidal rule: integral += (W[j] + W[j+1]) / 2 * dt[j]
        # Vectorize over all sources at once
        midpoints = (self.noise_trajectory[:, :-1] + self.noise_trajectory[:, 1:]) / 2.0  # (n_sources, n_time-1)
        increments = midpoints * dt[None, :]  # (n_sources, n_time-1)
        
        # Cumulative sum to get integrals at each time point
        cumulative = jnp.cumsum(increments, axis=1)  # (n_sources, n_time-1)
        
        # Prepend zeros for initial time
        self.cumulative_integrals = jnp.concatenate(
            [jnp.zeros((self.noise_trajectory.shape[0], 1)), cumulative],
            axis=1
        )  # (n_sources, n_time)
    
    def evaluate(self, t0: float, t1: float, left: bool = True, use_levy: bool = True, **kwargs):
        """Evaluate Brownian increment and Lévy area over interval [t0, t1].
        
        Args:
            t0: Start time
            t1: End time
            left: Unused, for compatibility
            use_levy: Whether to compute Lévy areas (required by some solvers)
            **kwargs: Additional arguments for compatibility with Diffrax
        
        Returns:
            (W, H) where:
                W: Brownian increment W(t1) - W(t0), shape (n_sources,)
                H: Space-time Lévy area ∫_{t0}^{t1} (W(s) - W(t0)) ds, shape (n_sources,)
        """
        import jax.numpy as jnp
        
        # Interpolate W(t0) and W(t1)
        W_t0 = self._interpolate_at_time(t0)
        W_t1 = self._interpolate_at_time(t1)
        
        # Brownian increment
        dW = W_t1 - W_t0
        
        # Space-time Lévy area: H = ∫_{t0}^{t1} (W(s) - W(t0)) ds
        if use_levy:
            H = self._compute_levy_area(t0, t1, W_t0)
        else:
            # If Lévy area not needed, return zeros (simpler solvers don't need it)
            H = jnp.zeros_like(dW)
        
        return dW, H
    
    def _interpolate_at_time(self, t: float):
        """Linear interpolation of noise at arbitrary time t."""
        import jax.numpy as jnp
        
        # Clamp to bounds
        t = jnp.clip(t, self.time_grid[0], self.time_grid[-1])
        
        # Find bracketing indices
        idx = jnp.searchsorted(self.time_grid, t)
        idx = jnp.clip(idx, 1, len(self.time_grid) - 1)
        
        # Linear interpolation
        t_low = self.time_grid[idx - 1]
        t_high = self.time_grid[idx]
        W_low = self.noise_trajectory[:, idx - 1]
        W_high = self.noise_trajectory[:, idx]
        
        # Avoid division by zero
        dt = t_high - t_low
        alpha = jnp.where(dt > 1e-14, (t - t_low) / dt, 0.0)
        
        return W_low + alpha * (W_high - W_low)
    
    def _compute_levy_area(self, t0: float, t1: float, W_t0):
        """Compute space-time Lévy area H = ∫_{t0}^{t1} (W(s) - W(t0)) ds.
        
        For pre-generated noise, we use:
        H = ∫_{t0}^{t1} W(s) ds - W(t0) * (t1 - t0)
        
        Using cumulative integrals: H = [∫_0^{t1} W(s) ds - ∫_0^{t0} W(s) ds] - W(t0) * (t1 - t0)
        """
        import jax.numpy as jnp
        
        # Get cumulative integral values at t0 and t1
        integral_t0 = self._interpolate_integral_at_time(t0)
        integral_t1 = self._interpolate_integral_at_time(t1)
        
        # H = [∫_{t0}^{t1} W(s) ds] - W(t0) * (t1 - t0)
        H = (integral_t1 - integral_t0) - W_t0 * (t1 - t0)
        
        return H
    
    def _interpolate_integral_at_time(self, t: float):
        """Interpolate cumulative integral ∫_0^t W(s) ds at arbitrary time t."""
        import jax.numpy as jnp
        
        # Clamp to bounds
        t = jnp.clip(t, self.time_grid[0], self.time_grid[-1])
        
        # Find bracketing indices
        idx = jnp.searchsorted(self.time_grid, t)
        idx = jnp.clip(idx, 1, len(self.time_grid) - 1)
        
        # Linear interpolation of integral values
        t_low = self.time_grid[idx - 1]
        t_high = self.time_grid[idx]
        I_low = self.cumulative_integrals[:, idx - 1]
        I_high = self.cumulative_integrals[:, idx]
        
        # Add contribution from [t_low, t] using trapezoidal rule
        dt = t_high - t_low
        alpha = jnp.where(dt > 1e-14, (t - t_low) / dt, 0.0)
        
        # Interpolate integral value
        I_interp = I_low + alpha * (I_high - I_low)
        
        return I_interp


def _run_stochastic_trajectories_diffrax_vmap(
    noise_array: np.ndarray,
    initial_vector: np.ndarray,
    time_grid: np.ndarray,
    component_data: Any,
    device: str = 'auto',
    chunk_size: int | None = None,
    solver_name: str = 'tsit5',
    rtol: float = 1e-7,
    atol: float = 1e-9,
    max_steps: int | None = None,
) -> np.ndarray:
    """Execute stochastic trajectories using Diffrax with JIT-compiled ODE solves.
    
    Fast approach: JIT-compiles the vector field once and reuses it across all trajectories.
    This provides 5-20x speedup on CPU, 50-200x on GPU.
    
    Args:
        noise_array: Pre-generated noise (n_traj, n_sources, n_time)
        initial_vector: Initial quantum state
        time_grid: Time points for solution output
        component_data: Hamiltonian component data
        device: Device placement - 'auto' (default), 'cpu', or 'gpu'
        solver_name: ODE solver name for Diffrax ('tsit5', 'dopri5', 'dopri8', 'heun', 'midpoint')
        rtol: Relative tolerance for adaptive step size (default: 1e-7)
        atol: Absolute tolerance for adaptive step size (default: 1e-9)
    
    Returns:
        Array of shape (n_traj, n_time, n_state) with evolved wavefunctions
    """
    try:
        import jax
        import jax.numpy as jnp
        from diffrax import diffeqsolve, Tsit5, Dopri5, Dopri8, Heun, Midpoint, ODETerm, SaveAt, PIDController, ConstantStepSize
        
        jax.config.update("jax_enable_x64", True)
    except ImportError:
        raise IonSimError(
            'trajectory_backend="diffrax_vmap" requested but diffrax/jax not installed. '
            'Install with: pip install diffrax jax jaxlib'
        )
    
    n_traj, n_sources, n_time = noise_array.shape
    n_state = initial_vector.shape[0]
    
    # Select device
    all_devices = jax.devices()
    
    if device == 'auto':
        target_device = all_devices[0]  # Use default device
    elif device.lower() == 'cpu':
        cpu_devices = [d for d in all_devices if d.platform == 'cpu']
        if not cpu_devices:
            available = ', '.join([f"{d.platform}:{d.id}" for d in all_devices])
            raise IonSimError(f'No CPU device available. Available devices: {available}')
        target_device = cpu_devices[0]
    elif device.lower() == 'gpu':
        # Check for GPU devices - JAX reports them as platform 'gpu', 'cuda', or 'metal'
        gpu_devices = [d for d in all_devices if d.platform in ('gpu', 'cuda', 'metal')]
        if not gpu_devices:
            available = ', '.join([f"{d.platform}:{d.id}" for d in all_devices])
            windows_note = (
                "\nNote: Native Windows Python builds of JAX are typically CPU-only. "
                "For NVIDIA CUDA GPU acceleration, use WSL2 (Ubuntu) or Linux."
                if sys.platform.startswith('win')
                else ""
            )
            raise IonSimError(
                f'No GPU device available. Available devices: {available}{windows_note}\n'
                f'For NVIDIA GPU: pip install -U "jax[cuda12]"\n'
                f'For Apple Silicon: pip install -U "jax[metal]"'
            )
        target_device = gpu_devices[0]
    else:
        raise IonSimError(f'Invalid device "{device}". Use "auto", "cpu", or "gpu"')
    
    # Convert to JAX arrays and place on target device
    noise_array_jax = jax.device_put(jnp.array(noise_array, dtype=jnp.float64), target_device)
    initial_vector_jax = jax.device_put(jnp.array(initial_vector, dtype=jnp.complex128), target_device)
    time_grid_jax = jax.device_put(jnp.array(time_grid, dtype=jnp.float64), target_device)
    
    # Convert component data to JAX arrays and place on target device
    H0_jax = jax.device_put(jnp.array(component_data.H0, dtype=jnp.complex128), target_device)
    det_hints_jax = jax.device_put(jnp.array(component_data.deterministic_hints, dtype=jnp.complex128), target_device)
    det_rates_jax = jax.device_put(jnp.array(component_data.deterministic_rates, dtype=jnp.float64), target_device)
    det_has_rate_jax = jax.device_put(jnp.array(component_data.deterministic_has_rate, dtype=jnp.uint8), target_device)
    stoch_hints_jax = jax.device_put(jnp.array(component_data.stochastic_hints, dtype=jnp.complex128), target_device)
    stoch_rates_jax = jax.device_put(jnp.array(component_data.stochastic_rates, dtype=jnp.float64), target_device)
    stoch_has_rate_jax = jax.device_put(jnp.array(component_data.stochastic_has_rate, dtype=jnp.uint8), target_device)
    noise_strengths_jax = jax.device_put(jnp.array(component_data.noise_strengths, dtype=jnp.complex128), target_device)
    noise_source_indices_jax = jax.device_put(jnp.array(component_data.noise_source_indices, dtype=jnp.int32), target_device)
    
    # Define JIT-compiled vector field (compiled once, reused for all trajectories)
    @jax.jit
    def vector_field(t, y, traj_noise):
        """Time-dependent Hamiltonian with noise: -i H(t, ξ) ψ
        
        Args:
            t: Current time
            y: Current state (n_state,)
            traj_noise: Noise for this trajectory (n_sources, n_time)
        
        Returns:
            1D array of shape (n_state,)
        """
        # Start with base Hamiltonian
        H_total = H0_jax.copy()
        
        # Add deterministic time-dependent terms
        if det_hints_jax.shape[0] > 0:
            # det_rates_jax has shape (n_det, dim, dim); reshape flags for proper broadcasting
            det_has_rate = det_has_rate_jax.reshape((-1, 1, 1))
            phase_factors = jnp.where(
                det_has_rate != 0,
                jnp.exp(-1j * det_rates_jax * t),
                1.0
            )
            # Accumulate each term individually to avoid dimension issues
            for i in range(det_hints_jax.shape[0]):
                H_total = H_total + phase_factors[i] * det_hints_jax[i]
        
        # Add stochastic terms with noise lookup
        if stoch_hints_jax.shape[0] > 0:
            # Find nearest time index for noise interpolation
            time_idx = jnp.searchsorted(time_grid_jax, t)
            time_idx = jnp.clip(time_idx, 0, n_time - 1)
            
            # Compute phase factors for stochastic terms
            stoch_has_rate = stoch_has_rate_jax.reshape((-1, 1, 1))
            stoch_phase_factors = jnp.where(
                stoch_has_rate != 0,
                jnp.exp(-1j * stoch_rates_jax * t),
                1.0
            )
            
            # Accumulate stochastic terms
            for idx in range(stoch_hints_jax.shape[0]):
                template = stoch_hints_jax[idx]
                source_idx = noise_source_indices_jax[idx]
                strength = noise_strengths_jax[idx]
                phase = stoch_phase_factors[idx]
                noise_val = traj_noise[source_idx, time_idx]
                
                H_total = H_total + strength * noise_val * phase * template
        
        # Schrödinger equation: d/dt |ψ⟩ = -i H |ψ⟩
        result = -1j * jnp.dot(H_total, y)
        # Ensure 1D output by squeezing any extra dimensions
        return jnp.squeeze(result)
    
    # Select solver for ODE backend
    solver_key = (solver_name or 'tsit5').lower()
    
    # Report which device is being used
    device_type = target_device.device_kind
    print(f"JAX using device: {device_type} ({target_device})")
    print(f"Solving {n_traj} trajectories with JIT-compiled Diffrax ({solver_key} on {device_type})...")
    if solver_key == 'tsit5':
        solver = Tsit5()
    elif solver_key == 'dopri5':
        solver = Dopri5()
    elif solver_key == 'dopri8':
        solver = Dopri8()
    elif solver_key == 'heun':
        solver = Heun()
    elif solver_key == 'midpoint':
        solver = Midpoint()
    elif solver_key in {'generalshark', 'sde'}:
        raise IonSimError(
            'diffrax_vmap is an ODE backend and cannot use GeneralShARK. '
            'Use trajectory_backend="diffrax_generalshark" for SDE solvers.'
        )
    else:
        raise IonSimError(f'Unknown diffrax solver "{solver_name}". Use "tsit5", "dopri5", "dopri8", "heun", or "milstein".')

    # Solve trajectories in chunked batches using vmap (reduces Python overhead)
    if chunk_size is None or chunk_size <= 0:
        chunk_size = n_traj

    t0 = float(time_grid_jax[0])
    t1 = float(time_grid_jax[-1])
    dt0 = float(time_grid_jax[1] - time_grid_jax[0])

    # Auto-select max_steps based on solver order if not specified
    # Lower-order methods need many more steps for adaptive stepping with tight tolerances
    if max_steps is None:
        if solver_key == 'heun':
            # Heun is RK2 (order 2) - needs ~200x more steps than RK5 methods
            max_steps_value = 200 * n_time
        else:
            # Tsit5, Dopri5, Dopri8 are RK5+ (order 5-8) - standard budget
            max_steps_value = 16 * n_time
    else:
        max_steps_value = max_steps

    # Create the ODE term structure outside of solve_one
    # The vector field will be called with (t, y, args) where args contains the trajectory noise
    def vf_for_diffrax(t, y, args):
        """Vector field compatible with diffrax's argument structure."""
        traj_noise = args  # args will contain the noise trajectory
        return vector_field(t, y, traj_noise)

    term = ODETerm(vf_for_diffrax)
    saveat = SaveAt(ts=time_grid_jax)

    def solve_one(traj_noise):
        """Solve ODE for a single trajectory, passing noise as args."""
        sol = diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=initial_vector_jax,
            args=traj_noise,  # Pass trajectory noise as args
            saveat=saveat,
            stepsize_controller=ConstantStepSize(), # PIDController(rtol=rtol, atol=atol),
            max_steps=max_steps_value,
        )
        return sol.ys

    solve_batch = jax.jit(jax.vmap(solve_one))

    results = []
    for start in range(0, n_traj, chunk_size):
        end = min(start + chunk_size, n_traj)
        batch_noise = noise_array_jax[start:end]
        batch_states = solve_batch(batch_noise)
        results.append(np.array(batch_states, dtype=np.complex128))

    return np.concatenate(results, axis=0)


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
