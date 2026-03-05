from ionsim.custom_math import trapz_for_matrix
from ionsim.custom_types import Vector, Matrix
from ionsim.noise import Noise
from ionsim.basis import DegreeOfFreedom, Basis, StandardBasis
from ionsim.ionsim_error import IonSimError
from ionsim.hamiltonian import Hamiltonian
from ionsim.state import State

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable
from abc import ABC
from scipy.signal import savgol_filter

from icecream import ic

@dataclass(frozen=True, eq=False)
class Process(ABC): 
    """A quantum process represented in a basis of states."""
    basis: Basis
    process_matrix: Matrix
        
    def compute_process_fidelity(self, target_process_matrix: Matrix):
        """Compute the process fidelity with respect to a target process matrix."""
        total = 0
        for basis_state_vector in np.eye(len(self.process_matrix), dtype='complex'):
            # TODO: is dtype='complex' necessary here? When is it?
            final_state_vector = self.process_matrix.dot(basis_state_vector)
            target_state_vector = target_process_matrix.dot(basis_state_vector)
            total += np.dot(target_state_vector.conj().T, final_state_vector).real
        return total/len(self.process_matrix)

@dataclass(frozen=True, eq=False)
class Gate(Process):
    """A quantum gate represented in a basis of states."""
    process_matrix_function: Callable | None = None
    parameters: dict[str, float] = field(default_factory=dict)

    unitary: Matrix | None = None

    @classmethod #TODO: let default target_dofs be all degrees of freedom
    def from_unitary(cls, basis: Basis, unitary: Matrix, target_dofs: list[DegreeOfFreedom]):
        """Build a gate from a unitary-gate matrix."""
        full_unitary = basis.enlarge_matrix(unitary, target_dofs)
        process_matrix = basis.compute_superoperator_from_unitary_operator(full_unitary)
        return cls(basis, process_matrix, unitary=full_unitary)

    @classmethod
    def from_unitary_function(cls, basis: Basis, unitary_function: Callable,
            parameters: dict[str, float], target_dofs: list[DegreeOfFreedom], noise: Noise | None = None):
        """Build a gate from a unitary-gate function and its arguments."""
        parameter_names, arguments = list(parameters.keys()), list(parameters.values())
        full_function = basis.enlarge_matrix_function(unitary_function, target_dofs)
        process_matrix_function = basis.create_superoperator_function_from_unitary_operator_function(full_function)
        if noise is not None:
            noisy_parameter_index = parameter_names.index(noise.parameter_name)
            process_matrix_function = noise.add_noise_to_matrix_function(process_matrix_function, noisy_parameter_index)
        return cls(basis, process_matrix_function(*arguments), process_matrix_function, parameters)

    @classmethod
    def from_process_matrix_function(cls, basis: Basis, process_matrix_function: Callable,
            parameters: dict[str, float], target_dofs: list[DegreeOfFreedom], noise: Noise | None = None):
        """Build a gate from a process-matrix function and its arguments."""
        # TODO: It looks like this function doesn't use the target_dofs input parameter. Should it?
        parameter_names, arguments = list(parameters.keys()), list(parameters.values())
        if noise is None or noise.parameter_name not in parameter_names:
            return cls(basis, process_matrix_function(*arguments), process_matrix_function, parameters)
        noisy_parameter_index = parameter_names.index(noise.parameter_name)
        process_matrix_function = noise.add_noise_to_matrix_function(process_matrix_function, noisy_parameter_index)
        return cls(basis, process_matrix_function(*arguments), process_matrix_function, parameters)

    @classmethod
    def from_hamiltonian(cls, basis: StandardBasis, hamiltonian: Hamiltonian, duration: float,
            dofs_to_trace_out: list[DegreeOfFreedom] | None = None,
            initial_wavefunctions_for_dofs_to_trace_out: list[Vector] | None = None,
            ode_solver: str = 'odeintz',
            **ode_solver_kwargs): # TODO: add an option for initial density matrices for the traced out DoFs.
        """Build a gate by solving the Schrodinger equation for a complete set of initial states.""" 
        if dofs_to_trace_out is not None:
            assert(initial_wavefunctions_for_dofs_to_trace_out is not None)
            assert(len(dofs_to_trace_out) == len(initial_wavefunctions_for_dofs_to_trace_out))
            assert(len(dofs_to_trace_out) == 1) # TODO: generlize for multiple traced out DoFs
            dof_to_trace_out = dofs_to_trace_out[0]
            initial_wavefunction_for_dof_to_trace_out = initial_wavefunctions_for_dofs_to_trace_out[0]
            # TODO: consider if this function should just accept a reduced basis...?

        if dofs_to_trace_out is None:
            reduced_basis = basis
        else:
            reduced_basis = StandardBasis([dof for dof in basis.degrees_of_freedom if dof not in dofs_to_trace_out])

        import time
        final_states = []
        ic(len(reduced_basis.vectors))
        ic(reduced_basis.vectors)
        for vector in reduced_basis.vectors:
            if dofs_to_trace_out is None:
                initial_state = State.from_wavefunction(basis, vector)
            else:
                initial_state = State.from_wavefunction_with_new_component(
                    basis, vector, initial_wavefunction_for_dof_to_trace_out, [dof_to_trace_out]
                )
            ic(len(initial_state.wavefunction))
            # start = time.perf_counter()
            final_states.append(
                initial_state.propagate_using_schrodinger_equation(
                    hamiltonian, duration,
                    ode_solver=ode_solver, **ode_solver_kwargs
                )
            )
            # end = time.perf_counter()
            # ic(f'State propagation took {end-start} seconds.')

        # TODO: how can we do multiprocessing outside of main?
        # from concurrent.futures import ProcessPoolExecutor
        # def propagate(vector):
        #     if traced_out_dofs is None:
        #         initial_state = State.from_wavefunction(basis, vector)
        #     else:
        #         # wavefunction = basis.build_wavefunction(vector, traced_out_initial_wavefunction, [traced_out_dof])
        #         # TODO: crate the method above and use instead of line below
        #         wavefunction = np.kron(vector, traced_out_initial_wavefunction)
        #         initial_state = State.from_wavefunction(basis, wavefunction)
        #     initial_state.propagate_using_schrodinger_equation(hamiltonian, duration)
        # with ProcessPoolExecutor() as executor:
        #     results = executor.map(propagate, reduced_basis.vectors)
        # final_states = list(results)

        if dofs_to_trace_out is None:
            final_wavefunctions = [fs.wavefunction for fs in final_states]
            unitary = np.array(final_wavefunctions).T
        else:
            unitary = None

        supervectors = []
        for final_state_p in final_states:
            for final_state in final_states: # iterate rows with inner loop for column-stacked supervectors
                density_matrix = np.outer(final_state.wavefunction, final_state_p.wavefunction.conj().T)
                if dofs_to_trace_out is None:
                    spin_state = State.from_density_matrix(basis, density_matrix)
                else:
                    spin_state = State.from_density_matrix(
                        basis, density_matrix
                    ).trace_out_degree_of_freedom(dof_to_trace_out)
                supervectors.append(spin_state.supervector)
        process_matrix = np.array(supervectors).T

        return cls(reduced_basis, process_matrix, unitary=unitary)

    @classmethod
    def from_stochastic_hamiltonian(cls, basis: StandardBasis, hamiltonian: Hamiltonian, duration: float,
            dofs_to_trace_out: list[DegreeOfFreedom] | None = None,
            initial_wavefunctions_for_dofs_to_trace_out: list[Vector] | None = None,
            noisy_trajectories: Any = None,
            time_evals: np.ndarray | None = None,
            return_time_series: bool = False,
            **sse_kwargs):
        """Build a stochastic process map via SSE on a spanning set of input states.

        Reconstructs the full process superoperator via the polarization identity,
        requiring d + 2·d·(d-1)/2 = d² SSE propagations (d = dim of reduced basis).

        Off-diagonal channel elements are extracted using:
            E(|i><j|) = ½ [(2·E_plus − E_i − E_j) + i·(2·E_y − E_i − E_j)]
        where
            E_plus = E(|+_ij><+_ij|),   |+_ij> = (|i> + |j>) / √2
            E_y    = E(|y_ij><y_ij|),   |y_ij> = (|i> + i|j>) / √2

        Returns
        -------
        Gate
            If `return_time_series=False` (default), returns the map at the final
            time sample.
        np.ndarray
            If `return_time_series=True`, returns the full time series of process
            matrices with shape (N_t, d², d²), where N_t = len(time_evals).
        """
        if dofs_to_trace_out is not None:
            assert initial_wavefunctions_for_dofs_to_trace_out is not None
            assert len(dofs_to_trace_out) == len(initial_wavefunctions_for_dofs_to_trace_out)
            assert len(dofs_to_trace_out) == 1  # TODO: generalise for multiple traced-out DoFs
            dof_to_trace_out = dofs_to_trace_out[0]
            initial_wavefunction_for_dof_to_trace_out = initial_wavefunctions_for_dofs_to_trace_out[0]

        if dofs_to_trace_out is None:
            reduced_basis = basis
        else:
            reduced_basis = StandardBasis(
                [dof for dof in basis.degrees_of_freedom if dof not in dofs_to_trace_out]
            )

        d = len(reduced_basis.vectors)
        _time_evals = time_evals if time_evals is not None else np.array([duration])
        n_t = len(_time_evals)

        def _propagate(wavefunction: np.ndarray) -> np.ndarray:
            """Propagate one wavefunction via SSE; return reduced density matrices over time."""
            if dofs_to_trace_out is None:
                initial_state = State.from_wavefunction(basis, wavefunction)
            else:
                initial_state = State.from_wavefunction_with_new_component(
                    basis, wavefunction, initial_wavefunction_for_dof_to_trace_out, [dof_to_trace_out]
                )
            final_states = initial_state.propagate_using_stochastic_schrodinger_equation(
                hamiltonian,
                noisy_trajectories=noisy_trajectories,
                time_evals=_time_evals,
                return_density_average=True,
                **sse_kwargs,
            )
            if dofs_to_trace_out is None:
                return np.array([state.density_matrix for state in final_states], dtype='complex')
            return np.array(
                [state.trace_out_degree_of_freedom(dof_to_trace_out).density_matrix for state in final_states],
                dtype='complex',
            )

        # ── Step 1: d diagonal propagations  E(|i><i|) ─────────────────────────
        rho_diag = [_propagate(v) for v in reduced_basis.vectors]

        # ── Step 2: d(d-1) superposition propagations for off-diagonal elements ─
        # For each pair (i, j) with i < j, propagate:
        #   |+_ij> = (|i> + |j>) / √2   →  rho_plus[(i, j)] = E(|+_ij><+_ij|)
        #   |y_ij> = (|i> + i|j>) / √2  →  rho_y[(i, j)]    = E(|y_ij><y_ij|)
        rho_plus: dict[tuple[int, int], np.ndarray] = {}
        rho_y:    dict[tuple[int, int], np.ndarray] = {}
        for i in range(d):
            for j in range(i + 1, d):
                vi, vj = reduced_basis.vectors[i], reduced_basis.vectors[j]
                rho_plus[(i, j)] = _propagate((vi + vj) / np.sqrt(2.0))
                rho_y[(i, j)]    = _propagate((vi + 1j * vj) / np.sqrt(2.0))

        # ── Step 3: assemble process matrix/matrices ───────────────────────────
        # Column (q + p·d) = svec(E(|q><p|))  [outer loop: p = bra, inner: q = ket]
        # For off-diagonal (p ≠ q), let i = min(p, q), j = max(p, q):
        #   A = 2·rho_plus[(i,j)] − rho_diag[i] − rho_diag[j]
        #   B = 2·rho_y[(i,j)]    − rho_diag[i] − rho_diag[j]
        #   E(|i><j|) = ½(A + iB)   [q < p: ket-index i < bra-index j]
        #   E(|j><i|) = ½(A − iB)   [q > p: ket-index j > bra-index i]
        def _assemble_process_matrix_at_time_index(t_idx: int) -> np.ndarray:
            supervectors = []
            for p in range(d):
                for q in range(d):
                    if p == q:
                        dm = rho_diag[p][t_idx]
                    else:
                        i, j = min(p, q), max(p, q)
                        A = 2.0 * rho_plus[(i, j)][t_idx] - rho_diag[i][t_idx] - rho_diag[j][t_idx]
                        B = 2.0 * rho_y[(i, j)][t_idx]    - rho_diag[i][t_idx] - rho_diag[j][t_idx]
                        dm = 0.5 * (A + 1j * B) if q < p else 0.5 * (A - 1j * B)
                    supervectors.append(reduced_basis.compute_supervector_from_density_matrix(dm))
            return np.array(supervectors).T

        if return_time_series:
            return np.array([
                _assemble_process_matrix_at_time_index(t_idx)
                for t_idx in range(n_t)
            ])

        process_matrix = _assemble_process_matrix_at_time_index(n_t - 1)
        return cls(reduced_basis, process_matrix)

# @dataclass(frozen=True, eq=False)
# class PauliGate(Gate):
#     """A quantum gate in the z-Pauli spin basis.""" # TODO: Should we say "qubit basis" instead?

#     # TODO: check if basis is a z-Pauli spin basis.
#     # TODO: does this class require a StandardBasis input?

#     @staticmethod
#     def get_unitary(name: str):
#         """Get a unitary-gate matrix from its name."""
#         return _UNITARY_GATES[name]

#     @classmethod
#     def from_named_unitary(cls, basis: Basis, name: str, *args, **kwargs):
#         """Build a gate from the name of a unitary gate."""
#         unitary = cls.get_unitary(name)
#         return cls.from_unitary(basis, unitary, *args, **kwargs)

#     @staticmethod
#     def get_unitary_function(name: str):
#         """Get a unitary-gate function from its name."""
#         return _UNITARY_GATE_FUNCTIONS[name]

#     @classmethod
#     def from_named_unitary_function(cls, basis: Basis, name: str, *args, **kwargs):
#         """Build a gate from the name of a unitary-gate function."""
#         unitary_function = cls.get_unitary_function(name)
#         return cls.from_unitary_function(basis, unitary_function, *args, **kwargs)


@dataclass(frozen=True, eq=False)
class Circuit(Process):
    """A quantum circuit (i.e., a series of gates) in a basis of states."""
    gates: list[Gate]

    @classmethod
    def from_gates(cls, gates: list[Gate], noise: Noise | None = None):
        """Build a circuit from a series of gates in the same basis."""
        if any(gate.basis is not gates[0].basis for gate in gates):
            raise IonSimError('All gates in a circuit must be in the same basis.')
        if noise is None or all([noise.parameter_name not in gate.parameters for gate in gates]):
            process_matrix = _combine_process_matrices([gate.process_matrix for gate in gates])
            return cls(gates[0].basis, process_matrix, gates)
        pmats_list = []
        for gate in gates:
            if gate.process_matrix_function is not None and noise.parameter_name in gate.parameters:
                arguments = np.array(list(gate.parameters.values()))
                vec = np.array([1 if noise.parameter_name == name else 0 for name in gate.parameters])
                pmats = [gate.process_matrix_function(*list(arguments + darg * vec)) for darg in noise.domain_arguments]
            else:
                pmats = [gate.process_matrix for darg in noise.domain_arguments]
            pmats_list.append(pmats)
        new_pmats_list = [[pmats[i] for pmats in pmats_list] for i in range(len(pmats_list[0]))]
        process_mats = [_combine_process_matrices(ps) for ps in new_pmats_list]
        probs = [noise.probability_density_function(darg) for darg in noise.domain_arguments]
        ys = np.array([p * chi for p, chi in zip(probs, process_mats)])
        process_matrix = trapz_for_matrix(ys, noise.domain_arguments) 
        return cls(gates[0].basis, process_matrix, gates)

def _combine_process_matrices(process_matrices: list[Matrix]):
    """Combine a series of process matrices (in chronological order) into a single process matrix for the whole circuit."""
    if len(process_matrices) == 1:
        return process_matrices[0]
    else:
        return np.linalg.multi_dot(process_matrices[::-1])


def compute_memory_kernel(
    E_series: np.ndarray,
    t: np.ndarray,
    n_talbot: int = 32,
    time_stride: int = 1,
    method: str = 'talbot',
) -> np.ndarray:
    """
    Compute the memory kernel K(t) from a precomputed time series of the process
    superoperator E(t).

    Two methods are available:

    1) 'talbot' (default): Laplace-domain inversion via fixed Talbot contour.
       The Nakajima-Zwanzig equation in Laplace space yields:

           K_hat(z) = z*I - E_hat(z)^{-1}

       where E_hat(z) is the numerical Laplace transform of E(t).

    2) 'volterra': direct discrete-time Volterra recursion from

           dE/dt ≈ Σ_{m=0}^n K[n-m] E[m] Δt,

       which is generally more robust for reconstruction diagnostics.

    Parameters
    ----------
    E_series : ndarray, shape (N, d2, d2)
        Time series of the process superoperator, E_series[n] = E(t[n]).
        E_series[0] must equal the identity (E(0) = I).
    t : ndarray, shape (N,)
        Uniformly spaced time grid with t[0] = 0.
    n_talbot : int
        Number of Talbot quadrature points (default 32; increase to e.g. 64
        for higher accuracy at greater cost).
    time_stride : int
        Evaluate kernel extraction every `time_stride` time points and linearly
        interpolate intermediate values. For 'talbot', this substantially
        reduces runtime; for 'volterra', this is equivalent to solving on a
        coarser grid then interpolating back.
    method : str
        One of {'talbot', 'volterra'}.

    Returns
    -------
    ndarray, shape (N, d2, d2)
        Memory kernel time series K_series[n] = K(t[n]).
        K_series[0] is estimated by nearest-neighbour extrapolation from K(t[1])
        because the Talbot method is singular at t = 0.
    """
    if time_stride < 1:
        raise ValueError("time_stride must be >= 1")

    if method not in {'talbot', 'volterra'}:
        raise ValueError("method must be one of {'talbot', 'volterra'}")

    # ------------------------------------------------------------------
    # Direct Volterra recursion (reconstruction-friendly)
    # ------------------------------------------------------------------
    if method == 'volterra':
        N, d2, _ = E_series.shape

        eval_indices = np.arange(0, N, time_stride, dtype=int)
        if eval_indices[-1] != N - 1:
            eval_indices = np.append(eval_indices, N - 1)

        t_work = np.asarray(t[eval_indices], dtype=float)
        E_work = np.asarray(E_series[eval_indices], dtype=complex)
        Nw = len(t_work)
        if Nw < 2:
            raise ValueError("Need at least two time points to compute memory kernel")

        # Enforce the exact Volterra initial condition E(0) = I.
        # This stabilizes the recursion against small numerical drift in E_series[0].
        E_work[0] = np.eye(d2, dtype=complex)

        dt_work = t_work[1] - t_work[0]
        # Use Savitzky-Golay derivative for improved noise robustness over np.gradient.
        max_window = min(11, Nw if (Nw % 2 == 1) else (Nw - 1))
        if max_window >= 3:
            polyorder = min(3, max_window - 1)
            dE_work = (
                savgol_filter(E_work.real, window_length=max_window, polyorder=polyorder, deriv=1, delta=dt_work, axis=0)
                + 1j * savgol_filter(E_work.imag, window_length=max_window, polyorder=polyorder, deriv=1, delta=dt_work, axis=0)
            )
        else:
            dE_work = np.gradient(E_work, dt_work, axis=0)

        try:
            E0_inv = np.linalg.inv(E_work[0])
        except np.linalg.LinAlgError:
            E0_inv = np.linalg.pinv(E_work[0], rcond=1e-12)

        K_work = np.zeros((Nw, d2, d2), dtype=complex)
        K_work[0] = dE_work[0] @ E0_inv

        for n in range(1, Nw):
            acc = dt_work * np.einsum('mij,mjk->ik', K_work[n-1::-1], E_work[1:n+1], optimize=True)
            K_work[n] = (dE_work[n] - acc) @ E0_inv

        if time_stride != 1:
            K_full = np.zeros((N, d2, d2), dtype=complex)
            K_flat_work = K_work.reshape(Nw, -1)
            K_flat_full = K_full.reshape(N, -1)
            for col in range(K_flat_full.shape[1]):
                K_flat_full[:, col] = np.interp(t, t_work, K_flat_work[:, col])
            return K_full

        return K_work

    N, d2, _ = E_series.shape
    dt = t[1] - t[0]
    I_mat = np.eye(d2, dtype=complex)
    M = n_talbot
    E_weighted_flat = E_series.reshape(N, -1).copy()
    E_weighted_flat[0] *= 0.5
    E_weighted_flat[-1] *= 0.5

    # ------------------------------------------------------------------
    # Precompute Talbot contour nodes and weights (independent of t_j).
    # For the k-th node (k = 1 … M-1):
    #   s_k = r * theta_k * (cot(theta_k) + i),   r = 2M / (5 * t_j)
    #   omega_k = exp(i*theta_k) * (1 + i*(theta_k*csc²(theta_k) - cot(theta_k)))
    # Reference: Abate & Valko (2004), ACM TOMS Algorithm 843.
    # ------------------------------------------------------------------
    k_arr = np.arange(1, M)                                    # (M-1,)
    theta = k_arr * np.pi / M                                  # (M-1,)
    cot_theta = np.cos(theta) / np.sin(theta)                  # (M-1,)
    sigma_norm = theta * (cot_theta + 1j)                      # (M-1,)  normalised nodes
    omega = np.exp(1j * theta) * (
        1.0 + 1j * (theta / np.sin(theta) ** 2 - cot_theta)
    )                                                          # (M-1,)  quadrature weights

    K_series = np.zeros((N, d2, d2), dtype=complex)
    eval_indices = np.arange(1, N, time_stride, dtype=int)
    if eval_indices[-1] != N - 1:
        eval_indices = np.append(eval_indices, N - 1)

    K_eval = np.zeros((len(eval_indices), d2, d2), dtype=complex)
    EXP_CLIP = 700.0  # float64-safe exp() range guard
    # These Talbot factors are independent of i because r * t_j = 2M/5.
    talbot_scale = 2.0 * M / 5.0
    exp_rt = np.exp(talbot_scale)
    weighted_sigma = np.exp(talbot_scale * sigma_norm) * omega  # (M-1,)

    for pos, i in enumerate(eval_indices):
        t_j = t[i]
        r = 2.0 * M / (5.0 * t_j)                             # Talbot scale factor

        # Collect all M contour points in a single batch: k=0 (real) + k=1…M-1
        s_all = np.empty(M, dtype=complex)
        s_all[0] = r                                           # k=0: real node
        s_all[1:] = r * sigma_norm                            # k=1…M-1: complex nodes

        # Numerical Laplace transform  E_hat(s_k) = ∫₀ᵀ exp(-s_k·t) E(t) dt
        # via trapezoidal weights pre-applied to E_weighted_flat.
        arg = -np.outer(s_all, t)                               # (M, N) complex
        arg = np.clip(arg.real, -EXP_CLIP, EXP_CLIP) + 1j * arg.imag
        exp_mat = np.exp(arg)
        E_hat = (exp_mat @ E_weighted_flat).reshape(M, d2, d2) * dt

        # K_hat(s) = s·I − E_hat(s)⁻¹
        # Fast path: batched inverse. Fallback to batched pseudo-inverse only if needed.
        try:
            E_hat_inv = np.linalg.inv(E_hat)
        except np.linalg.LinAlgError:
            E_hat_inv = np.linalg.pinv(E_hat, rcond=1e-12)
        K_hat = s_all[:, None, None] * I_mat[None] - E_hat_inv

        # ------------------------------------------------------------------
        # Talbot summation
        # k=0: half-weight real contribution  0.5 · exp(r·t_j) · K_hat[0]
        # k=1…M-1: Re[ exp(s_k·t_j) · omega_k · K_hat[k] ]
        # Overall prefactor: 2 / (5·t_j)
        # ------------------------------------------------------------------
        result = 0.5 * exp_rt * K_hat[0]                      # (d2, d2)

        contrib = weighted_sigma[:, None, None] * K_hat[1:]   # (M-1, d2, d2) complex
        result += np.sum(contrib, axis=0)                      # (d2, d2)

        K_eval[pos] = (2.0 / (5.0 * t_j)) * result

    if time_stride == 1:
        K_series[1:] = K_eval
    else:
        t_eval = t[eval_indices]
        K_flat_eval = K_eval.reshape(len(eval_indices), -1)
        K_flat_all = K_series.reshape(N, -1)
        for col in range(K_flat_all.shape[1]):
            K_flat_all[:, col] = np.interp(t, t_eval, K_flat_eval[:, col])

    K_series[0] = K_series[1]    # nearest-neighbour extrapolation to t = 0
    return K_series


def compute_kernel_length(
    K_series: np.ndarray,
    t: np.ndarray,
    threshold: float = 0.01,
    method: str = 'integral_tail',
    ignore_instantaneous: bool = True,
    noise_tail_fraction: float = 0.2,
    noise_multiplier: float = 10.0,
    min_consecutive: int = 1,
) -> float:
    """
    Estimate the memory kernel length (correlation time) τ_K from a kernel time series.

    By default (`method='integral_tail'`), this returns the integral timescale

        τ_K = (∫ t ||K(t)||_F dt) / (∫ ||K(t)||_F dt),

    evaluated on the memory tail (optionally excluding the first bin, which can
    contain an instantaneous/Markov-like contribution).

    For backward compatibility, `method='threshold_after_peak'` retains the
    previous threshold-crossing definition.

    Parameters
    ----------
    K_series : ndarray, shape (N, d2, d2)
        Memory kernel time series, typically the output of `compute_memory_kernel`.
    t : ndarray, shape (N,)
        Time grid corresponding to K_series.
    threshold : float
        Used only when `method='threshold_after_peak'`.
        Fraction of peak Frobenius norm used as the decay cutoff.
    method : str
        Kernel-length definition:
        - 'integral_tail' (default): weighted integral timescale of ||K||_F.
        - 'threshold_after_peak': first post-peak crossing below threshold.
        - 'support_above_noise': largest time where ||K||_F exceeds
          b + noise_multiplier·MAD, with b/MAD estimated from the last
          `noise_tail_fraction` of the (optionally instantaneous-excluded) tail.
    ignore_instantaneous : bool
        If True, exclude the first time bin (n=0) from the memory-tail
        calculation. This is useful when K[0] captures an instantaneous
        delta-like contribution.
    noise_tail_fraction : float
        Fraction of late-time tail used to estimate noise floor for
        `method='support_above_noise'`.
    noise_multiplier : float
        Multiplier for MAD in the support-above-noise cutoff:
            cutoff = median + noise_multiplier * MAD
    min_consecutive : int
        Minimum number of consecutive points above cutoff required for
        support-above-noise detection.

    Returns
    -------
    tau_K : float
        Estimated kernel length τ_K.
    norms : ndarray, shape (N,)
        Frobenius norm of the kernel at each time step, ||K(t_n)||_F.
    """
    # Frobenius norm at each time step: ||K(t_n)||_F = sqrt(sum |K_ij(t_n)|²)
    norms = np.linalg.norm(K_series, ord='fro', axis=(1, 2))
    finite = np.isfinite(norms)
    if not np.any(finite):
        return float('nan'), norms

    start_idx = 1 if (ignore_instantaneous and len(t) > 1) else 0

    if method == 'integral_tail':
        tail_t = np.asarray(t[start_idx:], dtype=float)
        tail_norms = np.asarray(norms[start_idx:], dtype=float)
        tail_finite = np.isfinite(tail_t) & np.isfinite(tail_norms)
        if np.count_nonzero(tail_finite) < 2:
            return float('nan'), norms
        tail_t = tail_t[tail_finite]
        tail_norms = tail_norms[tail_finite]

        denom = np.trapezoid(tail_norms, tail_t)
        if not np.isfinite(denom) or denom <= 0:
            return float('nan'), norms
        numer = np.trapezoid(tail_t * tail_norms, tail_t)
        return float(numer / denom), norms

    if method == 'threshold_after_peak':
        search_norms = np.where(finite, norms, -np.inf)
        if start_idx > 0:
            search_norms[:start_idx] = -np.inf

        peak_idx = int(np.nanargmax(search_norms))
        peak_val = float(norms[peak_idx])
        cutoff = threshold * peak_val

        # Search only after the peak to avoid trivial τ_K = 0 when the norm starts
        # below a later transient maximum.
        post_peak_idx = np.arange(peak_idx, len(t))
        below = post_peak_idx[np.where(finite[post_peak_idx] & (norms[post_peak_idx] < cutoff))[0]]
        return (float(t[below[0]]) if len(below) > 0 else float(t[-1])), norms

    if method == 'support_above_noise':
        if min_consecutive < 1:
            raise ValueError('min_consecutive must be >= 1')
        if not (0 < noise_tail_fraction <= 1):
            raise ValueError('noise_tail_fraction must be in (0, 1]')

        tail_t = np.asarray(t[start_idx:], dtype=float)
        tail_norms = np.asarray(norms[start_idx:], dtype=float)
        tail_finite = np.isfinite(tail_t) & np.isfinite(tail_norms)
        if np.count_nonzero(tail_finite) < 2:
            return float('nan'), norms
        tail_t = tail_t[tail_finite]
        tail_norms = tail_norms[tail_finite]

        n_tail = len(tail_norms)
        n_noise = max(1, int(np.ceil(noise_tail_fraction * n_tail)))
        noise_region = tail_norms[-n_noise:]
        baseline = float(np.median(noise_region))
        mad = float(np.median(np.abs(noise_region - baseline)))
        cutoff = baseline + noise_multiplier * mad

        above = tail_norms > cutoff
        if not np.any(above):
            return float('nan'), norms

        if min_consecutive == 1:
            return float(tail_t[np.where(above)[0][-1]]), norms

        valid = np.zeros_like(above, dtype=bool)
        run_start = None
        for i, flag in enumerate(above):
            if flag and run_start is None:
                run_start = i
            elif (not flag) and run_start is not None:
                if (i - run_start) >= min_consecutive:
                    valid[run_start:i] = True
                run_start = None
        if run_start is not None and (len(above) - run_start) >= min_consecutive:
            valid[run_start:len(above)] = True

        if not np.any(valid):
            return float('nan'), norms
        return float(tail_t[np.where(valid)[0][-1]]), norms

    raise ValueError("method must be one of {'integral_tail', 'threshold_after_peak', 'support_above_noise'}")
