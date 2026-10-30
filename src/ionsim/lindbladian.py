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

from ionsim.basis import StandardBasis
from ionsim.degree_of_freedom import AtomicStructure 
from ionsim.operator import Operator, Coupling, EnergyShift, GeneralOperator, EnergyShiftOperator, CouplingOperator
from ionsim.custom_types import Vector, Matrix, SparseMatrix, AnyMatrix, as_dense_matrix
from ionsim.custom_math import matrix_AYB_multiply_to_superoperator, solve_time_evolution_equation, solve_ode
from ionsim.ionsim_error import IonSimError
from ionsim.composite_operator import CompositeOperator
from ionsim.hamiltonian import Hamiltonian
from ionsim.atomic_internal_energy_level import AtomicInternalEnergyLevel, SinkLevel
from ionsim.atomic_internal_energy_level import compute_dipole_amplitude
from ionsim.config import NUMERICAL_EQUIVALENCE_THRESHOLD

REPRESENTATIONS = ('supervector', 'density_matrix')


@dataclass(frozen=True, eq=False)
class LindbladOperatorParts:
    """ A Lindblad operator in the rotating frame, precomputed once: L(t) = static + sum_k values_k e^{-i rates_k t} |rows_k><cols_k|.

        static: the diagonal (energy-shift) elements, which have no frame shift. The off-diagonal elements carry their
        frame-shifted oscillation rates. sparse selects the matrix type returned by matrix(t).
    """
    size: int
    static: AnyMatrix | None
    rows: np.ndarray
    columns: np.ndarray
    values: np.ndarray
    rates: np.ndarray
    sparse: bool = False

    def matrix(self, t: float) -> AnyMatrix:
        """ L(t) as an N x N matrix (sparse if self.sparse). """
        data = self.values * np.exp(-1j * self.rates * t) if self.values.size else self.values
        if self.sparse:
            L = csr_matrix((data, (self.rows, self.columns)), shape=(self.size, self.size), dtype=complex)
            return L + self.static if self.static is not None else L
        L = np.zeros((self.size, self.size), dtype=complex)
        if self.static is not None:
            L += self.static
        np.add.at(L, (self.rows, self.columns), data)
        return L

    @cached_property
    def common_rate(self) -> float | None:
        """ The single rate R if L(t) = e^{-iRt} L(0) (all elements share one rate; diagonal elements have rate 0), else None.
            Then the overall phase cancels in L rho L^dagger and L^dagger L, so the operator acts as if static. """
        rates = list(self.rates)
        if self.static is not None:
            has_static = self.static.count_nonzero() > 0 if hasattr(self.static, 'count_nonzero') else np.any(self.static != 0)
            if has_static:
                rates.append(0.)
        if not rates:
            return 0.
        rates = np.asarray(rates, dtype=float)
        tolerance = 1e-12 * max(1., float(np.max(np.abs(rates))))
        return float(rates[0]) if np.all(np.abs(rates - rates[0]) <= tolerance) else None


@dataclass(frozen=True, eq=False)
class Dissipator(CompositeOperator):
    """Class for dissipator object which implements dissipative phenomena in the context of open quantum systems. 

        Inherits from CompositeOperator parent class. Instantiation requires a basis and 
            list of operators corresponding to Lindblad operators.  
    """
    def __post_init__(self):
        super().__post_init__()

    @staticmethod
    def lindblad_matrix_to_superoperator(L_matrix: AnyMatrix) -> AnyMatrix:
        """ Method to convert a Lindblad operator in matrix form (N x N) to a superoperator (N^2 x N^2) 
            for matrix-vector multiplication on a flattened density matrix (shape: N^2 x 1) 
    
            Lindblad master equation contains Lindblad operators {L} with 3 contributions:
            1. L rho L^†
            2. -1/2 L^† L rho 
            3. -1/2 rho L^† L 
    
            - assumes any decay rates have been lumped into definition of L matrix.
    
            For matrix-matrix operations, the mapping to the (column-stacked) superoperator (y) is: 
            A rho --> (I kron A) y  
            rho A --> (A^{T} kron I) y  
            A rho B --> (B^{T} kron A) y 
        """
        LdaggerL = np.conj(L_matrix.T) @ L_matrix # L^† L: 

        superoperator = matrix_AYB_multiply_to_superoperator(A = LdaggerL, B = None) 
        superoperator += matrix_AYB_multiply_to_superoperator(A = None, B = LdaggerL) 
        superoperator *= -0.5
    
        superoperator += matrix_AYB_multiply_to_superoperator(L_matrix, np.conjugate(L_matrix.T)) 
        return superoperator 

    @cached_property
    def lindblad_matrix_functions(self) -> list[Callable]:
        """Creates and stores a list of callables that return (N x N) matrices as a function of time. 
            Each callable corresponds to a lindblad operator in the interaction picture, which may contain time-dependence as a result of frame shifting. 

            Defines and returns a list of callables as a function of time"""

        lindblad_functions = []

        for lindblad_op in self.operators:
            lindblad_functions.append(self.create_lindblad_matrix_function(lindblad_op))

        return lindblad_functions

    def create_lindblad_matrix_function(self, lindblad_operator: Operator) -> Callable:
        """Converts a lindblad Operator object to a callable that returns a matrix at a given time point.
            The frame-shifted elements are computed once (lindblad_operator_parts), not at every evaluation."""
        parts = self._lindblad_operator_parts(lindblad_operator)

        def _lindblad_function(t: float) -> AnyMatrix:
            return parts.matrix(t)

        return _lindblad_function

    @cached_property
    def lindblad_operator_parts(self) -> list[LindbladOperatorParts]:
        """ The frame-shifted elements of every Lindblad operator, computed once (see LindbladOperatorParts). """
        return [self._lindblad_operator_parts(operator) for operator in self.operators]

    def _lindblad_operator_parts(self, lindblad_operator: Operator) -> LindbladOperatorParts:
        """ Splits a Lindblad operator into its static diagonal part and its frame-shifted off-diagonal elements. """
        static = None
        # Static, diagonal contribution
        if isinstance(lindblad_operator, GeneralOperator):
            if lindblad_operator.energy_shift_operator_contribution:
                static = lindblad_operator.energy_shift_operator_contribution.static_matrix
        elif isinstance(lindblad_operator, EnergyShiftOperator):
            static = lindblad_operator.static_matrix
        if static is not None:
            static = csr_matrix(static, dtype=complex) if self.sparse else np.asarray(static.toarray(), dtype=complex)

        rows = columns = np.zeros(0, dtype=int)
        values, rates = np.zeros(0, dtype=complex), np.zeros(0)
        if isinstance(lindblad_operator, CouplingOperator) or isinstance(lindblad_operator, GeneralOperator):
            # Must account for frame shifts in coupling operators: L_int * exp(-1j * Rate * t), element-wise
            L_int, Rate = self._frame_shifted_coupling_matrix_and_rate_from_operator(lindblad_operator)
            L_int = csr_matrix(L_int, dtype=complex)
            L_int.sum_duplicates()
            L_int.eliminate_zeros()
            coo = L_int.tocoo()
            rows, columns, values = coo.row, coo.col, coo.data
            rates = np.asarray(csr_matrix(Rate)[rows, columns], dtype=float).ravel() if rows.size else np.zeros(0)
        return LindbladOperatorParts(self.size, static, rows, columns, values, rates, self.sparse)
            
    @cached_property
    def dissipator_matrix_function(self) -> Callable:
        """Creates N^2 x N^2 matrix representation of each Lindblad operator's action on a N^2 x 1 superoperator (y) from NxN density matrix (rho): 
                Dissipator acting on supervector from Lindblad operator i (L_i): d_i(t)y <==> L_i(t) rho L†_i(t) - 1/2 { L†_i(t) L_i(t) , rho }  

            Returns the sum of the dissipators: D(t) = sum_i d_i (t) 
        """
        # Transform to dissipator form for superoperator matrix-vector multiplication.  
        def _dissipator_matrix_function(t: float):
            if self.sparse:
                d_matrix = csr_matrix(([0], ([0], [0])), shape=(self.size**2, self.size**2), dtype='complex') 
            else:
                d_matrix = np.zeros((self.size**2, self.size**2), dtype=complex)

            for L_i in self.lindblad_matrix_functions:
                d_matrix += self.lindblad_matrix_to_superoperator(L_i(t))
            return d_matrix

        return _dissipator_matrix_function 


@dataclass(frozen=True, eq=False)
class DissipatorSpontaneousEmission(Dissipator):
    """Subclass for including spontaneous emission dynamics for a known atomic structure."""

    @classmethod
    def from_atomic_structure_data(cls, basis: StandardBasis, ground_levels: list[AtomicInternalEnergyLevel],
                                   excited_levels: list[AtomicInternalEnergyLevel], frame_energies: list[float] | None = None,
                                   sparse: bool = False, select_dofs: list[AtomicStructure] | None = None,
                                   decay_to_sink: bool = False):
        """Build the spontaneous-emission dissipator for decay from excited_levels to ground_levels.

        One Lindblad operator sqrt(Gamma_{e->g,q}) |g><e| is created per decay path (excited level e, ground level g,
        polarization q), in every selected AtomicStructure DOF. The rate of each path is exact:

            Gamma_{e->g,q} = (branching ratio of g's manifold) / (lifetime of e) * |<g| r_q |e>|^2 (2 J_e + 1) / (2 J_g + 1)

        where <g| r_q |e> is compute_dipole_amplitude in units of <J_g||r||J_e>. Summed over every sublevel of a
        ground manifold and over q, the factor after the branching ratio is exactly 1, so the rates do not depend on
        which levels are included.

        The lifetime is the level's total lifetime (all decay channels, as measured), and branching ratios are fractions
        of that total decay. Decay to states that are not included (omitted sublevels, manifolds outside the basis) 
        is not modeled by default, so the excited level decays more slowly than 1/lifetime. With decay_to_sink=True, that remaining decay,
        1/lifetime - (sum of modeled rates), goes to the DOF's SinkLevel instead, so every excited level decays at
        exactly 1/lifetime. Build the structure with AtomicStructure.from_species(..., include_sink=True).
        A level without branching_ratios is assumed to decay only to its one ground manifold, so only decay to omitted
        sublevels of that manifold reaches the sink. For levels with several decay channels, such as Rydberg states,
        give branching_ratios for the modeled manifolds, so the rest of the decay goes to the sink.

        Decay channels come from the excited level's branching_ratios, keyed by the ground manifold's term symbol.
        Ground manifolds missing from branching_ratios are not decay channels of that level. An excited level with no
        branching_ratios is assumed to decay entirely to a single ground manifold, and an error is raised if the
        ground levels span more than one manifold it could decay to.

        Only electric-dipole (rank-1) angular factors are used. A decay listed in branching_ratios that has no dipole-allowed
        paths (e.g. an E2 decay with Delta J = 2) raises an error. An E2 decay with |Delta J| <= 1 (e.g. D3/2 -> S1/2) gets
        the correct total rate but a dipole-like distribution over sublevels, since parity is not known here.

        Separate operators per path is the secular approximation: coherences between decay paths that emit the
        same polarization are dropped. This is accurate when those transition frequencies differ by much more than
        the decay rate, but not for nearly degenerate excited sublevels (e.g. Zeeman sublevels at very low field).

        Args:
            basis: The system basis.
            ground_levels: Levels that excited levels may decay to. Matched to each DOF's levels by name.
            excited_levels: Levels that decay. Each needs a lifetime (s). Matched to each DOF's levels by name.
            frame_energies: Rotating-frame energies of the basis states; zeros (the lab frame) if omitted.
            sparse: Whether to use sparse matrices.
            select_DOFs: AtomicStructure DOFs of the basis to add decay to; all of them if omitted. Every selected
                DOF must contain all of the given levels.
            decay_to_sink: Send decay that does not reach the given ground levels to each DOF's SinkLevel. With this set,
                ground_levels may be empty (all decay goes to the sink).
        """
        if frame_energies is None:
            frame_energies = [0.] * len(basis.states)

        if select_dofs is None:
            dof_list = basis.atomic_structure_dofs
        else:
            dof_list = select_dofs
            if any(not any(dof is basis_dof for basis_dof in basis.degrees_of_freedom) for dof in dof_list):
                raise IonSimError("Every DOF in select_DOFs must be an AtomicStructure degree of freedom of the basis.")
        if not dof_list:
            raise IonSimError("The basis has no AtomicStructure degrees of freedom to add spontaneous emission to.")

        lindblad_operators = []
        for dof in dof_list:
            dimension = len(dof.energy_levels)
            level_index = {level.name: index for index, level in enumerate(dof.energy_levels)}
            missing = [level.name for level in [*ground_levels, *excited_levels] if level.name not in level_index]
            if missing:
                raise IonSimError(f"Levels {missing} are not in the degree of freedom {dof.name or ''}. "
                                  f"Use select_dofs to choose which atomic structures to add decay to.")
            # Use this DOF's own level objects, so ions built separately (or at different fields) are handled correctly.
            ground = [dof.energy_levels[level_index[level.name]] for level in ground_levels]
            excited = [dof.energy_levels[level_index[level.name]] for level in excited_levels]
            if decay_to_sink:
                sinks = [index for index, level in enumerate(dof.energy_levels) if isinstance(level, SinkLevel)]
                if len(sinks) != 1:
                    raise IonSimError(f"decay_to_sink requires exactly one SinkLevel in each selected dof, found {len(sinks)}. "
                                      f"Build the structure with AtomicStructure.from_species(..., include_sink=True).")

            def add_decay(to_index: int, from_index: int, rate: float):
                lowering_matrix = np.zeros((dimension, dimension))
                lowering_matrix[to_index, from_index] = np.sqrt(rate)
                lindblad_operators.append(CouplingOperator.from_matrix(basis, basis.enlarge_matrix(lowering_matrix, [dof]), 0.))

            for e_level in excited:
                rates = cls._decay_rates(e_level, ground, require_paths=not decay_to_sink)
                for g_level, rate in rates:
                    add_decay(level_index[g_level.name], level_index[e_level.name], rate)
                if decay_to_sink:
                    remaining = 1. / e_level.lifetime - sum(rate for _, rate in rates)
                    if remaining > NUMERICAL_EQUIVALENCE_THRESHOLD / e_level.lifetime:
                        add_decay(sinks[0], level_index[e_level.name], remaining)

        return cls(basis, lindblad_operators, frame_energies, sparse)

    @staticmethod
    def _decay_rates(e_level: AtomicInternalEnergyLevel, ground_levels: list[AtomicInternalEnergyLevel],
                     require_paths: bool = True) -> list[tuple]:
        """ Decay rates (1/s) from e_level to each ground level, one entry per (ground level, polarization q) path.

            Raises if there are no paths and require_paths is True.
        """
        lifetime = e_level.lifetime
        if not isinstance(lifetime, (int, float)) or lifetime <= 0:
            raise IonSimError(f"Excited level {e_level.name} needs a positive lifetime (s) for spontaneous emission, got {lifetime!r}.")

        # Dipole-allowed paths, grouped by ground manifold. The fraction of e's decay into a complete manifold sums to 1.
        paths = {}  # term symbol -> list of (ground level, fraction)
        for g_level in ground_levels:
            for q in (-1, 0, 1):
                amplitude = compute_dipole_amplitude(g_level, e_level, q)
                if abs(amplitude) > NUMERICAL_EQUIVALENCE_THRESHOLD:
                    fraction = abs(amplitude)**2 * (2*e_level.j + 1) / (2*g_level.j + 1)
                    paths.setdefault(g_level.term_symbol, []).append((g_level, fraction))

        branching_ratios = e_level.branching_ratios
        if branching_ratios is not None:
            # A listed channel with no dipole-allowed paths (e.g. an E2 decay with Delta J = 2) would silently vanish.
            ground_manifolds = {g_level.term_symbol for g_level in ground_levels}
            non_dipole = [term_symbol for term_symbol, ratio in branching_ratios.items()
                          if ratio > 0 and term_symbol in ground_manifolds and term_symbol not in paths]
            if non_dipole:
                raise IonSimError(f"Excited level {e_level.name} decays to {non_dipole} (branching_ratios), but no electric-dipole "
                                  f"paths connect them; higher-multipole decays (e.g. E2) are not supported.")
        if branching_ratios is None:
            if len(paths) > 1:
                raise IonSimError(f"Excited level {e_level.name} has no branching ratios, but can decay to several ground "
                                  f"manifolds {sorted(paths)}. Add branching_ratios to its config entry.")
            branching_ratios = {term_symbol: 1. for term_symbol in paths}

        rates = []
        for term_symbol, manifold_paths in paths.items():
            branching_ratio = branching_ratios.get(term_symbol, 0.)
            for g_level, fraction in manifold_paths:
                if branching_ratio == 0.:
                    continue
                if e_level.energy <= g_level.energy:
                    raise IonSimError(f"Excited level {e_level.name} (energy {e_level.energy}) must be higher in energy than "
                                      f"ground level {g_level.name} (energy {g_level.energy}) to decay to it.")
                rates.append((g_level, branching_ratio / lifetime * fraction))

        if not rates and require_paths:
            raise IonSimError(f"No decay paths from excited level {e_level.name} to the given ground levels. Dipole-allowed "
                              f"ground manifolds: {sorted(paths)}; branching_ratios keys: "
                              f"{sorted(e_level.branching_ratios) if e_level.branching_ratios else None}.")
        return rates


@dataclass(frozen=True, eq=False)
class Lindbladian:
    hamiltonian: Hamiltonian | None  
    dissipator: Dissipator | None 

    def __post_init__(self):
        if self.hamiltonian and self.dissipator:
            # Check if Hamiltonian and Dissipator are the same dimensionality 
            if self.hamiltonian.size != self.dissipator.size:
                raise IonSimError('Hamiltonian and Dissipator objects must have the same size (dimensionality)')
            # Check if Hamiltonian and Dissipator have the same sparse setting.  
            if self.hamiltonian.sparse != self.dissipator.sparse: 
                raise IonSimError('Input error: Both Hamiltonian and Dissipator should have the same setting for sparse variable. e.g. Set sparse = True when constructing the Hamiltonian and Dissipator objects.')
        if self.hamiltonian is None and self.dissipator is None:
            raise IonSimError('Input error: Both Hamiltonian and Dissipator inputs are None')

    @property
    def size(self):
        # Lindbladian size is N^2, corresponding to an N^2 x N^2 superoperator representation
        if self.hamiltonian and self.dissipator:
            return self.hamiltonian.size**2
        elif self.hamiltonian and (self.dissipator is None):
            return self.hamiltonian.size**2
        elif self.dissipator and self.hamiltonian is None:
            return self.dissipator.size**2

    @cached_property
    def matrix_function(self) -> Callable:
        """ Lindbladian matrix function L(t), corresponding to an N^2 x N^2 superoperator at time t. """
        if self.hamiltonian:
            super_ham = lambda t: -1j*(
                  matrix_AYB_multiply_to_superoperator(A = self.hamiltonian.hamiltonian_function(t), B = None) 
                - matrix_AYB_multiply_to_superoperator(A = None, B = self.hamiltonian.hamiltonian_function(t))
                )
        else:
            super_ham = lambda t: 0. 

        if self.dissipator:
            super_dissipator = self.dissipator.dissipator_matrix_function
        else:
            super_dissipator = lambda t: 0. 

        # Lindbladian superoperator from hamiltonian and dissipation contributions: 
        lindbladian_function = lambda t: super_ham(t) + super_dissipator(t)
        return lindbladian_function

    def evolve_supervector(self, initial_supervector: Vector, duration: float, time_evals: Vector | None = None, **kwargs):
        """ Evolve a supervector by solving the time-dependent Lindblad master equation.
            e.g. evolves supervector "y" using dy/dt = Ly, where L is the N^2 x N^2 dissipator matrix. 
        """
        # solve_time_evolution_equation() assumes a Schrodinger equation form dy/dt = (-i*A)y, where i = sqrt(-1) and A <==> the function input, e.g. a Hamiltonian matrix. 
        # Therfore, we must compensate this form by multiplying the lindbladian by i 
        assert(self.size == len(initial_supervector))
        dynamical_matrix = lambda t: self.matrix_function(t) * 1j
        return solve_time_evolution_equation(dynamical_matrix, initial_supervector, duration, time_evals, **kwargs)

    @property
    def dimension(self) -> int:
        """ Dimension N of the Hilbert space (the density matrix is N x N). """
        return self.hamiltonian.size if self.hamiltonian else self.dissipator.size

    def evolve_density_matrix(self, initial_density_matrix: Matrix, duration: float, time_evals: Vector | None = None,
                              **solver_options):
        """ Evolve a density matrix by solving the time-dependent Lindblad master equation in matrix form,

                d rho / dt = -i [H(t), rho] + sum_k ( L_k(t) rho L_k(t)^dagger - 1/2 { L_k(t)^dagger L_k(t), rho } ),

            without building the N^2 x N^2 superoperator: each step costs O(N^3) per term (O(N^4) for the superoperator's
            matrix-vector product) and memory is O(N^2). The same equation as evolve_supervector, so results agree to
            solver tolerance.

            initial_density_matrix: N x N; need not be Hermitian (e.g. |i><j| when building a process matrix).
            solver_options: ode_solver and its options (see custom_math.solve_ode). Explicit solvers suit this form (e.g.
                ode_solver='solve_ivp', method='DOP853', or 'zvode' with method='adams'); for stiff problems prefer
                evolve_supervector, whose Jacobian (the superoperator) the implicit solvers use.

            Returns (times, density matrices).
        """
        n = self.dimension
        rho0 = initial_density_matrix.toarray() if hasattr(initial_density_matrix, 'toarray') else np.asarray(initial_density_matrix)
        if rho0.shape != (n, n):
            raise IonSimError(f"The initial density matrix must be {n} x {n}, got shape {rho0.shape}.")
        times, vectors = solve_ode(self.density_matrix_right_hand_side, rho0.astype(complex).ravel(), duration, time_evals,
                                   **solver_options)
        return times, [np.asarray(vector).reshape(n, n) for vector in vectors]

    @cached_property
    def density_matrix_right_hand_side(self) -> Callable:
        """ f(t, y) = d rho / dt for the row-major flattened density matrix y (see evolve_density_matrix).

            Lindblad operators whose elements share one oscillation rate (e.g. a single decay path) satisfy
            L(t) = e^{-iRt} L(0); the phase cancels, so L, L^dagger and their summed L^dagger L are precomputed. The other
            operators (e.g. coherent decay channels in a rotating frame) are evaluated at each time.
        """
        n = self.dimension
        hamiltonian_function = self.hamiltonian.hamiltonian_function if self.hamiltonian else None
        static_operators, time_dependent_operators = [], []
        static_LdL = None
        for parts in (self.dissipator.lindblad_operator_parts if self.dissipator else []):
            if parts.common_rate is not None:
                L = parts.matrix(0.)
                Ld = L.conj().T
                static_operators.append((L, Ld))
                static_LdL = Ld @ L if static_LdL is None else static_LdL + Ld @ L
            else:
                time_dependent_operators.append(parts)

        def right_hand_side(t: float, y: Vector) -> Vector:
            rho = y.reshape(n, n)
            drho = np.zeros((n, n), dtype=complex)
            if hamiltonian_function is not None:
                H = hamiltonian_function(t)
                drho += -1j * (_left_multiply(H, rho) - _right_multiply(rho, H))
            for L, Ld in static_operators:
                drho += _right_multiply(_left_multiply(L, rho), Ld)
            if static_LdL is not None:
                drho -= 0.5 * (_left_multiply(static_LdL, rho) + _right_multiply(rho, static_LdL))
            for parts in time_dependent_operators:
                L = parts.matrix(t)
                Ld = L.conj().T
                LdL = Ld @ L
                drho += _right_multiply(_left_multiply(L, rho), Ld) - 0.5 * (_left_multiply(LdL, rho) + _right_multiply(rho, LdL))
            return drho.ravel()

        return right_hand_side


def _left_multiply(A: AnyMatrix, rho: np.ndarray) -> np.ndarray:
    """ A @ rho as a dense array, for dense or sparse A. """
    return np.asarray(A @ rho)


def _right_multiply(rho: np.ndarray, A: AnyMatrix) -> np.ndarray:
    """ rho @ A as a dense array, for dense or sparse A (computed as (A^T rho^T)^T to keep sparse products sparse-dense). """
    return np.asarray(A.T @ rho.T).T
