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
from ionsim.custom_types import Vector, Matrix, SparseMatrix, AnyMatrix
from ionsim.custom_math import matrix_AYB_multiply_to_superoperator, solve_time_evolution_equation
from ionsim.ionsim_error import IonSimError
from ionsim.composite_operator import CompositeOperator
from ionsim.hamiltonian import Hamiltonian
from ionsim.atomic_internal_energy_level import AtomicInternalEnergyLevel, SinkLevel
from ionsim.atomic_internal_energy_level import compute_multipole_amplitude
from ionsim.config import NUMERICAL_EQUIVALENCE_THRESHOLD


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
        """Converts a lindblad Operator object to a callable that returns a matrix at a given time point"""

        def _lindblad_function(t: float) -> AnyMatrix:
            if self.sparse:
                L_matrix = csr_matrix(([0], ([0], [0])), shape=(self.size, self.size), dtype='complex') 
            else:
                L_matrix = np.zeros((self.size, self.size), dtype=complex) 
            
            # Static, diagonal contribution 
            if isinstance(lindblad_operator, GeneralOperator):
                if lindblad_operator.energy_shift_operator_contribution:
                    L_matrix += lindblad_operator.energy_shift_operator_contribution.static_matrix
            elif isinstance(lindblad_operator, EnergyShiftOperator):
                L_matrix += lindblad_operator.static_matrix

            if isinstance(lindblad_operator, CouplingOperator) or isinstance(lindblad_operator, GeneralOperator):
                # Must account for frame shifts in coupling operators 
                L_int, Rate = self._frame_shifted_coupling_matrix_and_rate_from_operator(lindblad_operator) 
    
                # Element-wise multiplication, compute L * exp(-1j * Rate * t)
                if self.sparse: 
                    phase_factor_minus_one = Rate.multiply(-1j*t).expm1() # equivalent to exp(-1j * Rate * t) - 1
                    Ltemp = L_int + L_int.multiply(phase_factor_minus_one) # necessary to cancel out the -1 contribution above 
                    L_matrix += Ltemp 
                else:
                    L_matrix += (L_int.toarray() * np.exp(-1j * Rate.toarray() * t))
            return L_matrix

        return _lindblad_function
            
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
                                   sparse: bool = False, select_DOFs: list[AtomicStructure] | None = None,
                                   decay_to_sink: bool = False, multipole_orders: dict[tuple[str, str], int] | None = None):
        """Build the spontaneous-emission dissipator for decay from excited_levels to ground_levels. """

        """ One Lindblad operator sqrt(Gamma_{e->g,q}) |g><e| is created per decay path (excited level e, ground level g,
        polarization q), in every selected AtomicStructure DOF. The rate of each path is exact:

            Gamma_{e->g,q} = (branching ratio of g's manifold) / (lifetime of e) * |<g| T^(k)_q |e>|^2 (2 J_e + 1) / (2 J_g + 1)

        where <g| T^(k)_q |e> is compute_multipole_amplitude of rank k in units of <J_g||T^(k)||J_e>, and q runs over
        -k..k. Summed over every sublevel of a ground manifold and over q, the factor after the branching ratio is
        exactly 1 for any k, so the rates do not depend on which levels are included.

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

        Each decay channel (excited manifold -> ground manifold) is electric dipole (E1, k = 1) or quadrupole (E2, k = 2).
        By default the order follows parity, taken as (-1)^L: E1 if the parity changes (e.g. P -> S), E2 if it does not
        (e.g. D5/2 -> S1/2 in Ca+ or Yb+). Levels without L (j1l2 coupling) are assumed to decay by E1. Set the order
        of any channel explicitly with multipole_orders. A decay listed in branching_ratios with no paths of its order
        (e.g. an E3 decay) raises an error.

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
            multipole_orders: Optional {(excited term symbol, ground term symbol): 1 or 2} overriding the parity-based
                multipole order of a decay channel, e.g. {('[3/2]1/2', 'D3/2'): 1}.
        """
        if frame_energies is None:
            frame_energies = [0.] * len(basis.states)

        if select_DOFs is None:
            DOF_list = basis.atomic_structure_DOFs
        else:
            DOF_list = select_DOFs
            if any(not any(dof is basis_dof for basis_dof in basis.degrees_of_freedom) for dof in DOF_list):
                raise IonSimError("Every DOF in select_DOFs must be an AtomicStructure degree of freedom of the basis.")
        if not DOF_list:
            raise IonSimError("The basis has no AtomicStructure degrees of freedom to add spontaneous emission to.")

        lindblad_operators = []
        for DOF in DOF_list:
            dimension = len(DOF.energy_levels)
            level_index = {level.name: index for index, level in enumerate(DOF.energy_levels)}
            missing = [level.name for level in [*ground_levels, *excited_levels] if level.name not in level_index]
            if missing:
                raise IonSimError(f"Levels {missing} are not in the degree of freedom {DOF.name or ''}. "
                                  f"Use select_DOFs to choose which atomic structures to add decay to.")
            # Use this DOF's own level objects, so ions built separately (or at different fields) are handled correctly.
            ground = [DOF.energy_levels[level_index[level.name]] for level in ground_levels]
            excited = [DOF.energy_levels[level_index[level.name]] for level in excited_levels]
            if decay_to_sink:
                sinks = [index for index, level in enumerate(DOF.energy_levels) if isinstance(level, SinkLevel)]
                if len(sinks) != 1:
                    raise IonSimError(f"decay_to_sink requires exactly one SinkLevel in each selected DOF, found {len(sinks)}. "
                                      f"Build the structure with AtomicStructure.from_species(..., include_sink=True).")

            def add_decay(to_index: int, from_index: int, rate: float):
                lowering_matrix = np.zeros((dimension, dimension))
                lowering_matrix[to_index, from_index] = np.sqrt(rate)
                lindblad_operators.append(CouplingOperator.from_matrix(basis, basis.enlarge_matrix(lowering_matrix, [DOF]), 0.))

            for e_level in excited:
                rates = cls._decay_rates(e_level, ground, require_paths=not decay_to_sink, multipole_orders=multipole_orders)
                for g_level, rate in rates:
                    add_decay(level_index[g_level.name], level_index[e_level.name], rate)
                if decay_to_sink:
                    remaining = 1. / e_level.lifetime - sum(rate for _, rate in rates)
                    if remaining > NUMERICAL_EQUIVALENCE_THRESHOLD / e_level.lifetime:
                        add_decay(sinks[0], level_index[e_level.name], remaining)

        return cls(basis, lindblad_operators, frame_energies, sparse)

    @staticmethod
    def _decay_rates(e_level: AtomicInternalEnergyLevel, ground_levels: list[AtomicInternalEnergyLevel],
                     require_paths: bool = True, multipole_orders: dict[tuple[str, str], int] | None = None) -> list[tuple]:
        """ Decay rates (1/s) from e_level to each ground level, one entry per (ground level, polarization q) path.

            Raises if there are no paths and require_paths is True.
        """
        lifetime = e_level.lifetime
        if not isinstance(lifetime, (int, float)) or lifetime <= 0:
            raise IonSimError(f"Excited level {e_level.name} needs a positive lifetime (s) for spontaneous emission, got {lifetime!r}.")

        # Allowed paths, grouped by ground manifold. The fraction of e's decay into a complete manifold sums to 1 for any rank k.
        paths = {}  # term symbol -> list of (ground level, fraction)
        orders = {}  # term symbol -> multipole order used for that channel
        for g_level in ground_levels:
            k = _decay_multipole_order(e_level, g_level, multipole_orders)
            orders[g_level.term_symbol] = k
            for q in range(-k, k + 1):
                amplitude = compute_multipole_amplitude(g_level, e_level, k, q)
                if abs(amplitude) > NUMERICAL_EQUIVALENCE_THRESHOLD:
                    fraction = abs(amplitude)**2 * (2*e_level.j + 1) / (2*g_level.j + 1)
                    paths.setdefault(g_level.term_symbol, []).append((g_level, fraction))

        branching_ratios = e_level.branching_ratios
        if branching_ratios is not None:
            # A listed channel with no allowed paths at its multipole order (e.g. an E3 decay) would silently vanish.
            unsupported = [f"{term_symbol} (E{orders[term_symbol]})" for term_symbol, ratio in branching_ratios.items()
                           if ratio > 0 and term_symbol in orders and term_symbol not in paths]
            if unsupported:
                raise IonSimError(f"Excited level {e_level.name} decays to {unsupported} (branching_ratios), but no paths of that "
                                  f"multipole order connect them. Only E1 and E2 decays are supported; if the order was "
                                  f"inferred incorrectly, set it with multipole_orders.")
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
            raise IonSimError(f"No decay paths from excited level {e_level.name} to the given ground levels. Allowed "
                              f"ground manifolds: {sorted(paths)}; branching_ratios keys: "
                              f"{sorted(e_level.branching_ratios) if e_level.branching_ratios else None}.")
        return rates


def _parity_of_ls_coupled_level(level: AtomicInternalEnergyLevel) -> int | None:
    """ Parity (+1 or -1) of an LS-coupled level, taken as (-1)^L; None if the level has no L (e.g. j1l2 coupling).

        (-1)^L equals the configuration parity (-1)^(sum of electron l) for one valence electron and for configurations
        such as sp and sd, which covers the current species configs. It is not general (e.g. p^2 3P has L = 1 but even
        parity); use multipole_orders for such levels.
    """
    l = getattr(level, 'l', None)
    return None if l is None else (-1)**int(round(l))


def _decay_multipole_order(e_level: AtomicInternalEnergyLevel, g_level: AtomicInternalEnergyLevel,
                           multipole_orders: dict[tuple[str, str], int] | None) -> int:
    """ Electric multipole order of the decay e_level -> g_level: 1 (E1) or 2 (E2).

        An entry (excited term symbol, ground term symbol) in multipole_orders takes precedence. Otherwise the order follows
        parity: E1 if it changes, E2 if not. If either parity is unknown, E1 is assumed.
    """
    if multipole_orders is not None:
        if not isinstance(multipole_orders, dict):
            raise IonSimError("multipole_orders must be a dict {(excited term symbol, ground term symbol): 1 or 2}, "
                              f"got {type(multipole_orders).__name__}.")
        k = multipole_orders.get((e_level.term_symbol, g_level.term_symbol))
        if k is not None:
            if k not in (1, 2):
                raise IonSimError(f"multipole_orders values must be 1 (E1) or 2 (E2), got {k} for "
                                  f"({e_level.term_symbol!r}, {g_level.term_symbol!r}).")
            return k
    e_parity, g_parity = _parity_of_ls_coupled_level(e_level), _parity_of_ls_coupled_level(g_level)
    if e_parity is None or g_parity is None or e_parity != g_parity:
        return 1
    return 2


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
