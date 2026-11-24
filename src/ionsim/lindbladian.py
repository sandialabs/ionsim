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
from typing import Callable, NamedTuple
from functools import cached_property

from ionsim.basis import StandardBasis
from ionsim.degree_of_freedom import AtomicStructure 
from ionsim.custom_types import Vector, Matrix, SparseMatrix, AnyMatrix
from ionsim.custom_math import matrix_AYB_multiply_to_superoperator, solve_time_evolution_equation, solve_ode
from ionsim.operators import Operator, Coupling, EnergyShift, GeneralOperator, EnergyShiftOperator, CouplingOperator 
from ionsim.ionsim_error import IonSimError
from ionsim.multi_operator import MultiOperator, to_MultiOperator
from ionsim.hamiltonian import Hamiltonian
from ionsim.atomic_internal_energy_level import AtomicInternalEnergyLevel, SinkLevel
from ionsim.atomic_internal_energy_level import compute_multipole_amplitude, reduced_dipole_matrix_element, _orbital_reduction_factor
from ionsim.config import NUMERICAL_EQUIVALENCE_THRESHOLD


@dataclass(frozen=True, eq=False)
class LindbladOperatorParts:
    """ A Lindblad operator (a MultiOperator in the dissipator's frame), decomposed once into
            L(t) = prefactor * sum_g f_g(t) e^{-i R_g t} L_g,
        the same operator as MultiOperator.get_matrix(t).

        groups: tuple of (f_g, R_g, L_g): the modulation function (or None), the frame-shifted rate and a static sparse
            matrix. Elements are grouped by (modulation function, floating-point equal rate); diagonal (energy-shift)
            elements have rate 0. A Lindblad operator built from one coupling, e.g. one decay path, has a single group.
    """
    size: int
    groups: tuple
    prefactor: complex = 1.
    sparse: bool = False

    @classmethod
    def from_multi_operator(cls, operator: MultiOperator) -> 'LindbladOperatorParts':
        """ Decompose a MultiOperator, using its parts exactly as MultiOperator.build_matrix_function does. """
        size = operator.size
        elements = []  # (modulation function, rate, row, column, value)
        for entry in operator.operators:
            parts = [entry]
            if isinstance(entry, GeneralOperator):
                parts = [part for part in (entry.energy_shift_operator_contribution if entry.energy_shifts else None,
                                           entry.coupling_operator_contribution if entry.couplings else None) if part is not None]
            for part in parts:
                if isinstance(part, EnergyShiftOperator):
                    matrix = csr_matrix(part.static_matrix, dtype=complex).tocoo()
                    elements.extend((part.modulation_function, 0., row, column, value)
                                    for row, column, value in zip(matrix.row, matrix.col, matrix.data))
                elif isinstance(part, CouplingOperator):
                    matrix, rate = operator._frame_shifted_coupling_matrix_and_rate_from_operator(part)
                    matrix, rate = csr_matrix(matrix, dtype=complex).tocoo(), csr_matrix(rate)
                    elements.extend((part.modulation_function, float(rate[row, column]), row, column, value)
                                    for row, column, value in zip(matrix.row, matrix.col, matrix.data))
                else:
                    raise IonSimError(f"Unsupported operator type {type(part).__name__} in a Lindblad operator.")

        tolerance = _rate_tolerance([element[1] for element in elements], operator.rotating_frame_energies)
        groups = []  # [modulation function, rate, rows, columns, values]
        for modulation, rate, row, column, value in elements:
            if value == 0:
                continue
            for group in groups:
                if group[0] is modulation and abs(group[1] - rate) <= tolerance:
                    group[2].append(row); group[3].append(column); group[4].append(value)
                    break
            else:
                groups.append([modulation, rate, [row], [column], [value]])
        return cls(size, tuple((modulation, rate, csr_matrix((values, (rows, columns)), shape=(size, size), dtype=complex))
                               for modulation, rate, rows, columns, values in groups),
                   complex(operator.prefactor), operator.sparse)

    def coefficients(self, t: float) -> list[complex]:
        """ prefactor * f_g(t) e^{-i R_g t} for every group. """
        return [self.prefactor * (1. if modulation is None else modulation(t)) * np.exp(-1j * rate * t)
                for modulation, rate, _ in self.groups]

    @cached_property
    def _dense_group_matrices(self) -> list[np.ndarray]:
        return [matrix.toarray() for _, _, matrix in self.groups]

    def matrix(self, t: float) -> AnyMatrix:
        """ L(t) as an N x N matrix (sparse if self.sparse). """
        coefficients = self.coefficients(t)
        if self.sparse:
            L = csr_matrix((self.size, self.size), dtype=complex)
            for coefficient, (_, _, matrix) in zip(coefficients, self.groups):
                L = L + coefficient * matrix
            return L
        L = np.zeros((self.size, self.size), dtype=complex)
        for coefficient, matrix in zip(coefficients, self._dense_group_matrices):
            L += coefficient * matrix
        return L

    @property
    def is_single_group(self) -> bool:
        """ True if L(t) = c(t) L_0 with one scalar coefficient c(t): the dissipator of L is |c(t)|^2 times that of L_0, so
            it is static when c(t) is a pure phase (no modulation function) and scales with |prefactor f(t)|^2 otherwise. """
        return len(self.groups) == 1


def _rate_tolerance(rates: list[float], rotating_frame_energies: list[float]) -> float:
    """ Tolerance for treating two frame-shifted rates as equal. A rate is oscillation_rate + E_row - E_column with frame
        energies E that can be optical (~1e15 rad/s), so its floating-point error scales with the frame energies, not with
        the (possibly much smaller) rate. """
    rate_scale = max([1.] + [abs(rate) for rate in rates])
    frame_scale = max([0.] + [abs(energy) for energy in rotating_frame_energies])
    return max(NUMERICAL_EQUIVALENCE_THRESHOLD * rate_scale, 64. * np.finfo(float).eps * frame_scale)


class SuperoperatorComponent(NamedTuple):
    """ One term of the precomputed dissipator superoperator,
            data * f_a(t) conj(f_b(t)) e^{-i rate t}   (f = 1 where the modulation is None),
        in coordinate form with unique (rows, columns). """
    rate: float
    rows: np.ndarray
    columns: np.ndarray
    data: np.ndarray
    modulation_a: Callable | None = None
    modulation_b: Callable | None = None


@dataclass(frozen=True, eq=False)
class Dissipator:
    """Class for dissipator object which implements dissipative phenomena in the context of open quantum systems. 

        D(t) rho = sum_k ( L_k(t) rho L_k(t)^dagger - 1/2 { L_k(t)^dagger L_k(t), rho } )

        operators: the Lindblad operators. Each entry is one Lindblad operator L_k: either a MultiOperator or a plain Operator, 
            which is wrapped as a single-term MultiOperator. 

        The list of operators may mix both. `operators` keeps exactly what was passed in.
        rotating_frame_energies: the frame of the Lindblad operators. Multi entries must use the same frame.
        sparse: whether to use sparse matrices; composite entries must use the same setting.
    """
    basis: StandardBasis
    operators: list[MultiOperator | Operator]
    rotating_frame_energies: list[float]
    sparse: bool = False

    def __post_init__(self):
        # Check frame shifts are compatible 
        if len(self.rotating_frame_energies) != self.size:
            raise IonSimError(f"Rotating_frame_energies has {len(self.rotating_frame_energies)} entries, but the basis has {self.size} states.")

        # Check Lindblad operators share the same basis  
        frame = np.asarray(self.rotating_frame_energies, dtype=float)
        for index, entry in enumerate(self.operators):
            if isinstance(entry, Operator):
                # Check basis matching 
                if not self.basis.has_same_dofs(entry.basis): 
                    raise IonSimError(f"Lindblad operator {index} ({type(entry).__name__}) does not live in the dissipator's basis.")
            elif isinstance(entry, MultiOperator):
                if entry.size != self.size:
                    raise IonSimError(f"Lindblad operator {index} ({type(entry).__name__}) has size {entry.size}, expected {self.size}.")
                if entry.sparse != self.sparse:
                    raise IonSimError(f"Lindblad operator {index} has sparse={entry.sparse}, but the dissipator has sparse={self.sparse}.")
                if not np.allclose(np.asarray(entry.rotating_frame_energies, dtype=float), frame, rtol=NUMERICAL_EQUIVALENCE_THRESHOLD, atol=0.):
                    raise IonSimError(f"Lindblad operator {index} ({type(entry).__name__}) uses a different rotating frame than "
                                      f"the dissipator; all Lindblad operators of a dissipator must share its frame.")
            else:
                raise IonSimError(f"Lindblad operator {index} must be an Operator or a MultiOperator, got {type(entry).__name__}.")

    @property
    def size(self):
        return len(self.basis.states)


    @cached_property
    def lindblad_operators(self) -> list[MultiOperator]:
        """ The Lindblad operators as MultiOperator (composites): MultiOperator entries unchanged, each plain Operator wrapped as a
            single-term MultiOperator in the dissipator's frame. 
        """
        return [entry if isinstance(entry, MultiOperator) else to_MultiOperator(entry, self.rotating_frame_energies, self.sparse)
                for entry in self.operators]

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
        return Dissipator.lindblad_pair_to_superoperator(L_matrix, L_matrix)

    @staticmethod
    def lindblad_pair_to_superoperator(L_a: AnyMatrix, L_b: AnyMatrix) -> AnyMatrix:
        """ Superoperator (column-stacked, N^2 x N^2) of rho -> L_a rho L_b^dagger - 1/2 { L_b^dagger L_a, rho }.

            With L_a = L_b = L this is the dissipator of L. For L(t) = sum_g e^{-i R_g t} L_g, the dissipator of L(t) is
            sum_{a, b} e^{-i (R_a - R_b) t} lindblad_pair_to_superoperator(L_a, L_b).
        """
        LbdaggerLa = L_b.conj().T @ L_a

        superoperator = matrix_AYB_multiply_to_superoperator(A = LbdaggerLa, B = None)
        superoperator += matrix_AYB_multiply_to_superoperator(A = None, B = LbdaggerLa)
        superoperator *= -0.5

        superoperator += matrix_AYB_multiply_to_superoperator(L_a, L_b.conj().T)
        return superoperator

    @cached_property
    def lindblad_matrix_functions(self) -> list[Callable]:
        """Creates and stores a list of callables that return (N x N) matrices as a function of time. 
            Each callable corresponds to a lindblad operator in the interaction picture, which may contain time-dependence
            as a result of frame shifting and modulation functions. 

            Defines and returns a list of callables as a function of time
        """
        return [lindblad_operator.get_matrix for lindblad_operator in self.lindblad_operators]

    @cached_property
    def lindblad_operator_parts(self) -> list[LindbladOperatorParts]:
        """ Every Lindblad operator decomposed once into L(t) = prefactor sum_g f_g(t) e^{-i R_g t} L_g (see LindbladOperatorParts). """
        return [LindbladOperatorParts.from_multi_operator(operator) for operator in self.lindblad_operators]

    @cached_property
    def build_superoperator_matrix_function(self) -> Callable:
        """Creates N^2 x N^2 matrix representation of each Lindblad operator's action on a N^2 x 1 superoperator (y) from NxN density matrix (rho): 
                Dissipator acting on supervector from Lindblad operator i (L_i): d_i(t)y <==> L_i(t) rho L†_i(t) - 1/2 { L†_i(t) L_i(t) , rho }  

            Returns the sum of the dissipators: D(t) = sum_i d_i (t) 
        """
        # The superoperators are precomputed (dissipator_superoperator_components):
        #     D(t) = S_0 + sum_c S_c f_a(t) conj(f_b(t)) e^{-i w_c t}.
        static, components = self.dissipator_superoperator_components

        def _dissipator_matrix_function(t: float):
            modulation_values = {}   # each modulation function is evaluated once per call

            def value(modulation):
                if modulation is None:
                    return 1.
                if id(modulation) not in modulation_values:
                    modulation_values[id(modulation)] = modulation(t)
                return modulation_values[id(modulation)]

            d_matrix = static.copy()
            for component in components:
                coefficient = (value(component.modulation_a) * np.conj(value(component.modulation_b))
                               * np.exp(-1j * component.rate * t))
                if self.sparse:
                    d_matrix = d_matrix + csr_matrix((component.data * coefficient, (component.rows, component.columns)),
                                                     shape=static.shape)
                else:
                    d_matrix[component.rows, component.columns] += component.data * coefficient
            return d_matrix

        return _dissipator_matrix_function

    @cached_property
    def dissipator_superoperator_components(self) -> tuple:
        """ The dissipator superoperator D(t) = S_0 + sum_c S_c f_a(t) conj(f_b(t)) e^{-i w_c t}, computed once.

            Each Lindblad operator L(t) = p sum_g f_g(t) e^{-i R_g t} L_g (LindbladOperatorParts) contributes
                |p|^2 f_a(t) conj(f_b(t)) e^{-i (R_a - R_b) t} lindblad_pair_to_superoperator(L_a, L_b)
            for every pair (a, b) of its groups. Unmodulated pairs with equal rates (e.g. every unmodulated operator with a
            single rate, such as one decay path) give the static part S_0. The others are kept as components, one per
            (f_a, f_b, rate), with the modulation functions matched by identity.

            Returns (S_0, [SuperoperatorComponent, ...]): S_0 dense or sparse as self.sparse, each component in
            coordinate form with unique (rows, columns).
        """
        n2 = self.size**2
        static = csr_matrix((n2, n2), dtype=complex)
        components = []  # [f_a, f_b, w, accumulated sparse S]
        all_rates = [rate for parts in self.lindblad_operator_parts for _, rate, _ in parts.groups]
        tolerance = _rate_tolerance(all_rates, self.rotating_frame_energies)
        for parts in self.lindblad_operator_parts:
            weight = abs(parts.prefactor)**2
            for modulation_a, rate_a, L_a in parts.groups:
                for modulation_b, rate_b, L_b in parts.groups:
                    superoperator = weight * csr_matrix(self.lindblad_pair_to_superoperator(L_a, L_b), dtype=complex)
                    w = rate_a - rate_b
                    if modulation_a is None and modulation_b is None and abs(w) <= tolerance:
                        static = static + superoperator
                        continue
                    for component in components:
                        if component[0] is modulation_a and component[1] is modulation_b and abs(component[2] - w) <= tolerance:
                            component[3] = component[3] + superoperator
                            break
                    else:
                        components.append([modulation_a, modulation_b, w, superoperator])
        coordinate_components = []
        for modulation_a, modulation_b, w, superoperator in components:
            superoperator = superoperator.tocoo()
            superoperator.sum_duplicates()
            coordinate_components.append(SuperoperatorComponent(w, superoperator.row, superoperator.col, superoperator.data,
                                                                modulation_a, modulation_b))
        static = static.tocsr() if self.sparse else static.toarray()
        return static, coordinate_components

    def get_matrix(self, t: float):
        """ Gets matrix form of dissipator at time t. """ 
        return self.build_superoperator_matrix_function(t)


@dataclass(frozen=True, eq=False)
class DissipatorSpontaneousEmission(Dissipator):
    """Subclass for including spontaneous emission dynamics for a known atomic structure."""

    @classmethod
    def from_atomic_structure_data(cls, basis: StandardBasis, ground_levels: list[AtomicInternalEnergyLevel],
                                   excited_levels: list[AtomicInternalEnergyLevel], frame_energies: list[float] | None = None,
                                   sparse: bool = False, select_dofs: list[AtomicStructure] | None = None, decay_to_sink: bool = False, 
                                   multipole_orders: dict[tuple[str, str], int] | None = None, coherent_channels: bool = False):
        """Build the spontaneous-emission dissipator for decay from excited_levels to ground_levels.

        One Lindblad operator sqrt(Gamma_{e->g,q}) |g><e| is created per decay path (excited level e, ground level g,
        polarization q), in every selected AtomicStructure DOF. The rate of each path is exact:

            Gamma_{e->g,q} = (branching ratio of g's manifold) / (lifetime of e) * |<e| T^(k)_q |g>|^2 (2 J_e + 1) / (2 J_g + 1)

        where <e| T^(k)_q |g> is compute_multipole_amplitude of rank k in units of <J_g||T^(k)||J_e> (same magnitude as
        <g| T^(k)_{-q} |e>), q = m_e - m_g, and q runs over -k..k. Summed over every sublevel of a ground manifold and over q, the factor after the branching ratio is
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

        coherent_channels=True instead builds one operator per decay channel (DOF, ground manifold, multipole order k,
        emitted polarization q), summing the signed amplitudes of every (e, g) path of that channel:

            L_{k,q} = sum_{e, g} sign(<J_g||T^(k)||J_e>) <g| T^(k)_q |e> sqrt( (branching ratio) / (lifetime of e) (2 J_e + 1) / (2 J_g + 1) ) |g><e|

        Each element has the same magnitude as the corresponding secular operator, so sum_{k,q} L^dagger L has the same
        diagonal (the decay rates). The relative signs keep the interference between paths that emit the same photon,
        which the secular operators drop. This is what adiabatic elimination of far-detuned excited states needs:
        scattered photons are then all near the laser frequency, so paths through different excited sublevels and
        fine-structure manifolds interfere (Raman and Rayleigh scattering). The sign of <J_g||T^(k)||J_e> is that of the
        (tabulated or derived) reduced dipole element for E1, and of the LS orbital reduction factor for E2.
        In a frame, each element keeps its own rotating-frame phase, so the coherent operators are also exact (no secular
        approximation) for direct simulation, at the cost of fast oscillating terms. Decay to the sink stays per path.

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
            coherent_channels: One operator per (ground manifold, k, q) channel with signed amplitudes (see above)
                instead of one per path.

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

            channels = {}  # (ground term symbol, k, q) -> coherent lowering matrix of this DOF
            for e_level in excited:
                amplitudes = cls._decay_amplitudes(e_level, ground, require_paths=not decay_to_sink, multipole_orders=multipole_orders)
                rates = [(g_level, abs(amplitude)**2) for g_level, _, _, amplitude in amplitudes]
                if coherent_channels:
                    for g_level, k, q, amplitude in amplitudes:
                        matrix = channels.setdefault((g_level.term_symbol, k, q), np.zeros((dimension, dimension)))
                        matrix[level_index[g_level.name], level_index[e_level.name]] += amplitude
                else:
                    for g_level, rate in rates:
                        add_decay(level_index[g_level.name], level_index[e_level.name], rate)
                if decay_to_sink:
                    remaining = 1. / e_level.lifetime - sum(rate for _, rate in rates)
                    if remaining > NUMERICAL_EQUIVALENCE_THRESHOLD / e_level.lifetime:
                        add_decay(sinks[0], level_index[e_level.name], remaining)
            for matrix in channels.values():
                if np.any(np.abs(matrix) > 0.):
                    lindblad_operators.append(CouplingOperator.from_matrix(basis, basis.enlarge_matrix(matrix, [dof]), 0.))

        return cls(basis, lindblad_operators, frame_energies, sparse)

    @classmethod
    def _decay_rates(cls, e_level: AtomicInternalEnergyLevel, ground_levels: list[AtomicInternalEnergyLevel],
                     require_paths: bool = True, multipole_orders: dict[tuple[str, str], int] | None = None) -> list[tuple]:
        """ Decay rates (1/s) from e_level to each ground level, one entry per (ground level, polarization q) path.

            Raises if there are no paths and require_paths is True.
        """
        return [(g_level, abs(amplitude)**2) for g_level, _, _, amplitude
                in cls._decay_amplitudes(e_level, ground_levels, require_paths, multipole_orders)]

    @staticmethod
    def _decay_amplitudes(e_level: AtomicInternalEnergyLevel, ground_levels: list[AtomicInternalEnergyLevel],
                          require_paths: bool = True, multipole_orders: dict[tuple[str, str], int] | None = None) -> list[tuple]:
        """ Signed decay amplitudes (1/sqrt(s)) from e_level, one entry (ground level, k, q, amplitude) per path, with
            |amplitude|^2 the partial decay rate of that path (see from_atomic_structure_data).

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
            sign = _reduced_element_sign(g_level, e_level, k)
            for q in range(-k, k + 1):
                amplitude = compute_multipole_amplitude(g_level, e_level, k, q)
                if abs(amplitude) > NUMERICAL_EQUIVALENCE_THRESHOLD:
                    # signed amplitude in units of sqrt(lifetime / branching ratio); its square is the old 'fraction'
                    unit_amplitude = sign * amplitude * np.sqrt((2*e_level.j + 1) / (2*g_level.j + 1))
                    paths.setdefault(g_level.term_symbol, []).append((g_level, k, q, unit_amplitude))

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
            for g_level, k, q, unit_amplitude in manifold_paths:
                if branching_ratio == 0.:
                    continue
                if e_level.energy <= g_level.energy:
                    raise IonSimError(f"Excited level {e_level.name} (energy {e_level.energy}) must be higher in energy than "
                                      f"ground level {g_level.name} (energy {g_level.energy}) to decay to it.")
                rates.append((g_level, k, q, np.sqrt(branching_ratio / lifetime) * unit_amplitude))

        if not rates and require_paths:
            raise IonSimError(f"No decay paths from excited level {e_level.name} to the given ground levels. Allowed "
                              f"ground manifolds: {sorted(paths)}; branching_ratios keys: "
                              f"{sorted(e_level.branching_ratios) if e_level.branching_ratios else None}.")
        return rates


def _reduced_element_sign(g_level: AtomicInternalEnergyLevel, e_level: AtomicInternalEnergyLevel, k: int) -> float:
    """ Sign of <J_g||T^(k)||J_e>: that of the tabulated or derived reduced dipole element for E1, otherwise of the LS
        orbital reduction factor; +1 if neither is available or the factor vanishes. """
    if k == 1:
        try:
            element = reduced_dipole_matrix_element(g_level, e_level)
        except IonSimError:
            element = None
        if element is not None and element != 0.:
            return float(np.sign(element))
    if hasattr(g_level, 'l') and hasattr(e_level, 'l'):
        factor = float(_orbital_reduction_factor(g_level, e_level, k))
        if factor != 0.:
            return float(np.sign(factor))
    return 1.


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
            # Check if Hamiltonian and Dissipator have the same basis
            if not (self.hamiltonian.basis).has_same_dofs(self.dissipator.basis): 
                raise IonSimError('Both Hamiltonian and Dissipator should have the same basis. Input bases contain different degrees of freedom.')

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
    def build_matrix_function(self) -> Callable:
        """ Lindbladian matrix function L(t), corresponding to an N^2 x N^2 superoperator at time t. """
        if self.hamiltonian:
            get_hamiltonian = self.hamiltonian.get_matrix

            def super_ham(t: float):
                H = get_hamiltonian(t)  # evaluated once for both sides of the commutator: -i (H rho - rho H^dagger)
                return -1j * (matrix_AYB_multiply_to_superoperator(A = H, B = None)
                              - matrix_AYB_multiply_to_superoperator(A = None, B = H.conj().T))
        else:
            super_ham = lambda t: 0. 

        if self.dissipator:
            super_dissipator = self.dissipator.get_matrix
        else:
            super_dissipator = lambda t: 0. 

        # Lindbladian superoperator from hamiltonian and dissipation contributions: 
        lindbladian_function = lambda t: super_ham(t) + super_dissipator(t)
        return lindbladian_function
    
    def get_matrix(self, t):
        """ Build Lindbladian matrix at time t. """
        return self.build_matrix_function(t)

    def evolve_supervector(self, initial_supervector: Vector, duration: float, time_evals: Vector | None = None, **kwargs):
        """ Evolve a supervector by solving the time-dependent Lindblad master equation.
            e.g. evolves supervector "y" using dy/dt = Ly, where L is the N^2 x N^2 dissipator matrix. 
        """
        # solve_time_evolution_equation() assumes a Schrodinger equation form dy/dt = (-i*A)y, where i = sqrt(-1) and A <==> the function input, e.g. a Hamiltonian matrix. 
        # Therfore, we must compensate this form by multiplying the lindbladian by i 
        assert(self.size == len(initial_supervector))
        dynamical_matrix = lambda t: self.get_matrix(t) * 1j
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

            solver_options: ode_solver and its options (see custom_math.solve_ode). Explicit solvers suit this form (e.g.
                ode_solver='solve_ivp', method='DOP853', or 'zvode' with method='adams'); for stiff problems prefer
                evolve_supervector, whose Jacobian (the superoperator) the implicit solvers use.
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

            Lindblad operators with a single group, L(t) = c(t) L_0 (LindbladOperatorParts.is_single_group, e.g. one decay
            path), are precomputed: unmodulated ones are static (|c(t)| = |prefactor|), and modulated ones are scaled by
            |prefactor f(t)|^2 at each time. The other operators (e.g. coherent decay channels in a rotating frame, or a
            modulated drive and its adjoint) are evaluated at each time.
        """
        n = self.dimension
        get_hamiltonian = self.hamiltonian.get_matrix if self.hamiltonian else None
        static_operators, modulated_operators, time_dependent_operators = [], [], []
        static_LdL = None
        for parts in (self.dissipator.lindblad_operator_parts if self.dissipator else []):
            if not parts.groups:
                continue
            if parts.is_single_group:
                modulation, _, matrix = parts.groups[0]
                L = parts.prefactor * (matrix if self.dissipator.sparse else matrix.toarray())
                Ld = L.conj().T
                if modulation is None:
                    static_operators.append((L, Ld))
                    static_LdL = Ld @ L if static_LdL is None else static_LdL + Ld @ L
                else:
                    modulated_operators.append((modulation, L, Ld, Ld @ L))
            else:
                time_dependent_operators.append(parts)

        def right_hand_side(t: float, y: Vector) -> Vector:
            rho = y.reshape(n, n)
            drho = np.zeros((n, n), dtype=complex)
            if get_hamiltonian is not None:
                H = get_hamiltonian(t)
                drho += -1j * (_left_multiply(H, rho) - _right_multiply(rho, H.conj().T))
            for L, Ld in static_operators:
                drho += _right_multiply(_left_multiply(L, rho), Ld)
            if static_LdL is not None:
                drho -= 0.5 * (_left_multiply(static_LdL, rho) + _right_multiply(rho, static_LdL))
            for modulation, L, Ld, LdL in modulated_operators:
                weight = abs(modulation(t))**2
                drho += weight * (_right_multiply(_left_multiply(L, rho), Ld)
                                  - 0.5 * (_left_multiply(LdL, rho) + _right_multiply(rho, LdL)))
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
