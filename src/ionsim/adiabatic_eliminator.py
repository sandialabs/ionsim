#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

""" Methods for adiabatic elimination, which may produce an effective Hamiltonian or Lindbladian from a higher-dimensional Hamiltonian / Lindbladian.

    Adiabatic elimination of far-detuned levels (e.g. the P manifolds of a Raman transition) at second order in the
    couplings to them, for continuous-wave drives. The kept states P and eliminated states Q are split by levels; the
    reduced basis is the product basis without those levels.

    Input. The Hamiltonian is used exactly as hamiltonian_function evaluates it,
        H(t) = diag(E) + sum_terms [ h |r><c| exp(-i R t) + h.c. ],
    with E the frame-shifted energies (H0 plus static energy shifts) and R the frame-shifted rate of each term, so no
    convention for the frame or the laser rates is assumed. Every P-Q term is written in the direction
        <q| H |p> = v exp(-i w t)
    (an element stored as <p|H|q> = h e^{-i R t} gives v = conj(h), w = -R). Spontaneous emission enters through
    Lindblad operators with elements <g| L |q> = lambda exp(-i r t), and Gamma_q = (sum_L L^dagger L)_qq.

    Effective operators (effective-operator formalism of Reiter and Sorensen, PRA 85, 032111 (2012), generalized to one
    resolvent per drive frequency). With the complex denominators
        D_k(p, q) = E_p + w_k - E_q + i Gamma_q / 2
    for the term k from p to q:
        H_eff(p, p') = H_PP(p, p') + 1/2 sum_q sum_{k, l} conj(v_l) v_k [1/D_k(p', q) + 1/conj(D_l(p, q))] exp(-i (w_k - w_l) t)
        L_eff(g, p)  = L_PP(g, p) + sum_q sum_k lambda_{g q} v_k / D_k(p, q) exp(-i (r_{gq} + w_k) t)
    where l runs over terms from p, k over terms from p'. For Gamma = 0, H_eff is second-order Schrieffer-Wolff theory for
    each drive frequency; it gives the two-photon couplings and the light shifts (e.g. Omega_1 conj(Omega_2) / (4 Delta)
    for a Lambda system with couplings Omega/2). L_eff describes Raman (p -> g != p) and Rayleigh (g = p) scattering.

    Effective terms are grouped by their total rate. A Lindblad operator's elements with the same total rate emit
    photons of the same frequency and are added coherently (one effective operator per group); groups at different rates
    are distinguishable and kept as separate operators. Rate differences below rate_tolerance are treated as equal.

    Decay among kept states (e.g. of a kept excited level) is passed through, split into operators of equal total rate
    (the secular approximation between distinguishable photon frequencies).

    Validity (checked and reported): |v / D| << 1 for every term, CW drives (no modulation functions on P-Q couplings),
    no couplings among eliminated states, and decay from eliminated states only to kept states. The eliminated block
    uses the diagonal of sum_L L^dagger L as its decay rates.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from ionsim.basis import StandardBasis
from ionsim.degree_of_freedom import AtomicStructure
from ionsim.energy_level import EnergyEigenstate, EnergyLevel
from ionsim.operator import Coupling, CouplingOperator, EnergyShift, EnergyShiftOperator, GeneralOperator
from ionsim.hamiltonian import Hamiltonian
from ionsim.lindbladian import Dissipator
from ionsim.ionsim_error import IonSimError
from ionsim.config import SMALLEST_ENERGY_SCALE

OSCILLATING_SHIFT_OPTIONS = ('drop', 'keep')


@dataclass(frozen=True, eq=False)
class AdiabaticEliminationResult:
    """ Result of adiabatically_eliminate.

        hamiltonian: effective Hamiltonian on reduced_basis, in the original frame restricted to the kept states.
        dissipator: effective Dissipator (scattering plus any decay among kept states), or None without a dissipator input.
        reduced_basis: product basis without the eliminated levels.
        kept_states / eliminated_states: states of the original basis; kept_states[i] corresponds to reduced_basis.states[i].
        light_shifts: static diagonal shift of each reduced state induced by the elimination (rad/s), i.e. the AC Stark
            shifts in the bare-state basis (not the eigenvalue shifts of resonantly coupled kept states).
        max_perturbation_ratio: max |v / D| over all eliminated couplings; the expansion parameter.
        dropped_oscillating_shifts: (reduced state, amplitude a, rate w) for diagonal terms a e^{-iwt} + c.c. that were
            dropped (oscillating_shifts='drop').
        dropped_fast_terms: (reduced state, reduced state, amplitude, rate) for terms above rwa_rate_cutoff (upper
            triangle; each term also has its Hermitian conjugate).
        scattering_operators: effective jump-operator matrices on the reduced basis (one per photon-frequency group), each
            with the total rate of its group; used by the scattering-rate methods.
    """
    hamiltonian: Hamiltonian
    dissipator: Dissipator | None
    reduced_basis: StandardBasis
    kept_states: list[EnergyEigenstate]
    eliminated_states: list[EnergyEigenstate]
    light_shifts: np.ndarray
    max_perturbation_ratio: float
    dropped_oscillating_shifts: list[tuple] = field(default_factory=list)
    dropped_fast_terms: list[tuple] = field(default_factory=list)
    scattering_operators: list[tuple[np.ndarray, float]] = field(default_factory=list)

    def _index(self, state: EnergyEigenstate | int) -> int:
        if isinstance(state, (int, np.integer)):
            return int(state)
        for index, reduced_state in enumerate(self.reduced_basis.states):
            if reduced_state is state:
                return index
        raise IonSimError(f"State {getattr(state, 'name', state)} is not a state of the reduced basis.")

    def light_shift(self, state: EnergyEigenstate | int) -> float:
        """ Static light shift (rad/s) of a reduced-basis state (or its index). """
        return float(self.light_shifts[self._index(state)])

    def scattering_rate(self, state: EnergyEigenstate | int) -> float:
        """ Total photon-scattering rate (1/s) out of a reduced-basis state, sum_L sum_g |L_eff(g, p)|^2. """
        p = self._index(state)
        return float(sum(np.sum(np.abs(matrix[:, p])**2) for matrix, _ in self.scattering_operators))

    def rayleigh_scattering_rate(self, state: EnergyEigenstate | int) -> float:
        """ Elastic (Rayleigh) scattering rate (1/s) of a reduced-basis state: sum_L |L_eff(p, p)|^2. """
        p = self._index(state)
        return float(sum(np.abs(matrix[p, p])**2 for matrix, _ in self.scattering_operators))

    def raman_scattering_rate(self, state: EnergyEigenstate | int, final_state: EnergyEigenstate | int | None = None) -> float:
        """ Inelastic (Raman) scattering rate (1/s) out of a reduced-basis state, to final_state or to all other states. """
        p = self._index(state)
        if final_state is None:
            return self.scattering_rate(p) - self.rayleigh_scattering_rate(p)
        g = self._index(final_state)
        if g == p:
            raise IonSimError("The final state of Raman scattering must differ from the initial state.")
        return float(sum(np.abs(matrix[g, p])**2 for matrix, _ in self.scattering_operators))

    def elastic_dephasing_rate(self, state_a: EnergyEigenstate | int, state_b: EnergyEigenstate | int) -> float:
        """ Decay rate (1/s) of the coherence between two reduced states caused by elastic scattering alone:
            1/2 sum_L |L_eff(a, a) - L_eff(b, b)|^2 (only scattering that distinguishes the states dephases them). """
        a, b = self._index(state_a), self._index(state_b)
        return float(0.5 * sum(np.abs(matrix[a, a] - matrix[b, b])**2 for matrix, _ in self.scattering_operators))


def adiabatically_eliminate(hamiltonian: Hamiltonian, levels_to_eliminate: list[EnergyLevel],
                            dissipator: Dissipator | None = None, *, validity_threshold: float = 0.1,
                            rate_tolerance: float = 1e3, oscillating_shifts: str = 'drop',
                            rwa_rate_cutoff: float | None = None) -> AdiabaticEliminationResult:
    """ Adiabatically eliminate levels from a Hamiltonian (and optionally a spontaneous-emission Dissipator).

        Args:
            hamiltonian: Hamiltonian on a StandardBasis, e.g. built from Laser coupling operators.
            levels_to_eliminate: atomic levels to eliminate, matched by name in every AtomicStructure DOF of the basis
                (as the Laser coupling builders do), so levels of one atom also select the same levels of identical atoms.
                Every basis state containing one of them is eliminated.
            dissipator: optional Dissipator on the same basis and in the same frame, e.g.
                DissipatorSpontaneousEmission.from_atomic_structure_data(..., coherent_channels=True). Use coherent
                channels: the secular (one operator per path) form drops the interference between scattering paths.
            validity_threshold: warn if max |v / D| exceeds this value.
            rate_tolerance: rates (rad/s) closer than max(rate_tolerance, 1e-12 |rate|) are treated as equal when grouping.
            oscillating_shifts: 'drop' (default) diagonal terms a e^{-iwt} + c.c. at nonzero beat rates w (a rotating-wave
                approximation, error of order |a|^2 / w); 'keep' them as modulated energy shifts.
            rwa_rate_cutoff: if given (rad/s), also drop effective Hamiltonian terms whose rate in the Hamiltonian's frame
                exceeds it, e.g. cross terms of two Raman beams at twice the qubit frequency, which make the effective
                model stiff. Dropped terms are reported in dropped_fast_terms.

        Returns:
            AdiabaticEliminationResult.
    """
    if oscillating_shifts not in OSCILLATING_SHIFT_OPTIONS:
        raise IonSimError(f"oscillating_shifts must be one of {OSCILLATING_SHIFT_OPTIONS}, got {oscillating_shifts!r}.")
    basis = hamiltonian.basis
    if not isinstance(basis, StandardBasis):
        raise IonSimError("Adiabatic elimination requires a Hamiltonian on a StandardBasis of energy eigenstates.")

    # ---- Partition the basis ----
    eliminated_levels = _match_levels(basis, levels_to_eliminate)
    eliminated_ids = {id(level) for level in eliminated_levels}
    Q = [i for i, state in enumerate(basis.states) if any(id(c) in eliminated_ids for c in state.components)]
    P = [i for i in range(len(basis.states)) if i not in set(Q)]
    if not P:
        raise IonSimError("Eliminating these levels leaves no states in the reduced basis.")
    reduced_basis = basis.build_subspace_basis_from_levels_to_project(eliminated_levels)
    reduced_index = _reduced_index_map(basis, reduced_basis, P)
    Q_set, P_set = set(Q), set(P)
    P_in_reduced_order = sorted(P, key=lambda i: reduced_index[i])
    frame = np.asarray(hamiltonian.rotating_frame_energies, dtype=float)
    reduced_frame = [float(frame[i]) for i in P_in_reduced_order]
    reduced_states = reduced_basis.states

    # ---- Energies (H0 plus static shifts) and pass-through shifts ----
    H0, H0_shifts, Hints, Rates = hamiltonian.H0_H0Shifts_Hints_and_Rates
    energies = np.real(np.asarray(H0.diagonal(), dtype=complex))
    output_operators = []
    for operator, shift_matrix in zip(hamiltonian.energy_shift_operators, H0_shifts):
        diagonal = np.real(np.asarray(shift_matrix.diagonal()).ravel())
        if operator.modulation_function is None:
            energies = energies + diagonal
        elif np.any(np.abs(diagonal[Q]) > SMALLEST_ENERGY_SCALE):
            raise NotImplementedError("Modulated energy shifts of eliminated states are not supported (CW only).")
        if np.any(np.abs(diagonal[P]) > SMALLEST_ENERGY_SCALE):
            # Keep the kept-state part of every energy shift, with its modulation; static ones too, as explicit operators.
            output_operators.append(EnergyShiftOperator.from_vector(reduced_basis, diagonal[P_in_reduced_order],
                                                                    modulation_function=operator.modulation_function))

    # ---- Couplings: pass-through P-P terms and directed P->Q terms ----
    directed = {q: [] for q in Q}  # q -> list of (p, v, w)
    for operator, Hint, Rate in zip(hamiltonian.coupling_operators, Hints, Rates):
        Hint, Rate = Hint.tocoo(), Rate.tocsr()
        passthrough = []
        for r, c, h in zip(Hint.row, Hint.col, Hint.data):
            if abs(h) < SMALLEST_ENERGY_SCALE:
                continue
            R = float(Rate[r, c])
            if r in P_set and c in P_set:
                passthrough.append(Coupling(row_state=reduced_states[reduced_index[r]], column_state=reduced_states[reduced_index[c]],
                                            strength=h, oscillation_rate=R - (frame[r] - frame[c])))
            elif r in Q_set and c in Q_set:
                raise NotImplementedError(f"Couplings among eliminated states ({basis.states[r].name} <-> {basis.states[c].name}) "
                                          f"are not supported; this needs the resolvent of the eliminated block.")
            else:
                if operator.modulation_function is not None:
                    raise NotImplementedError("Modulated couplings to eliminated states are not supported (CW drives only).")
                if r in Q_set:   # <q|H|p> = h e^{-iRt}
                    directed[r].append((c, h, R))
                else:            # <p|H|q> = h e^{-iRt}, so <q|H|p> = conj(h) e^{+iRt}
                    directed[c].append((r, np.conj(h), -R))
        if passthrough:
            output_operators.append(CouplingOperator(reduced_basis, passthrough, operator.modulation_function))

    # ---- Decay of eliminated states, scattering channels, pass-through decay ----
    gammas = np.zeros(len(basis.states))
    lindblad_channels = []      # per Lindblad operator: list of (g, q, lambda, r) with q eliminated, g kept
    passthrough_lindblad = []   # per Lindblad operator: list of (g, e, lambda, r) with g, e kept
    if dissipator is not None:
        _check_dissipator(hamiltonian, dissipator)
        for elements in _lindblad_elements(dissipator):
            channel, passthrough = [], []
            for g, e, lam, r in elements:
                gammas[e] += abs(lam)**2
                if e in Q_set and g in P_set:
                    channel.append((g, e, lam, r))
                elif e in Q_set and g in Q_set:
                    if g != e:
                        raise NotImplementedError(f"Decay among eliminated states ({basis.states[e].name} -> {basis.states[g].name}) "
                                                  f"is not supported.")
                elif e in P_set and g in Q_set:
                    raise NotImplementedError(f"Decay from a kept state into an eliminated state ({basis.states[e].name} -> "
                                              f"{basis.states[g].name}) is not supported.")
                else:
                    passthrough.append((g, e, lam, r))
            lindblad_channels.append(channel)
            passthrough_lindblad.append(passthrough)
        _warn_on_offdiagonal_decay(dissipator, Q)

    # ---- Complex denominators and validity ----
    def denominator(p: int, w: float, q: int) -> complex:
        return energies[p] + w - energies[q] + 0.5j * gammas[q]

    max_ratio = 0.
    for q, terms in directed.items():
        for p, v, w in terms:
            D = denominator(p, w, q)
            if abs(D) <= SMALLEST_ENERGY_SCALE:
                raise IonSimError(f"Coupling {basis.states[p].name} -> {basis.states[q].name} is resonant (denominator {D}); "
                                  f"the state cannot be adiabatically eliminated.")
            max_ratio = max(max_ratio, abs(v / D))
    if max_ratio > validity_threshold:
        warnings.warn(f"Adiabatic elimination: max |coupling / detuning| = {max_ratio:.3g} exceeds validity_threshold "
                      f"{validity_threshold}; second-order results may be inaccurate.", stacklevel=2)

    # ---- Effective Hamiltonian terms ----
    raw_terms = {}  # (p, p') with p <= p' -> list of (amplitude, rate)
    for q, terms in directed.items():
        for p_l, v_l, w_l in terms:            # bra side: <p_l| V^dagger |q>
            for p_k, v_k, w_k in terms:        # ket side: <q| V |p_k>
                a, b = reduced_index[p_l], reduced_index[p_k]
                if a > b:
                    continue  # the conjugate (p_k, p_l) term is generated by the swapped pair
                amplitude = 0.5 * np.conj(v_l) * v_k * (1. / denominator(p_k, w_k, q) + 1. / np.conj(denominator(p_l, w_l, q)))
                raw_terms.setdefault((a, b), []).append((amplitude, w_k - w_l))

    light_shifts = np.zeros(len(P))
    dropped, dropped_fast = [], []
    off_diagonal = []  # (a, b, amplitude, total rate)
    for (a, b), terms in raw_terms.items():
        for amplitude, rate in _group_by_rate(terms, rate_tolerance):
            if abs(amplitude) < SMALLEST_ENERGY_SCALE:
                continue
            if rwa_rate_cutoff is not None and abs(rate) > rwa_rate_cutoff:
                if a != b or rate > 0.:  # report each diagonal c.c. pair once
                    dropped_fast.append((reduced_states[a], reduced_states[b], amplitude, rate))
                continue
            if a != b:
                off_diagonal.append((a, b, amplitude, rate))
            elif _is_zero_rate(rate, rate_tolerance):
                light_shifts[a] += np.real(amplitude)
            elif rate > 0.:  # the term at -rate is its complex conjugate
                if oscillating_shifts == 'drop':
                    dropped.append((reduced_states[a], amplitude, rate))
                else:
                    vector = np.zeros(len(P))
                    vector[a] = 2. * abs(amplitude)
                    output_operators.append(EnergyShiftOperator.from_vector(
                        reduced_basis, vector, modulation_function=_cosine(rate, np.angle(amplitude))))
    if np.any(np.abs(light_shifts) > SMALLEST_ENERGY_SCALE):
        output_operators.append(EnergyShiftOperator.from_vector(reduced_basis, light_shifts))

    # One CouplingOperator per bare oscillation rate (a CouplingOperator has a single rate).
    by_bare_rate = []
    for a, b, amplitude, rate in off_diagonal:
        bare = rate - (reduced_frame[a] - reduced_frame[b])
        coupling = Coupling(row_state=reduced_states[a], column_state=reduced_states[b], strength=amplitude, oscillation_rate=bare)
        for group in by_bare_rate:
            if _rates_equal(group[0], bare, rate_tolerance):
                group[1].append(coupling)
                break
        else:
            by_bare_rate.append((bare, [coupling]))
    for bare, couplings in by_bare_rate:
        couplings = [Coupling(row_state=c.row_state, column_state=c.column_state, strength=c.strength, oscillation_rate=bare) for c in couplings]
        output_operators.append(CouplingOperator(reduced_basis, couplings))

    effective_hamiltonian = Hamiltonian(reduced_basis, output_operators, reduced_frame, hamiltonian.sparse)

    # ---- Effective Lindblad operators ----
    effective_dissipator = None
    scattering_operators = []
    if dissipator is not None:
        lindblad_operators = []
        for channel, passthrough in zip(lindblad_channels, passthrough_lindblad):
            elements = []  # (g, p) reduced indices, amplitude, total rate
            for g, q, lam, r in channel:
                for p, v, w in directed.get(q, []):
                    elements.append((reduced_index[g], reduced_index[p], lam * v / denominator(p, w, q), r + w))
            for group in _group_elements_by_rate(elements, rate_tolerance):
                matrix, rate = _group_matrix(group, len(P))
                if np.max(np.abs(matrix)) < SMALLEST_ENERGY_SCALE:
                    continue
                scattering_operators.append((matrix, rate))
                lindblad_operators.append(_lindblad_operator(reduced_basis, group, reduced_frame, rate_tolerance))
            if passthrough:
                kept = [(reduced_index[g], reduced_index[e], lam, r) for g, e, lam, r in passthrough]
                for group in _group_elements_by_rate(kept, rate_tolerance):
                    lindblad_operators.append(_lindblad_operator(reduced_basis, group, reduced_frame, rate_tolerance))
        if lindblad_operators:
            effective_dissipator = Dissipator(reduced_basis, lindblad_operators, reduced_frame, dissipator.sparse)

    return AdiabaticEliminationResult(
        hamiltonian=effective_hamiltonian, dissipator=effective_dissipator, reduced_basis=reduced_basis,
        kept_states=[basis.states[i] for i in P_in_reduced_order], eliminated_states=[basis.states[i] for i in Q],
        light_shifts=light_shifts, max_perturbation_ratio=float(max_ratio),
        dropped_oscillating_shifts=dropped, dropped_fast_terms=dropped_fast, scattering_operators=scattering_operators)


# ---------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------
def _match_levels(basis: StandardBasis, levels: list[EnergyLevel]) -> list[EnergyLevel]:
    """ The levels of every AtomicStructure DOF whose names match the requested levels. """
    if not levels:
        raise IonSimError("Specify at least one level to eliminate.")
    names = {level.name for level in levels}
    matched, found = [], set()
    for dof in basis.degrees_of_freedom:
        if not isinstance(dof, AtomicStructure):
            continue
        for level in dof.energy_levels:
            if level.name in names:
                matched.append(level)
                found.add(level.name)
    missing = sorted(names - found)
    if missing:
        raise IonSimError(f"Levels {missing} are not in any AtomicStructure degree of freedom of the basis.")
    return matched


def _reduced_index_map(basis: StandardBasis, reduced_basis: StandardBasis, P: list[int]) -> dict[int, int]:
    """ Map from kept-state indices of the full basis to indices of the reduced basis, matched by components. """
    if len(reduced_basis.states) != len(P):
        raise IonSimError("The reduced basis must be the product basis of the kept levels; eliminating these levels does "
                          "not leave a product basis.")
    lookup = {tuple(id(c) for c in state.components): index for index, state in enumerate(reduced_basis.states)}
    index_map = {}
    for i in P:
        key = tuple(id(c) for c in basis.states[i].components)
        if key not in lookup:
            raise IonSimError(f"Kept state {basis.states[i].name} is not in the reduced basis.")
        index_map[i] = lookup[key]
    return index_map


def _check_dissipator(hamiltonian: Hamiltonian, dissipator: Dissipator):
    if dissipator.basis is not hamiltonian.basis:
        if len(dissipator.basis.states) != len(hamiltonian.basis.states) or any(
                a is not b for a, b in zip(dissipator.basis.states, hamiltonian.basis.states)):
            raise IonSimError("The dissipator must be built on the same basis as the Hamiltonian.")
    if not np.allclose(dissipator.rotating_frame_energies, hamiltonian.rotating_frame_energies, rtol=0., atol=SMALLEST_ENERGY_SCALE):
        raise IonSimError("The dissipator and the Hamiltonian must use the same rotating_frame_energies.")


def _lindblad_elements(dissipator: Dissipator) -> list[list[tuple]]:
    """ For each Lindblad operator, its elements (row g, column e, strength, frame-shifted rate). """
    all_elements = []
    for operator in dissipator.operators:
        if operator.modulation_function is not None:
            raise NotImplementedError("Modulated Lindblad operators are not supported.")
        elements = []
        if isinstance(operator, (EnergyShiftOperator, GeneralOperator)):
            for shift in operator.energy_shifts:
                index = dissipator.basis.states.index(shift.row_state)
                elements.append((index, index, shift.strength, 0.))
        if isinstance(operator, (CouplingOperator, GeneralOperator)) and operator.couplings:
            matrix, rate = dissipator._frame_shifted_coupling_matrix_and_rate_from_operator(operator)
            matrix, rate = matrix.tocoo(), rate.tocsr()
            for g, e, lam in zip(matrix.row, matrix.col, matrix.data):
                if abs(lam) > 0.:
                    elements.append((int(g), int(e), lam, float(rate[g, e])))
        all_elements.append(elements)
    return all_elements


def _warn_on_offdiagonal_decay(dissipator: Dissipator, Q: list[int]):
    """ The eliminated block uses only the diagonal of sum_L L^dagger L; warn if its off-diagonal part is significant. """
    if len(Q) < 2:
        return
    total = np.zeros((len(Q), len(Q)), dtype=complex)
    for operator in dissipator.operators:
        matrix = operator.static_matrix.toarray()[:, Q]
        total += matrix.conj().T @ matrix
    diagonal = np.max(np.abs(np.diag(total))) if total.size else 0.
    off_diagonal = np.max(np.abs(total - np.diag(np.diag(total))))
    if diagonal > 0. and off_diagonal > 1e-6 * diagonal:
        warnings.warn(f"Adiabatic elimination: sum L^dagger L has off-diagonal elements among eliminated states "
                      f"({off_diagonal / diagonal:.2g} of the largest decay rate), e.g. from an incomplete set of ground "
                      f"levels; only its diagonal is used.", stacklevel=3)


def _rates_equal(a: float, b: float, tolerance: float) -> bool:
    return abs(a - b) <= max(tolerance, 1e-12 * max(abs(a), abs(b)))


def _is_zero_rate(rate: float, tolerance: float) -> bool:
    return abs(rate) <= tolerance


def _group_by_rate(terms: list[tuple[complex, float]], tolerance: float) -> list[tuple[complex, float]]:
    """ Sum amplitudes of terms with equal rates (within tolerance); the rate of a group is its first member's. """
    groups = []
    for amplitude, rate in sorted(terms, key=lambda term: term[1]):
        if groups and _rates_equal(groups[-1][1], rate, tolerance):
            groups[-1][0] += amplitude
        else:
            groups.append([amplitude, rate])
    return [(amplitude, rate) for amplitude, rate in groups]


def _group_elements_by_rate(elements: list[tuple], tolerance: float) -> list[list[tuple]]:
    """ Group (row, column, amplitude, rate) elements with equal rates; equal (row, column) entries are summed. """
    groups = []
    for element in sorted(elements, key=lambda element: element[3]):
        if groups and _rates_equal(groups[-1][0][3], element[3], tolerance):
            groups[-1].append(element)
        else:
            groups.append([element])
    merged = []
    for group in groups:
        summed = {}
        for row, column, amplitude, rate in group:
            summed[(row, column)] = summed.get((row, column), 0.) + amplitude
        reference = group[0][3]
        merged.append([(row, column, amplitude, reference) for (row, column), amplitude in summed.items()])
    return merged


def _group_matrix(group: list[tuple], size: int) -> tuple[np.ndarray, float]:
    matrix = np.zeros((size, size), dtype=complex)
    for row, column, amplitude, _ in group:
        matrix[row, column] += amplitude
    return matrix, group[0][3]


def _lindblad_operator(reduced_basis: StandardBasis, group: list[tuple], reduced_frame: list[float], tolerance: float) -> GeneralOperator:
    """ One Lindblad operator from elements that share a total rate R (an overall phase e^{-iRt}, which drops out of the
        master equation). Diagonal elements are energy-shift elements; off-diagonal elements are couplings with the bare
        rate that the Dissipator's frame shift turns back into a static element.
        A GeneralOperator is used because the couplings need not share one bare rate. """
    states = reduced_basis.states
    elements = []
    for row, column, amplitude, _ in group:
        if abs(amplitude) < SMALLEST_ENERGY_SCALE:
            continue
        if row == column:
            elements.append(EnergyShift(row_state=states[row], column_state=states[row], strength=complex(amplitude)))
        else:
            bare = -(reduced_frame[row] - reduced_frame[column])
            elements.append(Coupling(row_state=states[row], column_state=states[column], strength=complex(amplitude), oscillation_rate=bare))
    return GeneralOperator(reduced_basis, elements)


def _cosine(rate: float, phase: float):
    """ cos(rate t - phase), so that 2|a| cos(rate t - arg a) = a e^{-i rate t} + c.c. """
    def modulation(t: float) -> float:
        return np.cos(rate * t - phase)
    return modulation
