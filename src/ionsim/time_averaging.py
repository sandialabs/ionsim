#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

""" Time averaging of fast oscillating terms: the second-order (James) effective Hamiltonian.

    For an interaction-picture Hamiltonian made of oscillating terms,
        H(t) = sum_k [ h_k exp(-i w_k t) + h_k^dagger exp(+i w_k t) ],    w_k > 0,
    whose rates are large compared with the couplings (|h_k| << w_k), the slow dynamics follow, at second order in the
    couplings, the time-averaged Hamiltonian (D. F. V. James and J. Jerke, Can. J. Phys. 85, 625 (2007))
        H_avg = sum_{m, n} [h_m^dagger, h_n] exp(i (w_m - w_n) t) / w_mn,     1 / w_mn = (1/w_m + 1/w_n) / 2,
    whose static part (m = n, after combining terms of equal rate) is
        H_avg = sum_k [h_k^dagger, h_k] / w_k.
    For one transition, h = (Omega / 2) |e><g| at rate w = Delta (laser minus transition frequency), this gives the AC Stark
    shifts +Omega^2 / (4 Delta) on |g> and -Omega^2 / (4 Delta) on |e>.

    Two entry points:
        time_averaged_hamiltonian(matrices, rates): the static part for given couplings and rates (the matrix-level core).
        Hamiltonian.time_average(rate_cutoff) (time_average_hamiltonian here): replaces every coupling of an IonSim
            Hamiltonian that oscillates faster than rate_cutoff (in the interaction picture of its diagonal) by its static
            second-order effect, and returns a new Hamiltonian with the slow terms unchanged.
    Uses: Bloch-Siegert and counter-rotating shifts, off-resonant (e.g. two-photon) couplings and the fourth-order light shift
    of frequency combs (ionsim.frequency_comb), off-resonant carrier terms in sideband gates.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from ionsim.ionsim_error import IonSimError


@dataclass(frozen=True, eq=False)
class TimeAverageResult:
    """ Result of time_averaged_hamiltonian.

        hamiltonian: static time-averaged Hamiltonian (dense, Hermitian, same shape as the h_k).
        rates: the distinct positive rates (rad/s) after grouping, and group_couplings their combined h (one per rate).
        max_perturbation_ratio: max over groups of ||h||_2 / w, the expansion parameter.
        slow_terms: (matrix, rate) pairs [h_m^dagger, h_n] / w_mn with 0 < |w_m - w_n| <= slow_rate_cutoff, if requested:
            near-degenerate cross terms that oscillate slowly, at exp(i (w_m - w_n) t).
    """
    hamiltonian: np.ndarray
    rates: np.ndarray
    group_couplings: np.ndarray
    max_perturbation_ratio: float
    slow_terms: list[tuple[np.ndarray, float]] = field(default_factory=list)

    @property
    def level_shifts(self) -> np.ndarray:
        """ Diagonal of the static time-averaged Hamiltonian (rad/s) """
        return np.real(np.diag(self.hamiltonian))


def group_terms_by_rate(matrices: np.ndarray, rates: np.ndarray, rate_tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """ Combine oscillating terms of (nearly) equal rate.

        matrices: (K, n, n) couplings h_k; rates: (K,) rates w_k (any sign; a term with w < 0 is replaced by its
        Hermitian conjugate at -w, which describes the same Hermitian pair). Rates within rate_tolerance of each other
        (single-linkage on the sorted rates) form one group, at the mean rate of its members weighted by ||h||^2.
        Returns (group matrices (G, n, n), group rates (G,)), all rates > 0.
    """
    matrices = np.asarray(matrices, dtype=complex)
    rates = np.asarray(rates, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2] or len(matrices) != len(rates):
        raise IonSimError("matrices must have shape (K, n, n) with one rate per matrix.")
    if np.any(np.abs(rates) <= rate_tolerance):
        raise IonSimError(f"A term has |rate| <= rate_tolerance ({rate_tolerance} rad/s): it is resonant and cannot be "
                          f"time averaged; keep it in the Hamiltonian instead.")
    negative = rates < 0.
    matrices = matrices.copy()
    matrices[negative] = np.conj(np.transpose(matrices[negative], (0, 2, 1)))
    rates = np.abs(rates)
    order = np.argsort(rates)
    rates, matrices = rates[order], matrices[order]
    breaks = np.flatnonzero(np.diff(rates) > rate_tolerance) + 1
    starts = np.concatenate([[0], breaks])
    group_matrices = np.add.reduceat(matrices, starts, axis=0)
    weights = np.sum(np.abs(matrices)**2, axis=(1, 2)) + 1e-300
    group_rates = np.add.reduceat(weights * rates, starts) / np.add.reduceat(weights, starts)
    return group_matrices, group_rates


def time_averaged_hamiltonian(matrices, rates, rate_tolerance: float = 1e-6, slow_rate_cutoff: float | None = None
                              ) -> TimeAverageResult:
    """ Static time-averaged Hamiltonian sum_k [h_k^dagger, h_k] / w_k of H(t) = sum_k [h_k e^{-i w_k t} + h.c.].

        Args:
            matrices: couplings h_k, shape (K, n, n) (or a list of n x n matrices). Give each physical term once: either
                h at rate w or h^dagger at rate -w, not both.
            rates: w_k in rad/s (interaction picture). Terms of equal rate (within rate_tolerance) are combined first.
            rate_tolerance: rad/s; also the smallest |rate| accepted (resonant terms are an error).
            slow_rate_cutoff: if given (rad/s), also return the cross terms between groups whose rates differ by at most
                this much (slowly oscillating, not included in the static Hamiltonian).
        Returns:
            TimeAverageResult. Valid when max_perturbation_ratio << 1.
    """
    group_matrices, group_rates = group_terms_by_rate(np.asarray(matrices), np.asarray(rates), rate_tolerance)
    dagger = np.conj(np.transpose(group_matrices, (0, 2, 1)))
    commutators = dagger @ group_matrices - group_matrices @ dagger
    H = np.sum(commutators / group_rates[:, None, None], axis=0)
    H = 0.5 * (H + H.conj().T)
    norms = np.linalg.norm(group_matrices, ord=2, axis=(1, 2))
    ratio = float(np.max(norms / group_rates)) if len(group_rates) else 0.
    slow = []
    if slow_rate_cutoff is not None and len(group_rates) > 1:
        for m in range(len(group_rates)):
            for n in range(len(group_rates)):
                difference = group_rates[m] - group_rates[n]
                if m != n and abs(difference) <= slow_rate_cutoff:
                    inverse = 0.5 * (1. / group_rates[m] + 1. / group_rates[n])
                    slow.append(((dagger[m] @ group_matrices[n] - group_matrices[n] @ dagger[m]) * inverse, float(difference)))
    return TimeAverageResult(hamiltonian=H, rates=group_rates, group_couplings=group_matrices,
                             max_perturbation_ratio=ratio, slow_terms=slow)


def time_average_hamiltonian(hamiltonian, rate_cutoff: float, rate_tolerance: float = 1.,
                             validity_threshold: float = 0.1, return_result: bool = False):
    """ New Hamiltonian in which every coupling faster than rate_cutoff is replaced by its static second-order effect.

        The rate of a coupling element <r|H|c> = h exp(-i R t) is taken in the interaction picture of the Hamiltonian's
        static diagonal D (frame energies plus unmodulated energy shifts), w = R - (D_r - D_c): a term is fast when it is
        far from resonance, whatever the frame. Elements with |w| > rate_cutoff are time averaged together
        (time_averaged_hamiltonian): the diagonal of the result is added as an EnergyShiftOperator, and off-diagonal parts
        (couplings between states of equal D, if any) as couplings static in that interaction picture. All other operators
        (slow couplings, energy shifts, modulated operators) are kept as they are. Cross terms between fast terms of slightly
        different rates are dropped (second-order terms oscillating at the rate difference).

        Args:
            hamiltonian: an ionsim Hamiltonian.
            rate_cutoff: rad/s; couplings with |w| above it are averaged.
            rate_tolerance: rad/s; fast rates closer than this are combined (they interfere) before averaging.
            validity_threshold: warn if max |coupling / rate| of the averaged terms exceeds this value.
            return_result: also return the TimeAverageResult of the averaged terms (diagnostics).
        Returns:
            Hamiltonian (and TimeAverageResult if return_result). Modulated couplings with fast elements raise
            NotImplementedError (their time dependence is not a single rate).
    """
    from ionsim.hamiltonian import Hamiltonian
    from ionsim.operator import Coupling, CouplingOperator, EnergyShiftOperator
    from ionsim.config import SMALLEST_ENERGY_SCALE

    if rate_cutoff <= 0.:
        raise IonSimError(f"rate_cutoff must be positive (rad/s), got {rate_cutoff}.")
    basis = hamiltonian.basis
    states = basis.states
    n = len(states)
    frame = np.asarray(hamiltonian.rotating_frame_energies, dtype=float)
    H0, H0_shifts, Hints, Rates = hamiltonian.H0_H0Shifts_Hints_and_Rates
    D = np.real(np.asarray(H0.diagonal(), dtype=complex)).copy()
    for operator, shift in zip(hamiltonian.energy_shift_operators, H0_shifts):
        if operator.modulation_function is None:
            D += np.real(np.asarray(shift.diagonal()).ravel())

    kept = list(hamiltonian.energy_shift_operators)
    fast_matrices, fast_rates = [], []
    for operator, Hint, Rate in zip(hamiltonian.coupling_operators, Hints, Rates):
        Hint, Rate = Hint.tocoo(), Rate.tocsr()
        slow, fast = [], []
        for r, c, h in zip(Hint.row, Hint.col, Hint.data):
            if abs(h) < SMALLEST_ENERGY_SCALE:
                continue
            R = float(Rate[r, c])
            w = R - (D[r] - D[c])
            (fast if abs(w) > rate_cutoff else slow).append((int(r), int(c), complex(h), R))
        if fast and operator.modulation_function is not None:
            raise NotImplementedError("Time averaging of modulated couplings is not supported (their time dependence is "
                                      "not a single rate); raise rate_cutoff above their rates or remove the modulation.")
        if not fast:
            kept.append(operator)
            continue
        for r, c, h, R in fast:
            matrix = np.zeros((n, n), dtype=complex)
            matrix[r, c] = h
            fast_matrices.append(matrix)
            fast_rates.append(R - (D[r] - D[c]))
        if slow:
            couplings = [Coupling(row_state=states[r], column_state=states[c], strength=h,
                                  oscillation_rate=R - (frame[r] - frame[c])) for r, c, h, R in slow]
            kept.append(CouplingOperator(basis, couplings, operator.modulation_function))

    result = None
    if fast_matrices:
        result = time_averaged_hamiltonian(np.array(fast_matrices), np.array(fast_rates), rate_tolerance=rate_tolerance)
        if result.max_perturbation_ratio > validity_threshold:
            warnings.warn(f"Time averaging: max |coupling / rate| = {result.max_perturbation_ratio:.3g} exceeds "
                          f"validity_threshold {validity_threshold}; the second-order result may be inaccurate.", stacklevel=3)
        H_avg = result.hamiltonian
        diagonal = np.real(np.diag(H_avg))
        if np.any(np.abs(diagonal) > SMALLEST_ENERGY_SCALE):
            kept.append(EnergyShiftOperator.from_vector(basis, diagonal))
        # Off-diagonal parts are static in the interaction picture of D: total rate D_r - D_c in the frame
        couplings = []
        for r in range(n):
            for c in range(r + 1, n):
                if abs(H_avg[r, c]) > SMALLEST_ENERGY_SCALE:
                    couplings.append(Coupling(row_state=states[r], column_state=states[c], strength=complex(H_avg[r, c]),
                                              oscillation_rate=(D[r] - D[c]) - (frame[r] - frame[c])))
        for coupling in couplings:
            kept.append(CouplingOperator(basis, [coupling]))
    averaged = Hamiltonian(basis, kept, list(hamiltonian.rotating_frame_energies), hamiltonian.sparse)
    return (averaged, result) if return_result else averaged
