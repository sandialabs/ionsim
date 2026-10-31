#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

""" Effective (time-averaged) Hamiltonian of off-resonant drive terms.

    For an interaction-picture Hamiltonian made of oscillating terms,
        H(t) = sum_k [ h_k exp(-i w_k t) + h_k^dagger exp(+i w_k t) ],    w_k > 0,
    whose rates are large compared with the couplings (|h_k| << w_k), the slow dynamics follow, at second order in the
    couplings, the effective Hamiltonian (D. F. V. James and J. Jerke, Can. J. Phys. 85, 625 (2007))
        H_eff = sum_{m, n} [h_m^dagger, h_n] exp(i (w_m - w_n) t) / w_mn,     1 / w_mn = (1/w_m + 1/w_n) / 2.
    Its static part (m = n, after combining terms of equal rate) is
        H_eff = sum_k [h_k^dagger, h_k] / w_k.
    For one transition, h = (Omega / 2) |e><g| at rate w = Delta (laser minus transition frequency), this gives the
    familiar AC Stark shifts +Omega^2 / (4 Delta) on |g> and -Omega^2 / (4 Delta) on |e>.

    Typical uses: Bloch-Siegert and counter-rotating shifts, off-resonant two-photon (Raman) couplings such as the
    fourth-order light shift of a frequency comb (ionsim.frequency_comb), off-resonant carrier terms in sideband gates.
    The rates are those of the interaction picture with respect to the bare energies of the states the h_k connect, so
    that a term is 'off resonant' exactly when its rate is far from zero. Terms of equal rate must be combined before
    taking the commutator (their cross terms are static); effective_hamiltonian groups rates within rate_tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ionsim.ionsim_error import IonSimError


@dataclass(frozen=True, eq=False)
class EffectiveHamiltonianResult:
    """ Result of effective_hamiltonian.

        hamiltonian: static effective Hamiltonian (dense, Hermitian, same shape as the h_k).
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
        """ Diagonal of the static effective Hamiltonian (rad/s) """
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
                          f"treated as off resonant; keep it in the Hamiltonian instead.")
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


def effective_hamiltonian(matrices, rates, rate_tolerance: float = 1e-6, slow_rate_cutoff: float | None = None
                          ) -> EffectiveHamiltonianResult:
    """ Static effective Hamiltonian sum_k [h_k^dagger, h_k] / w_k of H(t) = sum_k [h_k e^{-i w_k t} + h.c.].

        Args:
            matrices: couplings h_k, shape (K, n, n) (or a list of n x n matrices). Give each physical term once: either
                h at rate w or h^dagger at rate -w, not both.
            rates: w_k in rad/s (interaction picture). Terms of equal rate (within rate_tolerance) are combined first.
            rate_tolerance: rad/s; also the smallest |rate| accepted (resonant terms are an error).
            slow_rate_cutoff: if given (rad/s), also return the cross terms between groups whose rates differ by at most
                this much (slowly oscillating, not included in the static Hamiltonian).
        Returns:
            EffectiveHamiltonianResult. Valid when max_perturbation_ratio << 1.
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
    return EffectiveHamiltonianResult(hamiltonian=H, rates=group_rates, group_couplings=group_matrices,
                                      max_perturbation_ratio=ratio, slow_terms=slow)
