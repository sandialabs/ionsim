#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

""" Two-photon (Raman) couplings and fourth-order light shifts driven by optical frequency combs (mode-locked lasers).

    A mode-locked laser is a frequency comb: tooth j of comb alpha has angular frequency w_alpha(j) = w_alpha + j 2 pi f_rep
    (w_alpha includes any AOM shift of that comb) and field amplitude proportional to sech(j 2 pi f_rep tau) for pulses of
    duration tau, normalized so that the tooth intensities add up to the time-averaged intensity of the beam (A. C. Lee et
    al., PRA 94, 042308 (2016), Eq. (1); Yale et al., IEEE TQE 6, 3102117 (2025), Eq. (12)). The beam itself is described by
    an ionsim Laser (frequency of tooth 0, polarization, direction, phase) whose peak field is the time-averaged field.

    Far detuned from the excited manifolds (single-photon detuning >> comb bandwidth), the excited states are eliminated at
    second order, exactly as in ionsim.adiabatic_eliminator but summed over tooth pairs. Absorbing from tooth (alpha, j) and
    emitting into tooth (beta, k) couples ground sublevels s -> s' with
        <s'| H |s> = sum_J sum_{e in J} conj(v_beta[e, s']) v_alpha[e, s] S_J[s', s](l),       l = j - k,
        S_J[s', s](l) = sum_k w_alpha(k + l) w_beta(k) (1/Delta_J,alpha(k + l, s) + 1/Delta_J,beta(k, s')) / 2,
    where v[e, s] = (Omega_es / 2) e^{i phase} is the absorption element of the beam (IonSim's Laser couplings, time-averaged
    field), w(j) the normalized tooth amplitudes, and Delta_J(j, s) = w(j) - (E_J - E_s) the detuning of tooth j from the
    fine-structure centroid of manifold J (Yale et al. Eq. (13) generalized to all sublevels and polarizations). The P-state
    hyperfine structure inside a manifold (~1e-4 of the detuning) and counter-rotating terms are neglected here; the
    adiabatic eliminator includes both for a single tooth pair. The coupling oscillates, in the interaction picture of the
    bare ground-state energies E_s, at rate
        R = (w_alpha - w_beta) + l 2 pi f_rep - (E_s' - E_s).
    One (or a few) of these terms is resonant (|R| small): the Raman drive. All others are off-resonant two-photon couplings;
    at second order in them (time averaging, ionsim.time_averaging) they shift the levels by sum |<s'|H|s>|^2 / R: the fourth-order
    light shift (Lee et al. Eqs. (6)-(10); Yale et al. Eq. (14)). Same-beam, same-harmonic terms (alpha = beta, l = 0) are
    ordinary second-order light shifts and are reported separately.

    A continuous-wave beam is passed as its Laser (one tooth at its frequency), so CW tones, combs, or both can be combined;
    two CW tones reproduce the second-order Raman coupling of ionsim.adiabatic_eliminator (rotating-wave, P hyperfine aside).

    Entry point: raman_effective_model(basis, levels_to_eliminate, beams) returns a RamanEffectiveModel whose .hamiltonian
    is an ordinary IonSim Hamiltonian on the ground levels (Raman drive couplings plus second- and fourth-order light shifts),
    with the diagnostics alongside, as Hamiltonian.adiabatically_eliminate does for one drive frequency per transition.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import correlate

from ionsim.atomic_internal_energy_level import AtomicInternalEnergyLevel
from ionsim.time_averaging import time_averaged_hamiltonian
from ionsim.ionsim_error import IonSimError
from ionsim.laser import Laser
from ionsim.basis import StandardBasis
from ionsim.degree_of_freedom import AtomicStructure
from ionsim.hamiltonian import Hamiltonian
from ionsim.operator import Coupling, CouplingOperator, EnergyShiftOperator
from ionsim.config import SMALLEST_ENERGY_SCALE

TPI = 2. * np.pi


@dataclass(frozen=True, eq=False)
class FrequencyComb:
    """ The frequency comb of a mode-locked (pulsed) laser: a Laser (tooth 0: frequency, polarization, direction, phase;
        time-averaged field) plus the pulse train's repetition rate (Hz) and pulse duration tau (s), with tooth amplitudes
        proportional to sech(j 2 pi f_rep tau). A continuous-wave beam is not a comb: pass its Laser to
        comb_raman_couplings directly.

        tooth_cutoff: teeth whose relative amplitude sech(j 2 pi f_rep tau) is below this are dropped.
    """
    laser: Laser
    repetition_rate: float
    pulse_duration: float
    tooth_cutoff: float = 1e-12

    def __post_init__(self):
        if not self.repetition_rate > 0.:
            raise IonSimError(f"repetition_rate must be positive (Hz), got {self.repetition_rate}.")
        if self.pulse_duration is None or not self.pulse_duration > 0.:
            raise IonSimError(f"pulse_duration must be positive (s), got {self.pulse_duration}. For a continuous-wave beam, "
                              f"use its Laser instead of a FrequencyComb.")

    @property
    def envelope_argument(self) -> float:
        """ x = 2 pi f_rep tau, the argument of the comb's sech envelope per tooth: tooth j has amplitude sech(j x). The
            envelope spans ~1/x teeth (a bandwidth of ~f_rep / x = 1 / (2 pi tau)). """
        return TPI * self.repetition_rate * self.pulse_duration

    @property
    def tooth_indices(self) -> np.ndarray:
        J = int(np.ceil(np.arccosh(1. / self.tooth_cutoff) / self.envelope_argument))
        return np.arange(-J, J + 1)

    @property
    def tooth_weights(self) -> np.ndarray:
        """ Normalized tooth amplitudes w(j), sum_j w(j)^2 = 1 (the time-averaged intensity is shared among the teeth) """
        amplitudes = 1. / np.cosh(self.tooth_indices * self.envelope_argument)
        return amplitudes / np.sqrt(np.sum(amplitudes**2))

    def tooth_frequencies(self) -> np.ndarray:
        return self.laser.frequency + self.tooth_indices * TPI * self.repetition_rate


Beam = Laser | FrequencyComb       # a beam: one continuous-wave tone (Laser) or a frequency comb


def _beam_laser(beam: Beam) -> Laser:
    return beam.laser if isinstance(beam, FrequencyComb) else beam


def _beam_teeth(beam: Beam) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ (tooth indices, normalized tooth amplitudes, tooth angular frequencies) of a beam; a CW Laser is one tooth """
    if isinstance(beam, FrequencyComb):
        return beam.tooth_indices, beam.tooth_weights, beam.tooth_frequencies()
    if isinstance(beam, Laser):
        return np.zeros(1, dtype=int), np.ones(1), np.array([beam.frequency])
    raise IonSimError(f"A beam must be a Laser (continuous wave) or a FrequencyComb, got {type(beam).__name__}.")


def frequency_comb_overlap(beam_a: Beam, beam_b: Beam, harmonics) -> np.ndarray:
    """ sum_k w_a(k + l) w_b(k) for each harmonic l: the comb-structure factor of the two-photon coupling, without the
        per-tooth detuning dependence. Beams are FrequencyCombs or (CW) Lasers; for two CW Lasers it is 1 at l = 0. For
        sech teeth it is close to (l x)/sinh(l x). """
    indices_a, wa, _ = _beam_teeth(beam_a)
    indices_b, wb, _ = _beam_teeth(beam_b)
    full = correlate(wa, wb, mode='full', method='auto')
    lags = np.arange(len(full)) - (len(wb) - 1) + indices_a[0] - indices_b[0]
    lookup = dict(zip(lags.tolist(), full))
    return np.array([lookup.get(int(l), 0.) for l in np.atleast_1d(harmonics)])


@dataclass(frozen=True, eq=False)
class RamanEffectiveModel:
    """ Result of raman_effective_model: the effective Hamiltonian of the ground levels driven by Raman beams (CW tones
        and/or frequency combs), with the diagnostics of its construction (as AdiabaticEliminationResult for
        adiabatically_eliminate).

        hamiltonian: Hamiltonian on reduced_basis, in the requested rotating frame, holding the resonant two-photon (Raman)
            couplings, the second-order light shifts (if included) and the fourth-order light shifts.
        reduced_basis: basis of the ground levels (the levels that were not eliminated).
        ground_levels: the atomic levels of reduced_basis, in its order.
        second_order_shifts: static light shifts of same-beam, same-harmonic terms (alpha = beta, l = 0), diagonal (rad/s).
        fourth_order_hamiltonian: static time-averaged Hamiltonian of every off-resonant two-photon term (rad/s, n x n);
            fourth_order_shifts is its diagonal.
        resonant_terms: [(alpha, beta, l, row, column, amplitude, rate)] with |rate| <= resonance_tolerance: the drive
            (alpha, beta: indices of the absorbing and emitting beams; l: harmonic; rate in the interaction picture of the
            bare ground energies).
        max_perturbation_ratio: max |coupling| / |rate| of the off-resonant terms (the expansion parameter).
        n_terms: number of off-resonant element terms included.
        harmonic_couplings: {(alpha, beta, l): (coupling matrix, rate matrix)} for the requested keep_harmonics.
    """
    hamiltonian: object
    reduced_basis: object
    ground_levels: list
    second_order_shifts: np.ndarray
    fourth_order_hamiltonian: np.ndarray
    resonant_terms: list
    max_perturbation_ratio: float
    n_terms: int
    harmonic_couplings: dict = field(default_factory=dict, repr=False)

    @property
    def fourth_order_shifts(self) -> np.ndarray:
        return np.real(np.diag(self.fourth_order_hamiltonian))

    def _index(self, level) -> int:
        if isinstance(level, (int, np.integer)):
            return int(level)
        names = [g.name for g in self.ground_levels]
        name = level if isinstance(level, str) else level.name
        return names.index(name)

    def resonant_coupling(self, row, column) -> complex:
        """ Sum of the resonant amplitudes <row| H |column> (rad/s): the Raman drive element (Rabi rate / 2). """
        r, c = self._index(row), self._index(column)
        return complex(sum(term[5] for term in self.resonant_terms if term[3] == r and term[4] == c))

    def differential_fourth_order_shift(self, upper, lower) -> float:
        """ Fourth-order shift of the transition upper - lower (rad/s) """
        return float(self.fourth_order_shifts[self._index(upper)] - self.fourth_order_shifts[self._index(lower)])


def _absorption_elements(laser: Laser, ground_levels, excited_levels, atomic_levels) -> np.ndarray:
    """ v[e, s] = (Omega_es / 2) e^{i phase}: the absorption element <e| H |s> of the laser (IonSim convention) """
    v = np.zeros((len(excited_levels), len(ground_levels)), dtype=complex)
    for i, e in enumerate(excited_levels):
        for k, s in enumerate(ground_levels):
            rabi = laser.rabi_frequency(s, e, 1)
            if abs(rabi) > 0.:
                v[i, k] = 0.5 * rabi * np.exp(1j * laser.phase)
    return v


def _two_photon_couplings(ground_levels: list[AtomicInternalEnergyLevel], excited_levels: list[AtomicInternalEnergyLevel],
                          beams: list[Beam], atomic_levels: list[AtomicInternalEnergyLevel] | None = None,
                          resonance_tolerance: float = TPI * 1e5, rate_tolerance: float = TPI * 1.,
                          coupling_cutoff: float = 1e-12, keep_harmonics: list | None = None) -> dict:
    """ Tooth-summed two-photon couplings between ground levels for every ordered pair of beams and every harmonic, and the
        fourth-order light shift of the off-resonant ones (module docstring).

        Args:
            ground_levels, excited_levels: levels of one atom (e.g. all S1/2 and all P1/2, P3/2 sublevels); excited levels
                are grouped into fine-structure manifolds by term symbol, and each manifold's detuning is taken from its
                fine-structure energy.
            beams: FrequencyComb objects (pulsed) and/or Laser objects (continuous wave, a single tone). All combs must share
                one repetition rate (one mode-locked laser, possibly split and shifted by AOMs); their Lasers' frequencies
                are the tooth-0 frequencies. With only CW Lasers, there is one harmonic, l = 0, and the result is the
                second-order (adiabatic-elimination) Raman coupling of the tones plus the fourth-order shift of their
                off-resonant pairs (e.g. Bloch-Siegert-type terms).
            atomic_levels: all levels of the atom (for the coupling builders); defaults to ground + excited levels.
            resonance_tolerance: rad/s; terms with |rate| at most this are resonant (returned as the drive, not shifted).
            rate_tolerance: rad/s; rates closer than this are combined before the effective Hamiltonian is taken.
            coupling_cutoff: terms with |coupling| < coupling_cutoff x (largest coupling) are dropped.
            keep_harmonics: optional list of (alpha, beta, l) whose coupling matrices are stored in harmonic_couplings.
        Returns:
            dict of the RamanEffectiveModel diagnostics (see raman_effective_model, which wraps this).
    """
    if not beams:
        raise IonSimError("Give at least one beam (FrequencyComb or Laser).")
    teeth = [_beam_teeth(beam) for beam in beams]          # also checks the beam types
    rates_hz = {beam.repetition_rate for beam in beams if isinstance(beam, FrequencyComb)}
    if len(rates_hz) > 1:
        raise IonSimError("All combs must share one repetition rate (one mode-locked laser).")
    f_rep = rates_hz.pop() if rates_hz else 0.           # CW tones only: every pair is at harmonic l = 0
    lasers = [_beam_laser(beam) for beam in beams]
    atomic_levels = list(atomic_levels) if atomic_levels is not None else list(ground_levels) + list(excited_levels)
    n = len(ground_levels)
    E_ground = np.array([g.energy for g in ground_levels])

    # Excited manifolds (fine-structure centroids) and absorption elements of each beam's laser
    manifolds = {}
    for i, e in enumerate(excited_levels):
        manifolds.setdefault(e.term_symbol, []).append(i)
    E_manifold = {J: excited_levels[idx[0]].fine_energy for J, idx in manifolds.items()}
    v = [_absorption_elements(laser, ground_levels, excited_levels, atomic_levels) for laser in lasers]
    # M_J[alpha, beta][s', s] = sum_{e in J} conj(v_beta[e, s']) v_alpha[e, s]
    M = {(a, b, J): np.conj(v[b][idx]).T @ v[a][idx] for a in range(len(beams)) for b in range(len(beams))
         for J, idx in manifolds.items()}

    # Tooth sums S_J[s', s](l) by correlation: 1/Delta depends on the tooth, the manifold and the ground level
    all_rows, all_cols, all_amps, all_rates = [], [], [], []
    second_order = np.zeros(n)
    resonant = []
    stored = {}
    for a in range(len(beams)):
        for b in range(len(beams)):
            (indices_a, wa, frequencies_a), (indices_b, wb, frequencies_b) = teeth[a], teeth[b]
            lags = np.arange(len(wa) + len(wb) - 1) - (len(wb) - 1) + indices_a[0] - indices_b[0]
            amplitude = np.zeros((len(lags), n, n), dtype=complex)
            for J, idx in manifolds.items():
                if not np.any(M[(a, b, J)]):
                    continue
                inv_a = 1. / (frequencies_a[:, None] - (E_manifold[J] - E_ground[None, :]))       # (teeth_a, s)
                inv_b = 1. / (frequencies_b[:, None] - (E_manifold[J] - E_ground[None, :]))       # (teeth_b, s')
                # S[l, s', s] = 1/2 sum_k w_a(k+l) w_b(k) (inv_a(k+l, s) + inv_b(k, s'))
                term_a = np.stack([correlate(wa * inv_a[:, s], wb, mode='full', method='auto') for s in range(n)], axis=1)
                term_b = np.stack([correlate(wa, wb * inv_b[:, sp], mode='full', method='auto') for sp in range(n)], axis=1)
                S = 0.5 * (term_a[:, None, :] + term_b[:, :, None])                             # (lags, s', s)
                amplitude += M[(a, b, J)][None, :, :] * S
            offset = lasers[a].frequency - lasers[b].frequency
            rates = offset + lags[:, None, None] * TPI * f_rep - (E_ground[None, :, None] - E_ground[None, None, :])
            if keep_harmonics:
                for key in keep_harmonics:
                    if key[0] == a and key[1] == b and key[2] in set(lags.tolist()):
                        stored[key] = (amplitude[list(lags).index(key[2])], rates[list(lags).index(key[2])])
            if a == b:
                zero = int(np.flatnonzero(lags == 0)[0])
                second_order += np.real(np.diag(amplitude[zero]))
                amplitude[zero] = 0.                          # same beam, same harmonic: second order, reported separately
            l_index, rows, cols = np.nonzero(np.abs(amplitude) > 0.)
            amps = amplitude[l_index, rows, cols]
            r = rates[l_index, rows, cols]
            is_resonant = np.abs(r) <= resonance_tolerance
            for k in np.flatnonzero(is_resonant):
                resonant.append((a, b, int(lags[l_index[k]]), int(rows[k]), int(cols[k]), complex(amps[k]), float(r[k])))
            keep = ~is_resonant & (r > 0.)                    # each Hermitian pair once (its partner has rate -r)
            all_rows.append(rows[keep]); all_cols.append(cols[keep]); all_amps.append(amps[keep]); all_rates.append(r[keep])

    rows, cols = np.concatenate(all_rows), np.concatenate(all_cols)
    amps, rates = np.concatenate(all_amps), np.concatenate(all_rates)
    if len(amps):
        keep = np.abs(amps) >= coupling_cutoff * np.abs(amps).max()
        rows, cols, amps, rates = rows[keep], cols[keep], amps[keep], rates[keep]
    # Coherent sums of identical (rate, element) terms (e.g. the same harmonic of two combs with the same offset)
    keys = np.round(rates / rate_tolerance).astype(np.int64)
    table = np.stack([keys, rows, cols], axis=1)
    unique, inverse = np.unique(table, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    summed = np.zeros(len(unique), dtype=complex)
    np.add.at(summed, inverse, amps)
    rate_sum = np.zeros(len(unique)); np.add.at(rate_sum, inverse, rates)
    counts = np.bincount(inverse, minlength=len(unique))
    u_rates, u_rows, u_cols = rate_sum / counts, unique[:, 1], unique[:, 2]
    # Diagonal elements commute with themselves; off-diagonal elements alone at their rate give |a|^2/R on the two levels;
    # rates shared by several off-diagonal elements (degenerate transitions) need the full commutator.
    off = u_rows != u_cols
    H = np.zeros((n, n), dtype=complex)
    key_of = unique[:, 0]
    off_keys, off_counts = np.unique(key_of[off], return_counts=True)
    shared = set(off_keys[off_counts > 1].tolist())
    single = off & ~np.isin(key_of, list(shared))
    contribution = np.abs(summed[single])**2 / u_rates[single]
    np.add.at(H, (u_cols[single], u_cols[single]), contribution)        # |a|^2/R on the initial level s (column)
    np.add.at(H, (u_rows[single], u_rows[single]), -contribution)       # and -|a|^2/R on the final level s' (row)
    ratio = float(np.max(np.abs(summed[off]) / u_rates[off])) if np.any(off) else 0.
    if shared:
        groups = sorted(shared)
        matrices = np.zeros((len(groups), n, n), dtype=complex)
        group_rates = np.zeros(len(groups))
        position = {k: g for g, k in enumerate(groups)}
        for k in np.flatnonzero(np.isin(key_of, groups)):
            g = position[int(key_of[k])]
            matrices[g, u_rows[k], u_cols[k]] += summed[k]
            group_rates[g] = u_rates[k]
        H += time_averaged_hamiltonian(matrices, group_rates, rate_tolerance=rate_tolerance).hamiltonian
    return dict(fourth_order_hamiltonian=0.5 * (H + H.conj().T), second_order_shifts=second_order, resonant_terms=resonant,
                max_perturbation_ratio=ratio, n_terms=int(len(summed)), harmonic_couplings=stored)


def raman_effective_model(basis: StandardBasis, levels_to_eliminate: list, beams: list[Beam],
                          rotating_frame_energies: list[float] | None = None, include_second_order: bool = True,
                          resonance_tolerance: float = TPI * 1e5, rate_tolerance: float = TPI * 1.,
                          coupling_cutoff: float = 1e-12, keep_harmonics: list | None = None) -> RamanEffectiveModel:
    """ Effective Hamiltonian of the ground levels of an atom driven by Raman beams (CW tones and/or frequency combs), with
        the excited levels eliminated at second order and every off-resonant two-photon term time averaged (module
        docstring). The analogue of Hamiltonian.adiabatically_eliminate for many drive frequencies.

        Args:
            basis: StandardBasis of a single AtomicStructure (e.g. 171Yb+ with S1/2, P1/2 and P3/2).
            levels_to_eliminate: the excited levels (matched by name), e.g. all P1/2 and P3/2 sublevels; their
                fine-structure manifolds (term symbols) set the detunings. The other levels are kept.
            beams: FrequencyComb objects (pulsed) and/or Lasers (continuous wave); combs share one repetition rate.
            rotating_frame_energies: frame of the reduced basis (rad/s, one per kept level); default the interaction
                picture of the bare energies, -E.
            include_second_order: include the second-order light shifts in the Hamiltonian (they are always reported in
                second_order_shifts). These neglect the P-state hyperfine structure and counter-rotating terms; to use the
                adiabatic eliminator's for a CW-equivalent pair of tones instead, set False and add them yourself.
            resonance_tolerance: rad/s; two-photon terms within this of resonance are kept as couplings (the Raman drive),
                all others are time averaged.
            rate_tolerance: rad/s; off-resonant rates closer than this are combined (they interfere) before averaging.
            coupling_cutoff: off-resonant terms with |coupling| < coupling_cutoff x (largest coupling) are dropped.
            keep_harmonics: optional list of (alpha, beta, l) whose coupling matrices are stored in harmonic_couplings.
        Returns:
            RamanEffectiveModel; simulate with e.g. Lindbladian(model.hamiltonian, None) or Gate.from_lindbladian. Same-beam,
            same-harmonic off-diagonal terms (e.g. a single beam's own vector light-shift coupling between hyperfine
            levels, rate ~ the hyperfine splitting) are not included.
    """
    atoms = [dof for dof in basis.degrees_of_freedom if isinstance(dof, AtomicStructure)]
    if len(atoms) != 1 or len(basis.degrees_of_freedom) != 1:
        raise NotImplementedError("raman_effective_model supports a basis of a single AtomicStructure (no motional modes "
                                  "or other atoms yet).")
    atom = atoms[0]
    names = {level.name for level in levels_to_eliminate}
    excited = [level for level in atom.energy_levels if level.name in names]
    missing = sorted(names - {level.name for level in excited})
    if missing:
        raise IonSimError(f"Levels {missing} are not levels of the atom in the basis.")
    reduced_basis = basis.build_subspace_basis_from_levels_to_project(excited)
    by_name = {level.name: level for level in atom.energy_levels}
    ground = [by_name[state.components[0].name] for state in reduced_basis.states]
    n = len(ground)
    terms = _two_photon_couplings(ground, excited, beams, atom.energy_levels, resonance_tolerance=resonance_tolerance,
                                  rate_tolerance=rate_tolerance, coupling_cutoff=coupling_cutoff, keep_harmonics=keep_harmonics)
    E = np.array([level.energy for level in ground])
    frame = -E if rotating_frame_energies is None else np.asarray(rotating_frame_energies, dtype=float)
    if len(frame) != n:
        raise IonSimError(f"rotating_frame_energies must have one entry per kept level ({n}), got {len(frame)}.")
    states = reduced_basis.states

    operators = []
    H4 = terms['fourth_order_hamiltonian']
    shifts = np.real(np.diag(H4)) + (terms['second_order_shifts'] if include_second_order else 0.)
    if np.any(np.abs(shifts) > SMALLEST_ENERGY_SCALE):
        operators.append(EnergyShiftOperator.from_vector(reduced_basis, shifts))
    # Off-diagonal time-averaged terms are static in the interaction picture of the bare energies: lab rate E_r - E_c
    for r in range(n):
        for c in range(r + 1, n):
            if abs(H4[r, c]) > SMALLEST_ENERGY_SCALE:
                operators.append(CouplingOperator(reduced_basis, [Coupling(row_state=states[r], column_state=states[c],
                                                  strength=complex(H4[r, c]), oscillation_rate=float(E[r] - E[c]))]))
    # Resonant (drive) terms, one orientation of each Hermitian pair (row < column), at their lab rates
    # R + (E_row - E_column); terms of equal rate on the same element add coherently. One CouplingOperator per rate.
    drive = {}
    for _, _, _, r, c, amplitude, rate in terms['resonant_terms']:
        if r < c:
            lab_rate = rate + (E[r] - E[c])
            key = (r, c, int(np.round(lab_rate / rate_tolerance)))
            total, _ = drive.get(key, (0j, lab_rate))
            drive[key] = (total + amplitude, lab_rate)
    by_rate = {}
    for (r, c, rate_key), (amplitude, lab_rate) in drive.items():
        if abs(amplitude) > SMALLEST_ENERGY_SCALE:
            by_rate.setdefault(rate_key, []).append(Coupling(row_state=states[r], column_state=states[c],
                                                             strength=complex(amplitude), oscillation_rate=float(lab_rate)))
    for couplings in by_rate.values():
        operators.append(CouplingOperator(reduced_basis, couplings))

    hamiltonian = Hamiltonian(reduced_basis, operators, list(frame))
    return RamanEffectiveModel(hamiltonian=hamiltonian, reduced_basis=reduced_basis, ground_levels=ground,
                               second_order_shifts=terms['second_order_shifts'], fourth_order_hamiltonian=H4,
                               resonant_terms=terms['resonant_terms'], max_perturbation_ratio=terms['max_perturbation_ratio'],
                               n_terms=terms['n_terms'], harmonic_couplings=terms['harmonic_couplings'])
