#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

from __future__ import annotations

import importlib.resources
import warnings
from abc import ABC
from dataclasses import dataclass, fields, replace
from fractions import Fraction
from functools import cached_property
from typing import Sequence

import numpy as np
import yaml

from ionsim.energy_level import EnergyLevel
from ionsim.atomic_internal_energy_level import (AtomicInternalEnergyLevel, SinkLevel, EigenBasis, LSFineLevel, LSHyperfineLevel, LSBackGoudsmitLevel,
                                                J1L2FineLevel, J1L2HyperfineLevel, J1L2BackGoudsmitLevel)
from ionsim.collective_motional_energy_level import CollectiveMotionalEnergyLevel
from ionsim.zeeman_solver import ZeemanHyperfineSolver
from ionsim.ionsim_error import IonSimError
from ionsim.config import NUMERICAL_EQUIVALENCE_THRESHOLD, STRUCTURAL_KEYS, PROJECTION_KEYS, ALLOWED_QUANTUM_NUMBER_KEYS

UNCOUPLED_OVERLAP_WARNING_THRESHOLD = 0.9


@dataclass(frozen=True, eq=False)
class DegreeOfFreedom(ABC):
    """A degree of freedom in a basis of states."""
    energy_levels: Sequence[EnergyLevel]
    name: str | None = None # TODO: will we use these names?

def _to_float(value) -> float:
    """Convert a user-supplied quantum number (int, float, Fraction, or string like '3/2') to float."""
    if isinstance(value, str):
        return float(Fraction(value))
    return float(value)

def _is_equal(a: float, b: float) -> bool:
    """True if a and b agree to within NUMERICAL_EQUIVALENCE_THRESHOLD (absolute)."""
    return abs(a - b) < NUMERICAL_EQUIVALENCE_THRESHOLD

def _is_half_integer_multiple(x: float) -> bool:
    """True if x is an integer or half-integer."""
    return _is_equal(2 * x, np.round(2 * x))


def _check_projection(m: float, total: float, m_name: str, total_name: str, qn: dict):
    """Check that a projection quantum number m is allowed for angular momentum `total`."""
    if not _is_half_integer_multiple(m):
        raise IonSimError(f"{m_name}={m} must be an integer or half-integer, got quantum numbers {qn}.")
    if abs(m) > total + NUMERICAL_EQUIVALENCE_THRESHOLD:
        raise IonSimError(f"|{m_name}| must be <= {total_name}={total}, got {m_name}={m} in {qn}.")
    if not _is_equal(total - m, np.round(total - m)):
        raise IonSimError(f"{total_name} - {m_name} must be an integer, got {total_name}={total}, {m_name}={m} in {qn}.")


@dataclass(frozen=True, eq=False)
class MotionalMode(DegreeOfFreedom):
    """An normal mode of motion for a linear chain of ions."""
    energy_levels: list[CollectiveMotionalEnergyLevel]

    @classmethod
    def from_frequency(cls, frequency: float, fock_dimension: int, name: str | None = None, level_aliases: list[str] | None=None):
        """Build a motional normal-mode degree of freedom for an ion chain."""
        if level_aliases:
            levels = [CollectiveMotionalEnergyLevel(frequency, fock_number, alias = level_aliases[fock_number]) for fock_number in range(fock_dimension)]
        else:
            levels = [CollectiveMotionalEnergyLevel(frequency, fock_number) for fock_number in range(fock_dimension)]
        return cls(levels, name)

@dataclass(frozen=True, eq=False)
class AtomicStructure(DegreeOfFreedom):
    """An atomic structure object, containing atomic internal energy levels corresponding to angular momentum eigenstates."""
    energy_levels: list[AtomicInternalEnergyLevel | SinkLevel]

    @classmethod
    def from_species(cls, species: str, manifolds: list[str] | None = None, level_names: list[str] | None = None,
            quantum_numbers: list[dict] | None = None, level_aliases: list[str] | None = None, name: str | None = None,
            magnetic_field: float = 0., term_symbols: list[str] | None = None, include_sink: bool = False, **kwargs):
        """Build the atomic structure degree of freedom for a particular species of atom.

        Args:
            species: Name of the species config file, e.g. '171Yb+'.
            manifolds: Term symbols of the config-file manifolds to include. ``term_symbols`` is accepted as an alias.
            level_names: Names of individual levels to keep, e.g. 'S1/2,1,0'. Mutually exclusive with ``quantum_numbers``.
            quantum_numbers: One dict per level. Structural keys (n, l, s, j for LS coupling; n, k, j, ... for j1l2)
                select the manifold; they only need to be specific enough to identify one manifold. Projection keys
                select the basis and the state:
                    {'f', 'mf'}  -> hyperfine level |F, mF>   (LSHyperfineLevel / J1L2HyperfineLevel)
                    {'mj', 'mi'} -> uncoupled level |mJ, mI>  (LSBackGoudsmitLevel / J1L2BackGoudsmitLevel), needs B != 0
                    {'mj'}       -> fine level |J, mJ>        (nuclear spin zero only)
                Levels are returned in the order given.
            level_aliases: Optional alias per requested level (same length as level_names / quantum_numbers).
            name: Name of the degree of freedom.
            magnetic_field: Static magnetic field used for Zeeman shifts, in the Zeeman solver's field units (gauss by default).
            include_sink: Append a SinkLevel (named 'sink') after the other levels, to collect population that decays to
                states outside the simulated levels (see DissipatorSpontaneousEmission, decay_to_sink=True).
            **kwargs: Options passed to ZeemanHyperfineSolver, e.g. ``approximation``.

            Without ``level_names`` or ``quantum_numbers``, every sublevel of the selected manifolds is included, in the
            |F, mF> basis (or |J, mJ> when the nuclear spin is zero).
        """
        # Import configuration data and levels for the species 
        config_data = cls.get_config_data(species)
        levels_data = config_data['levels']

        if level_names and quantum_numbers:
            raise IonSimError("Specify either level names or quantum numbers, not both.")
        if manifolds is not None and term_symbols is not None:
            raise IonSimError("Specify either manifolds or term_symbols (they are aliases), not both.")
        if manifolds is None:
            manifolds = term_symbols

        requested = level_names if level_names is not None else quantum_numbers
        if level_aliases:
            if requested is None:
                raise IonSimError("level_aliases requires level_names or quantum_numbers, so each alias maps to a specific level.")
            if len(level_aliases) != len(requested):
                raise IonSimError(f"Specify one level alias per requested level: expected {len(requested)}, got {len(level_aliases)}.")

        if manifolds is not None:
            available = [data['term_symbol'] for data in levels_data]
            missing = [ts for ts in manifolds if ts not in available]
            if missing:
                raise IonSimError(f"Term symbols {missing} not found in the {species} config data. Available: {available}.")
            levels_data = [data for data in levels_data if data['term_symbol'] in manifolds]

        # Manifolds of electronic levels may differ in character and thus quantum numbers; therefore we use a manifold builder 
        #  that can accommodate differences in character for each term symbols' levels. This depends on the angular momentum couplings and any magnetic fields. 
        builders = [_ManifoldBuilder(cls.get_fine_data(level_data), level_data['coupling_scheme'], config_data,
                                     magnetic_field, cls.get_level_factory, kwargs) for level_data in levels_data]

        # Build structure from a list of quantum numbers from each level or from specified level names 
        if quantum_numbers:
            levels = cls._levels_from_quantum_numbers(quantum_numbers, builders, config_data['nuclear_spin'], level_aliases)
        else:
            # Extract levels that are requested by the user OR include all levels if only manifold/term symbol is specified.  
            levels = [level for builder in builders for level in builder.all_levels()
                      if level_names is None or level.name in level_names]

        # Check for duplicate or levels not found in requested manifolds  
        if level_names:
            # Check for duplicate level names:
            duplicates = (len(level_names) != len(set(level_names))) 
            if duplicates:
                raise IonSimError(f"Level names should be unique but contains duplicates. Found {len(set(level_names))} unique level names.") 

            # Check for level names that are not found in the requested manifolds:
            missing = set(level_names) - {level.name for level in levels}
            if missing:
                raise IonSimError(f"Level names {sorted(missing)} were not found in the selected manifolds.")
            if level_aliases:
                levels = [replace(level, alias=level_aliases[level_names.index(level.name)]) for level in levels]

        return cls(levels + [SinkLevel()] if include_sink else levels, name)

    @classmethod
    def _levels_from_quantum_numbers(cls, quantum_numbers: list[dict], builders: list[_ManifoldBuilder],
                                     nuclear_spin: float, level_aliases: list[str] | None):
        """Build one level per quantum-number dictionary, preserving the user's order."""
        levels = []
        seen_names = set()
        for index, raw_qn in enumerate(quantum_numbers):
            qn = cls._parse_quantum_numbers(raw_qn, nuclear_spin)
            builder = cls._match_manifold(qn, builders)
            eigenbasis = cls._identify_basis(qn, nuclear_spin)

            if eigenbasis is EigenBasis.FINE:
                _check_projection(qn['mj'], builder.j, 'mj', 'j', raw_qn)
                level = builder.fine_level(qn['mj'])
            elif eigenbasis is EigenBasis.HYPERFINE:
                if not any(_is_equal(qn['f'], f) for f in builder.f_values):
                    raise IonSimError(f"f={qn['f']} is not allowed for manifold {builder.describe()} with I={nuclear_spin}; "
                                      f"allowed values are {[float(f) for f in builder.f_values]}. Got {raw_qn}.")
                _check_projection(qn['mf'], qn['f'], 'mf', 'f', raw_qn)
                level = builder.hyperfine_level(qn['f'], qn['mf'])
            elif eigenbasis is EigenBasis.UNCOUPLED:
                _check_projection(qn['mj'], builder.j, 'mj', 'j', raw_qn)
                _check_projection(qn['mi'], nuclear_spin, 'mi', 'i', raw_qn)
                level = builder.uncoupled_level(qn['mj'], qn['mi'])
            else:
                raise IonSimError(f"Unsupported level eigenbasis {eigenbasis}.")

            if level.name in seen_names:
                raise IonSimError(f"Quantum numbers {raw_qn} specify the level '{level.name}', which was already requested.")
            seen_names.add(level.name)

            if level_aliases:
                level = replace(level, alias=level_aliases[index])
            levels.append(level)
        return levels

    @staticmethod
    def _parse_quantum_numbers(raw_qn: dict, nuclear_spin: float) -> dict:
        """Parse a user-supplied quantum-number dict: check its keys against the allowed set, convert values to floats, and check any given nuclear spin against the species."""
        if not isinstance(raw_qn, dict):
            raise IonSimError(f"Each entry of quantum_numbers must be a dict, got {type(raw_qn).__name__}: {raw_qn}.")
        unknown = set(raw_qn) - ALLOWED_QUANTUM_NUMBER_KEYS
        if unknown:
            raise IonSimError(f"Unknown quantum-number keys {sorted(unknown)} in {raw_qn}. "
                              f"Allowed keys: {sorted(ALLOWED_QUANTUM_NUMBER_KEYS)}.")
        qn = {key: _to_float(value) for key, value in raw_qn.items()}
        if 'i' in qn and not _is_equal(qn['i'], nuclear_spin):
            raise IonSimError(f"Nuclear spin i={qn['i']} in {raw_qn} does not match the species nuclear spin {nuclear_spin}.")
        return qn

    @staticmethod
    def _identify_basis(qn: dict, nuclear_spin: float) -> EigenBasis:
        """Decide which basis a quantum-number dict refers to from its projection quantum numbers."""
        projections = frozenset(k for k in qn if k in PROJECTION_KEYS)
        if nuclear_spin == 0:
            if projections == {'mj'}:
                return EigenBasis.FINE
            raise IonSimError(f"This species has zero nuclear spin, so levels are specified by 'mj' only. Got {qn}.")
        if projections == {'f', 'mf'}:
            return EigenBasis.HYPERFINE
        if projections == {'mj', 'mi'}:
            return EigenBasis.UNCOUPLED
        if projections == {'ml', 'ms', 'mi'}:
            raise IonSimError(f"Fully decoupled |mL, mS, mI> (Paschen-Back) levels are not yet supported. Got {qn}.")
        raise IonSimError(f"Could not identify the basis from quantum numbers {qn}. Specify either "
                          f"('f', 'mf') for a hyperfine level or ('mj', 'mi') for an uncoupled |mJ, mI> level.")

    @staticmethod
    def _match_manifold(qn: dict, builders: list[_ManifoldBuilder]) -> _ManifoldBuilder:
        """Find the unique manifold consistent with the structural quantum numbers in qn."""
        structural = {k: v for k, v in qn.items() if k in STRUCTURAL_KEYS}
        matches = [b for b in builders
                   if all(b.fine_data.get(k) is not None and _is_equal(float(b.fine_data[k]), v)
                          for k, v in structural.items())]
        if len(matches) == 1:
            return matches[0]
        available = '; '.join(b.describe() for b in builders) or 'none'
        if not matches:
            raise IonSimError(f"No manifold matches quantum numbers {qn}. Available manifolds: {available}.")
        candidates = ', '.join(b.describe() for b in matches)
        raise IonSimError(f"Quantum numbers {qn} match more than one manifold: {candidates}. "
                          f"Add structural quantum numbers (e.g. 'n', 'l', 'j') or restrict `manifolds`.")

    @classmethod
    def get_fine_data(cls, level_data: dict) -> dict:
        """Fine-structure data for one manifold from its config entry, with energies converted from Hz to rad/s."""
        term_symbol = level_data['term_symbol']
        fine_data = {key: value for key, value in level_data.items() if key != 'coupling_scheme'}
        fine_data['fine_energy'] = 2 * np.pi * level_data['fine_energy']
        fine_data['hyperfine_A'] = 2 * np.pi * level_data['hyperfine_A']
        hyperfine_B = level_data.get('hyperfine_B')
        fine_data['hyperfine_B'] = None if hyperfine_B is None else 2 * np.pi * hyperfine_B
        fine_data['branching_ratios'] = level_data.get('branching_ratios')
        fine_data['j'] = cls.compute_j(term_symbol)
        if level_data['coupling_scheme'] == 'j1l2':
            fine_data['k'] = cls.compute_k(term_symbol)
            if fine_data.get('gj') is None:
                fine_data['gj'] = cls.compute_j1l2_gj(fine_data['j1'], fine_data['l2'], fine_data['s2'], fine_data['k'], fine_data['j'])
        else:
            fine_data['l'] = cls.compute_l(term_symbol)
        return fine_data

    @staticmethod
    def compute_j1l2_gj(j1: float, l2: float, s2: float, k: float, j: float) -> float:
        """Lande g-factor of a j1l2-coupled level (K = J1 + L2, J = K + S2).

        See p. 100 of B. G. Wybourne, Spectroscopic Properties of Rare Earths (Interscience, New York, 1965),
        and pp. 6-7 of https://nvlpubs.nist.gov/nistpubs/Legacy/NSRDS/nbsnsrds60.pdf
        """
        gj1 = 1. + (j1*(j1+1) + s2*(s2+1) - l2*(l2+1))/(2. * j1*(j1+1))  # from LS formula
        gj = 2. * (gj1 - 1.) * (k*(k+1) + j1*(j1+1) - l2*(l2 + 1))/((2*j + 1)*(2*k + 1))
        return gj + (3*j*(j+1) - k*(k+1) + s2*(s2+1))/(2.*j*(j+1))

    @classmethod
    def get_level_factory(cls, coupling_scheme: str):
        """Get a factory to build energy levels with a particular coupling scheme."""
        factories = {
            'ls': (cls.get_fine_data, LSFineLevel, LSHyperfineLevel, LSBackGoudsmitLevel),
            'j1l2': (cls.get_fine_data, J1L2FineLevel, J1L2HyperfineLevel, J1L2BackGoudsmitLevel),
            # 'ls1': (_get_ls1_fine_data, LS1FineLevel, LS1HyperfineLevel),
            # 'j1j2': (_get_j1j2_fine_data, J1J2FineLevel, J1J2HyperfineLevel),
        }
        return factories[coupling_scheme]

    @staticmethod
    def compute_l(term_symbol: str):
        """Compute the total electronic orbital angular momentum "l" from a term symbol."""
        orbitals = {'S': 0, 'P': 1, 'D': 2, 'F': 3}
        match = [k for k in orbitals if k in term_symbol]
        if not len(match) == 1:
            raise IonSimError(f"Computing L from the term symbol requires exactly one corresponding letter: {list(orbitals.keys())}. Found {match} in {term_symbol}.")
        return orbitals[match[0]]

    @staticmethod
    def compute_k(term_symbol: str):
        """Compute the intermediate electronic angluar momentum "k" from a term symbol."""
        if term_symbol[2] == '/':
            return float(Fraction(term_symbol[1:4]))
        return float(term_symbol[1])

    @staticmethod
    def compute_j(term_symbol: str): # term_symbol = S1/2, D3/2, [3/2]1/2, S0, D2, etc.
        """Compute the total electronic angluar momentum "j" from a term symbol."""
        if term_symbol[-2] == '/':
            return float(Fraction(term_symbol[len(term_symbol)-3:len(term_symbol)]))
        return float(term_symbol[-1])

    @staticmethod
    def get_config_data(species: str):
        """Load the configuration data for the internal energy levels of a particular species of atom."""
        with importlib.resources.files('ionsim.atomic_config_data').joinpath(f'{species}.yaml').open('r') as file:
            config_data = yaml.safe_load(file)
        return config_data

def levels_in_manifold(structure: AtomicStructure, term_symbol: str):
    return [level for level in structure.energy_levels if level.term_symbol == term_symbol]


def _make_level(level_class: type, fine_data: dict, **quantum_numbers) -> AtomicInternalEnergyLevel:
    """ Construct a level from the manifold's fine data, passing only the fields the level class declares.

        Config files can therefore carry data that only some level classes use (e.g. gj, or future polarizabilities).
    """
    field_names = {f.name for f in fields(level_class)}
    return level_class(**{k: v for k, v in fine_data.items() if k in field_names}, **quantum_numbers)


class _ManifoldBuilder:
    """ Builds the energy levels of one manifold (one entry of the species config file) at a given magnetic field.

        Every level's external energy shift comes from `_energy_shift`, the single place to extend when other static-field
        terms (e.g. a DC light shift) are added. The Zeeman solution is computed on first use and cached, so manifolds
        that contribute no levels are never diagonalized.
    """

    def __init__(self, fine_data: dict, coupling_scheme: str, species_data: dict, magnetic_field: float, get_level_factory, solver_kwargs: dict):
        self.fine_data = fine_data
        self.coupling_scheme = coupling_scheme
        #self.FineLevel, self.HyperfineLevel, self.UncoupledLevel = LEVEL_CLASSES[coupling_scheme]
        _, self.FineLevel, self.HyperfineLevel, self.UncoupledLevel = get_level_factory(self.coupling_scheme)
        self.nuclear_spin = species_data['nuclear_spin']
        self.species_data = species_data
        self.magnetic_field = magnetic_field
        self.solver_kwargs = dict(solver_kwargs)
        if self.nuclear_spin == 0:
            # ZeemanHyperfineSolver.lande_gi computes nuclear_moment / i, which fails for i = 0.
            # With no nuclear spin the nuclear Zeeman term vanishes anyway.
            self.solver_kwargs.setdefault('gi', 0.)

    @property
    def j(self) -> float:
        return self.fine_data['j']

    @property
    def term_symbol(self) -> str:
        return self.fine_data['term_symbol']

    @property
    def f_values(self) -> np.ndarray:
        """Allowed total angular momenta F = |J - I|, ..., J + I."""
        return np.arange(abs(self.j - self.nuclear_spin), self.j + self.nuclear_spin + 1)

    def describe(self) -> str:
        """Human-readable summary of the structural quantum numbers of this manifold (for error messages)."""
        items = ', '.join(f"{k}={self.fine_data[k]}" for k in sorted(STRUCTURAL_KEYS) if k in self.fine_data)
        return f"'{self.term_symbol}' ({items})"

    @cached_property
    def zeeman(self):
        """(solver, energy_shifts, eigenvecs) for this manifold at the magnetic field, or None at zero field."""
        if self.magnetic_field == 0.:
            return None
        data = self.fine_data
        hyperfine_A = data['hyperfine_A'] / (2. * np.pi)  # solver works in Hz
        hyperfine_B = None if data['hyperfine_B'] is None else data['hyperfine_B'] / (2. * np.pi)
        if self.coupling_scheme == 'j1l2':
            l, s, options = None, data['s2'], {'gj': data['gj']}
        else:
            l, s, options = data['l'], data['s'], {}
        solver = ZeemanHyperfineSolver(self.nuclear_spin, self.j, l, s, hyperfine_A, hyperfine_B,
                                       self.species_data['mass'], self.species_data['magnetic_moment'],
                                       self.species_data['Z'], **{**options, **self.solver_kwargs})
        energy_shifts, eigenvecs = solver.solve_at_field(self.magnetic_field)
        return solver, energy_shifts, eigenvecs

    def _energy_shift(self, **label) -> float:
        """External energy shift (rad/s) of the level labeled {'f', 'mf'} or {'mj', 'mi'}.

            For |F, mF> labels the hyperfine A shift is removed, since AtomicInternalEnergyLevel already includes it.
            For |mJ, mI> labels the full solver energy (Zeeman + hyperfine) is returned, since the level's
            hyperfine_energy_shift is 0.
        """
        if self.zeeman is None:
            return 0.
        solver, energy_shifts, eigenvecs = self.zeeman
        if 'mi' in label:
            shift_hz = solver.get_state_energy_from_mjmi_pair(energy_shifts, eigenvecs, **label)
        else:
            shift_hz = solver.get_state_energy(energy_shifts, eigenvecs, **label, subtract_hyperfineA_shift=True)
        return 2. * np.pi * shift_hz

    def all_levels(self) -> list[AtomicInternalEnergyLevel]:
        """Every sublevel of the manifold in the low-field basis: |J, mJ> if the nuclear spin is zero, else |F, mF>."""
        if self.nuclear_spin == 0:
            return [self.fine_level(mj) for mj in np.arange(-self.j, self.j + 1)]
        return [self.hyperfine_level(f, mf) for f in self.f_values for mf in np.arange(-f, f + 1)]

    def fine_level(self, mj: float):
        """|J, mJ> level (nuclear spin zero)."""
        # With I = 0, F = J and mF = mJ, so the solver's |F, mF> lookup gives the |J, mJ> energy.
        shift = self._energy_shift(f=self.j, mf=mj)
        return _make_level(self.FineLevel, self.fine_data, mj=mj, external_energy_shift=shift)

    def hyperfine_level(self, f: float, mf: float):
        """|F, mF> level (low-field basis)."""
        shift = self._energy_shift(f=f, mf=mf)
        return _make_level(self.HyperfineLevel, self.fine_data, i=self.nuclear_spin, f=f, mf=mf, external_energy_shift=shift)

    def uncoupled_level(self, mj: float, mi: float):
        """|mJ, mI> level (high-field / Back-Goudsmit basis)."""
        if self.zeeman is None:
            raise IonSimError(f"|mJ, mI> levels were requested for manifold '{self.term_symbol}' at zero magnetic field. "
                              f"These are not energy eigenstates at zero field; specify (f, mf) instead or set a nonzero magnetic_field.")
        solver, _, eigenvecs = self.zeeman
        if solver.approximation is not None:
            raise IonSimError(f"|mJ, mI> levels require the exact Zeeman solver, but approximation='{solver.approximation}' "
                              f"was requested. The weak-field approximation works in the |F, mF> basis; specify (f, mf) instead.")

        # Overlap of the requested |mJ, mI> with the eigenstate the solver assigns to it. If two labels were assigned
        # the same eigenstate, normalization forces at least one overlap <= 0.5, so this warning also catches that case.
        # Minimum |<mJ, mI|psi>|^2 between a requested |mJ, mI> label and its assigned energy eigenstate before a warning is issued.
        # Below this, mJ and mI are not good quantum numbers at the chosen field and the label is only nominal.
        overlaps = np.abs(eigenvecs[solver.basis_states.index((mj, mi)), :])**2
        max_overlap = float(np.max(overlaps))
        if max_overlap < UNCOUPLED_OVERLAP_WARNING_THRESHOLD:
            warnings.warn(f"|mJ={Fraction(mj)}, mI={Fraction(mi)}> in manifold '{self.term_symbol}' at B = {self.magnetic_field} "
                          f"has only {max_overlap:.1%} overlap with its assigned energy eigenstate; the label is nominal. "
                          f"Consider specifying (f, mf) at this field.", stacklevel=4)

        shift = self._energy_shift(mj=mj, mi=mi)
        return _make_level(self.UncoupledLevel, self.fine_data, i=self.nuclear_spin, mj=mj, mi=mi, external_energy_shift=shift)
