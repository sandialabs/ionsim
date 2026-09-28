#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

from ionsim.energy_level import EnergyLevel
from ionsim.atomic_internal_energy_level import AtomicInternalEnergyLevel
from ionsim.atomic_internal_energy_level import LSFineLevel, LSHyperfineLevel, J1L2FineLevel, J1L2HyperfineLevel, LSBackGoudsmitLevel, J1L2UncoupledLevel
from ionsim.collective_motional_energy_level import CollectiveMotionalEnergyLevel
from ionsim.zeeman_solver import ZeemanHyperfineSolver
from ionsim.ionsim_error import IonSimError

import importlib.resources
from pathlib import Path
from dataclasses import dataclass, replace
from abc import ABC
import yaml
from fractions import Fraction
import numpy as np
import warnings
from typing import Sequence

from icecream import ic

@dataclass(frozen=True, eq=False)
class DegreeOfFreedom(ABC):
    """A degree of freedom in a basis of states."""
    energy_levels: Sequence[EnergyLevel]
    name: str | None = None # TODO: will we use these names?

# Keys accepted in each quantum-number dictionary passed to AtomicStructure.from_species(quantum_numbers=...).
# Structural keys identify the manifold (config-file level); projection keys identify the state within it.
_STRUCTURAL_KEYS = {'n', 'l', 's', 'j', 'k', 'j1', 'l2', 's2'}
_PROJECTION_KEYS = {'f', 'mf', 'mj', 'mi', 'ml', 'ms'}
_ALLOWED_QUANTUM_NUMBER_KEYS = _STRUCTURAL_KEYS | _PROJECTION_KEYS | {'i'}

# Minimum |<mJ, mI|psi>|^2 for the eigenstate assigned to a requested |mJ, mI> label before a warning is issued.
# Below this, mJ and mI are not good quantum numbers at the chosen field and the label is only nominal.
UNCOUPLED_PURITY_WARNING_THRESHOLD = 0.9


def _to_float(value) -> float:
    """Convert a user-supplied quantum number (int, float, Fraction, or string like '3/2') to float."""
    if isinstance(value, str):
        return float(Fraction(value))
    return float(value)


def _is_half_integer_multiple(x: float) -> bool:
    """True if x is an integer or half-integer."""
    return np.isclose(2 * x, np.round(2 * x))


def _check_projection(m: float, total: float, m_name: str, total_name: str, qn: dict):
    """Check that a projection quantum number m is allowed for angular momentum `total`."""
    if not _is_half_integer_multiple(m):
        raise IonSimError(f"{m_name}={m} must be an integer or half-integer, got quantum numbers {qn}.")
    if abs(m) > total + 1e-9:
        raise IonSimError(f"|{m_name}| must be <= {total_name}={total}, got {m_name}={m} in {qn}.")
    if not np.isclose(total - m, np.round(total - m)):
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
    energy_levels: list[AtomicInternalEnergyLevel]

    @classmethod
    def from_species(cls, species: str, manifolds: list[str] | None = None, level_names: list[str] | None = None,
            quantum_numbers: list[dict] | None = None, level_aliases: list[str] | None = None, name: str | None = None,
            magnetic_field: float = 0., term_symbols: list[str] | None = None, **kwargs):
        """Build the atomic structure degree of freedom for a particular species of atom.

        Args:
            species: Name of the species config file, e.g. '171Yb+'.
            manifolds: Term symbols of the config-file manifolds to include. ``term_symbols`` is accepted as an alias.
            level_names: Names of individual levels to keep, e.g. 'S1/2,1,0'. Mutually exclusive with ``quantum_numbers``.
            quantum_numbers: One dict per level. Structural keys (n, l, s, j for LS coupling; n, k, j, ... for j1l2)
                select the manifold; they only need to be specific enough to identify one manifold. Projection keys
                select the basis and the state:
                    {'f', 'mf'}  -> hyperfine level |F, mF>   (LSHyperfineLevel / J1L2HyperfineLevel)
                    {'mj', 'mi'} -> uncoupled level |mJ, mI>  (LSBackGoudsmitLevel / J1L2UncoupledLevel), needs B != 0
                    {'mj'}       -> fine level |J, mJ>        (nuclear spin zero only)
                Levels are returned in the order given.
            level_aliases: Optional alias per requested level (same length as level_names / quantum_numbers).
            name: Name of the degree of freedom.
            magnetic_field: Static magnetic field used for Zeeman shifts.
            **kwargs: Passed to ZeemanHyperfineSolver.
        """
        config_data = cls.get_config_data(species)
        nuclear_spin = config_data['nuclear_spin']
        levels_data = config_data['levels']
        mass = config_data['mass'] # Daltons
        z = config_data['Z'] # Atomic number, number of protons
        magnetic_moment = config_data['magnetic_moment'] # units of \mu_{N}
        structure = 'fine' if nuclear_spin == 0 else 'hyperfine'

        if level_names and quantum_numbers:
            raise IonSimError(f"Specify either level names or quantum numbers, not both.")

        if manifolds is not None and term_symbols is not None:
            raise IonSimError("Specify either manifolds or term_symbols (they are aliases), not both.")
        if manifolds is None:
            manifolds = term_symbols

        if manifolds is not None:
            available = [data['term_symbol'] for data in levels_data]
            missing = [ts for ts in manifolds if ts not in available]
            if missing:
                raise IonSimError(f"Term symbols {missing} not found in the {species} config data. Available: {available}.")
            levels_data = cls.select_some_data(manifolds, levels_data)

        if level_aliases and level_names is None:
            raise IonSimError("level_aliases requires level_names or quantum_numbers, so each alias maps to a specific level.")

        if level_aliases:
            if level_names:
                if len(level_names) != len(level_aliases):
                    raise IonSimError(f'User should specify a level alias for each level in the atomic structure. Expected {len(level_names)} but have {level_aliases}')
            if quantum_numbers:
                if len(quantum_numbers) != len(level_aliases):
                    raise IonSimError(f'User should specify a level alias for each level in the atomic structure. Expected {len(quantum_numbers)} but have {level_aliases}')

        builders = []
        for level_data in levels_data:
            level_data['unique_term_symbol'] = level_data['term_symbol']
            level_data['unique_branching_ratios'] = level_data.get('branching_ratios', None)
            builders.append(_ManifoldBuilder(level_data, nuclear_spin, mass, magnetic_moment, z, magnetic_field,
                                             cls.get_level_factory, kwargs))

        # Atomic levels are built either by quantum numbers or by specified level names 
        if quantum_numbers is not None:
            levels = cls._levels_from_quantum_numbers(quantum_numbers, builders, nuclear_spin, level_aliases)
            return cls(levels, name)


        levels = []
        keep_all = level_names is None  # no filter: every level in the selected manifolds
        for builder in builders:
            j = builder.j
            # Construct levels based on coupling structure
            if structure == 'fine':
                candidates = (builder.fine_level(mj) for mj in np.arange(-j, j + 1))
            else:
                candidates = (builder.hyperfine_level(f, mf)
                              for f in np.arange(np.abs(j - nuclear_spin), j + nuclear_spin + 1)
                              for mf in np.arange(-f, f + 1))
            for level in candidates:
                if keep_all or level.name in level_names:
                    if level_aliases:
                        # Overwrite the level to include its alias
                        level = replace(level, alias=level_aliases[level_names.index(level.name)])
                    levels.append(level)
        if not keep_all:
            missing = set(level_names) - {level.name for level in levels}
            if missing:
                raise IonSimError(f"Level names {sorted(missing)} were not found in the selected manifolds.")

        return cls(levels, name)

    @classmethod
    def _levels_from_quantum_numbers(cls, quantum_numbers: list[dict], builders: list[_ManifoldBuilder],
                                     nuclear_spin: float, level_aliases: list[str] | None):
        """Build one level per quantum-number dictionary, preserving the user's order."""
        levels = []
        seen_names = set()
        for index, raw_qn in enumerate(quantum_numbers):
            qn = cls._parse_quantum_numbers(raw_qn, nuclear_spin)
            builder = cls._match_manifold(qn, builders)
            basis = cls._identify_basis(qn, nuclear_spin)
            j = builder.j

            if basis == 'fine':
                _check_projection(qn['mj'], j, 'mj', 'j', raw_qn)
                level = builder.fine_level(qn['mj'])
            elif basis == 'hyperfine':
                f, mf = qn['f'], qn['mf']
                f_min, f_max = abs(j - nuclear_spin), j + nuclear_spin
                if f < f_min - 1e-9 or f > f_max + 1e-9 or not np.isclose(f - f_min, np.round(f - f_min)):
                    allowed = [float(x) for x in np.arange(f_min, f_max + 1)]
                    raise IonSimError(f"f={f} is not allowed for manifold {builder.describe()} with I={nuclear_spin}; "
                                      f"allowed values are {allowed}. Got {raw_qn}.")
                _check_projection(mf, f, 'mf', 'f', raw_qn)
                level = builder.hyperfine_level(f, mf)
            elif basis == 'uncoupled':
                _check_projection(qn['mj'], j, 'mj', 'j', raw_qn)
                _check_projection(qn['mi'], nuclear_spin, 'mi', 'i', raw_qn)
                level = builder.uncoupled_level(qn['mj'], qn['mi'])
            else:  # pragma: no cover - _identify_basis only returns the cases above
                raise IonSimError(f"Unsupported basis '{basis}'.")

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
        unknown = set(raw_qn) - _ALLOWED_QUANTUM_NUMBER_KEYS
        if unknown:
            raise IonSimError(f"Unknown quantum-number keys {sorted(unknown)} in {raw_qn}. "
                              f"Allowed keys: {sorted(_ALLOWED_QUANTUM_NUMBER_KEYS)}.")
        qn = {key: _to_float(value) for key, value in raw_qn.items()}
        if 'i' in qn and not np.isclose(qn['i'], nuclear_spin):
            raise IonSimError(f"Nuclear spin i={qn['i']} in {raw_qn} does not match the species nuclear spin {nuclear_spin}.")
        return qn

    @staticmethod
    def _identify_basis(qn: dict, nuclear_spin: float) -> str:
        """Decide which basis a quantum-number dict refers to from its projection quantum numbers."""
        projections = frozenset(k for k in qn if k in _PROJECTION_KEYS)
        if nuclear_spin == 0:
            if projections == {'mj'}:
                return 'fine'
            raise IonSimError(f"This species has zero nuclear spin, so levels are specified by 'mj' only. Got {qn}.")
        if projections == {'f', 'mf'}:
            return 'hyperfine'
        if projections == {'mj', 'mi'}:
            return 'uncoupled'
        if projections == {'ml', 'ms', 'mi'}:
            raise IonSimError(f"Fully decoupled |mL, mS, mI> (Paschen-Back) levels are not yet supported. Got {qn}.")
        raise IonSimError(f"Could not identify the basis from quantum numbers {qn}. Specify either "
                          f"('f', 'mf') for a hyperfine level or ('mj', 'mi') for an uncoupled |mJ, mI> level.")

    @staticmethod
    def _match_manifold(qn: dict, builders: list[_ManifoldBuilder]) -> _ManifoldBuilder:
        """Find the unique manifold consistent with the structural quantum numbers in qn."""
        structural = {k: v for k, v in qn.items() if k in _STRUCTURAL_KEYS}
        matches = [b for b in builders
                   if all(k in b.fine_data and b.fine_data[k] is not None and np.isclose(float(b.fine_data[k]), v)
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
    def get_level_factory(cls, coupling_scheme: str):
        """Get a factory to build energy levels with a particular coupling scheme."""
        factories = {
            'ls': (cls.get_ls_fine_data, LSFineLevel, LSHyperfineLevel, LSBackGoudsmitLevel),
            'j1l2': (cls.get_j1l2_fine_data, J1L2FineLevel, J1L2HyperfineLevel, J1L2UncoupledLevel),
            # 'ls1': (_get_ls1_fine_data, LS1FineLevel, LS1HyperfineLevel),
            # 'j1j2': (_get_j1j2_fine_data, J1J2FineLevel, J1J2HyperfineLevel),
        }
        return factories[coupling_scheme]

    @classmethod
    def get_ls_fine_data(cls, level_data: dict):
        """Get fine-structure data from energy-level configuration data."""
        fine_data = dict(level_data)
        fine_data['fine_energy'] = 2 * np.pi * fine_data['fine_energy'] # convert from Hz to rad./s
        fine_data['hyperfine_A'] = 2 * np.pi * fine_data['hyperfine_A'] # convert from Hz to rad./s
        try: 
            hyperfine_B = fine_data['hyperfine_B'] * 2. * np.pi
        except:
            hyperfine_B = None
        fine_data['hyperfine_B'] = hyperfine_B 
        fine_data['l'] = cls.compute_l(level_data['term_symbol'])
        fine_data['j'] = cls.compute_j(level_data['term_symbol'])
        fine_data['term_symbol'] = level_data['unique_term_symbol']
        fine_data['branching_ratios'] = level_data['unique_branching_ratios']
        [fine_data.pop(key) for key in ['coupling_scheme', 'unique_term_symbol', 'unique_branching_ratios']]
        return fine_data

    @classmethod
    def get_j1l2_fine_data(cls, level_data: dict):
        """Get fine-structure data from energy-level configuration data."""
        fine_data = dict(level_data)
        fine_data['fine_energy'] = 2 * np.pi * fine_data['fine_energy'] # convert from Hz to rad./s
        fine_data['hyperfine_A'] = 2 * np.pi * fine_data['hyperfine_A'] # convert from Hz to rad./s
        try: 
            hyperfine_B = fine_data['hyperfine_B'] * np.pi * 2.
        except:
            hyperfine_B = None
        fine_data['hyperfine_B'] = hyperfine_B 
        fine_data['k'] = cls.compute_k(level_data['term_symbol'])
        fine_data['j'] = cls.compute_j(level_data['term_symbol'])
        fine_data['gj'] = fine_data.get('gj', None)
        fine_data['term_symbol'] = level_data['unique_term_symbol']
        fine_data['branching_ratios'] = level_data['unique_branching_ratios']
        [fine_data.pop(key) for key in ['coupling_scheme', 'unique_term_symbol', 'unique_branching_ratios']]
        return fine_data

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
    def select_some_data(term_symbols: list[str], levels_data: list[dict]):
        """Select a subset of data from the energy-levels configuration data."""
        selected_data = [data for data in levels_data if data['term_symbol'] in term_symbols]
        return selected_data

    @staticmethod
    def get_config_data(species: str):
        """Load the configuration data for the internal energy levels of a particular species of atom."""
        with importlib.resources.files('ionsim.atomic_config_data').joinpath(f'{species}.yaml').open('r') as file:
            config_data = yaml.safe_load(file)
        return config_data

    @staticmethod
    def check_uniqueness_of_term_symbol(term_symbol: str, levels_data: list[dict]):
        """Check whether a term symbol corresponds to a single energy level in the energy-levels configuration data."""
        all_term_symbols = [data['term_symbol'] for data in levels_data]
        assert(term_symbol in all_term_symbols)
        return all_term_symbols.count(term_symbol) == 1

class _ManifoldBuilder:
    """Builds energy levels belonging to one manifold (one entry of the species config file).

    The Zeeman solver is constructed lazily and cached, so it is only diagonalized for manifolds
    that actually contribute a level.
    """

    def __init__(self, level_data: dict, nuclear_spin: float, mass: float, magnetic_moment: float, z: int,
                 magnetic_field: float, get_level_factory, solver_kwargs: dict):
        self.level_data = level_data
        self.coupling_scheme = level_data['coupling_scheme']
        get_fine_data, self.FineLevel, self.HyperfineLevel, self.UncoupledLevel = get_level_factory(self.coupling_scheme)
        self.fine_data = get_fine_data(level_data)
        self.nuclear_spin = nuclear_spin
        self.mass = mass
        self.magnetic_moment = magnetic_moment
        self.z = z
        self.magnetic_field = magnetic_field
        self.solver_kwargs = dict(solver_kwargs)
        if nuclear_spin == 0:
            # ZeemanHyperfineSolver.lande_gi computes nuclear_moment / i, which fails for i = 0.
            # With no nuclear spin the nuclear Zeeman term vanishes anyway.
            self.solver_kwargs.setdefault('gi', 0.)
        self._zeeman = None
        self._assigned_eigenstates = {}  # eigenvector index -> (mj, mi) label, for |mJ, mI> levels  # (solver, energy_shifts, eigenvecs), filled on first use

    @property
    def j(self) -> float:
        return self.fine_data['j']

    @property
    def term_symbol(self) -> str:
        return self.fine_data['term_symbol']

    def describe(self) -> str:
        """Human-readable summary of the structural quantum numbers of this manifold (for error messages)."""
        items = ', '.join(f"{k}={self.fine_data[k]}" for k in sorted(_STRUCTURAL_KEYS) if k in self.fine_data)
        return f"'{self.term_symbol}' ({items})"

    def zeeman(self):
        """Return (solver, energy_shifts, eigenvecs), or None at zero magnetic field."""
        if self.magnetic_field == 0.:
            return None
        if self._zeeman is None:
            fine_data = self.fine_data
            j = self.j
            # Hyperfine A coefficient is converted to rad/s prior to this function
            if fine_data['hyperfine_B'] is None:
                hyperfine_B = None
            else:
                hyperfine_B = fine_data['hyperfine_B'] / (2. * np.pi)

            if self.coupling_scheme == 'j1l2':
                s2 = fine_data['s2']
                if fine_data['gj'] is None:
                    k = fine_data['k']
                    j1 = fine_data['j1']
                    l2 = fine_data['l2']
                    # See p. 100 of B. G. Wybourne, Spectroscopic Properties of Rare Earths (Interscience, New York, 1965).
                    # and p. 6 and 7 of https://nvlpubs.nist.gov/nistpubs/Legacy/NSRDS/nbsnsrds60.pdf
                    gj1 = 1. + (j1*(j1+1) + s2*(s2+1) - l2*(l2+1))/(2. * j1*(j1+1)) # from LS formula
                    gj = 2. * (gj1 - 1.) * (k*(k+1) + j1*(j1+1) - l2*(l2 + 1))/((2*j + 1)*(2*k + 1))
                    gj += (3*j*(j+1) - k*(k+1) + s2*(s2+1))/(2.*j*(j+1))
                    fine_data['gj'] = gj
                solver = ZeemanHyperfineSolver(self.nuclear_spin, j, None, s2, fine_data['hyperfine_A']/(2.*np.pi), hyperfine_B,
                                               self.mass, self.magnetic_moment, self.z, gj=fine_data['gj'], **self.solver_kwargs)
            else:
                solver = ZeemanHyperfineSolver(self.nuclear_spin, j, fine_data['l'], fine_data['s'], fine_data['hyperfine_A']/(2. * np.pi),
                                               hyperfine_B, self.mass, self.magnetic_moment, self.z, **self.solver_kwargs)
            energy_shifts, eigenvecs = solver.solve_at_field(self.magnetic_field)
            self._zeeman = (solver, energy_shifts, eigenvecs)
        return self._zeeman

    def fine_level(self, mj: float):
        """|J, mJ> level (nuclear spin zero)."""
        shift = 0.
        zeeman = self.zeeman()
        if zeeman is not None:
            solver, energy_shifts, eigenvecs = zeeman
            # For fine couplings, F = J since I = 0, so F <==> J and mf <==> mj labels are interchangable.
            shift = solver.get_state_energy(energy_shifts, eigenvecs, f=self.j, mf=mj)
        return self.FineLevel(**self.fine_data, mj=mj, external_energy_shift=shift * 2. * np.pi)

    def hyperfine_level(self, f: float, mf: float):
        """|F, mF> level (low-field basis)."""
        shift = 0.
        zeeman = self.zeeman()
        if zeeman is not None:
            solver, energy_shifts, eigenvecs = zeeman
            # Hyperfine A shift is already accounted for in AtomicInternalEnergyLevel
            shift = solver.get_state_energy(energy_shifts, eigenvecs, f=f, mf=mf, subtract_hyperfineA_shift=True)
        return self.HyperfineLevel(**self.fine_data, i=self.nuclear_spin, f=f, mf=mf, external_energy_shift=shift * 2. * np.pi)

    def uncoupled_level(self, mj: float, mi: float):
        """|mJ, mI> level (high-field / Back-Goudsmit basis)."""
        zeeman = self.zeeman()
        if zeeman is None:
            raise IonSimError(f"|mJ, mI> levels were requested for manifold '{self.term_symbol}' at zero magnetic field. "
                              f"These are not energy eigenstates at zero field; specify (f, mf) instead or set a nonzero magnetic_field.")
        solver, energy_shifts, eigenvecs = zeeman
        if solver.approximation is not None:
            raise IonSimError(f"|mJ, mI> levels require the exact Zeeman solver, but approximation='{solver.approximation}' "
                              f"was requested. The weak-field approximation works in the |F, mF> basis; specify (f, mf) instead.")

        # Identify which eigenstate the solver will assign to this label, so we can check that the
        # label is meaningful (purity) and that no two requested labels land on the same eigenstate.
        basis_index = solver.basis_states.index((mj, mi))
        overlaps = np.abs(eigenvecs[basis_index, :])**2
        eigen_index = int(np.argmax(overlaps))
        purity = float(overlaps[eigen_index])
        previous = self._assigned_eigenstates.get(eigen_index)
        if previous is not None and previous != (mj, mi):
            raise IonSimError(f"In manifold '{self.term_symbol}' at B = {self.magnetic_field}, |mJ, mI> = {(mj, mi)} and {previous} "
                              f"map to the same energy eigenstate. mJ, mI are not good quantum numbers at this field; "
                              f"use (f, mf) or a stronger field.")
        self._assigned_eigenstates[eigen_index] = (mj, mi)
        if purity < UNCOUPLED_PURITY_WARNING_THRESHOLD:
            warnings.warn(f"|mJ={Fraction(mj)}, mI={Fraction(mi)}> in manifold '{self.term_symbol}' at B = {self.magnetic_field} "
                          f"has only {purity:.1%} overlap with its assigned energy eigenstate; the label is nominal. "
                          f"Consider specifying (f, mf) at this field.", stacklevel=4)

        # Returns the full eigenvalue (Zeeman + hyperfine), relative to the fine-structure energy, in solver freq units (Hz).
        # LSBackGoudsmitLevel.hyperfine_energy_shift is 0, so all of it goes into the external shift.
        shift = solver.get_state_energy_from_mjmi_pair(energy_shifts, eigenvecs, mj=mj, mi=mi)
        return self.UncoupledLevel(**self.fine_data, i=self.nuclear_spin, mj=mj, mi=mi, external_energy_shift=shift * 2. * np.pi)


