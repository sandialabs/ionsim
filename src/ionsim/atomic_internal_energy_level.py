#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
import numpy as np
import sympy 
from sympy.physics.wigner import wigner_3j, clebsch_gordan, wigner_6j 
from scipy import constants as const
import re

from ionsim.ionsim_error import IonSimError
from ionsim.energy_level import EnergyLevel
from ionsim.custom_types import Vector

term_symbol_matcher = re.compile(r"^([0-9]+ )?[SPDF]?[0-9/\[\]]*$")
def check_n_from_term_symbol(term_symbol: str, n_expected: int):
    """ Checks whether a term symbol violates internal consistency """ 
    m = term_symbol_matcher.match(term_symbol)
    # The regex must match, and then if there is a principal quantum number in front it must match expectation
    if not m:
        raise IonSimError(f"Term symbols consist of an optional principal quantum number separated from the electronic manifold part of the term symbol by a space, got {term_symbol}.")
    if m.groups()[0] is not None and int(m.groups()[0]) != n_expected:
        raise IonSimError(f"Term symbol is inconsistent with principal quantum number specification. Term symbol gave {m.groups()[0]}, expected {n_expected}.")

class EigenBasis(Enum):
    """Angular-momentum labels used to specify an atomic angular momentum eigenstate (not the Hilbert-space basis of StandardBasis)."""
    FINE = '|J, mJ>'         # nuclear spin zero
    HYPERFINE = '|F, mF>'    # low field
    UNCOUPLED = '|mJ, mI>'   # high field (Back-Goudsmit)


@dataclass(frozen=True, eq=False)
class AtomicInternalEnergyLevel(EnergyLevel):
    """An internal energy level of an atom, i.e., an energy eigenstate of the electronic and nuclear degrees of freedom."""
    n: float 
    j: float
    term_symbol: str
    fine_energy: float 
    hyperfine_A: float
    alias: str | None = field(default=None, kw_only=True)

    def __post_init__(self):
        check_n_from_term_symbol(self.term_symbol, self.n)

    @property
    @abstractmethod
    def coupling_scheme(self):
        """The coupling scheme for the electronic orbital and spin angular momenta."""

    @property
    def hyperfine_energy_shift(self):
        """The energy shift of the level from the hyperfine interaction."""
        return self.hyperfine_A/2 * (
            + self.f * (self.f + 1)
            - self.j * (self.j + 1)
            - self.i * (self.i + 1)
        )

    @property
    def bare_energy(self): 
        """The field-free energy of the hyperfine-structure level."""
        if self.i == 0:
            return self.fine_energy
        else:
            return self.fine_energy + self.hyperfine_energy_shift

    @property
    def energy(self):
        # Total energy: bare energy + external shifts (e.g. Zeeman, light shifts)
        return self.bare_energy + self.external_energy_shift

@dataclass(frozen=True, eq=False)
class SinkLevel(EnergyLevel):
    """ Auxiliary level that collects population decaying to states outside the simulated levels.

        It has no internal structure and no coherent couplings; it only receives spontaneous emission
        (DissipatorSpontaneousEmission with decay_to_sink=True). Its energy only sets a phase and does not affect the dynamics.
    """
    energy: float = 0.
    name: str = 'sink'
    term_symbol: str = 'sink'  # lets code that filters levels by term symbol skip the sink without special-casing it
    alias: str | None = field(default='sink', kw_only=True)


@dataclass(frozen=True, eq=False)
class LSFineLevel(AtomicInternalEnergyLevel): 
    """A fine-structure energy level of an atom."""
    l: float
    s: float
    mj: float
    external_energy_shift : float = 0. # Energy shift from external fields, such as time-independent Zeeman or Stark shifts.
    lifetime: float | None=None 
    branching_ratios: dict[str, float] | None=None 
    hyperfine_B: float | None=None

    def __post_init__(self):
        super().__post_init__()

    @property
    def i(self):
        return 0

    @property
    def coupling_scheme(self):
        """The coupling scheme for the electronic orbital and spin angular momenta."""
        return 'ls'

    @property
    def name(self):
        """A unique name for the fine-structure level."""
        return ','.join([self.term_symbol, str(Fraction(self.mj))])
 
@dataclass(frozen=True, eq=False)
class LSHyperfineLevel(AtomicInternalEnergyLevel): 
    """A hyperfine-structure energy level of an atom."""
    l: float
    s: float
    i: float
    f: float
    mf: float
    external_energy_shift: float = 0.
    lifetime: float | None=None 
    branching_ratios: dict[str, float] | None=None 
    hyperfine_B: float | None=None

    def __post_init__(self):
        super().__post_init__()

    @property
    def coupling_scheme(self):
        """The coupling scheme for the electronic orbital and spin angular momenta."""
        return 'ls'

    @property
    def name(self):
        """A unique name for the hyperfine-structure level."""
        return ','.join([self.term_symbol, str(Fraction(self.f)), str(Fraction(self.mf))])


@dataclass(frozen=True, eq=False)
class LSBackGoudsmitLevel(AtomicInternalEnergyLevel): 
    """An energy level of an atom at strong magnetic field such that F no longer a good quantum number, described by mJ, mI quantum numbers."""
    l: float
    s: float
    i: float
    mj: float
    mi: float
    external_energy_shift: float = 0.
    lifetime: float | None=None 
    branching_ratios: dict[str, float] | None=None 
    hyperfine_B: float | None=None

    def __post_init__(self):
        super().__post_init__()

    @property
    def coupling_scheme(self):
        """The coupling scheme for the electronic orbital and spin angular momenta."""
        return 'ls' 

    @property
    def name(self):
        """A unique name for the uncoupled (mJ, mI) level."""
        return ','.join([self.term_symbol, str(Fraction(self.mj)), str(Fraction(self.mi))])

    @property
    def hyperfine_energy_shift(self):
        """The energy shift of the level from the hyperfine interaction."""
        return self.mj * self.mi * self.hyperfine_A 

    @property
    def bare_energy(self): 
        """The field-free energy of the level. Hyperfine shift is included in the external energy shift."""
        return self.fine_energy

    @property
    def energy(self):
        # Total energy: bare energy + external shifts (e.g. Zeeman, light shifts)
        return self.bare_energy + self.external_energy_shift

@dataclass(frozen=True, eq=False)
class LSPaschenBackLevel(AtomicInternalEnergyLevel): 
    """An energy level of an atom at very strong magnetic field such that J is no longer a good quantum number, 
        described by mS, mL, mI quantum numbers."""
    l: float
    s: float
    i: float
    mi: float
    ml: float
    ms: float
    external_energy_shift: float = 0.
    lifetime: float | None=None 
    branching_ratios: dict[str, float] | None=None 
    hyperfine_B: float | None=None

    def __post_init__(self):
        super().__post_init__()

    @property
    def coupling_scheme(self):
        """The coupling scheme for the electronic orbital and spin angular momenta."""
        return None 

    @property
    def name(self):
        """A unique name for the hyperfine-structure level."""
        return ','.join([self.term_symbol, str(Fraction(self.mi)), str(Fraction(self.ml)), str(Fraction(self.ms))])

    @property
    def hyperfine_energy_shift(self):
        """The energy shift of the level from the hyperfine interaction."""
        raise NotImplementedError(f"Not supported currently, requiring hyperfine A coefficients for mL, mS.")
        return (self.ml + self.ms) * self.mi * self.hyperfine_A 

    @property
    def bare_energy(self): 
        """The field-free energy of the level. Hyperfine shift is included in the external energy shift."""
        return self.fine_energy

    @property
    def energy(self):
        # Total energy: bare energy + external shifts (e.g. Zeeman, light shifts)
        return self.bare_energy + self.external_energy_shift

@dataclass(frozen=True, eq=False)
class J1L2FineLevel(AtomicInternalEnergyLevel): 
    """A fine-structure energy level of an atom."""
    j1: float
    l2: float
    k: float
    s2: float
    mj: float
    external_energy_shift : float = 0. # Energy shift from external fields, such as time-independent Zeeman or Stark shifts.
    lifetime: float | None=None 
    branching_ratios: dict[str, float] | None=None 
    hyperfine_B: float | None=None

    def __post_init__(self):
        super().__post_init__()

    @property
    def i(self):
        return 0

    @property
    def coupling_scheme(self):
        """The coupling scheme for the electronic orbital and spin angular momenta."""
        return 'j1l2'

    @property
    def name(self):
        """A unique name for the fine-structure level."""
        return ','.join([self.term_symbol, str(Fraction(self.mj))])
    
@dataclass(frozen=True, eq=False)
class J1L2HyperfineLevel(AtomicInternalEnergyLevel): 
    """A hyperfine-structure energy level of an atom: k = j1 + l2 ; J = k + s2 
        Corresponding term symbol: (2S_2 + 1)[K] """ 
    j1: float
    l2: float
    k: float
    s2: float
    i: float
    f: float
    mf: float
    gj: float
    external_energy_shift : float = 0. # Energy shift from external fields, such as time-independent Zeeman or Stark shifts.
    lifetime: float | None=None 
    branching_ratios: dict[str, float] | None = None 
    hyperfine_B: float | None=None

    def __post_init__(self):
        super().__post_init__()

    @property
    def coupling_scheme(self):
        """The coupling scheme for the electronic orbital and spin angular momenta."""
        return 'j1l2'

    @property
    def name(self):
        """A unique name for the hyperfine-structure level."""
        return ','.join([self.term_symbol, str(Fraction(self.f)), str(Fraction(self.mf))])

@dataclass(frozen=True, eq=False)
class J1L2BackGoudsmitLevel(AtomicInternalEnergyLevel): 
    """A hyperfine-structure energy level of an atom: k = j1 + l2 ; J = k + s2 
        Corresponding term symbol: (2S_2 + 1)[K] """ 
    j1: float
    l2: float
    k: float
    s2: float
    i: float
    mi: float
    mj: float
    gj: float
    external_energy_shift : float = 0. # Energy shift from external fields, such as time-independent Zeeman or Stark shifts.
    lifetime: float | None=None 
    branching_ratios: dict[str, float] | None = None 
    hyperfine_B: float | None=None

    def __post_init__(self):
        super().__post_init__()

    @property
    def coupling_scheme(self):
        """The coupling scheme for the electronic orbital and spin angular momenta."""
        return 'j1l2'

    @property
    def name(self):
        """A unique name for the hyperfine-structure level."""
        return ','.join([self.term_symbol, str(Fraction(self.mj)), str(Fraction(self.mi))])

    @property
    def hyperfine_energy_shift(self):
        """The energy shift of the level from the hyperfine interaction."""
        return self.mj * self.mi * self.hyperfine_A 

    @property
    def bare_energy(self): 
        """The field-free energy of the level. Hyperfine shift is included in the external energy shift."""
        return self.fine_energy

    @property
    def energy(self):
        # Total energy: bare energy + external shifts (e.g. Zeeman, light shifts)
        return self.bare_energy + self.external_energy_shift

# def _check_uniqueness_of_term_symbols(term_symbols: list[str], levels_data: list[dict]):
#     """Check whether the term symbol corresponds to a single energy level in the configuration data."""
#     return all([_check_uniqueness_of_term_symbol(term_symbol, levels_data) for term_symbol in term_symbols])

def _as_rational(x: float) -> sympy.Rational:
    """Exact sympy Rational for an integer or half-integer angular momentum quantum number."""
    frac = Fraction(float(x)).limit_denominator(2)
    return sympy.Rational(frac.numerator, frac.denominator)


def _uncoupled_components(level: AtomicInternalEnergyLevel) -> list[tuple[sympy.Rational, sympy.Rational, sympy.Expr]]:
    """Expand a level in the uncoupled basis |J, mJ> (x) |I, mI>.

    Returns a list of (mj, mi, coefficient) with coefficient = <J mJ; I mI | level>, using
    Condon-Shortley Clebsch-Gordan coefficients with J coupled before I (F = J + I).
    """
    if isinstance(level, (LSFineLevel, J1L2FineLevel)):
        return [(_as_rational(level.mj), sympy.Integer(0), sympy.Integer(1))]

    if isinstance(level, (LSBackGoudsmitLevel, J1L2BackGoudsmitLevel)):
        return [(_as_rational(level.mj), _as_rational(level.mi), sympy.Integer(1))]

    if isinstance(level, (LSHyperfineLevel, J1L2HyperfineLevel)):
        j, i, f, mf = (_as_rational(x) for x in (level.j, level.i, level.f, level.mf))
        components = []
        mi = -i
        while mi <= i:
            mj = mf - mi
            if abs(mj) <= j:
                coefficient = clebsch_gordan(j, i, f, mj, mi, mf)
                if coefficient != 0:
                    components.append((mj, mi, coefficient))
            mi += 1
        return components

    raise IonSimError(f"Multipole amplitudes are not supported for {type(level).__name__} levels.")


def _fine_structure_multipole_amplitude(j: sympy.Rational, mj: sympy.Rational, jp: sympy.Rational,
                                        mjp: sympy.Rational, k: int, q: int) -> sympy.Expr:
    """<J mJ| T^(k)_q |J' mJ'> / <J||T^(k)||J'>, Steck Eq. 34 convention generalized to rank k (multipole order)."""
    return (-1)**(jp - k + mj) * sympy.sqrt(2*j + 1) * wigner_3j(jp, k, j, mjp, sympy.Integer(q), -mj)


def _orbital_reduction_factor(ground_level: AtomicInternalEnergyLevel, excited_level: AtomicInternalEnergyLevel,
                              k: int) -> sympy.Expr:
    """<J||T^(k)||J'> / <L||T^(k)||L'> for LS-coupled levels; T^(k) acts on L only and S is a spectator.

    Uses Steck Eq. 36 with (F, I) -> (J, S): (-1)^(J'+L+k+S) sqrt((2J'+1)(2L+1)) {L L' k; J' J S}.
    """
    if not (hasattr(ground_level, 'l') and hasattr(excited_level, 'l')):
        raise IonSimError(f"Amplitudes in units of the orbital reduced matrix element <L||T||L'> need LS-coupled levels, "
                          f"got {type(ground_level).__name__} and {type(excited_level).__name__}.")
    l, lp = _as_rational(ground_level.l), _as_rational(excited_level.l)
    # The ground level's spin is used for both levels. For a spin-changing (intercombination) transition, e.g.
    # 1S0 -> 3P1, pure LS coupling gives zero; this factor then only provides the angular dependence, and the
    # reduced matrix element must be an effective value for that transition.
    s = _as_rational(ground_level.s)
    j, jp = _as_rational(ground_level.j), _as_rational(excited_level.j)
    return (-1)**(jp + l + k + s) * sympy.sqrt((2*jp + 1) * (2*l + 1)) * wigner_6j(l, lp, k, jp, j, s)


def compute_multipole_amplitude(ground_level: AtomicInternalEnergyLevel, excited_level: AtomicInternalEnergyLevel,
                                multipole_order: int, q: int, include_J_J_prime_matrix_element: bool = False) -> float:
    """Electric multipole amplitude <ground| T^(k)_q |excited> of rank k = multipole_order, up to a reduced matrix element.

        multipole_order = 1 (2) is an E1 dipole (E2 quadrupole) transition. Follows Steck's conventions
        (https://steck.us/alkalidata/rubidium87numbers.pdf), generalized from rank 1 to rank k. The amplitude is nonzero
        only when m_ground = m_excited + q.

        Each level is expanded in the uncoupled basis |J mJ> (x) |I mI>; T^(k) acts only on the electronic part (mI is
        conserved), and each electronic matrix element is (Steck Eq. 34 with 1 -> k):
            <J mJ| T^(k)_q |J' mJ'> = <J||T^(k)||J'> (-1)^(J'-k+mJ) sqrt(2J+1) (J' k J; mJ' q -mJ)
        For two |F, mF> levels this reproduces Steck Eqs. 35-36 with 1 -> k:
            <F mF| T^(k)_q |F' mF'> = <J||T^(k)||J'> (-1)^(F'-k+mF) sqrt(2F+1) (F' k F; mF' q -mF)
                                      x (-1)^(F'+J+k+I) sqrt((2F'+1)(2J+1)) {J J' k; F' F I}
        and it also covers |mJ, mI> (Back-Goudsmit) levels and any mixture of the two bases.

        Units:
            include_J_J_prime_matrix_element=False (default): units of <J||T^(k)||J'>. Summed over every sublevel of the
                excited manifold and over q, the total strength out of any ground sublevel is 1.
            include_J_J_prime_matrix_element=True: units of the orbital reduced matrix element <L||T^(k)||L'>, i.e. the
                result is multiplied by <J||T^(k)||J'>/<L||T^(k)||L'> (LS-coupled levels only).

        Levels are treated as pure states of their labeled basis. At fields where the label is only nominal
        (F or mJ, mI not good quantum numbers), the true eigenstate is a superposition and this amplitude is approximate.
    """
    k = multipole_order
    if k not in (1, 2):
        raise IonSimError(f"Multipole order must be 1 (E1 dipole) or 2 (E2 quadrupole), got {multipole_order}.")
    possible_q = list(range(-k, k + 1))
    if q not in possible_q:
        raise IonSimError(f"q={q} is not possible for multipole order k={k}; possible q values are {possible_q}.")
    if ground_level.i != excited_level.i:
        raise IonSimError(f"Nuclear spin must be the same in both levels, got {ground_level.i} and {excited_level.i}.")

    j, jp = _as_rational(ground_level.j), _as_rational(excited_level.j)
    amplitude = sympy.Integer(0)
    for mj, mi, c_ground in _uncoupled_components(ground_level):
        for mjp, mip, c_excited in _uncoupled_components(excited_level):
            # T^(k) does not act on the nucleus, and the 3j symbol vanishes unless mJ = mJ' + q.
            if mi != mip or mj != mjp + q:
                continue
            amplitude += c_ground * c_excited * _fine_structure_multipole_amplitude(j, mj, jp, mjp, k, q)
    if include_J_J_prime_matrix_element and amplitude != 0:
        amplitude *= _orbital_reduction_factor(ground_level, excited_level, k)
    return float(sympy.simplify(amplitude))


def compute_dipole_amplitude(ground_level: AtomicInternalEnergyLevel, excited_level: AtomicInternalEnergyLevel, q: int) -> float:
    """E1 dipole amplitude <ground| r_q |excited>, in units of the reduced matrix element <J||e r||J'>.

        See compute_multipole_amplitude (multipole_order = 1).
    """
    return compute_multipole_amplitude(ground_level, excited_level, 1, q)


def compute_hyperfine_clebsch_gordan_coefficient(ground_level: AtomicInternalEnergyLevel, excited_level: AtomicInternalEnergyLevel,
                                                 q: int, multipole_order: int) -> float:
    """Angular (Clebsch-Gordan) part of <ground| T^(k)_q |excited>, in units of <J||T^(k)||J'>.

        See compute_multipole_amplitude.
    """
    return compute_multipole_amplitude(ground_level, excited_level, multipole_order, q, include_J_J_prime_matrix_element=False)


def compute_rabi_frequency_between_atomic_levels(ground_level: AtomicInternalEnergyLevel, excited_level: AtomicInternalEnergyLevel,
                                                 polarization_components: dict[int, complex], atomic_levels: list[AtomicInternalEnergyLevel],
                                                 multipole_order: int, peak_E0_amplitude: float, wavenumber: float | None = None) -> complex | float:
    """ Computes the Rabi frequency between a ground level |g> and an excited level |e>, given polarization components and selection rules. 

        See compute_coupling_amplitude_between_atomic_levels; wavenumber (rad/m) is required for multipole_order = 2.
    """  
    return peak_E0_amplitude * compute_coupling_amplitude_between_atomic_levels(ground_level, excited_level, polarization_components,
                                                                                atomic_levels, multipole_order, wavenumber)


def compute_coupling_amplitude_between_atomic_levels(ground_level: AtomicInternalEnergyLevel, excited_level: AtomicInternalEnergyLevel,
                                                     polarization_components: dict[int, complex], atomic_levels: list[AtomicInternalEnergyLevel],
                                                     multipole_order: int, wavenumber: float | None = None) -> complex | float:
    """ Computes a coupling amplitude (up to a electric field amplitude) between a ground level |g> and an excited level |e>, given polarization components and selection rules. 

        Returns Omega/E0, the Rabi frequency up to the E_0 electric field amplitude scaling, in units of the orbital
        reduced matrix element <L||T^(k)||L'> in atomic units: <L||r||L'> / a0 for E1, <L||r^2 C^(2)||L'> / a0^2 for E2.

        polarization_components maps q to the polarization's rank-k spherical component: eps_q for E1
        (Polarization.dipole_components), C_q for E2 (Polarization.quadrupole_components). The absorption matrix element is
            <e| T^(k) . eps |g> = sum_q c_q <e| T^(k)_q |g> = sum_q c_q (-1)^q <g| T^(k)_{-q} |e>,
        using T_q^dagger = (-1)^q T_{-q} and real angular amplitudes <g| T_q |e> (compute_multipole_amplitude).

        E2 uses the field gradient of a plane wave, E0 eps exp(i k n.r): the coupling is i k E0 sum_ij eps_i n_j x_i x_j,
        and sum_ij eps_i n_j x_i x_j = sqrt(2/3) r^2 sum_q C_q C^(2)_q, so E2 carries an extra factor i k a0 sqrt(2/3)
        relative to E1. wavenumber (rad/m) is required for E2.
    """  
    if ground_level not in atomic_levels:
        raise ValueError(f"Ground level {ground_level.name} not found in the atomic structure {atomic_levels}.")
    if excited_level not in atomic_levels:
        raise ValueError(f"Excited level {excited_level.name} not found in the atomic structure {atomic_levels}.")
    if multipole_order != 1 and multipole_order != 2:
        raise ValueError(f"Multipole order be either 1 or 2, corresponding to E1 dipole or E2 quadrupole transitions. Received {multipole_order}.")
    q_values = range(-multipole_order, multipole_order + 1)
    if not isinstance(polarization_components, dict) or set(polarization_components) != set(q_values):
        raise IonSimError(f"polarization_components must be a dict keyed by q = {list(q_values)} for multipole order "
                          f"{multipole_order}, e.g. Polarization.dipole_components() or Polarization.quadrupole_components().")

    # Absorption matrix element <e| T . eps |g>, from the amplitudes <g| T_q |e>, in units of <L||T^(k)||L'>.
    matrix_element = sum(polarization_components[q] * (-1)**q
                         * compute_multipole_amplitude(ground_level, excited_level, multipole_order, -q, include_J_J_prime_matrix_element=True)
                         for q in q_values)

    scientific_consts = const.e * const.value('Bohr radius') / const.hbar # from dipole moment and definition of Rabi frequency from electric dipole operator 
    coupling = 2. * matrix_element * scientific_consts
    if multipole_order == 2:
        if wavenumber is None:
            raise IonSimError("E2 coupling amplitudes need the laser wavenumber (rad/m).")
        coupling *= 1j * wavenumber * const.value('Bohr radius') * np.sqrt(2. / 3.)
    return coupling
