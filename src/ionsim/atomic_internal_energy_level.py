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
from fractions import Fraction
from sympy.physics.wigner import wigner_3j, clebsch_gordan
import sympy 
from icecream import ic
import re

from ionsim.ionsim_error import IonSimError
from ionsim.energy_level import EnergyLevel

term_symbol_matcher = re.compile(r"^([0-9]+ )?[SPDF]?[0-9/\[\]]*$")
def check_n_from_term_symbol(term_symbol: str, n_expected: int):
    """ Checks whether a term symbol violates internal consistency """ 
    m = term_symbol_matcher.match(term_symbol)
    # The regex must match, and then if there is a principal quantum number in front it must match expectation
    if not m:
        raise IonSimError(f"Term symbols consist of an optional principal quantum number separated from the electronic manifold part of the term symbol by a space, got {term_symbol}.")
    if m.groups()[0] is not None and int(m.groups()[0]) != n_expected:
        raise IonSimError(f"Term symbol is inconsistent with principal quantum number specification. Term symbol gave {m.groups()[0]}, expected {n_expected}.")

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
class LSFineLevel(AtomicInternalEnergyLevel): 
    """A fine-structure energy level of an atom."""
    l: float
    s: float
    mj: float
    external_energy_shift : float = 0. # Energy shift from external fields, such as time-independent Zeeman or Stark shifts.
    lifetime: float | str='null'
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
    lifetime: float | str='null'
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
    lifetime: float | str='null'
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
        return 0. 

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
    lifetime: float | str='null'
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
        """ Hyperfine shift not included here."""
        return 0. 

    @property
    def bare_energy(self): 
        """The field-free energy of the hyperfine-structure level."""
        if self.i == 0:
            return self.fine_energy
        else:
            return self.fine_energy + self.hyperfine_energy_shift

@dataclass(frozen=True, eq=False)
class J1L2FineLevel(AtomicInternalEnergyLevel): 
    """A fine-structure energy level of an atom."""
    j1: float
    l2: float
    k: float
    s2: float
    mj: float
    external_energy_shift : float = 0. # Energy shift from external fields, such as time-independent Zeeman or Stark shifts.
    lifetime: float | str='null'
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
    lifetime: float | str = 'null'
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
    lifetime: float | str = 'null'
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

    raise IonSimError(f"Dipole amplitudes are not supported for {type(level).__name__} levels.")


def _fine_structure_dipole_amplitude(j: sympy.Rational, mj: sympy.Rational, jp: sympy.Rational,
                                     mjp: sympy.Rational, q: int) -> sympy.Expr:
    """<J mJ| r_q |J' mJ'> / <J||r||J'>, Steck Eq. 34 convention."""
    return (-1)**(jp - 1 + mj) * sympy.sqrt(2*j + 1) * wigner_3j(jp, 1, j, mjp, sympy.Integer(q), -mj)


def compute_dipole_amplitude(ground_level: AtomicInternalEnergyLevel, excited_level: AtomicInternalEnergyLevel, q: int) -> float:
    """E1 dipole amplitude <ground| r_q |excited>, in units of the reduced matrix element <J||e r||J'>.

        Follows Steck's conventions (https://steck.us/alkalidata/rubidium87numbers.pdf). The amplitude is nonzero
        only when m_ground = m_excited + q. With this normalization, the total strength out of any ground sublevel,
        summed over all excited sublevels of J' and over q, is 1.
    
        Each level is expanded in the uncoupled basis |J mJ> (x) |I mI>; the dipole operator acts only on the
        electronic part (mI is conserved), and each electronic matrix element is given by Steck Eq. 34:
            <J mJ| r_q |J' mJ'> = <J||r||J'> (-1)^(J'-1+mJ) sqrt(2J+1) (J' 1 J; mJ' q -mJ)
        For two |F, mF> levels this reproduces Steck Eqs. 35-36:
            <F mF| r_q |F' mF'> = <J||r||J'> (-1)^(F'-1+mF) sqrt(2F+1) (F' 1 F; mF' q -mF)
                                  x (-1)^(F'+J+1+I) sqrt((2F'+1)(2J+1)) {J J' 1; F' F I}
        and it also covers |mJ, mI> (Back-Goudsmit) levels and any mixture of the two bases, e.g. an |F, mF>
        state coupled to an |mJ, mI> state.
    
        Levels are treated as pure states of their labeled basis. At fields where the label is only nominal
        (F or mJ, mI not good quantum numbers), the true eigenstate is a superposition and this amplitude is
        approximate.
    """
    if ground_level.i != excited_level.i:
        raise IonSimError(f"Nuclear spin must be the same in both levels, got {ground_level.i} and {excited_level.i}.")

    j, jp = _as_rational(ground_level.j), _as_rational(excited_level.j)
    amplitude = sympy.Integer(0)
    for mj, mi, c_ground in _uncoupled_components(ground_level):
        for mjp, mip, c_excited in _uncoupled_components(excited_level):
            # E1 does not act on the nucleus, and the 3j symbol vanishes unless mJ = mJ' + q.
            if mi != mip or mj != mjp + q:
                continue
            amplitude += c_ground * c_excited * _fine_structure_dipole_amplitude(j, mj, jp, mjp, q)
    return float(sympy.simplify(amplitude))
