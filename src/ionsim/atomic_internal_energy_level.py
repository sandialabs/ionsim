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

<<<<<<< HEAD
def compute_multipole_amplitude(ground_level: AtomicInternalEnergyLevel, excited_level: AtomicInternalEnergyLevel, 
                                multipole_order:int, q: int, include_J_J_prime_matrix_element: bool=True) -> float: 
    """ Return the angular amplitude for <e|T^(k)_q|g>, up to a common reduced matrix element. 

        Method to compute 'E-multipole-order' transition operator between two states using the Clebsch-Gordan
        or Wigner-3,6j coefficients.

        e.g. multipole_order = 1 (2) corresponds to a dipole (quadrupole) transition 

        Based on Steck conventions (https://steck.us/alkalidata/rubidium87numbers.pdf):
            - 3j part (Eq. 35 style): (-1)^(Fp-k+mf) sqrt(2F+1) (Fp k F; mp q -mf)
            - 6j part (Eq. 36 style): (-1)^(Fp+J+k+I) sqrt((2Fp+1)(2J+1)) {J Jp k; Fp F I}
    """
    # Check possible q values are consistent with specified "k" value 
    k = multipole_order
    possible_q = list(np.arange(-k, k+1))
    if q not in possible_q:
        raise ValueError(f"Specified q={q} is not possible for transition k={k}; possible q values are {possible_q}")
    
    # Extract angular momentum quantum numbers for each state: 
    i = ground_level.i
    assert i == excited_level.i, 'Error: Nuclear angular momentum should be the same in both excited and ground levels.'

    if isinstance(ground_level, LSFineLevel) or isinstance(ground_level, J1L2FineLevel): 
        f, mf = ground_level.j, ground_level.mj
        assert ground_level.i == 0.
    else:
        f, mf = ground_level.f, ground_level.mf 

    if isinstance(excited_level, LSFineLevel) or isinstance(excited_level, J1L2FineLevel): 
        fp, mp = excited_level.j, excited_level.mj
        assert excited_level.i == 0.
    else:
        fp, mp = excited_level.f, excited_level.mf 

    if isinstance(ground_level, J1L2HyperfineLevel): 
        j = ground_level.k + ground_level.s2 
    else:
        j = ground_level.j 

    if isinstance(excited_level, J1L2HyperfineLevel): 
        jp = excited_level.k + excited_level.s2 
    else:
        jp = excited_level.j 

    f = sympy.S(f)
    mf = sympy.S(mf)
    fp = sympy.S(fp)
    mp = sympy.S(mp)

    k = sympy.S(k)
    q = sympy.S(q)

    j = sympy.S(j)
    jp = sympy.S(jp)
    i = sympy.S(i)

    wigner_3j_term = ((-1)**(fp - k + mf)) * sympy.sqrt(2*f + 1) * wigner_3j(fp, k, f, mp, sympy.Integer(q), -mf)
    wigner_6j_term = ((-1)**(fp + j + k + i)) * sympy.sqrt((2*fp + 1) * (2*j + 1)) * wigner_6j(j, jp, k, fp, f, i)
    # TODO: Generalize to non LS Fine/hyperfine couplings 
    if include_J_J_prime_matrix_element:
        l = ground_level.l
        s = ground_level.s
        lp = excited_level.l
        l = sympy.S(l)
        lp = sympy.S(lp)
        s = sympy.S(s)
    
        J_L_wigner_6j_term = ((-1)**(jp + l + k + s)) * sympy.sqrt((2*jp + 1) * (2*l + 1)) * wigner_6j(l, lp, k, jp, j, s)
        return float(sympy.simplify(J_L_wigner_6j_term * wigner_3j_term * wigner_6j_term))
    else:
        return float(sympy.simplify(wigner_3j_term * wigner_6j_term))
    

def compute_dipole_amplitude(ground_level: AtomicInternalEnergyLevel, excited_level: AtomicInternalEnergyLevel, q: int) -> float:
    ''' Method to compute E1 dipole transition operator between two states using the Clebsch-Gordan
        or Wigner-3,6j coefficients. 

        Based on Steck conventions (https://steck.us/alkalidata/rubidium87numbers.pdf):
            - 3j part (Eq. 35 style): (-1)^(Fp-1+mf) sqrt(2F+1) (Fp 1 F; mp q -mf)
            - 6j part (Eq. 36 style): (-1)^(Fp+J+1+I) sqrt((2Fp+1)(2J+1)) {J Jp 1; Fp F I}
    '''
    # Extract angular momentum quantum numbers for each state: 
    return compute_multipole_amplitude(ground_level, excited_level, 1, q)


def compute_hyperfine_clebsch_gordan_coefficient(ground_level: AtomicInternalEnergyLevel, excited_level: AtomicInternalEnergyLevel, q: int, multipole_order) -> float:
    ''' Method to compute E1 dipole transition operator between two states using the Clebsch-Gordan
        or Wigner-3,6j coefficients. 

        Based on Steck conventions (https://steck.us/alkalidata/rubidium87numbers.pdf):
            - 3j part (Eq. 35 style): (-1)^(Fp-1+mf) sqrt(2F+1) (Fp 1 F; mp q -mf)
            - 6j part (Eq. 36 style): (-1)^(Fp+J+1+I) sqrt((2Fp+1)(2J+1)) {J Jp 1; Fp F I}
    '''
    # Extract angular momentum quantum numbers for each state: 
    return compute_multipole_amplitude(ground_level, excited_level, multipole_order, q, False)

def compute_rabi_frequency_between_atomic_levels(ground_level: AtomicInternalEnergyLevel, excited_level: AtomicInternalEnergyLevel, polarization_components: Vector, 
                                            atomic_levels: list[AtomicInternalEnergyLevel], multipole_order: int, peak_E0_amplitude: float) -> complex | float:
    """ Computes the Rabi frequency between a ground level |g> and an excited level |e>, given polarization components and selection rules. """  
    return peak_E0_amplitude * compute_coupling_amplitude_between_atomic_levels(ground_level, excited_level, polarization_components, atomic_levels, multipole_order)


def compute_coupling_amplitude_between_atomic_levels(ground_level: AtomicInternalEnergyLevel, excited_level: AtomicInternalEnergyLevel, polarization_components: Vector, 
                                            atomic_levels: list[AtomicInternalEnergyLevel], multipole_order: int) -> complex | float:
    """ Computes a coupling amplitude (up to a electric field amplitude) between a ground level |g> and an excited level |e>, given polarization components and selection rules. 

        Returns Omega/E0, the Rabi frequency up to the E_0 electric field amplitude scaling. 

    """  
    if ground_level not in atomic_levels:
        raise ValueError(f"Ground level {ground_level.name} not found in the atomic structure {atomic_levels}.")
    if excited_level not in atomic_levels:
        raise ValueError(f"Excited level {excited_level.name} not found in the atomic structure {atomic_levels}.")
    
    q = list(np.arange(-multipole_order, multipole_order+1))
    if multipole_order != 1 and multipole_order != 2:
        raise ValueError(f"Multipole order be either 1 or 2, corresponding to E1 dipole or E2 quadrupole transitions. Received {multipole_order}.")
    
    # Estimate rabi frequency from laser polarization and multipole amplitude components  
    # Compute dot product w.r.t q of spherical polarization components and multipole amplitude components 
    coupling_amplitudes = {}
    for _q in q: 
        coupling_amplitudes[_q] = compute_multipole_amplitude(ground_level, excited_level, multipole_order, _q) 
    
    # Compute dot product with laser field polarization vector 
    # TODO: should we use vdot? 
    scientific_consts = const.e * const.value('Bohr radius') / const.hbar # from dipole moment and definition of Rabi frequency from electric dipole operator 
    coupling = 0. + 1j*0.
    coupling = 2.*np.dot(polarization_components, np.array(list(coupling_amplitudes.values()))) * scientific_consts 
    return coupling 
=======
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
>>>>>>> main
