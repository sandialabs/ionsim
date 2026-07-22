#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

from abc import ABC, abstractmethod
from dataclasses import dataclass
from fractions import Fraction
from sympy.physics.wigner import wigner_3j, wigner_6j 
import sympy 
from icecream import ic

from ionsim.ionsim_error import IonSimError
from ionsim.energy_level import EnergyLevel

@dataclass(frozen=True, eq=False)
class AtomicInternalEnergyLevel(EnergyLevel):
    """An internal energy level of an atom, i.e., an energy eigenstate of the electronic and nuclear degrees of freedom."""
    n: float 
    j: float
    term_symbol: str
    fine_energy: float 
    hyperfine_A: float

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
 
# @update_annotations   
@dataclass(frozen=True, eq=False)
class LSHyperfineLevel(AtomicInternalEnergyLevel): 
    """A hyperfine-structure energy level of an atom."""
    # ct.update_annotations(__annotations__, [AtomicInternalEnergyLevel]) #TODO: is this proper?
    # why doesn't this stop me from passing None to branching ratios and lifetime?

    l: float
    s: float
    i: float
    f: float
    mf: float
    external_energy_shift: float = 0.
    lifetime: float | str='null'
    branching_ratios: dict[str, float] | None=None 
    hyperfine_B: float | None=None

    @property
    def coupling_scheme(self):
        """The coupling scheme for the electronic orbital and spin angular momenta."""
        return 'ls'

    @property
    def name(self):
        """A unique name for the hyperfine-structure level."""
        return ','.join([self.term_symbol, str(Fraction(self.f)), str(Fraction(self.mf))])

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
        return ','.join([self.term_symbol, str(Fraction(self.mf))])
    
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


    @property
    def coupling_scheme(self):
        """The coupling scheme for the electronic orbital and spin angular momenta."""
        return 'j1l2'

    @property
    def name(self):
        """A unique name for the hyperfine-structure level."""
        return ','.join([self.term_symbol, str(Fraction(self.f)), str(Fraction(self.mf))])


# def _check_uniqueness_of_term_symbols(term_symbols: list[str], levels_data: list[dict]):
#     """Check whether the term symbol corresponds to a single energy level in the configuration data."""
#     return all([_check_uniqueness_of_term_symbol(term_symbol, levels_data) for term_symbol in term_symbols])

def compute_dipole_amplitude(ground_level: AtomicInternalEnergyLevel, excited_level: AtomicInternalEnergyLevel, q: int) -> float:
    ''' Method to compute E1 dipole transition operator between two states using the Clebsch-Gordan
        or Wigner-3,6j coefficients. 

        Based on Steck conventions (https://steck.us/alkalidata/rubidium87numbers.pdf):
            - 3j part (Eq. 35 style): (-1)^(Fp-1+mf) sqrt(2F+1) (Fp 1 F; mp q -mf)
            - 6j part (Eq. 36 style): (-1)^(Fp+J+1+I) sqrt((2Fp+1)(2J+1)) {J Jp 1; Fp F I}
    '''
    # Extract angular momentum quantum numbers for each state: 
    i = ground_level.i
    assert i == excited_level.i, 'Error: Nuclear angular momentum should be the same in both excited and ground levels.'

    if isinstance(ground_level, (LSFineLevel, J1L2FineLevel)): 
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

    wigner_3j_term = (-1)**(fp - 1 + mf) * sympy.sqrt(2*f + 1) * wigner_3j(fp, 1, f, mp, sympy.Integer(q), -mf)
    wigner_6j_term = (-1)**(fp + j + 1 + i) * sympy.sqrt((2*f + 1) * (2*j + 1)) * wigner_6j(j, jp, 1, fp, f, i)
    return float(sympy.simplify(wigner_3j_term * wigner_6j_term))
