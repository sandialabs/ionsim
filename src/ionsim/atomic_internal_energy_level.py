from ionsim.ionsim_error import IonSimError
from ionsim.energy_level import EnergyLevel

from abc import ABC, abstractmethod
from dataclasses import dataclass
from fractions import Fraction

from icecream import ic

@dataclass(frozen=True, eq=False)
class AtomicInternalEnergyLevel(EnergyLevel):
    """An internal energy level of an atom, i.e., an energy eigenstate of the electronic and nuclear degrees of freedom."""
    n: float 
    j: float
    term_symbol: str
    fine_energy: float
    hyperfine_A: float
    lifetime: float
    branching_ratios: dict[str, float] # TODO: Shouldn't this need " | None "? Why doesn't this cause a mypy error?

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
    def bare_energy(self): # TODO: handle the dressed energy and A/C Stark shifts in the atom class since you need other levels
        """The field-free energy of the hyperfine-structure level."""
        if self.i == 0:
            return self.fine_energy
        else:
            return self.fine_energy + self.hyperfine_energy_shift

    @property
    def energy(self): # TODO: see comment next to 'bare_energy'
        return self.bare_energy

@dataclass(frozen=True, eq=False)
class LSFineLevel(AtomicInternalEnergyLevel): 
    """A fine-structure energy level of an atom."""
    l: float
    s: float
    mj: float

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
    """A hyperfine-structure energy level of an atom."""
    j1: float
    l2: float
    k: float
    s2: float
    i: float
    f: float
    mf: float

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



