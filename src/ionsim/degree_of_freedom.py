from ionsim.energy_level import EnergyLevel
from ionsim.atomic_internal_energy_level import AtomicInternalEnergyLevel
from ionsim.atomic_internal_energy_level import LSFineLevel, LSHyperfineLevel, J1L2FineLevel, J1L2HyperfineLevel
from ionsim.collective_motional_energy_level import CollectiveMotionalEnergyLevel
from ionsim.zeeman_solver import ZeemanHyperfineSolver

import importlib.resources
from pathlib import Path
from dataclasses import dataclass
from abc import ABC
import yaml
from fractions import Fraction
import numpy as np
from typing import Sequence

from icecream import ic

@dataclass(frozen=True, eq=False)
class DegreeOfFreedom(ABC):
    """A degree of freedom in a basis of states."""
    energy_levels: Sequence[EnergyLevel]
    name: str | None = None # TODO: will we use these names?

@dataclass(frozen=True, eq=False)
class AtomicStructure(DegreeOfFreedom):
    """An atomic structure object, containing atomic internal energy levels corresponding to angular momentum eigenstates.""" 
    energy_levels: list[AtomicInternalEnergyLevel]

    @classmethod
    def from_species(cls, species: str, term_symbols: list[str] | None = None, level_names: list[str] | None = None, 
            name: str | None = None, magnetic_field: float=0.):
        """Build the atomic structure degree of freedom for a particular species of atom."""
        config_data = cls.get_config_data(species)
        nuclear_spin = config_data['nuclear_spin']
        levels_data = config_data['levels']
        mass = config_data['mass'] # Daltons
        z = config_data['Z'] # Atomic number, number of protons 
        magnetic_moment = config_data['magnetic_moment'] # units of \mu_{N}
        structure = 'fine' if nuclear_spin == 0 else 'hyperfine' 

        if term_symbols is not None:
            levels_data = cls.select_some_data(term_symbols, levels_data)

        levels = []
        for level_data in levels_data:

            level_data['unique_term_symbol'] = level_data['term_symbol']
            level_data['unique_branching_ratios'] = level_data.get('branching_ratios', None)
            # level_data['unique_term_symbol'] = _get_unique_term_symbol(level_data, levels_data)

            get_fine_data, FineLevel, HyperfineLevel = cls.get_level_factory(level_data['coupling_scheme'])
            fine_data = get_fine_data(level_data)
            j = fine_data['j']

            # ic(FineLevel.__annotations__, FineLevel(**fine_data, mj={'key': 1}))
            # TODO: why is mypy not catching this??

            # Use Zeeman Solver based on level manifold to compute Zeeman shifts 
            if magnetic_field != 0. :
                if level_data['coupling_scheme'] == 'j1l2': 
                    s2 = fine_data['s2']
                    if fine_data['gj'] is None:
                        k = fine_data['k']
                        j1 = fine_data['j1']
                        l2 = fine_data['l2']
                        s2 = fine_data['s2']
                        # See p. 100 of B. G. Wybourne, Spectroscopic Properties of Rare Earths (Interscience, New York, 1965). 
                        # and p. 6 and 7 of https://nvlpubs.nist.gov/nistpubs/Legacy/NSRDS/nbsnsrds60.pdf
                        gj1 = 1. + (j1*(j1+1) + s2*(s2+1) - l2*(l2+1))/(2. * j1*(j1+1)) # from LS formula 
                        gj = 2. * (gj1 - 1.) * (k*(k+1) + j1*(j1+1) - l2*(l2 + 1))/((2*j + 1)*(2*k + 1))
                        gj += (3*j*(j+1) - k*(k+1) + s2*(s2+1))/(2.*j*(j+1)) 
                        fine_data['gj'] = gj
                    Zeeman_solver = ZeemanHyperfineSolver(nuclear_spin, j, None, s2, fine_data['hyperfine_A']*2.*np.pi, mass, magnetic_moment, z, gj = gj)
                else:
                    s = fine_data['s']
                    l = fine_data['l']
                    Zeeman_solver = ZeemanHyperfineSolver(nuclear_spin, j, l, s, fine_data['hyperfine_A']*2.*np.pi, mass, magnetic_moment, z)
                zeeman_energy_shifts, zeeman_eigenvecs = Zeeman_solver.solve_at_field(magnetic_field)
                zeeman_energy_shifts *= np.pi*2. # convert to rad/s 

            # Construct levels based on coupling structure 
            if structure == 'fine':
                for mj in np.arange(-j, j + 1):
                    # Extract any Zeeman shifts for this state 
                    zeeman_shift_energy = 0.
                    if magnetic_field != 0. :
                        # For fine couplings, F = J since I = 0, so F <==> J and mf <==> mj labels are interchangable. 
                        zeeman_shift_energy = Zeeman_solver.get_state_energy(zeeman_energy_shifts, zeeman_eigenvecs, f = j, mf = mj)
                    # Create the level 
                    level = FineLevel(**fine_data, mj=mj, external_energy_shift=zeeman_shift_energy)
                    if level_names is None or level.name in level_names: 
                        levels.append(level)
            else:
                for f in np.arange(np.abs(j - nuclear_spin), j + nuclear_spin + 1):
                    for mf in np.arange(-f, f + 1):
                        # Extract any Zeeman shifts for this mF state 
                        zeeman_shift_energy = 0.
                        if magnetic_field != 0. :
                            zeeman_shift_energy = Zeeman_solver.get_state_energy(zeeman_energy_shifts, zeeman_eigenvecs, f = f, mf = mf)

                        # Create the level 
                        level = HyperfineLevel(**fine_data, i=nuclear_spin, f=f, mf=mf, external_energy_shift = zeeman_shift_energy)
                        if level_names is None or level.name in level_names:
                            levels.append(level)
        return cls(levels, name)

    @classmethod
    def get_level_factory(cls, coupling_scheme: str):
        """Get a factory to build energy levels with a particular coupling scheme."""
        factories = {
            'ls': (cls.get_ls_fine_data, LSFineLevel, LSHyperfineLevel),
            'j1l2': (cls.get_j1l2_fine_data, J1L2FineLevel, J1L2HyperfineLevel),
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
        return {'S': 0, 'P': 1, 'D': 2, 'F': 3}[term_symbol[0]]

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

@dataclass(frozen=True, eq=False)
class MotionalMode(DegreeOfFreedom):
    """An normal mode of motion for a linear chain of ions."""
    energy_levels: list[CollectiveMotionalEnergyLevel]

    @classmethod
    def from_frequency(cls, frequency: float, fock_dimension: int, name: str | None = None):
        """Build a motional normal-mode degree of freedom for an ion chain."""
        levels = [CollectiveMotionalEnergyLevel(frequency, fock_number) for fock_number in range(fock_dimension)]
        return cls(levels, name)
