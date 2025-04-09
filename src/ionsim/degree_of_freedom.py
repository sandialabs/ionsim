from ionsim.energy_level import EnergyLevel
from ionsim.atomic_internal_energy_level import AtomicInternalEnergyLevel
from ionsim.atomic_internal_energy_level import LSFineLevel, LSHyperfineLevel, J1L2FineLevel, J1L2HyperfineLevel
from ionsim.collective_motional_energy_level import CollectiveMotionalEnergyLevel

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
class AtomicSpin(DegreeOfFreedom):
    """An atomic spin degree of freedom, i.e. its coupled spin and orbital angular momentum."""
    energy_levels: list[AtomicInternalEnergyLevel]

    @classmethod
    def from_species(cls, species: str, term_symbols: list[str] | None = None, level_names: list[str] | None = None,
            name: str | None = None):
        """Build the atomic spin degree of freedom for a particular species of atom."""
        config_data = cls.get_config_data(species)
        nuclear_spin = config_data['nuclear_spin']
        levels_data = config_data['levels']
        structure = 'fine' if nuclear_spin == 0 else 'hyperfine'

        if term_symbols is not None:
            levels_data = cls.select_some_data(term_symbols, levels_data)

        levels = []
        for level_data in levels_data:

            # TODO: add a unique term_symbol and corresponding branching ratios!
            level_data['unique_term_symbol'] = level_data['term_symbol']
            level_data['unique_branching_ratios'] = level_data.get('branching_ratios', None)
            # level_data['unique_term_symbol'] = _get_unique_term_symbol(level_data, levels_data)

            get_fine_data, FineLevel, HyperfineLevel = cls.get_level_factory(level_data['coupling_scheme'])
            fine_data = get_fine_data(level_data)
            j = fine_data['j']

            # ic(FineLevel.__annotations__, FineLevel(**fine_data, mj={'key': 1}))
            # TODO: why is mypy not catching this??

            if structure == 'fine':
                for mj in np.arange(-j, j + 1):
                    level = FineLevel(**fine_data, mj=mj)
                    if level_names is None or level.name in level_names: 
                        levels.append(level)
            else:
                for f in np.arange(-(j + nuclear_spin), j + nuclear_spin + 1):
                    for mf in np.arange(-f, f + 1):
                        level = HyperfineLevel(**fine_data, i=nuclear_spin, f=f, mf=mf)
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

def main():
    """Script to execute if module is ran directly."""

    # spin_a = AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2'])
    # ic(spin_a)

    # spin_b = AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
    # ic(spin_b)

    mode_0 = MotionalMode.from_frequency(frequency=3e6*2*np.pi, fock_dimension=3)
    ic(mode_0)


if __name__ == "__main__":
    main()

