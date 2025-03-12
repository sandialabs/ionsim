from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence
import numpy as np

from icecream import ic

@dataclass(frozen=True, eq=False)
class EnergyLevel(ABC):
    """An energy level."""
    @property
    @abstractmethod
    def energy(self):
        """The energy of the level."""

    @property
    @abstractmethod
    def name(self):
        """A unique name for the energy level."""

@dataclass(frozen=True, eq=False)
class EnergyEigenstate(EnergyLevel): #TODO: consider renaming this something like "BasisState" to avoid confusion with the State class
    """An energy eigenstate with arbitrary degrees of freedom."""
    components: Sequence[EnergyLevel]

    @property
    def energy(self):
        """The energy of the state."""
        return np.sum([component.energy for component in self.components])

    @property
    def name(self):
        """A unique name for the state."""
        return ' : '.join([component.name for component in self.components])

def main():
    """Script to execute if module is ran directly."""
    from ionsim.atomic_internal_energy_level import build_internal_levels

    levels_a = build_internal_levels('171Yb+', ['S1/2'], ['S1/2,0,0', 'S1/2,1,0'])
    levels_b = build_internal_levels('171Yb+', ['S1/2'], ['S1/2,0,0', 'S1/2,1,0'])

    level = EnergyEigenstate([levels_a[0], levels_b[1]])
    ic(level)

if __name__ == '__main__':
    main()

