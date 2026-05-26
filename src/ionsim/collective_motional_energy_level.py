from ionsim.atomic_internal_energy_level import EnergyLevel

from abc import ABC
from dataclasses import dataclass
import numpy as np

from icecream import ic

@dataclass(frozen=True, eq=False)
class CollectiveMotionalEnergyLevel(EnergyLevel):
    """An energy level of a normal mode of motion of an ion chain."""
    mode_frequency: float
    fock_number: float # TODO: someday we'll need to make a motional basis of coherent states
    alias: str | None=None

    @property
    def energy(self):
        """Energy of the motional level."""
        return self.mode_frequency * (1/2 + self.fock_number)

    @property
    def name(self):
        """A unique name for the collective motional energy level."""
        return str(self.fock_number)
    
