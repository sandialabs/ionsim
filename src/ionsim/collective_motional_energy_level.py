#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

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
    
