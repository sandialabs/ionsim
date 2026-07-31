#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

import unittest

import numpy as np

from ionsim.process import Gate, Circuit
from ionsim.degree_of_freedom import AtomicStructure
from ionsim.basis import StandardBasis
from ionsim.laser import Laser, Polarization 
from ionsim.named_operators import Unitary
from ionsim.noise import Noise

class TestProcess(unittest.TestCase):

    def setUp(self):
        """Set up the necessary objects for testing and test constructors."""
        levels = ['S1/2,0,0', 'S1/2,1,-1', 'S1/2,1,0', 'S1/2,1,1']
        self.atom_a = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=levels)
        #self.atom_a = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.atom_b = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=levels)
        self.basis = StandardBasis([self.atom_a, self.atom_b])

        propagation_vector = np.array([np.cos(np.pi/4.), np.sin(np.pi/4.), 0.])
        phase = np.pi 
        # Create polarization 
        laser_polarization = Polarization.circular(propagation_vector, '+')

        # Create Gaussian beam profile  
        wavelength = 355*1E-9 # nm -> meters 
        beam_waist = 20 * 1E-6 # µm -> meters 
        laser_power = 1e-3 # mWatt -> Watt  
        self.laser = Laser.gaussian_from_wavelength(wavelength, laser_power, beam_waist, propagation_vector, laser_polarization, phase) 

        # Test attributes:
        self.assertAlmostEqual(beam_waist, self.laser.beam_profile.waist, places=10)



    def test_laser_coupling_builder(self):
        """Test the process fidelity of the extra noisy gate."""
        print('test')
    #def build_atom_laser_coupling_operators(self, basis: Basis, ground_levels: list[AtomicInternalEnergyLevel], excited_levels: list[AtomicInternalEnergyLevel], 
    #                                            multipole_order: int, all_atoms_are_same: bool = True) -> list[Operator]: 
        # Test building coupling operators between 
        ground_levels = [self.atom_a.energy_levels[0]] 
        excited_levels = [*self.atom_a.energy_levels[1:]] 
        atom_a_coupling_operators = self.laser.build_individual_atom_laser_coupling_operators(self.basis, self.atom_a, ground_levels, excited_levels, 1) 
        all_atom_coupling_operators = self.laser.build_laser_coupling_operators_multiple_atoms(self.basis, [self.atom_a, self.atom_b], ground_levels, excited_levels, 1, True) 

        print(len(atom_a_coupling_operators))
        
        
if __name__ == '__main__':
    unittest.main()
