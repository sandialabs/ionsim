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
from scipy import constants as const 

from ionsim.process import Gate, Circuit
from ionsim.degree_of_freedom import AtomicStructure
from ionsim.basis import StandardBasis
from ionsim.laser import Laser, Polarization 
from ionsim.named_operators import Unitary
from ionsim.noise import Noise

class TestProcess(unittest.TestCase):

    def setUp(self):
        """Set up the necessary objects for testing and test constructors."""
        a_levels = ['S1/2,0,0', 'S1/2,1,-1', 'S1/2,1,0', 'S1/2,1,1']
        #levels = ['S1/2,1,0', 'P1/2,1,-1', 'P1/2,1,0', 'P1/2,1,1']
        self.atom_a = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=a_levels)
        #self.atom_a = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        b_levels = ['S1/2,1,0', 'P1/2,1,-1', 'P1/2,1,0', 'P1/2,1,1']
        self.atom_b = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2'], level_names=b_levels)
        self.basis = StandardBasis([self.atom_a, self.atom_b])

        c_levels = ['S0,1/2,1/2', 'P1,3/2,3/2']
        self.atom_c = AtomicStructure.from_species(species='171Yb', term_symbols=['S0', 'P1'], level_names=c_levels) # 3P1
        self.neutral_basis = StandardBasis([self.atom_c]) 

        propagation_vector = np.array([0., 0., 1.])
        #propagation_vector = np.array([np.cos(np.pi/4.), np.sin(np.pi/4.), 0.])
        phase = np.pi 
        # Create polarization 
        laser_polarization = Polarization.circular(propagation_vector, '+')

        # Create Gaussian beam profile  
        wavelength = 355*1E-9 # nm -> meters 
        beam_waist = 3 * 1E-6 # 3 µm -> meters 
        laser_power = 20e-3 # 20 mWatt -> Watt  
        self.laser = Laser.gaussian_from_wavelength(wavelength, laser_power, beam_waist, propagation_vector, laser_polarization, phase) 

        wavelength_laser_c = 556*1E-9 # nm -> meters
        laser_c_power = 1e-3 # Watt 
        laser_c_waist = (1e-3)/2 # m 
        laser_c_polarization = Polarization.linear(propagation_vector, angle=0.)
        phase = np.pi
        self.laser_c = Laser.gaussian_from_wavelength(wavelength_laser_c, laser_c_power, laser_c_waist, propagation_vector, laser_c_polarization, phase) 

        # Test attributes:
        self.assertAlmostEqual(beam_waist, self.laser.beam_profile.waist, places=10)
        # Expect about 1.4kV/m 
        self.assertAlmostEqual(self.laser_c.peak_electric_field_magnitude*1E-3, 1.3851612653205, places=10)
        self.assertAlmostEqual(self.laser_c.beam_profile.peak_electric_field_magnitude(laser_c_power)*1E-3, 1.3851612653205, places=10)

    def test_laser_coupling_builder(self):
        """Test the process fidelity of the extra noisy gate."""
        # Test building coupling operators between 
        ground_levels = [self.atom_a.energy_levels[0]] 
        excited_levels = [*self.atom_a.energy_levels[1:]] 
        atom_a_coupling_operators = self.laser.build_individual_atom_laser_coupling_operators(self.basis, self.atom_a, ground_levels, excited_levels, 1) 
        self.assertEqual(len(atom_a_coupling_operators), 0)

        ground_levels = [self.atom_b.energy_levels[0]] 
        excited_levels = [*self.atom_b.energy_levels[1:]] 
        atom_b_coupling_operators = self.laser.build_individual_atom_laser_coupling_operators(self.basis, self.atom_b, ground_levels, excited_levels, 1) 
        #all_atom_coupling_operators = self.laser.build_laser_coupling_operators_multiple_atoms(self.basis, [self.atom_a, self.atom_b], ground_levels, excited_levels, 1, True) 

 #        print(len(atom_a_coupling_operators))
 #        for op in atom_a_coupling_operators:
 #            print(f"Coupling operator contains {len(op.couplings)} couplings.")
 #            for coupling in op.couplings:
 #                print(f"Coupling between {coupling.row_state.name} and {coupling.column_state.name}.")
 #                print(f"Strength: {coupling.strength}\n")
 #            print()
 #
 #        print(len(atom_b_coupling_operators))
 #        for op in atom_b_coupling_operators:
 #            print(f"Coupling operator contains {len(op.couplings)} couplings.")
 #            for coupling in op.couplings:
 #                print(f"Coupling between {coupling.row_state.name} and {coupling.column_state.name}.")
 #                print(f"Strength: {coupling.strength}\n")
 #            print()


    def test_1S0_3P1_transition_coupling_from_laser(self):
        """ Following the reference https://arxiv.org/pdf/2509.04416v1, testing rabi frequency for 556 nm laser on 3P1 transition"""
        TPI = 2.*np.pi
        ground_levels = [self.atom_c.energy_levels[0]] 
        excited_levels = [self.atom_c.energy_levels[1]] 
        energy_difference = excited_levels[0].energy - ground_levels[0].energy 

        laser_detuning = self.laser_c.detuning_from_transition_frequency(energy_difference) 
        laser_detuning_v2 = self.laser_c.detuning_from_level_transition(ground_levels[0], excited_levels[0]) 
        self.assertEqual(laser_detuning, laser_detuning_v2)
        expected_detuning = (const.c / self.laser_c.wavelength)*2.*np.pi - energy_difference
        self.assertAlmostEqual(laser_detuning, expected_detuning, places=10)
        # Detuning should be negative because F=3/2 is shifted above the fine energy 
        self.assertAlmostEqual(laser_detuning/TPI/1E9, -195.33742087632817, places=8) 

        # Expecting 5.523 MHz Rabi frequency  
        atom_c_coupling_operators = self.laser_c.build_individual_atom_laser_coupling_operators(self.neutral_basis, self.atom_c, ground_levels, excited_levels, 1) 
        assert len(atom_c_coupling_operators) == 1

        # We expect a Rabi frequency of about ~5 MHz, but this is not precise  
        dipole_factor = 0.5398 # Table 10 of reference 
        rabi_frequency = dipole_factor * atom_c_coupling_operators[0].couplings[0].strength/TPI/1E6
        self.assertAlmostEqual(np.abs(rabi_frequency), 3.905827783900461, places = 8) 

        
        
if __name__ == '__main__':
    unittest.main()
