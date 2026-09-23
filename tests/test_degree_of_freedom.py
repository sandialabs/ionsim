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

from ionsim.degree_of_freedom import AtomicStructure, MotionalMode
from ionsim.atomic_internal_energy_level import LSHyperfineLevel, J1L2HyperfineLevel, compute_hyperfine_clebsch_gordan_coefficient, compute_multipole_amplitude
from ionsim.collective_motional_energy_level import CollectiveMotionalEnergyLevel

class TestDegreeOfFreedom(unittest.TestCase):

    def setUp(self):
        """Set up the necessary objects for testing."""
        self.spin_a = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2'])
        self.spin_b = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.spin_c = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', '[3/2]1/2'], level_names=['S1/2,0,0', 'S1/2,1,0', '[3/2]1/2,0,0'])
        neutral_171Yb_levels = ['S0,1/2,-1/2', 'S0,1/2,1/2', 'P1,1/2,-1/2', 'P1,1/2,1/2', 'P1,3/2,-3/2', 'P1,3/2,-1/2', 'P1,3/2,1/2', 'P1,3/2,3/2']
        self.Yb_atom = AtomicStructure.from_species(species='171Yb', term_symbols=['S0', 'P1'], level_names=neutral_171Yb_levels)
        neutral_171Yb_levels2 = ['S0,1/2,-1/2', 'S0,1/2,1/2', 'P0,1/2,-1/2', 'P0,1/2,1/2']
        self.Yb_atom2 = AtomicStructure.from_species(species='171Yb', term_symbols=['S0', 'P0'], level_names=neutral_171Yb_levels2)
        self.atom_a = AtomicStructure.from_species(species='87Rb', term_symbols=['S1/2', '6 P3/2', '53 S1/2'], level_names=['S1/2,1,0', 'S1/2,2,0', '6 P3/2,3,-1','53 S1/2,1,0'], magnetic_field = 0.01) 
        self.mode_0 = MotionalMode.from_frequency(frequency=3e6 * 2 * np.pi, fock_dimension=3)

        # Ca40+ 
        Ca_levels = ['S1/2,1/2','D3/2,3/2']
        self.Ca_ion = AtomicStructure.from_species(species='40Ca+', term_symbols=['S1/2', 'D3/2'], level_names=Ca_levels)

    def test_spin_a_energy_levels(self):
        """Test the energy levels of spin_a."""
        expected_levels_count = 8  # Based on the output for spin_a
        self.assertEqual(len(self.spin_a.energy_levels), expected_levels_count)

        # Check specific properties of the first energy level
        first_level = self.spin_a.energy_levels[0]
        self.assertIsInstance(first_level, LSHyperfineLevel)
        self.assertEqual(first_level.term_symbol, 'S1/2')
        self.assertAlmostEqual(first_level.hyperfine_A, 79437131344.39122, places=5)

    def test_spin_b_energy_levels(self):
        """Test the energy levels of spin_b."""
        expected_levels_count = 2  # Based on the output for spin_b
        self.assertEqual(len(self.spin_b.energy_levels), expected_levels_count)

        # Check specific properties of the first energy level
        first_level = self.spin_b.energy_levels[0]
        self.assertIsInstance(first_level, LSHyperfineLevel)
        self.assertEqual(first_level.term_symbol, 'S1/2')
        self.assertAlmostEqual(first_level.hyperfine_A, 79437131344.39122, places=5)

    def test_spin_c(self):
        """Test the energy levels of spin_c."""
        expected_levels_count = 3  # Based on the output for spin_c
        self.assertEqual(len(self.spin_c.energy_levels), expected_levels_count)

        # Check specific properties of the third energy level
        third_level = self.spin_c.energy_levels[2]
        self.assertIsInstance(third_level, J1L2HyperfineLevel)
        self.assertEqual(third_level.term_symbol, '[3/2]1/2')

    def test_neutral_atom_clebsch_gordan_coeffs(self):
        """Test the energy levels of spin_c."""
        expected_levels_count = 8  # Based on the output for spin_c
        self.assertEqual(len(self.Yb_atom.energy_levels), expected_levels_count)

        # Check specific properties of the third energy level
        third_level = self.Yb_atom.energy_levels[2]
        self.assertIsInstance(third_level, LSHyperfineLevel)
        self.assertEqual(third_level.term_symbol, 'P1')

        cg_coeffs = {}
        for ground_level in self.Yb_atom.energy_levels[0:2]:
            for excited_level in self.Yb_atom.energy_levels[2:]:
                cg_coeffs[(ground_level.name, excited_level.name)] = {} 
                for q in range(-1,2):
                    cg_coeffs[(ground_level.name, excited_level.name)][q] = compute_hyperfine_clebsch_gordan_coefficient(ground_level, excited_level, q, 1) 

        # Using this arxiv for CG coefficient tests: https://arxiv.org/pdf/2509.04416v1            
        self.assertAlmostEqual(cg_coeffs[('S0,1/2,-1/2','P1,1/2,-1/2')][0], -1/3, places = 8)
        self.assertAlmostEqual(cg_coeffs[('S0,1/2,-1/2','P1,1/2,-1/2')][1], 0., places = 8)
        # TODO: do we need to flip the q sign convention? 
        self.assertAlmostEqual(cg_coeffs[('S0,1/2,-1/2','P1,1/2,1/2')][-1], np.sqrt(2/9), places = 8)
        self.assertAlmostEqual(cg_coeffs[('S0,1/2,-1/2','P1,1/2,1/2')][-1], np.sqrt(2/9), places = 8)
        self.assertAlmostEqual(cg_coeffs[('S0,1/2,-1/2','P1,3/2,-1/2')][0], -np.sqrt(2/9), places = 8)

    def test_neutral_atom_clebsch_gordan_coeffs_quadrupole(self):
        """Test the energy levels of spin_c."""
        expected_levels_count = 4  # Based on the output for spin_c
        self.assertEqual(len(self.Yb_atom2.energy_levels), expected_levels_count)

        # Check specific properties of the third energy level
        cg_coeffs = {}
        for ground_level in self.Yb_atom2.energy_levels[0:2]:
            for excited_level in self.Yb_atom2.energy_levels[2:]:
                cg_coeffs[(ground_level.name, excited_level.name)] = {} 
                for q in range(-1,2):
                    cg_coeffs[(ground_level.name, excited_level.name)][q] = compute_hyperfine_clebsch_gordan_coefficient(ground_level, excited_level, q, 1) 
                    self.assertAlmostEqual(cg_coeffs[(ground_level.name, excited_level.name)][q], 0., places=8)

 #        print(self.Yb_atom2.energy_levels[2].name)
 #        cg_coeffs = {}
 #        for ground_level in self.Yb_atom2.energy_levels[0:2]:
 #            for excited_level in self.Yb_atom2.energy_levels[2:]:
 #                cg_coeffs[(ground_level.name, excited_level.name)] = {} 
 #                for q in range(-2,3):
 #                    cg_coeffs[(ground_level.name, excited_level.name)][q] = compute_hyperfine_clebsch_gordan_coefficient(ground_level, excited_level, q, 2) 
 #                    print((ground_level.name, excited_level.name))
 #                    self.assertAlmostEqual(cg_coeffs[(ground_level.name, excited_level.name)][q], 0., places=8)
 #                    print(cg_coeffs[(ground_level.name, excited_level.name)][q])

    def test_Ca40_transition_amplitudes(self):
        """Test the energy levels of spin_c."""
        expected_levels_count = 2  # Based on the output for spin_c
        self.assertEqual(len(self.Ca_ion.energy_levels), expected_levels_count)

        # Check specific properties of the third energy level
        # Should be no dipole allowed couplings between S1/2 and D3/2 
        cg_coeffs = {}
        for ground_level in [self.Ca_ion.energy_levels[0]]:
            for excited_level in [self.Ca_ion.energy_levels[1]]:
                cg_coeffs[(ground_level.name, excited_level.name)] = {} 
                for q in range(-1,2):
                    cg_coeffs[(ground_level.name, excited_level.name)][q] = compute_multipole_amplitude(ground_level, excited_level, 1, q, True) 
                    self.assertAlmostEqual(cg_coeffs[(ground_level.name, excited_level.name)][q], 0., places=8)

        cg_coeffs = {}
        # Should be a quadrupole allowed coupling  
        for ground_level in [self.Ca_ion.energy_levels[0]]:
            for excited_level in [self.Ca_ion.energy_levels[1]]:
                cg_coeffs[(ground_level.name, excited_level.name)] = {} 
                for q in range(-2,3):
                    #cg_coeffs[(ground_level.name, excited_level.name)][q] = compute_hyperfine_clebsch_gordan_coefficient(ground_level, excited_level, q, 2) 
                    cg_coeffs[(ground_level.name, excited_level.name)][q] = compute_multipole_amplitude(ground_level, excited_level, 2, q, True) 
                    if q == -1:
                        self.assertAlmostEqual(cg_coeffs[(ground_level.name, excited_level.name)][q], 0.2, places=8)
                    else:
                        self.assertAlmostEqual(cg_coeffs[(ground_level.name, excited_level.name)][q], 0., places=8)
    
    def test_Rydberg_Rb_atom(self):
        """Test the energy levels of spin_c."""
        expected_levels_count = 4  # Based on the output for atom_a 
        # Check specific properties of the fourth energy level
        fourth_level = self.atom_a.energy_levels[3]
        self.assertEqual(fourth_level.term_symbol, '53 S1/2')
        self.assertEqual(fourth_level.n, 53)
        
    def test_mode_0_energy_levels(self):
        """Test the energy levels of mode_0."""
        expected_levels_count = 3  # Based on the output for mode_0
        self.assertEqual(len(self.mode_0.energy_levels), expected_levels_count)

        # Check specific properties of the first energy level
        first_level = self.mode_0.energy_levels[0]
        self.assertIsInstance(first_level, CollectiveMotionalEnergyLevel)
        self.assertAlmostEqual(first_level.mode_frequency, 18849555.92153876, places=5)

if __name__ == '__main__':
    unittest.main()
