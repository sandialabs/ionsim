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
from ionsim.atomic_internal_energy_level import LSHyperfineLevel, J1L2HyperfineLevel
from ionsim.collective_motional_energy_level import CollectiveMotionalEnergyLevel
from ionsim.ionsim_error import IonSimError

class TestDegreeOfFreedom(unittest.TestCase):

    def setUp(self):
        """Set up the necessary objects for testing."""
        self.spin_a = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2'])
        self.spin_b = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.spin_c = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', '[3/2]1/2'], level_names=['S1/2,0,0', 'S1/2,1,0', '[3/2]1/2,0,0'])
        # Test Rydberg level parsing  
        self.atom_a = AtomicStructure.from_species(species='87Rb', term_symbols=['S1/2', '6 P3/2', '53 S1/2'], level_names=['S1/2,1,0', 'S1/2,2,0', '6 P3/2,3,-1','53 S1/2,1,0'], magnetic_field = 0.001) 

        # Alternative method using quantum numbers to specify the level 
        ground_level_dict = {'n' : 5, 'l' : 0, 's' : 0.5, 'j' : 1/2, 'f' : 1, 'mf' : 0}
        ground_level_dict2 = {'n' : 5, 'l' : 0, 's' : 0.5, 'j' : 1/2, 'f' : 2, 'mf' : 0}
        excited_level_dict = {'n' : 6, 'l' : 1, 's' : 0.5, 'j' : 3/2, 'f' : 3, 'mf' : -1}
        rydberg_level_dict = {'n' : 53, 'l' : 0, 's' : 0.5, 'j' : 1/2, 'mj' : -1/2, 'mi' : 1/2}
        rydberg_level2_dict = {'n' : 53, 'l' : 0, 's' : 0.5, 'j' : 1/2, 'mj' : +1/2, 'mi' : 1/2}
        quantum_numbers = [ground_level_dict, ground_level_dict2, excited_level_dict, rydberg_level_dict, rydberg_level2_dict]
        self.atom_b = AtomicStructure.from_species(species='87Rb', term_symbols=['S1/2', '6 P3/2', '53 S1/2'], quantum_numbers=quantum_numbers, magnetic_field = 13.6) 

        # Test motional mode 
        self.mode_0 = MotionalMode.from_frequency(frequency=3e6 * 2 * np.pi, fock_dimension=3)

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

    def test_Rydberg_Rb_atom(self):
        """Test the energy levels of spin_c."""
        expected_levels_count = 4  # Based on the output for atom_a 

        # Check specific properties of the fourth energy level
        fourth_level = self.atom_a.energy_levels[3]
        self.assertEqual(fourth_level.term_symbol, '53 S1/2')
        self.assertEqual(fourth_level.n, 53)

        # Check zeeman splitting of two rydberg levels in the high-field regime:
        r1 = self.atom_b.energy_levels[-2]
        r2 = self.atom_b.energy_levels[-1]
        energy_diff = r2.energy - r1.energy
        self.assertAlmostEqual(energy_diff/(2. * np.pi * 1E6), 38.18975921748056, places=8)
        
    def test_mode_0_energy_levels(self):
        """Test the energy levels of mode_0."""
        expected_levels_count = 3  # Based on the output for mode_0
        self.assertEqual(len(self.mode_0.energy_levels), expected_levels_count)

        # Check specific properties of the first energy level
        first_level = self.mode_0.energy_levels[0]
        self.assertIsInstance(first_level, CollectiveMotionalEnergyLevel)
        self.assertAlmostEqual(first_level.mode_frequency, 18849555.92153876, places=5)


class TestGetManifoldConfig(unittest.TestCase):
    """AtomicStructure.get_manifold_config returns one config-file entry, unconverted."""

    def test_returns_raw_entry(self):
        entry = AtomicStructure.get_manifold_config('87Rb', '53 S1/2')
        self.assertEqual(entry['term_symbol'], '53 S1/2')
        self.assertEqual(entry['n'], 53)
        # Values are as written in 87Rb.yaml (Hz), not converted to rad/s as on built levels.
        level = AtomicStructure.from_species(species='87Rb', term_symbols=['53 S1/2']).energy_levels[0]
        self.assertAlmostEqual(level.hyperfine_A, 2 * np.pi * entry['hyperfine_A'])

    def test_missing_manifold_lists_available(self):
        with self.assertRaises(IonSimError) as context:
            AtomicStructure.get_manifold_config('87Rb', 'P5/2')
        self.assertIn("'53 S1/2'", str(context.exception))

if __name__ == '__main__':
    unittest.main()
