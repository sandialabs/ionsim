import unittest

import numpy as np

from ionsim.degree_of_freedom import AtomicSpin, MotionalMode
from ionsim.atomic_internal_energy_level import LSHyperfineLevel, J1L2HyperfineLevel
from ionsim.collective_motional_energy_level import CollectiveMotionalEnergyLevel

class TestDegreeOfFreedom(unittest.TestCase):

    def setUp(self):
        """Set up the necessary objects for testing."""
        self.spin_a = AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2'])
        self.spin_b = AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.spin_c = AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2', '[3/2]1/2'], level_names=['S1/2,0,0', 'S1/2,1,0', '[3/2]1/2,0,0'])
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
