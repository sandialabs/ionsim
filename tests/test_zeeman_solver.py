import unittest
import numpy as np
from scipy.linalg import expm
from ionsim.zeeman_solver import Zeeman_Hyperfine_Solver 
from ionsim.testing import assert_array_close
from ionsim.degree_of_freedom import AtomicSpin

class TestZeemanSolver(unittest.TestCase):

    def setUp(self):
        """Set up the necessary objects for testing."""
        # Test 1: Manually specifying parameters, e.g. Rubidium 87
        J = 0.5; I = 1.5; L = 0.; S = 0.5 
        A_hf = 3417.34 * 1E-3 # GHz 
        mass = 86.909 # Daltons 
        gI = -0.0009951414 
        Z = 37
        frequency_units = 'GHz'
        
        self.solver_Rb87 = Zeeman_Hyperfine_Solver(I, J, L, S, A_hf, mass, Z = Z, gI = gI, freq_units = frequency_units)


        # Test 2: Neutral atom 171Yb  
        # 171Yb example
        J = 0
        S = 0
        I = 0.5
        Z = 70
        L = 0.
        mass = 170.936323 # Daltons
        magnetic_moment = 0.4919 # Nuclear magneton \mu_{N} units
        hyperfine_A = 0. 
        
        self.solver_Yb171 = Zeeman_Hyperfine_Solver(I, J, L, S, hyperfine_A, mass, Z = Z, nuclear_moment = magnetic_moment, freq_units = "kHz", mode='weak field')
        # Test 3: Use AtomicSpin from_species()  
        self.spin = AtomicSpin.from_species(species='171Yb', term_symbols=['S0'], level_names=['S0,1/2,1/2', 'S0,1/2,-1/2'], magnetic_field = 400.) 


    def test_Zeeman_shift_Rb87(self):
        """Test the Zeemaen shift for Rb87 in the ground manifold"""
        B_value = 2000 # Gauss 
        shifts, eigvecs = self.solver_Rb87.solve_at_field(B_value)
        
        zeeman_shift = self.solver_Rb87.get_state_energy(shifts, eigvecs, F = 1, m_F = 1.0)
        zeeman_shift_2 = self.solver_Rb87.get_state_energy_from_mJmI_pair(shifts, eigvecs, mJ = 0.5, mI = 1.5)
        expected_results = np.array([4.539345, 5.361322])
        assert_array_close(np.array([zeeman_shift, zeeman_shift_2]), expected_results) 


    def test_Yb171_neutral_frequency(self):
        """Test the Zeemaen shift for 171Yb in the nuclear ground manifold"""

        expected_frequency = 149.98214416459987*2. # kHz
        B_field = 400 # Gauss 
        shifts, eigvecs = self.solver_Yb171.solve_at_field(B_field)
        F_labels, mF_labels = self.solver_Yb171.F_mF_labels(eigvecs)
        
        zeeman_shift_minus = self.solver_Yb171.get_state_energy(shifts, eigvecs, F = 0.5, m_F = -0.5)
        zeeman_shift_plus = self.solver_Yb171.get_state_energy(shifts, eigvecs, F = 0.5, m_F = +0.5)

        qubit_frequency = np.abs(zeeman_shift_plus - zeeman_shift_minus)
  
        self.assertAlmostEqual(np.abs(qubit_frequency),expected_frequency, places=6) 
 

    def test_AtomicSpin_ZeemanShift(self): 
        """Test the Zeemaen shift functionality within AtomicSpin class."""
        expected_frequency = 149.98214416459987*2. # kHz
        qubit_frequency = self.spin.energy_levels[1].energy - self.spin.energy_levels[0].energy
        qubit_frequency /= (2.* np.pi) # convert from rad/s to Hz 
        self.assertAlmostEqual(expected_frequency, np.abs(qubit_frequency)*1E-3, places=6)



if __name__ == '__main__':
    unittest.main()
