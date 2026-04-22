import unittest
import numpy as np
from scipy.linalg import expm
#from ionsim.zeeman_solver import ZeemanHyperfineSolver 
from ionsim.trapped_ion_mode_analysis import TrappedIonModeAnalysis 
from ionsim.testing import assert_array_close
from ionsim.degree_of_freedom import AtomicSpin

class TestModeAnalysis(unittest.TestCase):

    def setUp(self):
        TPI = 2*np.pi
        """Set up the necessary objects for testing."""
        # Set up tests with a list of dictionaries: 
        self.test_cases = [
            {
                'test name' : 'single ion 171Yb',
                'mass' : 170.936, # amu  
                'omega x' : 10. * TPI * 1E6,  # rad/s 
                'omega y' : 1.9 * TPI * 1E6,
                'omega z' : 0.5 * TPI * 1E6,
                'N' : 1, # number of ions 
                'wavevector' : TPI / (355.0*1E-9), # 1/m
                'Z' : 70 
            },
            {
                'test name' : 'two ion 171Yb',
                'mass' : 170.936, # amu  
                'omega x' : 2. * TPI * 1E6,  # rad/s 
                'omega y' : 1.5 * TPI * 1E6,
                'omega z' : 0.5 * TPI * 1E6,
                'N' : 2, # number of ions 
                'wavevector' : TPI / (355.0*1E-9), # 1/m
                'Z' : 70 
            }
        ]
        self.mode_analyzers = {
            case['test name'] : TrappedIonModeAnalysis(case['N'], case['omega x'], case['omega y'], case['omega z'], case['mass'], case['Z']) 
                for case in self.test_cases
            }

        # Test 3: Use AtomicSpin from_species()  
        self.spin = AtomicSpin.from_species(species='171Yb', term_symbols=['S0'], level_names=['S0,1/2,1/2', 'S0,1/2,-1/2'], magnetic_field = 400.) 

    def test_mode_analysis_solvers(self):
        # Test functionality of mode analysis for computing Lamb-Dicke parameters as compared to a reference  
        # Create tests for each solver. Each atom case has a list of tests:  
        #tests = {case['test name'] : [] for case in self.test_cases}


        # Compute Lamb-Dicke parameters for both test cases 
        for case, mode_analyzer in zip(self.test_cases, self.mode_analyzers):
            # Solve the ion-trap equilibrium problem
            mode_analyzer.solve_ion_trap_equilibrium()
            # Compute and store Lamb-Dicke parameters: 
            case['Lamb Dicke parameters'] = case['wavevector'] * mode_analyzer.calculate_mode_participation_factors()
            print(f"Lamb Dicke parameters: {case['Lamb Dicke parameters']}")


        # For magnetic field strength, verify energy shift for a state: 
 #        tests['171Yb'].append({'Magnetic field' : 400, # Gauss
 #                'state type' : 'hyperfine',
 #                'F,mF' : (0.5,-0.5),
 #                'Zeeman shift' : 149.98214416459987 # kHz
 #            })    
 #
 #        tests['171Yb'].append({'Magnetic field' : 400, # Gauss
 #                'state type' : 'hyperfine',
 #                'F,mF' : (0.5,0.5),
 #                'Zeeman shift' : -149.98214416459987 # kHz
 #            })        
 #
 #        # Loop through each solver and loop through each tests
 #        for name, solver in (self.solvers).items():
 #            test_list = tests[name] 
 #            for test in test_list:
 #                shifts, eigenvecs = solver.solve_at_field(test['Magnetic field'])
 #                if test['state type'] == 'hyperfine':
 #                    f, mF = test['F,mF']
 #                    calculated_value = solver.get_state_energy(shifts, eigenvecs, f = f, mf = mF)
 #                else:
 #                    mj, mi = test['mJ,mI']
 #                    calculated_value = solver.get_state_energy_from_mjmi_pair(shifts, eigenvecs, mj = mj, mi = mi)
 #
 #                self.assertAlmostEqual(calculated_value, test['Zeeman shift'], places=6) 


 #    def test_AtomicSpin_ZeemanShift(self): 
 #        """Test the Zeemaen shift functionality within AtomicSpin class."""
 #        expected_frequency = 149.98214416459987*2. # kHz
 #        qubit_frequency = self.spin.energy_levels[1].energy - self.spin.energy_levels[0].energy
 #        qubit_frequency /= (2.* np.pi) # convert from rad/s to Hz 
 #        self.assertAlmostEqual(expected_frequency, np.abs(qubit_frequency)*1E-3, places=6)


if __name__ == '__main__':
    unittest.main()
