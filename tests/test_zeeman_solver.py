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
from scipy.linalg import expm
from ionsim.zeeman_solver import ZeemanHyperfineSolver 
from ionsim.testing import assert_array_close
from ionsim.degree_of_freedom import AtomicSpin

class TestZeemanSolver(unittest.TestCase):

    def setUp(self):
        """Set up the necessary objects for testing."""
        # Set up tests with a list of dictionaries: 
        self.test_cases = [
            {
                'test_name' : '87Rb',
                'J' : 0.5, 
                'I' : 1.5, 
                'L' : 0,
                'S' : 0.5,
                'hyperfine_A' : 3417.34 * 1E-3, # GHz
                'mass' : 86.909, #daltons 
                'gI' : -0.0009951414,
                'Z' : 37, 
                'frequency units' : 'GHz',
                'approximation' : None,
                'magnetic moment' : None
            },
            {
                'test_name' : '171Yb',
                'J' : 0.0, 
                'I' : 0.5, 
                'L' : 0,
                'S' : 0.0,
                'hyperfine_A' : 0., # 
                'mass' : 170.936323, #daltons 
                'gI' : None,
                'Z' : 70, 
                'frequency units' : 'kHz',
                'approximation' : 'weak field',
                'magnetic moment' : 0.4919
            }
        ]
        self.solvers = {
            case['test_name'] : ZeemanHyperfineSolver(
                case['I'], case['J'], case['L'], case['S'], case['hyperfine_A'], case['mass'], z = case['Z'], nuclear_moment = case['magnetic moment'],
                gi = case['gI'], freq_units = case['frequency units'], approximation = case['approximation'])
                for case in self.test_cases
            }

        # Test 3: Use AtomicSpin from_species()  
        self.spin = AtomicSpin.from_species(species='171Yb', term_symbols=['S0'], level_names=['S0,1/2,1/2', 'S0,1/2,-1/2'], magnetic_field = 400.) 

    def test_explicit_zeeman_solvers(self):
        # Test functionality when using explicit construction of Zeeman Solver objects for each test case. 
        # Create tests for each solver. Each atom case has a list of tests:  
        tests = {case['test_name'] : [] for case in self.test_cases}

        # For magnetic field strength, verify energy shift for a state: 
       tests['87Rb'].append({'Magnetic field' : 2000, # Gauss
                'state type' : 'hyperfine',
                'F,mF' : (1,1.0),
                'Zeeman shift' : -6.253586054112444  
            })

        tests['87Rb'].append({'Magnetic field' : 2000, # Gauss
            'state type' : 'high field',
            'mJ,mI' : (0.5, 1.5),
            'Zeeman shift' : 5.361322
        })

        tests['171Yb'].append({'Magnetic field' : 400, # Gauss
                'state type' : 'hyperfine',
                'F,mF' : (0.5,-0.5),
                'Zeeman shift' : 149.98214416459987 # kHz
            })    

        tests['171Yb'].append({'Magnetic field' : 400, # Gauss
                'state type' : 'hyperfine',
                'F,mF' : (0.5,0.5),
                'Zeeman shift' : -149.98214416459987 # kHz
            })        

        # Loop through each solver and loop through each tests
        for name, solver in (self.solvers).items():
            test_list = tests[name] 
            for test in test_list:
                shifts, eigenvecs = solver.solve_at_field(test['Magnetic field'])
                if test['state type'] == 'hyperfine':
                    f, mF = test['F,mF']
                    calculated_value = solver.get_state_energy(shifts, eigenvecs, f = f, mf = mF)
                else:
                    mj, mi = test['mJ,mI']
                    calculated_value = solver.get_state_energy_from_mjmi_pair(shifts, eigenvecs, mj = mj, mi = mi)

                self.assertAlmostEqual(calculated_value, test['Zeeman shift'], places=6) 


    def test_AtomicSpin_ZeemanShift(self): 
        """Test the Zeemaen shift functionality within AtomicSpin class."""
        expected_frequency = 149.98214416459987*2. # kHz
        qubit_frequency = self.spin.energy_levels[1].energy - self.spin.energy_levels[0].energy
        qubit_frequency /= (2.* np.pi) # convert from rad/s to Hz 
        self.assertAlmostEqual(expected_frequency, np.abs(qubit_frequency)*1E-3, places=6)


if __name__ == '__main__':
    unittest.main()
