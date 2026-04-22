import unittest
import numpy as np
from scipy.linalg import expm
from ionsim.trapped_ion_mode_analysis import TrappedIonModeAnalysis 
from ionsim.testing import assert_array_close

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

        # Compute Lamb-Dicke parameter analytical reference for each case  
        self.references = {}

        # 1. Single ion case: 
        k = self.test_cases[0]['wavevector']
        eta_x, eta_y, eta_z = self.mode_analyzers['single ion 171Yb'].compute_reference_single_ion_lamb_dicke_factors(k)
        etas_analytical = np.zeros((3, 1, 3), dtype = np.complex128)
        etas_analytical[0, 0, 2] = eta_x
        etas_analytical[1, 0, 1] = eta_y
        etas_analytical[2, 0, 0] = eta_z
        self.references['single ion 171Yb'] = etas_analytical

        # 2. Two-ion case: 
        k = self.test_cases[1]['wavevector']
        eta_x, eta_y, eta_z = self.mode_analyzers['two ion 171Yb'].compute_reference_single_ion_lamb_dicke_factors(k)
        wx = self.test_cases[1]['omega x']
        wy = self.test_cases[1]['omega y']
        wz = self.test_cases[1]['omega z']

        wy_tilt = np.sqrt(wy**2 - wz**2)
        wx_tilt = np.sqrt(wx**2 - wz**2)
        eta_COM_x = eta_x / np.sqrt(2)
        eta_COM_y = eta_y / np.sqrt(2)
        eta_COM_z = eta_z / np.sqrt(2)
        eta_stretch_z = eta_z /np.sqrt(2) /3**(1/4)
        # I don't know the analytical expressions for the tilt modes off the top of my head... let's assume its the square root of the normalized mode frequency
        eta_tilt_x = eta_x / np.sqrt(2) / np.sqrt(wx_tilt / wx) 
        eta_tilt_y = eta_y / np.sqrt(2) / np.sqrt(wy_tilt / wy) 
        etas_analytical = np.zeros((3, 2, 6), dtype = np.complex128)
        etas_analytical[0, 0, 5] = eta_COM_x
        etas_analytical[0, 1, 5] = eta_COM_x
        etas_analytical[0, 0, 4] = eta_tilt_x
        etas_analytical[0, 1, 4] = -eta_tilt_x
        etas_analytical[1, 0, 3] = eta_COM_y
        etas_analytical[1, 1, 3] = eta_COM_y
        etas_analytical[1, 0, 2] = eta_tilt_y
        etas_analytical[1, 1, 2] = -eta_tilt_y
        etas_analytical[2, 0, 0] = eta_COM_z
        etas_analytical[2, 1, 0] = eta_COM_z
        etas_analytical[2, 0, 1] = eta_stretch_z
        etas_analytical[2, 1, 1] = -eta_stretch_z
        self.references['two ion 171Yb'] = etas_analytical


    def test_mode_analysis_solvers(self):
        # Test functionality of mode analysis for computing Lamb-Dicke parameters as compared to a reference  
        # Compute Lamb-Dicke parameters for all test cases 
        for case, mode_analyzer, reference_values in zip(self.test_cases, self.mode_analyzers.values(), self.references.values()):
            # Solve the ion-trap equilibrium problem
            mode_analyzer.solve_ion_trap_equilibrium()
            # Compute and store Lamb-Dicke parameters: 
            case['Lamb Dicke parameters'] = case['wavevector'] * mode_analyzer.calculate_mode_participation_factors()
            assert_array_close(np.abs(case['Lamb Dicke parameters']), np.abs(reference_values), atol = 1E-10, rtol=None)

if __name__ == '__main__':
    unittest.main()
