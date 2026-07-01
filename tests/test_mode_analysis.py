import unittest
import numpy as np
from scipy.linalg import expm
from ionsim.trapped_ion_mode_analysis import TrappedIonModeAnalysis, LinearIonChainAnalysis
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
                'wavenumber' : TPI / (355.0*1E-9), # 1/m
                'charge' : 1 
            },
            {
                'test name' : 'two ion 171Yb',
                'mass' : 170.936, # amu  
                'omega x' : 2. * TPI * 1E6,  # rad/s 
                'omega y' : 1.5 * TPI * 1E6,
                'omega z' : 0.5 * TPI * 1E6,
                'N' : 2, # number of ions 
                'wavenumber' : TPI / (355.0*1E-9), # 1/m
                'charge' : 1 
            }
        ]
        self.mode_analyzers = {
            case['test name'] : TrappedIonModeAnalysis(case['N'], case['omega x'], case['omega y'], case['omega z'], case['mass'], case['charge']) 
                for case in self.test_cases
            }

        # Compute Lamb-Dicke parameter analytical reference for each case  
        self.references = {}

        # 1. Single ion case: 
        k = self.test_cases[0]['wavenumber']
        eta_x, eta_y, eta_z = self.mode_analyzers['single ion 171Yb'].compute_reference_single_ion_lamb_dicke_factors(k)
        etas_analytical = np.zeros((3, 1, 3), dtype = np.complex128)
        etas_analytical[0, 0, 2] = eta_x
        etas_analytical[1, 0, 1] = eta_y
        etas_analytical[2, 0, 0] = eta_z
        self.references['single ion 171Yb'] = etas_analytical

        # 2. Two-ion case: 
        k = self.test_cases[1]['wavenumber']
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
            computed_Lamb_Dicke_parameters = case['wavenumber'] * mode_analyzer.calculate_mode_participation_factors()
            assert_array_close(np.abs(computed_Lamb_Dicke_parameters), np.abs(reference_values), atol = 1E-10, rtol=None)

    def test_linear_ion_chain_analysis(self):
        """Test the LinearIonChainAnalysis class with a 5-ion Yb+ chain."""
        # Create a 5-ion linear chain of Yb+ ions
        num_ions = 5
        TPI = 2*np.pi
        omega_x = TPI * 9e6  # 10 MHz
        omega_y = TPI * 10e6  # 10 MHz
        omega_z = TPI * 1e6   # 1 MHz (axial frequency typically lower)
        atomic_mass = 170.936  # Yb+ mass in amu
        atomic_charge = 1      

        # Create and analyze the linear chain
        chain = LinearIonChainAnalysis(num_ions, omega_x, omega_y, omega_z, atomic_mass, atomic_charge)
        chain.solve_ion_trap_equilibrium()
        #chain.print_chain_summary()

        # Test that we can get axial and radial modes
        axial_eigvals, axial_eigvecs = chain.get_axial_modes()
        radial_x_eigvals, radial_x_eigvecs = chain.get_radial_modes('x')
        radial_y_eigvals, radial_y_eigvecs = chain.get_radial_modes('y')

        # Verify shapes
        self.assertEqual(axial_eigvals.shape, (num_ions,))
        self.assertEqual(axial_eigvecs.shape, (6*num_ions, num_ions))
        self.assertEqual(radial_x_eigvals.shape, (num_ions,))
        self.assertEqual(radial_x_eigvecs.shape, (6*num_ions, num_ions))

        # Test that we can get ion spacing
        spacing = chain.get_ion_spacing()
        self.assertEqual(spacing.shape, (num_ions-1,))
        self.assertTrue(np.all(spacing > 0))  # All spacings should be positive

        # Test that we can get center of mass
        com_x, com_y, com_z = chain.get_center_of_mass_position()
        self.assertIsInstance(com_x, (float, np.floating))
        self.assertIsInstance(com_y, (float, np.floating))
        self.assertIsInstance(com_z, (float, np.floating))

        # Test that we can get mode frequencies
        axial_freqs = chain.get_axial_mode_frequencies()
        radial_x_freqs = chain.get_radial_mode_frequencies('x')
        radial_y_freqs = chain.get_radial_mode_frequencies('y')

        self.assertEqual(axial_freqs.shape, (num_ions,))
        self.assertEqual(radial_x_freqs.shape, (num_ions,))
        self.assertEqual(radial_y_freqs.shape, (num_ions,))

        # Test mode participation factors by branch
        mode_pf_by_branch = chain.get_mode_participation_factors_by_branch()
        self.assertIn('x', mode_pf_by_branch)
        self.assertIn('y', mode_pf_by_branch)
        self.assertIn('z', mode_pf_by_branch)

        # Test that frequencies are positive
        self.assertTrue(np.all(axial_freqs > 0))
        self.assertTrue(np.all(radial_x_freqs > 0))
        self.assertTrue(np.all(radial_y_freqs > 0))

        # Test Lamb-Dicke parameter calculations
        wavenumber = 2 * np.pi / (355.0 * 1e-9)  # example laser wavelength used with Yb+

        # Test full Lamb-Dicke parameter matrix
        wavevector = wavenumber * np.array([np.cos(np.pi/4.), np.sin(np.pi/4.), 0.]) # perpendicular to the axial chain 
        full_ld_params = chain.calculate_lamb_dicke_parameters_full(wavevector)
        num_modes = 3*num_ions
        self.assertEqual(full_ld_params.shape, (num_ions, num_modes))

        # Test branch-organized Lamb-Dicke parameters
        ld_by_branch = chain.calculate_lamb_dicke_parameters_by_branch(wavevector)
        self.assertIn('x', ld_by_branch)
        self.assertIn('y', ld_by_branch)
        self.assertIn('z', ld_by_branch)
        self.assertEqual(ld_by_branch['x'].shape, (num_ions, num_ions))
        self.assertEqual(ld_by_branch['y'].shape, (num_ions, num_ions))
        self.assertEqual(ld_by_branch['z'].shape, (num_ions, num_ions))

        # Test axial Lamb-Dicke parameters
        axial_ld = chain.get_axial_lamb_dicke_parameters(wavevector)
        self.assertEqual(axial_ld.shape, (num_ions, num_ions))

        # Test radial Lamb-Dicke parameters
        radial_x_ld = chain.get_radial_lamb_dicke_parameters(wavevector, 'x')
        radial_y_ld = chain.get_radial_lamb_dicke_parameters(wavevector, 'y')
        self.assertEqual(radial_x_ld.shape, (num_ions, num_ions))
        self.assertEqual(radial_y_ld.shape, (num_ions, num_ions))

        # Verify that branch-organized LD params match the corresponding parts of full matrix
        n_modes_per_branch = num_ions
        np.testing.assert_array_almost_equal(ld_by_branch['x'], full_ld_params[:, :n_modes_per_branch])
        np.testing.assert_array_almost_equal(ld_by_branch['y'], full_ld_params[:, n_modes_per_branch:2*n_modes_per_branch])
        np.testing.assert_array_almost_equal(ld_by_branch['z'], full_ld_params[:, 2*n_modes_per_branch:3*n_modes_per_branch])

        # Test that Lamb-Dicke parameters are related to mode participation factors
        mode_pf = chain.get_mode_participation_factors_by_branch()
        #print(mode_pf['x'])
        #print(ld_by_branch['z'])
        #mode_pf = chain.get_mode_participation_factors_by_branch()
        #expected_full_ld = wavenumber * mode_pf
        #np.testing.assert_array_almost_equal(full_ld_params, expected_full_ld)


if __name__ == '__main__':
    # Run the unit tests
    unittest.main()
