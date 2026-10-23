import unittest
import numpy as np
import scipy.constants as const
from scipy.linalg import eigh
from ionsim.trapped_ion_mode_analysis import TrappedIonCrystal, IonChain
from ionsim.ionsim_error import IonSimError
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
            case['test name'] : TrappedIonCrystal(case['N'], case['omega x'], case['omega y'], case['omega z'], case['mass'], case['charge']) 
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
        # Two-ion tilt (rocking) modes have w_tilt^2 = w_r^2 - w_z^2, so eta_tilt = k sqrt(hbar / (2 m w_tilt)) / sqrt(2)
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
            # With the phase convention (largest component real and positive, ties -> lowest index), the
            # parameters are real and the relative signs of the reference (COM +/+, tilt/stretch +/-) hold.
            self.assertTrue(np.isrealobj(computed_Lamb_Dicke_parameters))
            assert_array_close(computed_Lamb_Dicke_parameters, np.real(reference_values), atol = 1E-10, rtol=None)

    def test_linear_ion_chain_analysis(self):
        """Test the IonChain class with a 5-ion Yb+ chain."""
        # Create a 5-ion linear chain of Yb+ ions
        num_ions = 5
        TPI = 2*np.pi
        omega_x = TPI * 9e6  # 10 MHz
        omega_y = TPI * 10e6  # 10 MHz
        omega_z = TPI * 1e6   # 1 MHz (axial frequency typically lower)
        atomic_mass = 170.936  # Yb+ mass in amu
        atomic_charge = 1      

        # Create and analyze the linear chain
        chain = IonChain(num_ions, omega_x, omega_y, omega_z, atomic_mass, atomic_charge)
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

        # Test that Lamb-Dicke parameters are related to mode participation factors: eta = k . r_zpf
        mode_pf = chain.calculate_mode_participation_factors()
        np.testing.assert_allclose(full_ld_params, np.einsum('d,dim->im', wavevector, mode_pf), rtol=1e-12, atol=0)


TPI = 2 * np.pi
YB_MASS = 170.936
BA_MASS = 137.905
BE_MASS = 9.012
MG_MASS = 24.986
K_355 = TPI / 355e-9


def reference_lamb_dicke(crystal, wavevector_trap):
    """ Independent reference: generalized symmetric eigenproblem K b = w^2 M b (mass-weighted, real). """
    K = crystal.hessian(crystal.equilibrium_positions)
    M = np.diag(np.tile(crystal.m, 3))
    w2, B = eigh(K, M)
    w = np.sqrt(w2)
    zpf = np.sqrt(const.hbar / (2 * crystal.mass_scale * crystal.trap_freq_scale * w))[np.newaxis, :] * B
    N = crystal.num_ions
    eta = sum(wavevector_trap[d] * zpf[d * N:(d + 1) * N, :] for d in range(3))
    return w * crystal.trap_freq_scale, eta


def rotation_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


class TestModeAnalysisPhysics(unittest.TestCase):

    def test_lamb_dicke_matches_mass_weighted_reference(self):
        """ |eta| and frequencies agree with an independent eigh solution, for single and mixed species. """
        k = K_355 * np.array([1., 1., 1.]) / np.sqrt(3)
        cases = [
            IonChain(3, TPI * 3e6, TPI * 3.3e6, TPI * 0.6e6, YB_MASS, 1),
            IonChain(3, TPI * 3e6, TPI * 3.3e6, TPI * 0.6e6, [YB_MASS, BA_MASS, YB_MASS], 1),
            IonChain(2, TPI * 8e6, TPI * 9e6, TPI * 2e6, [BE_MASS, MG_MASS], 1),
        ]
        for crystal in cases:
            crystal.solve_ion_trap_equilibrium()
            w_ref, eta_ref = reference_lamb_dicke(crystal, k)
            order = np.argsort(crystal.normal_mode_frequencies)
            np.testing.assert_allclose(crystal.normal_mode_frequencies[order], w_ref, rtol=1e-10)
            eta = crystal.calculate_lamb_dicke_parameters(k)[:, order]
            np.testing.assert_allclose(np.abs(eta), np.abs(eta_ref), rtol=1e-6, atol=1e-12)

    def test_two_species_axial_modes_analytic(self):
        """ Two-ion mixed-species axial modes: w^2 = wz^2 (1 + 1/mu -/+ sqrt(1 + 1/mu^2 - 1/mu)), mu = m2 / m1. """
        wz = TPI * 0.6e6
        chain = IonChain(2, TPI * 3e6, TPI * 3.3e6, wz, [YB_MASS, BA_MASS], 1)
        mu = BA_MASS / YB_MASS
        expected = wz * np.sqrt(1 + 1 / mu + np.array([-1, 1]) * np.sqrt(1 + 1 / mu**2 - 1 / mu))
        np.testing.assert_allclose(chain.get_axial_mode_frequencies(), expected, rtol=1e-10)

    def test_lamb_dicke_phase_convention(self):
        """ LD parameters are real, and in each mode the largest-magnitude position component is positive. """
        chain = IonChain(2, TPI * 3e6, TPI * 3.3e6, TPI * 0.6e6, YB_MASS, 1)
        k = K_355 * np.array([0., 0., 1.])
        eta = chain.get_axial_lamb_dicke_parameters(k)
        self.assertTrue(np.isrealobj(eta))
        # COM (lowest axial mode): both ions move in phase, positive
        self.assertTrue(np.all(eta[:, 0] > 0))
        # Stretch: equal magnitude tie broken by the lowest index -> ion 0 positive, ion 1 negative
        self.assertGreater(eta[0, 1], 0)
        self.assertLess(eta[1, 1], 0)
        np.testing.assert_allclose(eta[0, 1], -eta[1, 1], rtol=1e-10)

        # General rule over all modes of a mixed-species chain
        chain = IonChain(3, TPI * 3e6, TPI * 3.3e6, TPI * 0.6e6, [YB_MASS, BA_MASS, YB_MASS], 1)
        pf = chain.calculate_mode_participation_factors()
        self.assertTrue(np.isrealobj(pf))
        flat = pf.reshape(3 * chain.num_ions, -1)
        for n in range(flat.shape[1]):
            column = flat[:, n]
            largest = np.flatnonzero(np.abs(column) >= (1 - 1e-6) * np.abs(column).max())[0]
            self.assertGreater(column[largest], 0)

    def test_principal_axes_rotation(self):
        """ Rotating the trap and the wavevector together leaves eta unchanged; lab-frame factors are consistent. """
        k_trap = K_355 * np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
        R = rotation_matrix([1., 2., 0.5], 0.7)
        aligned = IonChain(3, TPI * 3e6, TPI * 3.3e6, TPI * 0.6e6, YB_MASS, 1)
        rotated = IonChain(3, TPI * 3e6, TPI * 3.3e6, TPI * 0.6e6, YB_MASS, 1, principal_axes=R)
        k_lab = R.T @ k_trap
        np.testing.assert_allclose(rotated.calculate_lamb_dicke_parameters(k_lab),
                                   aligned.calculate_lamb_dicke_parameters(k_trap), rtol=1e-9, atol=1e-14)
        pf_lab = rotated.calculate_mode_participation_factors(frame='lab')
        np.testing.assert_allclose(np.einsum('d,dim->im', k_lab, pf_lab),
                                   rotated.calculate_lamb_dicke_parameters(k_lab), rtol=1e-12, atol=1e-14)
        # positions in the lab frame lie along the lab-frame z principal axis
        x, y, z = np.split(rotated.return_equilibrium_positions(dimensionless=True, frame='lab'), 3)
        positions = np.vstack((x, y, z)).T
        np.testing.assert_allclose(np.cross(positions, R[2]), 0, atol=1e-9)
        # Non-normalized but orthogonal rows are accepted
        IonChain(1, TPI * 3e6, TPI * 3.3e6, TPI * 0.6e6, YB_MASS, 1, principal_axes=2 * R)
        with self.assertRaises(ValueError):
            IonChain(1, TPI * 3e6, TPI * 3.3e6, TPI * 0.6e6, YB_MASS, 1, principal_axes=[[1, 0, 0], [1, 1, 0], [0, 0, 1]])
        with self.assertRaises(ValueError):
            IonChain(1, TPI * 3e6, TPI * 3.3e6, TPI * 0.6e6, YB_MASS, 1, principal_axes=np.diag([1., 1., -1.]))

    def test_rf_paul_trap_species_scaling(self):
        """ Per-species frequencies: DC axial ~ sqrt(q/m), RF radial ~ q/m with DC defocusing. """
        wx, wy, wz = TPI * 3e6, TPI * 3.3e6, TPI * 0.6e6
        crystal = IonChain(2, wx, wy, wz, [YB_MASS, BA_MASS], 1)
        r = YB_MASS / BA_MASS
        np.testing.assert_allclose(crystal.omega_z[1], np.sqrt(r) * wz, rtol=1e-12)
        np.testing.assert_allclose(crystal.omega_x[1], np.sqrt(r**2 * (wx**2 + 0.5 * wz**2) - 0.5 * r * wz**2), rtol=1e-12)
        np.testing.assert_allclose(crystal.omega_y[1], np.sqrt(r**2 * (wy**2 + 0.5 * wz**2) - 0.5 * r * wz**2), rtol=1e-12)

        # Choosing the other species as the reference gives back the same trap
        swapped = IonChain(2, crystal.omega_x[1], crystal.omega_y[1], crystal.omega_z[1], [BA_MASS, YB_MASS], 1)
        np.testing.assert_allclose([swapped.omega_x[1], swapped.omega_y[1], swapped.omega_z[1]], [wx, wy, wz], rtol=1e-12)
        swapped.solve_ion_trap_equilibrium()
        np.testing.assert_allclose(np.sort(swapped.normal_mode_frequencies), np.sort(crystal.normal_mode_frequencies), rtol=1e-9)

        # Asymmetric DC split must satisfy Laplace, and 'surface' uses the same scaling laws
        surface = IonChain(2, wx, wy, wz, [YB_MASS, BA_MASS], 1, trap_type='surface', dc_radial_split=(0.8, 0.2))
        np.testing.assert_allclose(surface.omega_x[1], np.sqrt(r**2 * (wx**2 + 0.8 * wz**2) - 0.8 * r * wz**2), rtol=1e-12)
        with self.assertRaises(ValueError):
            IonChain(2, wx, wy, wz, YB_MASS, 1, dc_radial_split=(0.5, 0.6))

        # Custom per-ion frequencies are used as given
        custom = IonChain(2, [wx, 1.1 * wx], [wy, 1.1 * wy], [wz, 0.9 * wz], [YB_MASS, BA_MASS], 1, trap_type='custom')
        np.testing.assert_allclose(custom.omega_z, [wz, 0.9 * wz])
        with self.assertRaises(ValueError):
            IonChain(2, [wx, wx], wy, wz, YB_MASS, 1)  # arrays require trap_type='custom'

    def test_charges(self):
        """ Charge states enter the Coulomb interaction and the trap scaling. """
        wz = TPI * 0.6e6
        single = IonChain(2, TPI * 3e6, TPI * 3.3e6, wz, 40., 1)
        double = IonChain(2, TPI * 3e6, TPI * 3.3e6, wz, 40., 2)
        np.testing.assert_allclose(double.get_ion_spacing(), 4**(1 / 3) * single.get_ion_spacing(), rtol=1e-9)
        # COM and stretch frequencies are independent of the (common) charge
        np.testing.assert_allclose(double.get_axial_mode_frequencies(), [wz, np.sqrt(3) * wz], rtol=1e-9)

        # Mixed charges: analytic Hessian matches a finite-difference derivative of the force
        mixed = TrappedIonCrystal(3, TPI * 3e6, TPI * 3.3e6, wz, [40., 40., 40.], [1, 2, 1])
        u = np.array([0.1, -0.2, 0.05, 0.3, 0.0, -0.1, -1.2, 0.1, 1.4])
        h = 1e-6
        numerical = np.array([(mixed.force(u + h * e) - mixed.force(u - h * e)) / (2 * h) for e in np.eye(9)]).T
        np.testing.assert_allclose(mixed.hessian(u), numerical, rtol=1e-6, atol=1e-8)

        with self.assertRaises(IonSimError):
            IonChain(2, TPI * 3e6, TPI * 3.3e6, wz, 40., 0)
        with self.assertRaises(IonSimError):
            IonChain(2, TPI * 3e6, TPI * 3.3e6, wz, 40., [1, -1])

    def test_reindexing_consistency(self):
        """ All per-ion arrays are permuted together and the physics does not depend on the strategy. """
        masses = [BA_MASS, YB_MASS, YB_MASS]
        results = {}
        for strategy in ('none', 'distance', 'z_axis'):
            crystal = TrappedIonCrystal(3, TPI * 3e6, TPI * 3.3e6, TPI * 0.6e6, masses, 1, reindexing_strategy=strategy)
            crystal.solve_ion_trap_equilibrium()
            np.testing.assert_allclose(crystal.atomic_masses / const.u, np.array(masses)[crystal.ion_indices])
            # each stored ion keeps the frequencies of its species
            reference = TrappedIonCrystal(3, TPI * 3e6, TPI * 3.3e6, TPI * 0.6e6, masses, 1)
            np.testing.assert_allclose(crystal.omega_x, reference.omega_x[crystal.ion_indices])
            np.testing.assert_allclose(crystal.wz, reference.wz[crystal.ion_indices])
            results[strategy] = np.sort(crystal.normal_mode_frequencies)
        np.testing.assert_allclose(results['distance'], results['none'], rtol=1e-10)
        np.testing.assert_allclose(results['z_axis'], results['none'], rtol=1e-10)

        # IonChain: the user-supplied order is the order along +z, deterministically
        for order in ([BE_MASS, MG_MASS, BE_MASS], [MG_MASS, BE_MASS, BE_MASS]):
            chain = IonChain(3, TPI * 8e6, TPI * 9e6, TPI * 1e6, order, 1)
            chain.solve_ion_trap_equilibrium()
            np.testing.assert_allclose(chain.atomic_masses / const.u, order)
            np.testing.assert_array_equal(chain.ion_indices, [0, 1, 2])
            self.assertTrue(np.all(np.diff(chain.return_equilibrium_positions(True)[6:]) > 0))

    def test_build_mode_dofs(self):
        chain = IonChain(3, TPI * 3e6, TPI * 3.3e6, TPI * 0.6e6, YB_MASS, 1)
        modes = chain.build_mode_DOFs(list(range(9)), 4)
        self.assertEqual(len(modes), 9)
        for mode, frequency in zip(modes, chain.normal_mode_frequencies):
            self.assertEqual(len(mode.energy_levels), 4)
            np.testing.assert_allclose(mode.energy_levels[1].energy - mode.energy_levels[0].energy, frequency, rtol=1e-12)
        branch_modes = chain.build_mode_DOFs_from_branch('x', [0, 2], [3, 5])
        self.assertEqual([len(m.energy_levels) for m in branch_modes], [3, 5])
        np.testing.assert_allclose(branch_modes[1].energy_levels[1].energy - branch_modes[1].energy_levels[0].energy,
                                   chain.get_radial_mode_frequencies('x')[2], rtol=1e-12)
        with self.assertRaises(ValueError):
            chain.build_mode_DOFs([0, 1, 2], [3, 4])
        with self.assertRaises(ValueError):
            chain.build_mode_DOFs_from_branch('w', [0], 3)

    def test_find_axial_frequency_for_separation(self):
        chain = IonChain(2, TPI * 3e6, TPI * 3.3e6, TPI * 0.3e6, [YB_MASS, BA_MASS], 1)
        target = chain.get_two_central_ion_separation()
        wz = chain.find_axial_frequency_for_desired_central_ion_separation(target, bounds=(0.1e6, 0.5e6))
        np.testing.assert_allclose(wz, TPI * 0.3e6, rtol=1e-6)


if __name__ == '__main__':
    # Run the unit tests
    unittest.main()
