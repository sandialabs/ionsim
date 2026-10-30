#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

import unittest
import math
import numpy as np

from ionsim.state import State
from ionsim.degree_of_freedom import AtomicStructure, MotionalMode
from ionsim.basis import StandardBasis, XPauliBasis
from ionsim.named_operators import Pauli
from ionsim.testing import assert_array_close
from ionsim.ionsim_error import IonSimError

class TestState(unittest.TestCase):

    def setUp(self):
        """Set up the necessary objects for testing."""
        self.atomic_structure_a = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.atomic_structure_b = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.basis = StandardBasis([self.atomic_structure_a, self.atomic_structure_b])
        self.basis_x = XPauliBasis([self.atomic_structure_a, self.atomic_structure_b])
        self.eig_x = np.array([[1, 0], [0, -1]])
        self.state = State.from_density_matrix(self.basis_x, np.kron(self.eig_x, Pauli.I))

        # Set up atomic structure - motional state with 2 structure and 1 motional mode  
        fock_dim = 5
        self.mode = MotionalMode.from_frequency(frequency=2*np.pi * 3e6, fock_dimension=fock_dim)
        self.atomic_structure_motion_basis = StandardBasis([self.atomic_structure_a, self.atomic_structure_b, self.mode])
        state_coefficients = np.zeros(len(self.atomic_structure_motion_basis.states))
        state_coefficients[0] = 1.
        self.atomic_structure_motion_state = State.from_coefficients(self.atomic_structure_motion_basis, list(state_coefficients))

        # Test a coherent state at alpha = 1 to test motional displacement calculation  
        fock_dim = 12
        self.mode2 = MotionalMode.from_frequency(frequency=2*np.pi * 3e6, fock_dimension=fock_dim)
        self.atomic_structure_motion_basis2 = StandardBasis([self.atomic_structure_a, self.atomic_structure_b, self.mode2])
        state_coefficients = np.zeros(len(self.atomic_structure_motion_basis2.states))
        state_coefficients[0] = 1.
        for n in range(fock_dim):
            state_coefficients[n] = np.sqrt(np.exp(-1))/np.sqrt(math.factorial(n))  # |00>; n
        self.atomic_structure_coherent_state = State.from_coefficients(self.atomic_structure_motion_basis2, list(state_coefficients))

    def test_project_out_levels_builds_reduced_basis(self):
        """project_out_levels without new_basis builds the subspace basis and keeps the remaining amplitudes."""
        atom = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2'], level_names=['S1/2,0,0', 'S1/2,1,0', 'P1/2,1,0'])
        g0, g1, e = atom.energy_levels
        basis = StandardBasis([atom])
        wavefunction = np.array([0.6, 0.0, 0.8], dtype=complex)
        projected = State.from_wavefunction(basis, wavefunction).project_out_levels([e])
        self.assertEqual([state.components for state in projected.basis.states], [[g0], [g1]])
        assert_array_close(projected.wavefunction, wavefunction[:2])

    def test_project_to_qubit_subspace(self):
        """project_to_qubit_subspace returns self in a qubit basis and projects out non-qubit levels otherwise."""
        qubit_state = State.from_coefficients(self.basis, [1., 0., 0., 0.])
        self.assertIs(qubit_state.project_to_qubit_subspace([]), qubit_state)

        atom = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2'], level_names=['S1/2,0,0', 'S1/2,1,0', 'P1/2,1,0'])
        basis = StandardBasis([atom])
        projected = State.from_coefficients(basis, [1., 0., 0.]).project_to_qubit_subspace([atom.energy_levels[2]])
        self.assertTrue(projected.basis.is_qubit_basis)
        assert_array_close(projected.wavefunction, np.array([1., 0.]))

    def test_wavefunction_with_new_component(self):
        """Test adding new-DoF components at the end, in the middle, and for several DoFs at once."""
        spin_psi = np.array([0.6, 0.8j])
        other_spin_psi = np.array([1, -1j])/np.sqrt(2)
        mode_psi = np.zeros(len(self.mode.energy_levels)); mode_psi[1] = 1.

        # Add new dof last
        basis = StandardBasis([self.atomic_structure_a, self.atomic_structure_b, self.mode])
        current = np.kron(spin_psi, other_spin_psi)
        state = State.from_wavefunction_with_new_component(basis, current, mode_psi, self.mode)
        assert_array_close(state.wavefunction, np.kron(current, mode_psi))

        # Add a new dof in the middle
        basis = StandardBasis([self.atomic_structure_a, self.mode, self.atomic_structure_b])
        state = State.from_wavefunction_with_new_component(basis, current, mode_psi, self.mode)
        assert_array_close(state.wavefunction, np.kron(np.kron(spin_psi, mode_psi), other_spin_psi))

        # Several new dofs; the dict order does not need to match the basis ordering.
        expected = np.kron(np.kron(spin_psi, mode_psi), other_spin_psi)
        for new_components in [
            {self.mode: mode_psi, self.atomic_structure_a: spin_psi},
            {self.atomic_structure_a: spin_psi, self.mode: mode_psi},
        ]:
            state = State.from_wavefunction_with_new_components(basis, other_spin_psi, new_components)
            assert_array_close(state.wavefunction, expected)

        # Two new dofs with equal dimension but different wavefunctions are placed by key, not by position.
        mode_b = MotionalMode.from_frequency(frequency=2*np.pi * 2.5e6, fock_dimension=len(self.mode.energy_levels))
        mode_b_psi = np.zeros(len(mode_b.energy_levels)); mode_b_psi[0] = 1.
        basis = StandardBasis([self.atomic_structure_a, self.mode, mode_b])
        state = State.from_wavefunction_with_new_components(
            basis, spin_psi, {mode_b: mode_b_psi, self.mode: mode_psi}
        )
        assert_array_close(state.wavefunction, np.kron(np.kron(spin_psi, mode_psi), mode_b_psi))

    def test_wavefunction_with_new_component_errors(self):
        """Test that inconsistent inputs raise an IonSimError."""
        basis = StandardBasis([self.atomic_structure_a, self.atomic_structure_b, self.mode])
        current = np.kron([1., 0.], [1., 0.])
        mode_psi = np.zeros(len(self.mode.energy_levels)); mode_psi[0] = 1.
        with self.assertRaises(IonSimError):  # no new dofs
            State.from_wavefunction_with_new_components(basis, current, {})
        with self.assertRaises(IonSimError):  # dof not in basis
            State.from_wavefunction_with_new_component(basis, current, mode_psi, self.mode2)
        with self.assertRaises(IonSimError):  # component has the wrong length for its dof
            State.from_wavefunction_with_new_component(basis, current, mode_psi[:-1], self.mode)
        with self.assertRaises(IonSimError):  # current wavefunction has the wrong length
            State.from_wavefunction_with_new_component(basis, current[:2], mode_psi, self.mode)

    def test_density_matrix_in_new_basis(self):
        """Test the density matrix in the new basis."""
        rho_p = State.from_state(self.basis, self.state).density_matrix
        expected_rho_p = np.kron(Pauli.X, Pauli.I)

        # Check that the density matrix matches the expected value
        assert_array_close(rho_p, expected_rho_p)

    def test_wigner_function(self):
        """ Test Wigner function computation from a motional density matrix in the Fock state basis""" 
        domain_limit = 2.
        size = 25 
        x_grid = np.linspace(-domain_limit, domain_limit, size)
        p_grid = np.linspace(-domain_limit, domain_limit, size)
        W_distribution = self.atomic_structure_motion_state.compute_wigner_distribution(x_grid, p_grid)[0]        

        # Test just one slice of the array and half of its contents (U(1) symmetric about origin)) 
        slice_indx = 12
        W_expected_at_slice = np.array([0.00583005, 0.0110443, 0.0197914, 0.03354962, 0.05379861, 0.08160694, 0.11709966, 0.15894861, 0.20409406, 0.24789999, 0.2848362, 0.30958962, 0.31830989]) 
        norm = np.trapezoid(np.trapezoid(W_distribution, x_grid, axis=0), p_grid)
        expected_norm = 0.9902872798196112
        self.assertAlmostEqual(norm, expected_norm, places=7)
        expected_W_max = 0.31830988618379075
        W_max = np.max(W_distribution)
        self.assertAlmostEqual(W_max, expected_W_max, places=7)
        assert_array_close(W_distribution[slice_indx, 0:13], W_expected_at_slice, rtol=1e-04, atol=1E-7) 

    def test_characteristic_function(self):
        """ Test Wigner function computation from a motional density matrix in the Fock state basis""" 
        domain_limit = 2.
        size = 25 
        x_grid = np.linspace(-domain_limit, domain_limit, size)
        p_grid = np.linspace(-domain_limit, domain_limit, size)
        re_beta, im_beta, chi_distributions = self.atomic_structure_motion_state.compute_characteristic_functions(x_grid, p_grid)
        chi = chi_distributions[0].reshape(size, size)

        # Test just one slice of the array and half of its contents (U(1) symmetric about origin)) 
        chi_max = np.max(chi)
        expected_max = 0.9937179749826316 
        #print(chi[slice_indx, 0:13])
        self.assertAlmostEqual(chi_max, expected_max, places=7)

    def test_quadrature_computation(self):
        """ Test computation of x, p, and x^2, and p^2 expectations """ 
        include_squared_quantities = True
        # method 1
        use_partial_trace = False 
        x, p, x2, p2 = self.atomic_structure_motion_state.compute_quadratures(include_squared_quantities, use_partial_trace)

        x_expected = 0.
        p_expected = 0.
        x2_expected = 0.5
        p2_expected = 0.5
        self.assertAlmostEqual(x[0].real, x_expected, places=6) 
        self.assertAlmostEqual(p[0].real, p_expected, places=6) 
        self.assertAlmostEqual(x2[0].real, x2_expected, places=6) 
        self.assertAlmostEqual(p2[0].real, p2_expected, places=6) 

        # method 2
        use_partial_trace = True
        x, p, x2, p2 = self.atomic_structure_motion_state.compute_quadratures(include_squared_quantities, use_partial_trace)
        self.assertAlmostEqual(x[0].real, x_expected, places=6) 
        self.assertAlmostEqual(p[0].real, p_expected, places=6) 
        self.assertAlmostEqual(x2[0].real, x2_expected, places=6) 
        self.assertAlmostEqual(p2[0].real, p2_expected, places=6) 

    def test_coherent_displacement_calculation(self):
        """ Test calculation of a coherent displacement for a state |00> otimes |alpha=1>"""
        # For a coherent state with magnitude alpha = 1, we expect the displacement to be 1 for the |00> projection 
        # Note: We expect some error between our state and expectation for a finite Fock cutoff
        displacements = self.atomic_structure_coherent_state.compute_coherent_displacements(self.mode2) 
        expected_displacements = np.array([1., 0., 0., 0.])
        assert_array_close(np.array(displacements), expected_displacements, atol = 1e-7) 


if __name__ == '__main__':
    unittest.main()
