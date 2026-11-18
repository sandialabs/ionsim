#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************
""" Tests for Lindblad operators as composite operators (LindbladOperator), the Operator hermitian flag, and modulation
    functions in the Dissipator. """
import unittest
import numpy as np
from scipy.integrate import solve_ivp

from ionsim.basis import StandardBasis
from ionsim.operator import CouplingOperator, EnergyShiftOperator, GeneralOperator
from ionsim.named_operators import Pauli
from ionsim.degree_of_freedom import AtomicStructure
from ionsim.hamiltonian import Hamiltonian
from ionsim.lindbladian import Dissipator, Lindbladian, LindbladOperator
from ionsim.state import State
from ionsim.ionsim_error import IonSimError

TPI = 2. * np.pi
RABI = TPI * 100e3          # rad/s


def qubit():
    return AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])


def amplitude_and_phase_modulation(t):
    """ A complex modulation with both amplitude and phase dependence. """
    return (1. + 0.3 * np.sin(TPI * 50e3 * t)) * np.exp(1j * 0.7 * np.sin(TPI * 80e3 * t))


def other_modulation(t):
    return np.exp(-1j * 0.4 * np.cos(TPI * 30e3 * t))


class TestLindbladOperators(unittest.TestCase):

    def setUp(self):
        self.atoms = [qubit(), qubit()]
        self.basis = StandardBasis(self.atoms)
        self.frame = [-state.energy for state in self.basis.states]
        levels = self.atoms[0].energy_levels
        self.omega = levels[1].energy - levels[0].energy   # resonant drive frequency
        self.times = np.array([0., 0.37e-6, 1.3e-6, 4.9e-6])

    def drive_operator(self, atom, modulation=None, hermitian=None, storage='raising'):
        """ Resonant drive (Omega/2) |1><0| on one atom, stored as the raising element only or as the full Hermitian
            matrix (both triangles with the same oscillation rate, as the laser coupling builders do). """
        raising = 0.5 * RABI * self.basis.enlarge_matrix(Pauli.plus, [atom])
        matrix = raising if storage == 'raising' else raising + raising.conj().T
        return CouplingOperator.from_matrix(self.basis, matrix, self.omega, modulation_function=modulation, hermitian=hermitian)

    def expected_as_stored(self, operator, t, modulation=None):
        """ Lindblad matrix of a coupling operator used as stored: element (i, j) * exp(-i (rate + E_i - E_j) t) * f(t). """
        expected = np.zeros((self.basis.dimension if hasattr(self.basis, 'dimension') else len(self.basis.states),) * 2, dtype=complex)
        for coupling in operator.couplings:
            i, j = self.basis.states.index(coupling.row_state), self.basis.states.index(coupling.column_state)
            rate = coupling.oscillation_rate + self.frame[i] - self.frame[j]
            expected[i, j] = coupling.strength * np.exp(-1j * rate * t)
        return expected * (1. if modulation is None else modulation(t))

    # ---- Backward compatibility: plain operators, no flag ----
    def test_plain_operators_are_wrapped_and_used_as_stored(self):
        decay = CouplingOperator.from_matrix(self.basis, 0.2 * self.basis.enlarge_matrix(Pauli.minus, [self.atoms[0]]), 0.)
        flip = CouplingOperator.from_matrix(self.basis, 0.3 * self.basis.enlarge_matrix(Pauli.X, [self.atoms[1]]), 0.)
        dephasing = GeneralOperator.from_matrix(self.basis, 0.1 * self.basis.enlarge_matrix(Pauli.Z, [self.atoms[0]]))
        dissipator = Dissipator(self.basis, [decay, flip, dephasing], self.frame)

        self.assertEqual(len(dissipator.operators), 3)
        self.assertIs(dissipator.operators[0], decay)      # operators keeps exactly what was passed in
        self.assertTrue(all(isinstance(lop, LindbladOperator) for lop in dissipator.lindblad_operators))
        for t in self.times:
            L_decay, L_flip, L_dephasing = [function(t) for function in dissipator.lindblad_matrix_functions]
            np.testing.assert_allclose(L_decay, self.expected_as_stored(decay, t), atol=1e-12)
            np.testing.assert_allclose(L_flip, self.expected_as_stored(flip, t), atol=1e-12)
            np.testing.assert_allclose(L_dephasing, dephasing.static_matrix.toarray(), atol=1e-12)

    def test_modulation_is_applied_to_non_hermitian_operators(self):
        for hermitian in (None, False):
            decay = CouplingOperator.from_matrix(self.basis, 0.2 * self.basis.enlarge_matrix(Pauli.minus, [self.atoms[0]]), 0.,
                                                 modulation_function=amplitude_and_phase_modulation, hermitian=hermitian)
            (L,) = Dissipator(self.basis, [decay], self.frame).lindblad_matrix_functions
            for t in self.times:
                np.testing.assert_allclose(L(t), self.expected_as_stored(decay, t, amplitude_and_phase_modulation), atol=1e-12)

    def test_modulated_energy_shift_operator(self):
        shift = EnergyShiftOperator.from_matrix(self.basis, self.basis.enlarge_matrix(Pauli.Z, [self.atoms[0]]),
                                                modulation_function=other_modulation)
        self.assertIs(shift.modulation_function, other_modulation)   # was dropped by from_matrix before
        (L,) = Dissipator(self.basis, [shift], self.frame).lindblad_matrix_functions
        for t in self.times:
            np.testing.assert_allclose(L(t), self.basis.enlarge_matrix(Pauli.Z, [self.atoms[0]]) * other_modulation(t), atol=1e-12)

    # ---- Hermitian flag: the Lindblad operator is the same operator as the Hamiltonian term ----
    def test_hermitian_lindblad_operator_equals_hamiltonian_coupling(self):
        for storage in ('raising', 'hermitian'):
            for sparse in (False, True):
                with self.subTest(storage=storage, sparse=sparse):
                    operators = [self.drive_operator(atom, amplitude_and_phase_modulation, True, storage) for atom in self.atoms]
                    hamiltonian = Hamiltonian(self.basis, operators, self.frame)
                    lindblad = LindbladOperator(self.basis, operators, self.frame, sparse=sparse)
                    H0 = np.diag(hamiltonian.energies)
                    for t in self.times:
                        coupling = hamiltonian.hamiltonian_function(t) - H0
                        L = lindblad.lindblad_matrix_function(t)
                        L = L.toarray() if sparse else L
                        np.testing.assert_allclose(L, coupling, atol=1e-9 * RABI)
                        np.testing.assert_allclose(L, L.conj().T, atol=1e-9 * RABI)

    def test_operator_sum_reproduces_hamiltonian(self):
        """ The shared evaluation (CompositeOperator.build_operator_sum_function) plus H0 is the Hamiltonian, for a mix of
            couplings with different modulations, a modulated energy shift and a general operator. """
        operators = [self.drive_operator(self.atoms[0], amplitude_and_phase_modulation),
                     self.drive_operator(self.atoms[1], other_modulation, storage='hermitian'),
                     self.drive_operator(self.atoms[1]),
                     EnergyShiftOperator.from_matrix(self.basis, 1e4 * self.basis.enlarge_matrix(Pauli.Z, [self.atoms[0]]),
                                                     modulation_function=lambda t: np.cos(TPI * 20e3 * t)),
                     GeneralOperator.from_matrix(self.basis, 2e4 * self.basis.enlarge_matrix(Pauli.Z, [self.atoms[1]]))]
        hamiltonian = Hamiltonian(self.basis, operators, self.frame)
        operator_sum = hamiltonian.build_operator_sum_function()
        for t in self.times:
            np.testing.assert_allclose(hamiltonian.hamiltonian_function(t), np.diag(hamiltonian.energies) + operator_sum(t),
                                       atol=1e-9 * RABI)

    # ---- Composite Lindblad operators ----
    def test_collective_operator_differs_from_independent_operators(self):
        K = [self.drive_operator(atom, amplitude_and_phase_modulation, True) for atom in self.atoms]
        collective = Dissipator(self.basis, [LindbladOperator(self.basis, K, self.frame)], self.frame)
        independent = Dissipator(self.basis, K, self.frame)
        t = self.times[2]
        L_A, L_B = [function(t) for function in independent.lindblad_matrix_functions]
        np.testing.assert_allclose(collective.dissipator_matrix_function(t),
                                   Dissipator.lindblad_matrix_to_superoperator(L_A + L_B), atol=1e-6 * RABI**2)
        difference = np.abs(collective.dissipator_matrix_function(t) - independent.dissipator_matrix_function(t)).max()
        self.assertGreater(difference, 0.1 * RABI**2)   # cross terms between the atoms

    def test_prefactor_and_mixed_entries(self):
        K = self.drive_operator(self.atoms[0], amplitude_and_phase_modulation, True)
        decay = CouplingOperator.from_matrix(self.basis, self.basis.enlarge_matrix(Pauli.minus, [self.atoms[1]]), 0., hermitian=False)
        scaled = LindbladOperator(self.basis, [K], self.frame, prefactor=0.25j)
        dissipator = Dissipator(self.basis, [scaled, decay], self.frame)   # a composite and a plain operator
        unscaled = LindbladOperator(self.basis, [K], self.frame).lindblad_matrix_function
        for t in self.times:
            np.testing.assert_allclose(dissipator.lindblad_matrix_functions[0](t), 0.25j * unscaled(t), atol=1e-12 * RABI)

    # ---- Dynamics ----
    def single_qubit_setup(self, modulation=None):
        atom = qubit()
        basis = StandardBasis([atom])
        frame = [-state.energy for state in basis.states]
        raising = 0.5 * RABI * basis.enlarge_matrix(Pauli.plus, [atom])
        drive = CouplingOperator.from_matrix(basis, raising, self.omega, modulation_function=modulation, hermitian=True)
        return basis, frame, drive

    def test_white_amplitude_noise_dephasing(self):
        """ Resonant drive H = (Omega/2) sigma_x with white amplitude noise, Omega -> Omega (1 + xi), <xi xi'> = kappa delta:
            L = sqrt(kappa) H damps the Bloch components perpendicular to x at kappa Omega^2 / 2, so starting in |0>,
            P_1(t) = (1 - cos(Omega t) exp(-kappa Omega^2 t / 2)) / 2. """
        basis, frame, drive = self.single_qubit_setup()
        kappa = 2. / (RABI**2 * 20e-6)   # damping time 20 us
        noise = LindbladOperator(basis, [drive], frame, prefactor=np.sqrt(kappa))
        lindbladian = Lindbladian(Hamiltonian(basis, [drive], frame), Dissipator(basis, [noise], frame))
        times = np.linspace(0., 30e-6, 61)
        rho_t = State.from_coefficients(basis, [1., 0.]).propagate_using_master_equation(lindbladian, times[-1], times)
        population = np.array([rho.density_matrix[1, 1].real for rho in rho_t])
        expected = 0.5 * (1. - np.cos(RABI * times) * np.exp(-0.5 * kappa * RABI**2 * times))
        np.testing.assert_allclose(population, expected, atol=1e-6)

    def test_phase_modulated_noise_against_direct_integration(self):
        """ Amplitude- and phase-modulated drive with white amplitude noise, against a direct integration of
            d rho/dt = -i [K, rho] + kappa (K rho K - {K^2, rho} / 2),  K(t) = (Omega/2) (f(t) |1><0| + h.c.). """
        basis, frame, drive = self.single_qubit_setup(amplitude_and_phase_modulation)
        kappa = 2. / (RABI**2 * 20e-6)
        noise = LindbladOperator(basis, [drive], frame, prefactor=np.sqrt(kappa))
        lindbladian = Lindbladian(Hamiltonian(basis, [drive], frame), Dissipator(basis, [noise], frame))
        duration = 12e-6
        rho_ionsim = State.from_coefficients(basis, [1., 0.]).propagate_using_master_equation(
            lindbladian, duration, np.linspace(0., duration, 401))[-1].density_matrix

        def K(t):
            f = amplitude_and_phase_modulation(t)
            return 0.5 * RABI * np.array([[0., np.conj(f)], [f, 0.]])

        def rhs(t, y):
            rho, k = y.reshape(2, 2), K(t)
            return (-1j * (k @ rho - rho @ k) + kappa * (k @ rho @ k - 0.5 * (k @ k @ rho + rho @ k @ k))).ravel()

        rho0 = np.array([[1., 0.], [0., 0.]], dtype=complex)
        rho_direct = solve_ivp(rhs, (0., duration), rho0.ravel(), method='DOP853', rtol=1e-11, atol=1e-13).y[:, -1].reshape(2, 2)
        np.testing.assert_allclose(rho_ionsim, rho_direct, atol=1e-6)

    # ---- Input validation ----
    def test_input_validation(self):
        K = self.drive_operator(self.atoms[0], amplitude_and_phase_modulation, True)
        other_frame = [0.] * len(self.basis.states)
        with self.assertRaises(IonSimError):   # composite in another frame
            Dissipator(self.basis, [LindbladOperator(self.basis, [K], other_frame)], self.frame)
        with self.assertRaises(IonSimError):   # sparse setting mismatch
            Dissipator(self.basis, [LindbladOperator(self.basis, [K], self.frame, sparse=True)], self.frame)
        with self.assertRaises(IonSimError):   # not an operator
            Dissipator(self.basis, [np.eye(4)], self.frame)
        decay = CouplingOperator.from_matrix(self.basis, self.basis.enlarge_matrix(Pauli.minus, [self.atoms[0]]), 0., hermitian=False)
        with self.assertRaises(IonSimError):   # a Hamiltonian term cannot be flagged non-Hermitian
            Hamiltonian(self.basis, [decay], self.frame)
        inconsistent = self.basis.enlarge_matrix(np.array([[0., 2.], [1., 0.]]), [self.atoms[0]])
        flagged = CouplingOperator.from_matrix(self.basis, inconsistent, 0., hermitian=True)
        with self.assertRaises(IonSimError):   # both triangles stored but not complex conjugates
            Dissipator(self.basis, [flagged], self.frame).lindblad_matrix_functions


if __name__ == '__main__':
    unittest.main()
