#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************
import unittest
import warnings

import numpy as np

from ionsim.process import Gate
from ionsim.degree_of_freedom import AtomicStructure, MotionalMode
from ionsim.basis import StandardBasis
from ionsim.named_operators import Pauli, Fock, Unitary
from ionsim.operator import CouplingOperator
from ionsim.hamiltonian import Hamiltonian
from ionsim.laser import Laser, Polarization
from ionsim.testing import assert_array_close
from ionsim.ionsim_error import IonSimError


# Shared gate parameters: an MS(0, THETA) gate in LOOPS phase-space loops of duration TAU with Lamb-Dicke parameter ETA.
THETA = np.pi / 8
ETA = 0.1
TAU = 125.0 * 1e-6
LOOPS = 2


def _ms_detuning_and_rabi_rate(theta: float, eta: float, tau: float, loops: int) -> tuple[float, float]:
    """ Detuning delta and carrier Rabi rate Omega (rad/s) of a bichromatic MS gate that closes `loops` phase-space loops
        in time tau and gives Unitary.MS(phi, theta):
            delta = 2 pi loops / tau,   eta Omega = sqrt(theta / (pi/2)) * delta / (2 sqrt(loops)).
        For theta = pi/2 this is the maximally entangling condition eta Omega = delta / (2 sqrt(loops)). """
    detuning = 2*np.pi * loops / tau
    rabi_rate = np.sqrt(theta/(np.pi/2)) * np.pi * np.sqrt(loops) / (eta * tau)
    return detuning, rabi_rate


def _ms_hamiltonian(basis, spins, mode, eta, rabi_rate, detuning):
    qubit_frequency = spins[0].energy_levels[1].energy - spins[0].energy_levels[0].energy
    mode_frequency = mode.energy_levels[1].energy - mode.energy_levels[0].energy
    omega_b = qubit_frequency + mode_frequency + detuning
    omega_r = qubit_frequency - mode_frequency - detuning

    fock_dimension = len(mode.energy_levels)
    raise_motion = basis.enlarge_one_dof_matrix(Fock.raising(fock_dimension), mode)
    lower_motion = basis.enlarge_one_dof_matrix(Fock.lowering(fock_dimension), mode)
    prefactor = np.exp(-1j*np.pi/2) * rabi_rate/2 * (1j*eta)

    operators = []
    for spin in spins:
        raise_spin = basis.enlarge_one_dof_matrix(Pauli.plus, spin)
        operators.extend([
            CouplingOperator.from_matrix(basis, prefactor * raise_spin @ raise_motion, omega_b),
            CouplingOperator.from_matrix(basis, prefactor * raise_spin @ lower_motion, omega_r),
        ])
    interaction_frame_energies = [-state.energy for state in basis.states]
    return Hamiltonian(basis, operators, interaction_frame_energies, sparse=False)


class TestMSGateTracingOutMotion(unittest.TestCase):

    def setUp(self):
        self.spins = [
            AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
            for _ in range(2)
        ]
        # Small-angle gate: the motional displacement is small, so a small Fock space suffices.
        self.fock_dimension = 4
        self.mode = MotionalMode.from_frequency(frequency=2*np.pi * 3e6, fock_dimension=self.fock_dimension)
        self.basis = StandardBasis([*self.spins, self.mode])
        self.psi_motion_0 = np.eye(self.fock_dimension)[0]

        self.theta, self.eta, self.tau = THETA, ETA, TAU
        self.detuning, self.rabi_rate = _ms_detuning_and_rabi_rate(THETA, ETA, TAU, LOOPS)

        self.hamiltonian = _ms_hamiltonian(self.basis, self.spins, self.mode, self.eta, self.rabi_rate, self.detuning)
        self.gate = Gate.from_hamiltonian(self.basis, self.hamiltonian, self.tau, [self.mode], [self.psi_motion_0])

    def test_gate_is_on_qubit_basis(self):
        """ The traced-out gate acts on the two-qubit basis only. """
        self.assertEqual(self.gate.process_matrix.shape, (16, 16))
        self.assertEqual(len(self.gate.basis.degrees_of_freedom), 2)
        self.assertTrue(self.gate.basis.is_qubit_basis)

    def test_small_angle_ms_gate_matches_ideal(self):
        """ The small-angle MS gate matches Unitary.MS(0, theta) once the motion is traced out. """
        ideal_ms_process_matrix = Gate.from_unitary(self.gate.basis, Unitary.MS(0., self.theta), self.spins).process_matrix
        fidelity = self.gate.compute_process_fidelity(ideal_ms_process_matrix)
        self.assertGreater(fidelity, 0.9999)
        # Sanity check that the test discriminates the angle: MS(0, 0) = identity and MS(0, 2 theta) are clearly worse matches.
        for wrong_theta in [0., 2*self.theta]:
            wrong_ms_process_matrix = Gate.from_unitary(self.gate.basis, Unitary.MS(0., wrong_theta), self.spins).process_matrix
            self.assertLess(self.gate.compute_process_fidelity(wrong_ms_process_matrix), 0.97)

    def test_tracing_out_multiple_modes_with_spectator_in_middle(self):
        """ Adding an uncoupled spectator mode in the middle of the basis does not change the traced-out gate. """
        spectator = MotionalMode.from_frequency(frequency=2*np.pi * 2.5e6, fock_dimension=2)
        basis = StandardBasis([self.spins[0], spectator, self.spins[1], self.mode])
        hamiltonian = _ms_hamiltonian(basis, self.spins, self.mode, self.eta, self.rabi_rate, self.detuning)
        gate = Gate.from_hamiltonian(
            basis, hamiltonian, self.tau, [spectator, self.mode], [np.eye(2)[0], self.psi_motion_0]
        )
        self.assertTrue(all(a is b for a, b in zip(gate.basis.degrees_of_freedom, self.spins)))
        assert_array_close(gate.process_matrix, self.gate.process_matrix, rtol=0, atol=1e-6)


class TestMSGateFromLaserSidebands(unittest.TestCase):
    """ MS gate built from Laser sideband couplings on 40Ca+ S1/2 -> D5/2 (729 nm, E2), with the same gate parameters
        (theta, eta, tau, loops, hence the same Rabi rate) as TestMSGateTracingOutMotion.

        Two ions, one mode. Two lasers made with gaussian_from_target_rabi_frequency at detunings +/- (omega_mode + delta)
        from the qubit; the blue sideband of the blue tone and the red sideband of the red tone. The lasers propagate
        along x with polarization along z, which couples |S1/2, mJ=1/2> to |D5/2, mJ=3/2>.
    """
    MODE_FREQUENCY = 2*np.pi * 1e6    # rad/s; arbitrary trap frequency
    FOCK_DIMENSION = 6                # small-angle gate: small motional displacement
    WAIST = 10e-6                     # m
    N_HAT = np.array([1., 0., 0.])
    QN_G, QN_E = {'n': 4, 'l': 0, 'mj': 0.5}, {'n': 3, 'l': 2, 'j': 2.5, 'mj': 1.5}

    def build(self, mode_first: bool):
        atoms = [AtomicStructure.from_species(species='40Ca+', term_symbols=['S1/2', 'D5/2'], name=f'ion{i}',
                                              quantum_numbers=[self.QN_G, self.QN_E]) for i in range(2)]
        mode = MotionalMode.from_frequency(self.MODE_FREQUENCY, self.FOCK_DIMENSION, name='mode')
        basis = StandardBasis([mode, *atoms] if mode_first else [*atoms, mode])
        g = next(level for level in atoms[0].energy_levels if level.term_symbol == 'S1/2')
        e = next(level for level in atoms[0].energy_levels if level.term_symbol == 'D5/2')
        return atoms, mode, basis, g, e

    def laser(self, rabi_rate, g, e, detuning):
        """ Laser with |Omega| = rabi_rate on g -> e whose phase cancels the complex phase of Omega and adds -pi/2, so the
            sideband raising element (Omega/2) e^{i phase} (i eta) equals that of _ms_hamiltonian, which gives MS(0, theta). """
        polarization = Polarization(np.array([0., 0., 1.]), self.N_HAT)
        def build(phase):
            return Laser.gaussian_from_target_rabi_frequency(rabi_rate, g, e, 2, self.WAIST, self.N_HAT, polarization,
                                                             detuning=detuning, phase=phase)
        rabi = build(0.).rabi_frequency(g, e, 2)
        self.assertAlmostEqual(abs(rabi) / rabi_rate, 1., places=10)
        laser = build(-np.pi/2 - np.angle(rabi))
        self.assertAlmostEqual(laser.frequency, e.energy - g.energy + detuning, delta=1e-9 * laser.frequency)
        return laser

    def gate(self, mode_first: bool, theta: float = THETA):
        atoms, mode, basis, g, e = self.build(mode_first)
        detuning, rabi_rate = _ms_detuning_and_rabi_rate(theta, ETA, TAU, LOOPS)
        blue = self.laser(rabi_rate, g, e, +(self.MODE_FREQUENCY + detuning))
        red = self.laser(rabi_rate, g, e, -(self.MODE_FREQUENCY + detuning))
        etas = [[ETA], [ETA]]   # rows: atoms (target_atoms=None -> basis order), columns: [mode]
        operators = (blue.build_blue_sideband_coupling_operators(basis, None, [mode], [g], [e], 2, etas)
                     + red.build_red_sideband_coupling_operators(basis, None, [mode], [g], [e], 2, etas))
        hamiltonian = Hamiltonian(basis, operators, [-state.energy for state in basis.states], sparse=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gate = Gate.from_hamiltonian(basis, hamiltonian, TAU, [mode], [np.eye(self.FOCK_DIMENSION)[0]])
        return atoms, gate

    def test_same_rabi_rate_as_reference_test(self):
        _, rabi_rate = _ms_detuning_and_rabi_rate(THETA, ETA, TAU, LOOPS)
        reference = TestMSGateTracingOutMotion()
        reference.setUp()
        self.assertEqual(rabi_rate, reference.rabi_rate)

    def test_small_angle_ms_gate_matches_ideal_for_any_basis_ordering(self):
        """ After LOOPS loops the mode is disentangled and the traced-out gate is Unitary.MS(0, THETA), for either basis
            ordering; MS(0, 0) and MS(0, 2 THETA) are clearly worse matches. """
        for mode_first in (False, True):
            with self.subTest(mode_first=mode_first):
                atoms, gate = self.gate(mode_first)
                self.assertTrue(gate.basis.is_qubit_basis)
                fidelity = lambda theta: gate.compute_process_fidelity(
                    Gate.from_unitary(gate.basis, Unitary.MS(0., theta), atoms).process_matrix)
                self.assertGreater(fidelity(THETA), 0.9999)
                for wrong_theta in [0., 2*THETA]:
                    self.assertLess(fidelity(wrong_theta), 0.97)

    def test_maximally_entangling_condition(self):
        """ theta = pi/2 (eta Omega = delta / (2 sqrt(loops))) gives MS(0, pi/2). """
        atoms, gate = self.gate(False, theta=np.pi/2)
        ideal = Gate.from_unitary(gate.basis, Unitary.MS(0., np.pi/2), atoms).process_matrix
        self.assertGreater(gate.compute_process_fidelity(ideal), 0.999)

if __name__ == '__main__':
    unittest.main()
