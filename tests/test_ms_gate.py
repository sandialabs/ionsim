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

from ionsim.process import Gate
from ionsim.degree_of_freedom import AtomicStructure, MotionalMode
from ionsim.basis import StandardBasis
from ionsim.named_operators import Pauli, Fock, Unitary
from ionsim.operator import CouplingOperator
from ionsim.hamiltonian import Hamiltonian
from ionsim.testing import assert_array_close
from ionsim.ionsim_error import IonSimError


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

        self.theta = np.pi/8
        self.eta = 0.1
        self.tau = 125e-6
        loops = 2
        self.detuning = 2*np.pi * loops / self.tau
        self.rabi_rate = np.sqrt(self.theta/(np.pi/2)) * np.pi * np.sqrt(loops) / (self.eta * self.tau)

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

    def test_traced_out_gate_is_trace_preserving_and_completely_positive(self):
        """ The reduced channel is CPTP. """
        process_matrix = self.gate.process_matrix
        vectorized_identity = np.eye(4).T.flatten()
        assert_array_close(vectorized_identity @ process_matrix, vectorized_identity, rtol=0, atol=1e-6)

        choi = sum(
            np.kron(np.outer(np.eye(4)[i], np.eye(4)[j]), process_matrix[:, i + 4*j].reshape(4, 4).T)
            for i in range(4) for j in range(4)
        )
        assert_array_close(choi, choi.conj().T, rtol=0, atol=1e-10)
        self.assertGreater(np.linalg.eigvalsh(choi).min(), -1e-8)

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


if __name__ == '__main__':
    unittest.main()
