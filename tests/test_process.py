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

from ionsim.process import Gate, Circuit
from ionsim.degree_of_freedom import AtomicStructure
from ionsim.basis import StandardBasis
from ionsim.named_operators import Unitary, Pauli
from ionsim.noise import Noise
from ionsim.operator import EnergyShiftOperator, CouplingOperator
from ionsim.hamiltonian import Hamiltonian
from ionsim.state import State 

class TestProcess(unittest.TestCase):

    def setUp(self):
        """Set up the necessary objects for testing."""
        self.spin_a = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.spin_b = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.spin_c = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2','P1/2'], level_names=['S1/2,0,0', 'S1/2,1,0','P1/2,1,-1'])
        self.basis = StandardBasis([self.spin_a, self.spin_b])

        self.Sx = Gate.from_unitary(self.basis, Unitary.sqrtX, [self.spin_a])

        xs = np.linspace(-np.pi, np.pi, 21)
        self.phi_noise = Noise.from_named_pdf('phi', 'gaussian', {'standard_deviation': np.pi/10}, xs)
        self.noisy_phi_gate = Gate.from_unitary_function(
            self.basis, Unitary.R, {'phi': 0, 'theta': np.pi/2}, [self.spin_a], self.phi_noise,
        )

        self.theta_noise = Noise.from_named_pdf('theta', 'gaussian', {'standard_deviation': np.pi/10}, xs)
        self.noisy_theta_gate = Gate.from_unitary_function(
            self.basis, Unitary.R, {'phi': 0, 'theta': np.pi/2}, [self.spin_a], self.theta_noise,
        )

    def test_noisy_phi_gate_process_fidelity(self):
        """Test the process fidelity of the noisy phi gate."""
        fidelity = self.noisy_phi_gate.compute_process_fidelity(self.Sx.process_matrix)
        self.assertAlmostEqual(fidelity, 0.9535335189419549, places=14)

    def test_noisy_theta_gate_process_fidelity(self):
        """Test the process fidelity of the noisy theta gate."""
        fidelity = self.noisy_theta_gate.compute_process_fidelity(self.Sx.process_matrix)
        self.assertAlmostEqual(fidelity, 0.9759249157026244, places=14)

    def test_extra_noisy_gate_process_fidelity(self):
        """Test the process fidelity of the extra noisy gate."""
        extra_noisy_gate = Gate.from_process_matrix_function(
            self.basis, self.noisy_phi_gate.process_matrix_function, {'phi': 0, 'theta': np.pi/2}, self.theta_noise,
        )
        fidelity = extra_noisy_gate.compute_process_fidelity(self.Sx.process_matrix)
        self.assertAlmostEqual(fidelity, 0.9306176541502549, places=14)

    def test_ramsey_circuit_process_fidelity(self):
        """Test the process fidelity of the Ramsey circuit."""
        ramsey = Circuit.from_gates(
            [
                Gate.from_unitary(self.basis, Unitary.sqrtX, [self.spin_a]),
                Gate.from_unitary_function(self.basis, Unitary.R, {'phi': 0, 'theta': np.pi/2}, [self.spin_a], self.phi_noise),
            ],
            self.theta_noise,
        )
        fidelity = ramsey.compute_process_fidelity(Gate.from_unitary(self.basis, Unitary.X, [self.spin_a]).process_matrix)
        self.assertAlmostEqual(fidelity, 0.9306176541502548, places=14)

        # Test computing outcome probabilities 
        outcome_operator = EnergyShiftOperator.from_matrix(self.basis, np.kron(Pauli.projector_1, Pauli.projector_0)) 
        initial_state = State.from_coefficients(self.basis, [1., 0., 0., 0.]) 

        outcome_probability = ramsey.predict_outcome_probabilities(initial_state, [outcome_operator]) 
        self.assertAlmostEqual(outcome_probability[0], 0.9530090510307307, places = 10)

    def test_pauli_transfer_matrix_computations(self):
        noisy_gate = self.noisy_phi_gate
        noisy_gate_pauli = noisy_gate.convert_to_pauli_basis()
        Sx_pauli = self.Sx.convert_to_pauli_basis()

        fidelity = noisy_gate_pauli.compute_process_fidelity(Sx_pauli.process_matrix)
        self.assertAlmostEqual(fidelity, 0.9535335189419549, places=14)

    def test_Raman_gate(self):
        full_basis = StandardBasis([self.spin_c])
        target_rabi_rate = 10 * 2 * np.pi * 1E3 # rad/s 
        detuning = -2. * 2. * np.pi * 1E9
        rabi_rate = np.sqrt(np.abs(detuning) * 2. * target_rabi_rate) 

        omega1 = self.spin_c.energy_levels[2].energy - self.spin_c.energy_levels[0].energy + detuning
        omega2 = self.spin_c.energy_levels[2].energy - self.spin_c.energy_levels[1].energy + detuning
        omega_qubit = self.spin_c.energy_levels[1].energy - self.spin_c.energy_levels[0].energy
        assert (np.abs(omega1 - omega2) - omega_qubit)/(2*np.pi)/1E3 < 1.E-2, 'Raman resonance condition is violated.'

        size = len(full_basis.states)
        static_operator1 = np.zeros((size, size))
        static_operator1[2, 0] = rabi_rate/2. 
        static_operator2 = np.zeros((size, size))
        static_operator2[2, 1] = -rabi_rate/2.  # phase of pi

        raising_op1 = CouplingOperator.from_matrix(full_basis, static_operator1, omega1, [self.spin_c])
        raising_op2 = CouplingOperator.from_matrix(full_basis, static_operator2, omega2, [self.spin_c])
        
        interaction_frame_energies = [-1 * state.energy for state in full_basis.states]
        hamiltonian = Hamiltonian(full_basis, [raising_op1, raising_op2], interaction_frame_energies, sparse=False)

        theta = np.pi/16.
        duration = (np.pi/16.)/target_rabi_rate
        undesired_levels = [self.spin_c.energy_levels[2]] # P1/2 level to project out        
        projection_input = {'levels' : undesired_levels}
        dt = 0.002 * 1E-6 
        times = np.arange(0, duration + dt, dt)
        X_pi16_Raman_gate = Gate.from_hamiltonian(full_basis, hamiltonian, duration, projection_input = projection_input, ode_solver = 'odeintz', time_evals = times) 
        reduced_basis = X_pi16_Raman_gate.basis

        # Process fidelity 
        X_pi16_ref = Unitary.R(0., theta) 
        global_phase_correction = np.exp(-1j*theta)
        X_pi16_ref *= global_phase_correction
        X_pi16_ref = Gate.from_unitary(reduced_basis, X_pi16_ref, [self.spin_c])

        process_fidelity = X_pi16_Raman_gate.compute_process_fidelity(X_pi16_ref.process_matrix)
        self.assertAlmostEqual(process_fidelity, 0.999998720951, places=8)

        # Test pauli transfer matrix and pauli error rate calculations 
        import scipy
        error_channel_as_gate = Gate(reduced_basis, X_pi16_Raman_gate.process_matrix @ scipy.linalg.inv(X_pi16_ref.process_matrix))   
        error_rates = error_channel_as_gate.compute_pauli_error_rates()        
        # I <==> gate occurs as intended (no error); should be equal to process fidelity 
        self.assertAlmostEqual(0.9999987209516402, error_rates["I"].real, places=10)
        self.assertAlmostEqual(process_fidelity, error_rates["I"].real, places=10)


if __name__ == '__main__':
    unittest.main()
