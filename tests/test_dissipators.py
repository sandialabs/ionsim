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
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy.sparse import kron as skron

from ionsim.zeeman_solver import ZeemanHyperfineSolver 
from ionsim.basis import StandardBasis
from ionsim.operators import CouplingOperator, EnergyShiftOperator, GeneralOperator, adjoint
from ionsim.named_operators import Pauli, Fock
from ionsim.degree_of_freedom import AtomicStructure, MotionalMode
from ionsim.hamiltonian import Hamiltonian
from ionsim.lindbladian import Dissipator, Lindbladian, LindbladOperator
from ionsim.state import State
from ionsim.ionsim_error import IonSimError

TPI = 2*np.pi
# Helper functions for dissipator references 
def oscillitory_exponential_decay(t, A, gamma, C, w, phase):
    return A*np.exp(-t*gamma)*np.cos(w*t + phase) + C 

def exponential_decay(t, A, gamma, C): 
    return A*np.exp(-t*gamma) + C 

def exponential_warmup(t, y0, gamma):
    return y0 + (0. - y0)*np.exp(-gamma*t)

def spin_flip_exp_decay(t, Omega, gamma):
    return 0.5 + 0.5 * np.exp(-2*gamma*t) * np.cos(Omega*t)


class TestDissipators(unittest.TestCase):

    def setUp(self):
        """Set up the necessary objects for testing."""
        # Set up tests with a list of dictionaries: 
        self.test_cases = [
            {
                'test name' : 'Dephasing',
                'characteristic time' : 250.0E-6,
                'number of qubits' : 1,
                'rabi rate' : 100E3, 
                'detuning' : 0., 
                'initial state coefficients' : np.array([0., 1.]),
                'reference function' : oscillitory_exponential_decay, 
                'reference function guess parameters' : [1.0, 100E3, -0.5, 100E3, 0.]
            },
            {
                'test name' : 'Spontaneous Emission',
                'characteristic time' : 10.0E-9,
                'number of qubits' : 1,
                'rabi rate' : 0., 
                'detuning' : 0., 
                'initial state coefficients' : np.array([0., 1.]), 
                'reference function' : exponential_decay, 
                'reference function guess parameters' : [1., 100*100E3, 0.] 
            },
            {
                'test name' : 'Ion Heating',
                'characteristic time' : 0.001 , # ( quanta / seconds )^{-1}
                'N thermal' : 1.5 , # Equilibrium number of phonons  
                'number of qubits' : 1,
                'rabi rate' : 0., 
                'detuning' : 0., 
                'fock dimension' : 6,
                'initial state coefficients' : np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), 
                'reference function' : exponential_warmup, 
            },
            {
                'test name' : 'Spin Flipping',
                'characteristic time' : 5.0E-6,
                'number of qubits' : 1,
                'rabi rate' : 100E3*2.*np.pi, 
                'detuning' : 0., 
                'initial state coefficients' : np.array([1., 0.]),
                'reference function' : spin_flip_exp_decay, 
                'reference function guess parameters' : [50E3*np.pi*2., 10000]
            },
        ]

        # Set up the qubits, hamiltonian, dissipator, and lindbladian for each test case.  
        for case in self.test_cases:
            # Create spins and basis. Store for each case 
            case['qubits'] = [
                AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
                for _ in range(case['number of qubits'])
                ]

            case['qubit frequency'] = case['qubits'][0].energy_levels[1].energy - case['qubits'][0].energy_levels[0].energy # assuems all qubits are the same 
            case['omega_laser'] = case['qubit frequency'] + case['detuning'] 

            decay_rate = 1./(case['characteristic time'])

            if case['test name'] == 'Dephasing':
                # Dephasing qubit 1 during Rabi flopping 
                case['basis'] = StandardBasis([*case['qubits']])

                # Rabi flopping Hamiltonian:                 
                raise_qubit = case['rabi rate'] * 0.5 * case['basis'].enlarge_matrix(Pauli.plus, [case['qubits'][0]]) 
                coupling_operator = CouplingOperator.from_matrix(case['basis'], raise_qubit, case['omega_laser']) 
                frame_energies = [-state.energy for state in case['basis'].states] 
                case['hamiltonian'] = Hamiltonian(case['basis'], [coupling_operator], frame_energies) 

                case['duration'] = TPI * 4. / case['rabi rate'] 
                # Dephasing with Lindblad operator L = Z on qubit 1:                 
                dephase_q1 = np.sqrt(decay_rate) * case['basis'].enlarge_matrix(Pauli.Z , [case['qubits'][0]]) 
                case['lindblad operators'] = [GeneralOperator.from_matrix(case['basis'], dephase_q1)]                
            elif case['test name'] == 'Spontaneous Emission':
                # Spontaneous emission of a qubit in the |1> state; Decay from |1> to |0>.
                case['basis'] = StandardBasis([*case['qubits']])
                frame_energies = [-state.energy for state in case['basis'].states] 
                case['hamiltonian'] = None

                case['duration'] = 5./decay_rate

                # Dephasing with Lindblad operator L = Z on qubit 1:                 
                lower_qubit1 = np.sqrt(decay_rate) * case['basis'].enlarge_matrix(Pauli.minus, [case['qubits'][0]]) 
                case['lindblad operators'] = [CouplingOperator.from_matrix(case['basis'], lower_qubit1, 0.) ]

            if case['test name'] == 'Spin Flipping':
                # Dephasing qubit 1 during Rabi flopping 
                case['basis'] = StandardBasis([*case['qubits']])

                # Rabi flopping Hamiltonian:                 
                raise_qubit = case['rabi rate'] * 0.5 * case['basis'].enlarge_matrix(Pauli.plus, [case['qubits'][0]]) 
                coupling_operator = CouplingOperator.from_matrix(case['basis'], raise_qubit, case['omega_laser']) 
                frame_energies = [-state.energy for state in case['basis'].states] 
                case['hamiltonian'] = Hamiltonian(case['basis'], [coupling_operator], frame_energies) 

                case['duration'] = case['characteristic time']*0.25 
                # Dephasing with Lindblad operator L = Z on qubit 1:                 
                flip_q1 = np.sqrt(decay_rate) * case['basis'].enlarge_matrix(Pauli.X , [case['qubits'][0]]) 
                case['lindblad operators'] = [CouplingOperator.from_matrix(case['basis'], flip_q1, 0.)] 
            elif case['test name'] == 'Ion Heating':
                # Ion heating of 1 mode by lindblad operators: L1 = sqrt(gamma*(n+1)) a ; L2 = sqrt(gamma*n) a^† 
                # gamma is the heating rate and n is the average number of phonons in thermal equilibrium.  
                case['hamiltonian'] = None

                # Accurate heating at all times requires a large fock dimension - this slows simulations significantly. 
                # Therefore, we restrict to short times for the sake of a fast test. 
                case['duration'] = 0.25/decay_rate

                mode_frequency = 3E6 
                fock_dimension = case['fock dimension'] 
                modes = [MotionalMode.from_frequency(frequency=TPI*mode_frequency, fock_dimension=fock_dimension)]

                case['basis'] = StandardBasis([*case['qubits'], *modes])
                frame_energies = [-state.energy for state in case['basis'].states] 
                motional_basis = StandardBasis([*modes])
                spin_basis = StandardBasis([*case['qubits']])

                # Phonon raising and lowering operators 
                mode_lowering_matrix = motional_basis.enlarge_matrix(Fock.lowering(fock_dimension), [modes[0]]) 
                mode_raising_matrix = motional_basis.enlarge_matrix(Fock.raising(fock_dimension), [modes[0]]) 

                # Build spin basis identity 
                spin_identities = []
                for s, spin in enumerate(case['qubits']):
                    # Identity matrix for mode m in Fock space, enlarged to fit dimensionality of M modes 
                    spin_identities.append(spin_basis.enlarge_matrix(Pauli.I, [spin]))
            
                spin_identity = np.sum(spin_identities, axis=0)

                # Basis is of the form spins x modes, so we tensor in the that order: 
                N_thermal = case['N thermal']
                lowering_operator = np.sqrt(decay_rate*(N_thermal+1)) * skron(spin_identity, mode_lowering_matrix) 
                raising_operator = np.sqrt(decay_rate*N_thermal) * skron(spin_identity, mode_raising_matrix) 

                case['lindblad operators'] = []
                case['lindblad operators'].append(CouplingOperator.from_matrix(case['basis'], lowering_operator, 0.)) 
                case['lindblad operators'].append(CouplingOperator.from_matrix(case['basis'], raising_operator, 0.)) 

            # Create dissipator and lindbladian
            if case['test name'] == 'Spin Flipping':
                # Lindblad operator is already in the interaction picture as specified 
                case['dissipator'] = Dissipator(case['basis'], case['lindblad operators'], list(np.zeros_like(frame_energies))) 
            else:
                case['dissipator'] = Dissipator(case['basis'], case['lindblad operators'], frame_energies) 
            case['lindbladian'] = Lindbladian(hamiltonian = case['hamiltonian'], dissipator = case['dissipator'])

            # Set up initial state for master equation dynamics 
            case['initial state'] = State.from_coefficients(case['basis'], list(case['initial state coefficients'])) 

    def test_dissipation(self):
        ''' Test functionality and accuracy of master equation dynamics with dissipators.''' 
        for test_case in self.test_cases:
            # Evolve master equation in time 
            times = np.linspace(0., test_case['duration'], 50)

            init_state = test_case['initial state']
            rho_t = init_state.propagate_using_master_equation(test_case['lindbladian'], test_case['duration'], times) 

            if test_case['test name'] == 'Ion Heating':
                fock_dimension = test_case['fock dimension'] 
                # Trace out motional modes 
                populations = np.array([rho.compute_basis_state_probabilities() for rho in rho_t] )
            
                # Compare population of |1> to reference 
                motional_rhos = rho_t
                for spin in test_case['qubits']:
                    motional_rhos = [rho.trace_out_degree_of_freedom(spin) for rho in motional_rhos]
    
                fock_populations = np.array([rho.compute_basis_state_probabilities() for rho in motional_rhos])
                n_t = np.zeros(len(times), dtype=complex)
                for i, t in enumerate(times):
                    n_t[i] = np.trace( Fock.number(fock_dimension).dot(motional_rhos[i].density_matrix)) 
                Y = n_t.real
                Y_ref = test_case['reference function'](times, test_case['N thermal'], 1000.)  
                # Check sum of squared errors  
                self.assertAlmostEqual(np.mean(np.abs(Y - Y_ref)**2), 0., places=7)
                continue 

            else:
                populations = np.array([rho.compute_basis_state_probabilities() for rho in rho_t] )
                # Compare population of |1> to reference 
                Y = populations[:, 1]
            reference_function = test_case['reference function']

            if test_case['test name'] == 'Spin Flipping':
                # Simply check equality of |0> population with the reference
                P0_final = populations[-1, 0]
                P0_final_ref = spin_flip_exp_decay(test_case['duration'], test_case['rabi rate'], 1./test_case['characteristic time'])
                self.assertAlmostEqual(P0_final, P0_final_ref, places = 4)
            else:
                guess_params = test_case['reference function guess parameters']
                popt, pcov = curve_fit(reference_function, times, Y, p0 = guess_params) 
                gamma_fit = popt[1]
                t_decay_fit = 1./gamma_fit
    
                self.assertAlmostEqual(t_decay_fit, test_case['characteristic time'], places=7)

    def test_modulated_composite_lindblad_operator(self):
        """ Amplitude- and phase-modulated resonant drive with white amplitude noise, Omega -> Omega (1 + xi),
            <xi(t) xi(t')> = kappa delta(t - t'). The noise is one composite Lindblad operator sqrt(kappa) K(t) built from the
            drive C and adjoint(C), with K(t) = (Omega/2) (f(t) |1><0| + f*(t) |0><1|) in the qubit frame. The Bloch vector from the
            master equation is checked against a direct integration of
                d rho/dt = -i [K, rho] + kappa (K rho K - {K^2, rho} / 2).
        """
        def modulation(t):
            return (1. + 0.3 * np.sin(TPI * 50e3 * t)) * np.exp(1j * 0.7 * np.sin(TPI * 80e3 * t))

        rabi_rate = TPI * 100e3
        kappa = 2. / (rabi_rate**2 * 20e-6)   # damping time 20 µs for an unmodulated drive
        duration = 12e-6
        times = np.linspace(0., duration, 7)

        qubit = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        basis = StandardBasis([qubit])
        frame_energies = [-state.energy for state in basis.states]
        qubit_frequency = qubit.energy_levels[1].energy - qubit.energy_levels[0].energy

        # Build coupling operator for raising |g> --> |e> 
        raising = 0.5 * rabi_rate * basis.enlarge_matrix(Pauli.plus, [qubit])
        C = CouplingOperator.from_matrix(basis, raising, qubit_frequency, modulation_function=modulation)
        C_dagger = adjoint(C) 

        # Build Lindblad operator using drive and drive^dagger:
        L = LindbladOperator(basis, [C, C_dagger], frame_energies, prefactor=np.sqrt(kappa))
        # Build the Lindbladian from the Hamiltonian and the Dissipator; dissipator takesin the Lindblad operator 
        lindbladian = Lindbladian(Hamiltonian(basis, [C], frame_energies), Dissipator(basis, [L], frame_energies))
        rho_t = State.from_coefficients(basis, [1., 0.]).propagate_using_master_equation(lindbladian, duration, times)

        # Bloch-vector observables as static operators 
        observables = [CouplingOperator.from_matrix(basis, basis.enlarge_matrix(Pauli.X, [qubit]), 0.),
                       CouplingOperator.from_matrix(basis, basis.enlarge_matrix(Pauli.Y, [qubit]), 0.),
                       EnergyShiftOperator.from_matrix(basis, basis.enlarge_matrix(Pauli.Z, [qubit]))]
        bloch_ionsim = np.array([[rho.compute_operator_observable_expectation(O) for O in observables] for rho in rho_t])

        # Reference: direct integration in the qubit frame, where the resonant drive has no oscillating phase 
        def K(t):
            f = modulation(t)
            return 0.5 * rabi_rate * np.array([[0., np.conj(f)], [f, 0.]])

        def right_hand_side(t, y):
            rho, k = y.reshape(2, 2), K(t)
            return (-1j * (k @ rho - rho @ k) + kappa * (k @ rho @ k - 0.5 * (k @ k @ rho + rho @ k @ k))).ravel()

        rho0 = np.array([[1., 0.], [0., 0.]], dtype=complex)
        solution = solve_ivp(right_hand_side, (0., duration), rho0.ravel(), t_eval=times, method='DOP853', rtol=1e-11, atol=1e-13)
        paulis = [Pauli.X, Pauli.Y, Pauli.Z]
        bloch_direct = np.array([[np.trace(P @ y.reshape(2, 2)) for P in paulis] for y in solution.y.T])

        np.testing.assert_allclose(bloch_ionsim, bloch_direct, atol=1e-6)
        self.assertGreater(np.abs(bloch_direct[-1]).max(), 0.1)   # nontrivial final state 

        # An oscillating (non-static) operator is not accepted as an observable 
        with self.assertRaises(IonSimError):
            rho_t[-1].compute_operator_observable_expectation(C)


if __name__ == '__main__':
    unittest.main()
