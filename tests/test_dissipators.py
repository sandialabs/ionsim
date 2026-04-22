import unittest
import numpy as np
from scipy.linalg import expm
from scipy.optimize import curve_fit
from scipy.sparse import kron as skron

from ionsim.zeeman_solver import ZeemanHyperfineSolver 
from ionsim.basis import StandardBasis
from ionsim.operator import CouplingOperator, EnergyShiftOperator
from ionsim.named_operators import Pauli, Fock
from ionsim.degree_of_freedom import AtomicSpin, MotionalMode
from ionsim.hamiltonian import Hamiltonian
from ionsim.dissipator import Dissipator, Lindbladian
from ionsim.state import State

TPI = 2*np.pi
# Helper functions for dissipator references 
def oscillitory_exponential_decay(t, A, gamma, C, w, phase):
    return A*np.exp(-t*gamma)*np.cos(w*t + phase) + C 

def exponential_decay(t, A, gamma, C): 
    return A*np.exp(-t*gamma) + C 

def exponential_warmup(t, y0, gamma):
    return y0 + (0. - y0)*np.exp(-gamma*t)


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
            }
        ]

        # Set up the qubits, hamiltonian, dissipator, and lindbladian for each test case.  
        for case in self.test_cases:
            # Create spins and basis. Store for each case 
            case['qubits'] = [
                AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
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
                case['lindblad operators'] = [EnergyShiftOperator.from_matrix(case['basis'], dephase_q1)]                
            elif case['test name'] == 'Spontaneous Emission':
                # Spontaneous emission of a qubit in the |1> state; Decay from |1> to |0>.
                case['basis'] = StandardBasis([*case['qubits']])
                frame_energies = [-state.energy for state in case['basis'].states] 
                case['hamiltonian'] = None

                case['duration'] = 5./decay_rate

                # Dephasing with Lindblad operator L = Z on qubit 1:                 
                lower_qubit1 = np.sqrt(decay_rate) * case['basis'].enlarge_matrix(Pauli.minus, [case['qubits'][0]]) 
                case['lindblad operators'] = [CouplingOperator.from_matrix(case['basis'], lower_qubit1, 0.)] 
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

                # Tensor raising/lowering operators with spin identity for each mode that is heating  
                lowering_operators = []
                raising_operators = []

                # Basis is of the form spins x modes, so we tensor in the that order: 
                N_thermal = case['N thermal']
                lowering_operator = np.sqrt(decay_rate*(N_thermal+1)) * skron(spin_identity, mode_lowering_matrix) 
                raising_operator = np.sqrt(decay_rate*N_thermal) * skron(spin_identity, mode_raising_matrix) 

                case['lindblad operators'] = []
                case['lindblad operators'].append(CouplingOperator.from_matrix(case['basis'], lowering_operator, 0.)) 
                case['lindblad operators'].append(CouplingOperator.from_matrix(case['basis'], raising_operator, 0.)) 

            # Create dissipator and lindbladian
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

            guess_params = test_case['reference function guess parameters']
            popt, pcov = curve_fit(reference_function, times, Y, p0 = guess_params) 
            gamma_fit = popt[1]
            t_decay_fit = 1./gamma_fit

            self.assertAlmostEqual(t_decay_fit, test_case['characteristic time'], places=7)


if __name__ == '__main__':
    unittest.main()
