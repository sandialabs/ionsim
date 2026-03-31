from pathlib import Path

import ionsim as sm

import numpy as np
from scipy.sparse import kron as skron
import h5py

from icecream import ic
import sys

sparse = False

modulate_amplitude = False

num_spins = 1

spins = [
    sm.AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
    for _ in range(num_spins)
]
basis = sm.StandardBasis([*spins])

target_spins = [spins[0]]

def R_hamiltonian(basis, phi, rabi_rate, omega, sparse=False, mod=None):

    phase = phi
    prefactor = np.exp(1j*phase) * rabi_rate/2  

    raise_target_spins = [basis.enlarge_matrix(sm.Pauli.plus, [spin]) for spin in target_spins]

    operator = prefactor * raise_target_spins[0]

    operators = [
        sm.CouplingOperator.from_matrix(basis, operator, omega, modulation_function=mod),
    ]
    interaction_frame_energies = [-state.energy for state in basis.states] # implement arbitrary hamiltonian (with time-dependence? need an adiabatic intertwiner)
    return sm.Hamiltonian(basis, operators, interaction_frame_energies, sparse=sparse)

rabi_rate = 100e3 * 2*np.pi # rad./s
detuning = 0

omega = (
    + target_spins[0].energy_levels[1].energy - target_spins[0].energy_levels[0].energy
    + detuning
)

# TODO: fix for all theta and tau
# if modulate_amplitude:
#     def amp_mod(t):
#         width = 33.1e-6
#         gaussian = np.exp(-(t - tau/2)**2 / (2 * width**2))
#         return np.sqrt(gaussian)
# else:
#     amp_mod = None

amp_mod = None

def main():
    import time
    from matplotlib import pyplot as plt

    # dphis = np.linspace(-np.pi, np.pi, 21)
    # phi_noise = sm.Noise.from_named_pdf('dphi', 'gaussian', {'standard_deviation': np.pi/10}, dphis)

    def simulated_R(phi, theta, domega):
        """ Builds R(phi, theta) Hamiltonian for a frequency change omega + domega, returns gate """ 
        tau = abs(theta)/rabi_rate
        hamiltonian = R_hamiltonian(basis, phi, rabi_rate, omega + domega, sparse=sparse, mod=amp_mod)
        start = time.perf_counter()
        ic(hamiltonian.hamiltonian_function(0))
        end = time.perf_counter()
        ic(f'Building Hamiltonian took {end - start} s.')
        return sm.Gate.from_hamiltonian(basis, hamiltonian, tau)

    def R(phi, theta, domega, half_box_width):
        """ Builds a process matrix function, then a gate by adding optional noise to it """ 
        def process_matrix_function(domega):
            gate = simulated_R(phi, theta, domega) # builds Hamiltonian and returns gate 
            return gate.process_matrix
        if half_box_width == 0:
            omega_noise = None
        else:
            domegas = np.linspace(-half_box_width, half_box_width, 21)
            omega_noise = sm.Noise.from_named_pdf('domega', 'box', {'half_width': half_box_width}, domegas)
        return sm.Gate.from_process_matrix_function(
                basis, process_matrix_function, {'domega': domega}, omega_noise,
            )
                #basis, process_matrix_function, {'domega': domega}, spins, omega_noise,

    def ideal_R(phi, theta):
        return sm.Gate.from_unitary(basis, sm.Unitary.R(phi, theta), target_spins)

    def process_fidelity(phi, theta, dx, dy):
        return R(phi, theta, dx, dy).compute_process_fidelity(ideal_R(phi, theta).process_matrix)

    compute_state_fidelity = False
    compute_process_fidelity = False
    compute_gate_on_grid = True
    compute_interpolated_gate = True

    data_directory = Path.home() / "tmp" / "ionsim_examples_data"
    if not data_directory.exists():
        data_directory.mkdir(parents=True, exist_ok=True)

    data_filename = data_directory / "simr.hdf5"

    if compute_state_fidelity:
        phi = 0
        theta = np.pi/2
        tau = abs(theta)/rabi_rate
        target_wavefunction = 1/np.sqrt(2) * np.array([1, -1j])

        hamiltonian = R_hamiltonian(basis, phi, rabi_rate, omega, sparse=sparse, mod=amp_mod)

        start = time.perf_counter()
        ic(hamiltonian.hamiltonian_function(0))
        end = time.perf_counter()
        ic(f'Building Hamiltonian took {end - start} s.')

        coefs = np.zeros(len(basis.states))
        coefs[0] = 1
        initial_state = sm.State.from_coefficients(basis, list(coefs))

        times = np.linspace(0, tau, 41) # setting to None will return only the final spin state

        start = time.perf_counter()
        psis = initial_state.propagate_using_schrodinger_equation(hamiltonian, tau, times)
        end = time.perf_counter()
        ic(f'Propagating state took {end - start} s.')

        probs = np.array([psi.compute_basis_state_probabilities() for psi in psis])
        ic(probs[-1,:])

        target_psi = sm.State.from_wavefunction(basis, target_wavefunction)
        fidelity = psis[-1].compute_state_fidelity(target_psi.density_matrix)
        ic(fidelity)

        for i,state in enumerate(basis.states):
            plt.plot(times, probs[:, i], label=state.name)
        plt.ylabel('Probabilities')
        plt.xlabel('Gate Duration (s)')
        plt.legend()
        plt.show()

    if compute_process_fidelity:

        phi = 0
        theta = np.pi/2

        # dphi = 0
        # half_box_width = np.pi/10

        domega = 0
        half_box_width = 50 * 2*np.pi*1e3

        dx = domega
        dy = half_box_width

        start = time.perf_counter()
        ic(process_fidelity(phi, theta, dx, dy))
        end = time.perf_counter()
        ic(f'Simulating process fidelity took {end - start} s.')

    # Step 1: Set up a grid where you actually build the gates. 
    if compute_gate_on_grid:

        from itertools import product

        phi = 0
        theta = np.pi/2

        # Computing gate on a grid where x is a frequency offset from resonance 
        #  and y is a noise width. 
        #  Ex] So y = 0 corresponds to no noise. 
        #  Ex] x = 0 corresponds to being on resonance with some noise (unless y=0). 
        domegas = np.linspace(-50 * 2*np.pi*1e3, 50 * 2*np.pi*1e3, 5) 
        half_box_widths = np.linspace(0, 50 * 2*np.pi*1e3, 3) 

        dxs = domegas
        dys = half_box_widths

        gate_name = 'sqrtX'
        dx_name = 'domega'
        dy_name = 'half_box_width'

        grid_axes = {dx_name : dxs, dy_name : dys} 

        # Define a gate function to build the gate interpolant. 
        def R_function(domega, half_box_width):
            """ Gate function of the interpolation parameters; returns a Gate object """ 
            return R(phi, theta, domega, half_box_width)

        print("Building gate interoplant using process matrix function")
        R_gate_interpolant = sm.GateInterpolant.from_gate_function(R_function, grid_axes, gate_name) 
        grids = R_gate_interpolant.grids 
        grid = R_gate_interpolant.grid
        lens = R_gate_interpolant.grid_lengths 

        # Set up comparisons of the parametrized gate to an ideal R gate 
        size = len(basis.states)**2

        # Compute gate residuals using inverse of ideal R gate  
        chi_inv = np.linalg.inv(ideal_R(phi, theta).process_matrix)

        # Define a functional of the gate to return the desired property  
        def relative_err_gate_functional(gate):
            return gate.process_matrix.dot(chi_inv) - np.eye(size)

        gate_residual_data = R_gate_interpolant.compute_functional_of_gates(relative_err_gate_functional) 

        ic(gate_residual_data)

        # Set up matrix-valued residuals as a function of the parameter grid  
        F_data = np.empty((size, size, *lens), dtype='complex')
        for i in range(size):
            for j in range(size):
                F_data[i,j] = np.array([gd[i,j] for gd in gate_residual_data]).reshape(*lens)
        attributes = {
            'gate_name': gate_name,
            'dx_name': dx_name,
            'dy_name': dy_name,
        }

        # Set up a dictionary of results and write to an hdf5 file  
        results_dictionary = {'dx' : dxs, 'dy': dys, 'relative_error': F_data}
        sm.io.write_results_to_file(data_filename, results_dictionary, attributes)
                
    # Step 2: Use the grid of gates to interpolate. 
    if compute_interpolated_gate:
        from csaps import NdGridCubicSmoothingSpline

        phi = 0
        theta = np.pi/2

        gate_name = 'sqrtX'
        dx_name = 'domega'
        dy_name = 'half_box_width'

        size = len(basis.states)**2

        # Read results from file  
        results_dictionary = sm.io.read_results_from_file(data_filename)        
        dxs = results_dictionary['dx']
        dxy = results_dictionary['dy']
        F_data = results_dictionary['relative_error']

        # F_data <==> Gate-valued (process matrix) residuals. For every x,y gate parameter, there's a d^2 x d^2 process matrix .
        F_spline_reals, F_spline_imags = R_gate_interpolant.construct_spline_for_gate_derived_matrix_property(F_data, complex_data=True)

        # Using the interpolants, build a function to return F(x,y) for arbitary x,y pairs
        F_function = R_gate_interpolant.interpolant_function_from_splines([F_spline_reals, F_spline_imags], 'relative_error')

        def interpolated_R(phi, theta, dx, dy):
            return sm.Gate(basis, (F_function(dx, dy) + np.eye(size)).dot(ideal_R(phi, theta).process_matrix))

 #        def interpolated_R(phi, theta, dx, dy):
 #            def R_process_matrix_from_error_functional(dx, dy): 
 #                return (F_function(dx, dy) + np.eye(size)).dot(ideal_R(phi, theta).process_matrix) 
 #            return R_gate_interpolant.interpolated_gate_from_process_matrix_interpolant(basis, R_process_matrix_from_error_functional)  

        dxs2 = np.linspace(dxs[0], dxs[-1], (len(dxs)-1)*2 + 1)
        dy = dys[-1]
        ms_gates = []
        for dx in dxs:
            ms_gates.append(R(phi, theta, dx, dy))
        ms_gates2 = []
        for dx in dxs2:
            ms_gates2.append(R(phi, theta, dx, dy))
        fidelities = [gate.compute_process_fidelity(ideal_R(phi, theta).process_matrix) for gate in ms_gates]
        fidelities2 = [gate.compute_process_fidelity(ideal_R(phi, theta).process_matrix) for gate in ms_gates2]
        approx_fids = [
            interpolated_R(phi, theta, dx, dy).compute_process_fidelity(
                ideal_R(phi, theta).process_matrix
            ) for dx in dxs2
        ]

        plt.rcParams.update({'font.size': 16})

        dx_scale = 1/(2*np.pi*1e3)
        plt.plot(dxs * dx_scale, 1-np.array(fidelities), 'o', label='simulation: grid point')
        plt.plot(dxs2 * dx_scale, 1-np.array(approx_fids), '-', label='interpolation')
        plt.plot(dxs2 * dx_scale, 1-np.array(fidelities2), '.', label='simulation: off grid')
        # plt.xlabel(f'Error Parameter: {dx_name}')
        plt.xlabel(f'Frequency Error (kHz)')
        plt.ylabel('Infidelity')
        plt.legend()
        plt.savefig(data_directory / f'infidelity_vs_{dx_name}.pdf', bbox_inches='tight')
        plt.show()

        dx = dxs[-1]
        dys2 = np.linspace(dys[0], dys[-1], (len(dys)-1)*2 + 1)
        ms_gates = []
        for dy in dys:
            ms_gates.append(R(phi, theta, dx, dy))
        ms_gates2 = []
        for dy in dys2:
            ms_gates2.append(R(phi, theta, dx, dy))
        fidelities = [gate.compute_process_fidelity(ideal_R(phi, theta).process_matrix) for gate in ms_gates]
        fidelities2 = [gate.compute_process_fidelity(ideal_R(phi, theta).process_matrix) for gate in ms_gates2]
        approx_fids = [
            interpolated_R(phi, theta, dx, dy).compute_process_fidelity(
                ideal_R(phi, theta).process_matrix
            ) for dy in dys2
        ]

        dy_scale = 1/(2*np.pi*1e3)
        plt.plot(dys * dy_scale, 1-np.array(fidelities), 'o', label='simulation: grid point')
        plt.plot(dys2 * dy_scale, 1-np.array(approx_fids), '-', label='interpolation')
        plt.plot(dys2 * dy_scale, 1-np.array(fidelities2), '.', label='simulation: off grid')
        # plt.xlabel(f'Error Parameter: {dy_name}')
        plt.xlabel(f'Half-Width of Boxed White Noise (kHz)')
        plt.ylabel('Infidelity')
        plt.legend()
        plt.savefig(data_directory / f'infidelity_vs_{dy_name}.pdf', bbox_inches='tight')
        plt.show()

if __name__ == '__main__':
    main()
