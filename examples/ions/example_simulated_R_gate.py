# from basis import StandardBasis, XPauliBasis, XPauliAndFockBasis
# from degree_of_freedom import AtomicSpin, MotionalMode
# from hamiltonian import Hamiltonian
# from coupling import CouplingOperator
# from state import State
# from named_operators import Pauli, Fock

from pathlib import Path

import ionsim as sm

import numpy as np
from scipy.sparse import kron as skron
import h5py

from icecream import ic

sparse = False

modulate_amplitude = False

# number of qubits: 
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
    # Extract carrier / resonant frequencies of each state for interaction frame: 
    interaction_frame_energies = [-state.energy for state in basis.states] # implement arbitrary hamiltonian (with time-dependence? need an adiabatic intertwiner)

    # Create a Hamiltonian from the list of basis states (basis), the list of operators (operators).
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
        tau = abs(theta)/rabi_rate
        hamiltonian = R_hamiltonian(basis, phi, rabi_rate, omega + domega, sparse=sparse, mod=amp_mod)
        start = time.perf_counter()
        ic(hamiltonian.hamiltonian_function(0))
        end = time.perf_counter()
        ic(f'Building Hamiltonian took {end - start} s.')
        return sm.Gate.from_hamiltonian(basis, hamiltonian, tau)

    def R(phi, theta, domega, half_box_width):
        def process_matrix_function(domega):
            gate = simulated_R(phi, theta, domega)
            return gate.process_matrix
        if half_box_width == 0:
            omega_noise = None
        else:
            domegas = np.linspace(-half_box_width, half_box_width, 21)
            omega_noise = sm.Noise.from_named_pdf('domega', 'box', {'half_width': half_box_width}, domegas)
        return sm.Gate.from_process_matrix_function(
                basis, process_matrix_function, {'domega': domega}, spins, omega_noise,
            )

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

    if compute_gate_on_grid:

        from itertools import product

        phi = 0
        theta = np.pi/2

        # dphis = np.linspace(-np.pi/10, np.pi/10, 5)
        # half_box_widths = np.linspace(0, np.pi/10, 3)

        domegas = np.linspace(-50 * 2*np.pi*1e3, 50 * 2*np.pi*1e3, 5) 
        half_box_widths = np.linspace(0, 50 * 2*np.pi*1e3, 3) 

        dxs = domegas
        dys = half_box_widths

        gate_name = 'sqrtX'
        dx_name = 'domega'
        dy_name = 'half-box-width'

        size = len(basis.states)**2

        chi_inv = np.linalg.inv(ideal_R(phi, theta).process_matrix)
        def build_gate_data(val):
            gate = R(phi, theta, *val)
            return gate.process_matrix.dot(chi_inv) - np.eye(size)

        grids =[dxs, dys]
        vals = list(product(*grids))
        lens = [len(grid) for grid in grids]

        start = time.perf_counter()
        gate_data = np.array([build_gate_data(val) for val in vals])
        end = time.perf_counter()
        ic(f'Simulation took {end-start} s.')

        ic(gate_data)

        F_data = np.empty((size, size, *lens), dtype='complex')
        for i in range(size):
            for j in range(size):
                F_data[i,j] = np.array([gd[i,j] for gd in gate_data]).reshape(*lens)
        # ic(F_data)

        attributes = {
            'gate_name': gate_name,
            'dx_name': dx_name,
            'dy_name': dy_name,
        }

        # Opening the file with 'w' allows reading and writing and
        # truncates existing data. See
        # https://docs.h5py.org/en/stable/high/file.html
        with h5py.File(data_filename, 'w') as datafile:
            save_matrix(datafile, dxs, 'dx', attributes)
            save_matrix(datafile, dys, 'dy', attributes)
            save_matrix(datafile, F_data, 'relative_error', attributes)

    if compute_interpolated_gate:
        from csaps import NdGridCubicSmoothingSpline

        phi = 0
        theta = np.pi/2

        gate_name = 'sqrtX'
        dx_name = 'domega'
        dy_name = 'half-box-width'

        size = len(basis.states)**2

        # This time open the data file read-only
        with h5py.File(data_filename, 'r') as datafile:
            dxs, _ = load_matrix(datafile, 'dx')
            dys, _ = load_matrix(datafile, 'dy')
            F_data, _ = load_matrix(datafile, 'relative_error')

        # ic(dphi0s, half_box_widths)
        # ic(F_data)

        grids =[dxs, dys]

        F_spline_reals = {}
        F_spline_imags = {}
        for i in range(size):
            for j in range(size):
                F_spline_reals[i,j] = NdGridCubicSmoothingSpline(grids, F_data[i,j].real, smooth=1)
                F_spline_imags[i,j] = NdGridCubicSmoothingSpline(grids, F_data[i,j].imag, smooth=1)

        # ic(
        #     F_spline_reals[0,0]([0, np.pi/10]).item(),
        #     F_spline_imags[0,0]([0, np.pi/10]).item()
        # )

        def F(dx, dy):
            return np.array([
                [
                    F_spline_reals[i,j]([dx, dy]).item()
                    + 1j * F_spline_imags[i,j]([dx, dy]).item()
                    for j in range(size)
                ]
                for i in range(size)
            ])

        # ic(F(np.pi/10, np.pi/10))

        def interpolated_R(phi, theta, dx, dy):
            return sm.Gate(basis, (F(dx, dy) + np.eye(size)).dot(ideal_R(phi, theta).process_matrix))

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


def save_matrix(datafile, matrix, pathname, attributes=None):
    """Save a matrix in as a dataset in an HDF5 file."""
    dataset = datafile.require_dataset(pathname, shape=matrix.shape, dtype=matrix.dtype, data=matrix)
    if attributes:
        for name, value in attributes.items():
            dataset.attrs[name] = value
    return dataset


def load_matrix(datafile, pathname):
    """Load a matrix into a numpy array and return the attributes
    associated with the HDF5 dataset."""
    dataset = datafile[pathname]
    arr = np.empty(dataset.shape, dtype=dataset.dtype)
    dataset.read_direct(arr)
    attributes = {name: value for name, value in dataset.attrs.items()}
    return arr, attributes


if __name__ == '__main__':
    main()
