#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************
from pathlib import Path
import numpy as np
from scipy.sparse import kron as skron
from icecream import ic
import time
from matplotlib import pyplot as plt

import ionsim as sm
    
# R-gate helper functions 
# Basic R hamiltonian function: 
def R_hamiltonian(basis, phi, rabi_rate, omega, target_spins, sparse=False, mod=None):
    """ Returns an IonSim Hamiltonian for the R(theta, phi) gate.

        - theta is achieved by dynamic evolution of this Hamiltonian with the specified Rabi rate 
        - phi is the gate phase that sets the spin rotation axis (phi = 0 <==> Rx gate)
        - omega is the driving frequency in rad/s  
        - target_spins is a list of IonSim degrees of freedom corresponding to driven qubits   
     """ 
    prefactor = np.exp(1j*phi) * rabi_rate/2  

    raise_target_spins = [basis.enlarge_matrix(sm.Pauli.plus, [spin]) for spin in target_spins]

    operator = prefactor * raise_target_spins[0]

    operators = [
        sm.CouplingOperator.from_matrix(basis, operator, omega, modulation_function=mod),
    ]
    interaction_frame_energies = [-state.energy for state in basis.states] # implement arbitrary hamiltonian (with time-dependence? need an adiabatic intertwiner)
    return sm.Hamiltonian(basis, operators, interaction_frame_energies, sparse=sparse)

def simulated_R(basis, phi, theta, omega, domega, rabi_rate, target_spins, sparse=False, mod=None):
    """ Builds R(phi, theta) Hamiltonian for a frequency change omega + domega, returns gate """ 
    tau = abs(theta)/rabi_rate
    hamiltonian = R_hamiltonian(basis, phi, rabi_rate, omega + domega, target_spins, sparse, mod)
    start = time.perf_counter()
    ic(hamiltonian.hamiltonian_function(0))
    end = time.perf_counter()
    ic(f'Building Hamiltonian took {end - start} s.')
    return sm.Gate.from_hamiltonian(basis, hamiltonian, tau)

def R(basis, phi, theta, domega, half_box_width, omega, rabi_rate, target_spins, sparse, mod):
    """ Builds a process matrix function, then a gate by adding optional noise to it """ 
    def process_matrix_function(domega):
        gate = simulated_R(basis, phi, theta, omega, domega, rabi_rate, target_spins, sparse, mod) # builds Hamiltonian and returns gate 
        return gate.process_matrix
    if half_box_width == 0:
        omega_noise = None
    else:
        domegas = np.linspace(-half_box_width, half_box_width, 21)
        omega_noise = sm.Noise.from_named_pdf('domega', 'box', {'half_width': half_box_width}, domegas)
    return sm.Gate.from_process_matrix_function(
            basis, process_matrix_function, {'domega': domega}, omega_noise,
        )

def ideal_R(basis, phi, theta, target_spins):
    return sm.Gate.from_unitary(basis, sm.Unitary.R(phi, theta), target_spins)

def process_fidelity(basis, phi, theta, dx, dy, target_spins):
    return R(basis, phi, theta, dx, dy, rabi_rate, target_spins).compute_process_fidelity(ideal_R(phi, theta).process_matrix)


def main():

    sparse = False
    modulate_amplitude = False
    num_spins = 1
    
    # Create a basis of 1 qubit: 
    spins = [
        sm.AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        for _ in range(num_spins)
    ]
    basis = sm.StandardBasis([*spins])
    target_spins = [spins[0]]

    # Fix the rabi rate and detuning 
    rabi_rate = 100e3 * 2*np.pi # rad./s
    detuning = 0
    
    omega = (
        + target_spins[0].energy_levels[1].energy - target_spins[0].energy_levels[0].energy
        + detuning
    )
    amp_mod = None
    import_interpolant = False  # option if the interpolant has previously been built  

    # Define directory where interpolant and output plots will be stored 
    data_directory = Path.home() / "tmp" / "ionsim_examples_data"
    if not data_directory.exists():
        data_directory.mkdir(parents=True, exist_ok=True)

    # Filename for the interpolant 
    data_filename = data_directory / "test_R.hdf5"

    ## R Gate interpolation: 
    # Step 1: Set up a grid where you actually build the gates. 
    phi = 0
    theta = np.pi/2

    # Computing gate on a grid where x is a frequency offset from resonance 
    #  and y is a noise width. 
    #  Ex] So y = 0 corresponds to no noise. 
    #  Ex] x = 0 corresponds to being on resonance with some noise (unless y=0). 
    dx_name = 'domega'
    dy_name = 'half_box_width'

    if import_interpolant : 
        ic(f" -- Reading R gate data from file {data_filename} --- ")
        # Optional: Write interpolant to a file using gate interpolant class 
        #R_gate_interpolant.write_to_file(data_filename)
        R_gate_interpolant_v2 = sm.GateInterpolant.from_file(data_filename, basis)
        interpolated_R = R_gate_interpolant_v2.interpolated_gate_function

        grid_axes = R_gate_interpolant_v2.grid_axes
        dxs = grid_axes[dx_name]
        dys = grid_axes[dy_name]
    else:
        # Define a gate function to build the gate interpolant. 
        def R_function(domega, half_box_width):
            """ Gate function of the interpolation parameters; returns a Gate object """ 
            # domega and half_box_width are the interpolant parameters 
            return R(basis, phi, theta, domega, half_box_width, omega, rabi_rate, target_spins, sparse, amp_mod)
    
        # 1. Construct the gate interpolant class instance 
        ic("Building gate interpolant using gate function")
        gate_name = 'sqrtX'
        domegas = np.linspace(-50 * 2*np.pi*1e3, 50 * 2*np.pi*1e3, 5) 
        half_box_widths = np.linspace(0, 50 * 2*np.pi*1e3, 3) 

        dxs = domegas
        dys = half_box_widths

        grid_axes = {dx_name : dxs, dy_name : dys} 
        R_gate_interpolant = sm.GateInterpolant.from_gate_function(R_function, grid_axes, gate_name) 
    
        # 2 Build a gate interpolating function (this uses cubic splines): returns Gate evaluated at grid / off-grid parameter values  
        """ Ex] interpolated_R(x = 0.5 * 2π * 1E3, y = 2.) returns an R Gate object at domega = 0.5 * 2π * 1E3, half_box_width 2. """ 
        interpolated_R = R_gate_interpolant.interpolated_gate_function # returns a Gate object at a grid point 

        # 3. Save to a file to load in the future 
        attributes = {
            'gate_name': gate_name,
            'dx_name': dx_name,
            'dy_name': dy_name,
        }
        ic(f" -- Writing R gate data to file {data_filename} --- ")
        R_gate_interpolant.write_to_file(data_filename, attributes)

    dxs2 = np.linspace(dxs[0], dxs[-1], (len(dxs)-1)*2 + 1)
    # We have interpolated the R gate over a two-dimensional grid. 
    # Let's plot a slice in each direction to observe the interpolation and impact of each parameter on the gate infidelity  
    dy = dys[-1] # Pick the final half box width value as constant to study impact as a funciton of domega (dx) 
    ms_gates = []

    # Check that the interpolation is consistent / performing well by studying gate fidelity 
    #   on the grid (dx's) and off-the-grid (dx2): 
    for dx in dxs:
        ms_gates.append(R(basis, phi, theta, dx, dy, omega, rabi_rate, target_spins, sparse, amp_mod))
    ms_gates2 = []
    for dx in dxs2:
        ms_gates2.append(R(basis, phi, theta, dx, dy, omega, rabi_rate, target_spins, sparse, amp_mod))
    fidelities = [gate.compute_process_fidelity(ideal_R(basis, phi, theta, target_spins).process_matrix) for gate in ms_gates]
    fidelities2 = [gate.compute_process_fidelity(ideal_R(basis, phi, theta, target_spins).process_matrix) for gate in ms_gates2]
    approx_fids = [
        interpolated_R(dx, dy).compute_process_fidelity(
            ideal_R(basis, phi, theta, target_spins).process_matrix
        ) for dx in dxs2
    ]

    plt.rcParams.update({'font.size': 16})

    dx_scale = 1/(2*np.pi*1e3)
    plt.plot(dxs * dx_scale, 1-np.array(fidelities), 'o', label='simulation: grid point')
    plt.plot(dxs2 * dx_scale, 1-np.array(approx_fids), '-', label='interpolation')
    plt.plot(dxs2 * dx_scale, 1-np.array(fidelities2), '.', label='simulation: off grid')
    plt.xlabel(f'Frequency Error (kHz)')
    plt.ylabel('Infidelity')
    plt.legend()
    plt.savefig(data_directory / f'infidelity_vs_{dx_name}.pdf', bbox_inches='tight')
    plt.show()

    # Now studying a slice at constant dx (domega) to observe the interpolation and impact of dy (half box width) on the gate infidelity  
    dx = dxs[-1] # Pick the final domega value as constant to study impact as a funciton of half box width (dy) 
    dys2 = np.linspace(dys[0], dys[-1], (len(dys)-1)*2 + 1)
    # Check that the interpolation is consistent / performing well by studying gate fidelity 
    #   on the grid (dy's) and off-the-grid (dy2): 
    ms_gates = []
    for dy in dys:
        ms_gates.append(R(basis, phi, theta, dx, dy, omega, rabi_rate, target_spins, sparse, amp_mod))
    ms_gates2 = []
    for dy in dys2:
        ms_gates2.append(R(basis, phi, theta, dx, dy, omega, rabi_rate, target_spins, sparse, amp_mod))
    fidelities = [gate.compute_process_fidelity(ideal_R(basis, phi, theta, target_spins).process_matrix) for gate in ms_gates]
    fidelities2 = [gate.compute_process_fidelity(ideal_R(basis, phi, theta, target_spins).process_matrix) for gate in ms_gates2]
    approx_fids = [
        interpolated_R(dx, dy).compute_process_fidelity(
            ideal_R(basis, phi, theta, target_spins).process_matrix
        ) for dy in dys2
    ]

    dy_scale = 1/(2*np.pi*1e3)
    plt.plot(dys * dy_scale, 1-np.array(fidelities), 'o', label='simulation: grid point')
    plt.plot(dys2 * dy_scale, 1-np.array(approx_fids), '-', label='interpolation')
    plt.plot(dys2 * dy_scale, 1-np.array(fidelities2), '.', label='simulation: off grid')
    plt.xlabel(f'Half-Width of Boxed White Noise (kHz)')
    plt.ylabel('Infidelity')
    plt.legend()
    plt.savefig(data_directory / f'infidelity_vs_{dy_name}.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
