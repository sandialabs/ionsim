import numpy as np
import ionsim as ism


""" Example containing gate simulators that take in a state and return a state. """

def R_hamiltonian(basis, phi, rabi_rate, omega, sparse=False, mod=None):
    phase = phi
    prefactor = np.exp(1j*phase) * rabi_rate/2  
    target_spins = [basis.degrees_of_freedom[0]] 
    raise_target_spins = [basis.enlarge_matrix(ism.Pauli.plus, [spin]) for spin in target_spins]
    operator = prefactor * raise_target_spins[0]
    operators = [
        ism.CouplingOperator.from_matrix(basis, operator, omega, modulation_function=mod),
    ]
    interaction_frame_energies = [-state.energy for state in basis.states] # implement arbitrary hamiltonian (with time-dependence? need an adiabatic intertwiner)
    return ism.Hamiltonian(basis, operators, interaction_frame_energies, sparse=sparse)


def X_pi_2_state_propagator(rho: ism.State) -> ism.State:
    """ Takes an input state and propagates it under an X_pi/2 gate, returns the resulting state. """
    # Sets up an X_pi_2 Hamiltonian where the Rabi rate and duration are set to achieve theta = pi/2 
    theta = np.pi/2.
    levels = rho.basis.degrees_of_freedom[0].energy_levels
    qubit_frequency = np.abs(levels[1].energy - levels[0].energy)
    phase = 0. # for X gate  
    rabi_rate = 1. 
    X_pi2_hamiltonian = R_hamiltonian(rho.basis, phase, rabi_rate, qubit_frequency) 
    gate_duration = theta
    
    rho_propagated = rho.propagate_using_schrodinger_equation(X_pi2_hamiltonian, gate_duration)
    return rho_propagated


def Y_pi_2_state_propagator(rho: ism.State) -> ism.State:
    """ Takes an input state and propagates it under an Y_pi/2 gate, returns the resulting state. """
    # Sets up an Y_pi_2 Hamiltonian where the Rabi rate and duration are set to achieve theta = pi/2 
    theta = np.pi/2.
    levels = rho.basis.degrees_of_freedom[0].energy_levels
    qubit_frequency = np.abs(levels[1].energy - levels[0].energy)
    phase = np.pi/2. # for Y gate  
    rabi_rate = 1. 
    Y_pi2_hamiltonian = R_hamiltonian(rho.basis, phase, rabi_rate, qubit_frequency) 
    gate_duration = theta
    
    rho_propagated = rho.propagate_using_schrodinger_equation(Y_pi2_hamiltonian, gate_duration)
    return rho_propagated


def idle_state_propagator(rho: ism.State) -> ism.State:
    return rho 

