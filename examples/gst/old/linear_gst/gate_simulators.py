import numpy as np
import ionsim as ism
    
num_spins = 1
    
spins = [
    ism.AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
    for _ in range(num_spins)
]

basis = ism.StandardBasis([*spins])
target_spins = [spins[0]]

qubit_frequency = target_spins[0].energy_levels[1].energy - target_spins[0].energy_levels[0].energy


""" Example containing gate simulators that take in a state and return a state. """
def idle_state_propagator(rho: ism.State) -> ism.State:
    theta = 0.0035248
    I = np.eye(2,dtype=complex)
    I[0,0] = np.exp( - 1j * theta ) 
    I[1,1] = np.exp( 1j * theta ) 
    return ism.State.from_density_matrix(rho.basis, I @ rho.density_matrix @ (I.T).conjugate()) 

def R_gate_lindbladian_function(rabi_rate: float, phi: float, phase_mean: float, phase_variance: float):
    """ Builds a Lindbladian as a function of R gate rotation angle and phase, as well as phase noise mean and variance """ 
    # Onus is on the user to achieve theta = rabi_rate * gate_duration outside of this function.  
    raise_qubit_matrix = 0.5 * rabi_rate * np.exp(1j*phi)*basis.enlarge_matrix(ism.Pauli.plus, target_spins) 
    laser_freq = qubit_frequency

    # Driving field is on resonance with qubit frequency: 
    coupling_operator = ism.CouplingOperator.from_matrix(basis, raise_qubit_matrix, qubit_frequency) 

    # Include shift from mean of phase noise term  
    coherent_Z_shift = ism.EnergyShiftOperator.from_matrix(basis, ism.Pauli.Z * phase_mean * 0.5) 

    frame_energies = [-state.energy for state in basis.states] 
    H_0 = ism.Hamiltonian(basis, [coupling_operator, coherent_Z_shift], frame_energies) 

    # Dephasing with Lindblad operator L = sqrt(sigma^2) Z = sigma Z, on qubit 1:                 
    dephasing_matrix = np.sqrt(phase_variance) * basis.enlarge_matrix(ism.Pauli.Z , target_spins) 
    lindblad_ops = [ism.EnergyShiftOperator.from_matrix(basis, dephasing_matrix)]               

    dephasing_dissipator = ism.Dissipator(basis, lindblad_ops, frame_energies) 
    return ism.Lindbladian(hamiltonian = H_0, dissipator = dephasing_dissipator)


def X_pi2_state_propagator(rho: ism.State) -> ism.State: 
    """ Process matrix from Lindbladian for X_pi/2 rotation gate as a fxn of over/under rotation (X_rot), phase mean, phase standard deviation""" 
    gate_phase = 0. # for X rotation gate 
    X_rot = 0.015
    phase_mean = 0.025
    phase_std_deviation = 0.01

    theta = np.pi/2. + X_rot
    rabi_rate = theta 
    # Rabi rate is set to theta, so gate_duration is 1. (Omega*t = theta)
    gate_duration = 1. 
    #gate_duration = theta/rabi_rate 

    phase_variance = phase_std_deviation**2 
    # Build X_pi/2 Lindbladian from generalized R gate 
    X_pi2_lindbladian = R_gate_lindbladian_function(rabi_rate, gate_phase, phase_mean, phase_variance) 

    # Propagate the state: 
    return rho.propagate_using_master_equation(X_pi2_lindbladian, gate_duration)

 #    # Option 1: Creates a gate at every function evaluation. The gate is discarded at the end of this function call.   
 #    X_pi2_gate = sm.Gate.from_lindbladian(basis, X_pi2_lindbladian, gate_duration, lindbladian_time_independent=True)
 #    #X_pi2_gate = sm.Gate.from_lindbladian(basis, R_gate_lindbladian_function(rabi_rate, gate_phase, phase_mean, phase_variance), gate_duration)
 #    return X_pi2_gate.process_matrix 


### Note: we could precompute the gate and store it, then just use it during GST circuit simulation. e.g. instead of running a simulation with every time the gate appears in the GSt circuit, we could build the gate once and then just supply it every time. This would mean the gate simulator only does the matrix multiplication to propagate rho -> rho'  


# Returns process matrix for noisy Y_pi/2 gate from the previous functions.  
def Y_pi2_state_propagator(rho: ism.State) -> ism.State: 
    """ Process matrix from Lindbladian for Y_pi/2 rotation gate as a fxn of over/under rotation (Y_rot), phase mean, phase variance """ 
    gate_phase = np.pi/2. # for Y rotation gate 
    Y_rot = 0.0075
    phase_mean = 0.125
    phase_std_deviation = 0.05 
    theta = np.pi/2. + Y_rot
    rabi_rate = theta 
    # Rabi rate is set to theta, so gate_duration is 1. (Omega*t = theta)
    gate_duration = 1. 

    phase_variance = phase_std_deviation**2 
    # Build Y_pi/2 Lindbladian from generalized R gate 
    Y_pi2_lindbladian = R_gate_lindbladian_function(rabi_rate, gate_phase, phase_mean, phase_variance) 
    return rho.propagate_using_master_equation(Y_pi2_lindbladian, gate_duration)

 #    # Option 1: Creates a gate at every function evaluation. The gate is discarded at the end of this function call.   
 #    Y_pi2_gate = sm.Gate.from_lindbladian(basis, Y_pi2_lindbladian, gate_duration, lindbladian_time_independent=True)
 #    return Y_pi2_gate.process_matrix 


 #def X_pi_2_state_propagator(rho: ism.State) -> ism.State:
 #    """ Takes an input state and propagates it under an X_pi/2 gate, returns the resulting state. """
 #    # Sets up an X_pi_2 Hamiltonian where the Rabi rate and duration are set to achieve theta = pi/2 
 #    eps_X = 0.045 # overroration  
 #
 #    x_angle = np.pi/2. + eps_X 
 #    y_angle = 0.01 
 #    z_angle = -0.025 
 #    Rxpi2 = ism.Unitary.R_bloch([x_angle/2., y_angle/2., z_angle/2.]) 
 #    return ism.State.from_density_matrix(rho.basis, Rxpi2 @ rho.density_matrix @ (Rxpi2.T).conjugate())
 #
 #
 #def Y_pi_2_state_propagator(rho: ism.State) -> ism.State:
 #    """ Takes an input state and propagates it under an Y_pi/2 gate, returns the resulting state. """
 #    eps_Y = -0.042   # overroration  
 #
 #    x_angle = -0.033
 #    y_angle = np.pi/2. + eps_Y 
 #    z_angle = 0.0545 
 #    Rypi2 = ism.Unitary.R_bloch([x_angle/2., y_angle/2., z_angle/2.]) 
 #    return ism.State.from_density_matrix(rho.basis, Rypi2 @ rho.density_matrix @ (Rypi2.T).conjugate())
 #
 #
 #def idle_state_propagator(rho: ism.State) -> ism.State:
 #    theta = 0.0035248
 #    I = np.eye(2,dtype=complex)
 #    I[0,0] = np.exp( - 1j * theta ) 
 #    I[1,1] = np.exp( 1j * theta ) 
 #    return ism.State.from_density_matrix(rho.basis, I @ rho.density_matrix @ (I.T).conjugate()) 
