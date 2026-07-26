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


""" Helper functions for creating gate models / gate model functions in IonSim """ 
def R_gate_lindbladian_function(rabi_rate: float, phi: float, coherent_Z_error: float, dephasing_probability: float):
    """ Builds a Lindbladian as a function of R gate rotation angle and phase, as well as detuning error and qubit dephasing. """ 
    # Onus is on the user to achieve theta = rabi_rate * gate_duration outside of this function.  
    raise_qubit_matrix = 0.5 * rabi_rate * np.exp(1j*phi)*basis.enlarge_matrix(ism.Pauli.plus, target_spins) 
    laser_freq = qubit_frequency

    # Driving field is on resonance with qubit frequency: 
    coupling_operator = ism.CouplingOperator.from_matrix(basis, raise_qubit_matrix, qubit_frequency) 

    # Include shift from mean of phase noise term  
    coherent_Z_shift = ism.EnergyShiftOperator.from_matrix(basis, ism.Pauli.Z * coherent_Z_error * 0.5) 

    frame_energies = [-state.energy for state in basis.states] 
    H_0 = ism.Hamiltonian(basis, [coupling_operator, coherent_Z_shift], frame_energies) 

    # Dephasing with Lindblad operator L = sqrt(gamma) Z = sigma Z, on qubit 1, gamma is the dephasing rate                  
    # Since the Lindbladian dynamics are quadratic / bilinear in L, the overall dephasing is gamma x gate duration
    # Therefore, GST estimates gamma x gate duration = dephasing probability, instead of gamma on its own.  
    dephasing_matrix = np.sqrt(dephasing_probability) * basis.enlarge_matrix(ism.Pauli.Z , target_spins) 
    lindblad_ops = [ism.EnergyShiftOperator.from_matrix(basis, dephasing_matrix)]               

    dephasing_dissipator = ism.Dissipator(basis, lindblad_ops, frame_energies) 
    return ism.Lindbladian(hamiltonian = H_0, dissipator = dephasing_dissipator)


""" Example gate model functions, which take in error parameters and return 4 x 4 process matrices representing a single qubit operation. """
def shuttle(Z_error: float):
    """ Returns d^2 x d^2 process matrix in standard basis for Z-rotation by theta """  
    # Build identity matrix with Z rotation by theta:
    theta = Z_error
    I = np.eye(2,dtype=complex)
    I[0,0] = np.exp( - 1j * theta ) 
    I[1,1] = np.exp( 1j * theta ) 
    # Promote to a d^2 x d^2 superoperator 
    return basis.compute_superoperator_from_unitary_operator(I)

def shuttle_w_dephasing(Z_error: float, dephasing_probability: float):
    """ Returns d^2 x d^2 process matrix in standard basis for Z-rotation by theta """  
    # Build identity matrix with Z rotation by theta:
    dephasing_lindbladian = R_gate_lindbladian_function(0., 0., Z_error, dephasing_probability)
    gate_duration = 1. 

    # Compute superoperator form  
    X_pi2_gate = ism.Gate.from_lindbladian(basis, dephasing_lindbladian, gate_duration, lindbladian_time_independent=True)
    return X_pi2_gate.process_matrix


def X_pi2_process_matrix(excess_X_rot: float, dephasing_probability: float):
    """ Process matrix from Lindbladian for X_pi/2 rotation gate as a fxn of over/under rotation (excess_X_rot), dephasing rate""" 
    gate_phase = 0. # for X rotation gate 
    theta = np.pi/2. + excess_X_rot
    rabi_rate = theta 
    # Rabi rate is set to theta, so gate_duration is 1. (Omega*t = theta)
    gate_duration = 1. 
    #gate_duration = theta/rabi_rate 

    coherent_Z_error = 0.
    #dephasing_probability = dephasing_probability**2 
    # Build X_pi/2 Lindbladian from generalized R gate 
    X_pi2_lindbladian = R_gate_lindbladian_function(rabi_rate, gate_phase, coherent_Z_error, dephasing_probability) 

    # Option 1: Creates a gate at every function evaluation. The gate is discarded at the end of this function call.   
    X_pi2_gate = ism.Gate.from_lindbladian(basis, X_pi2_lindbladian, gate_duration, lindbladian_time_independent=True)
    return X_pi2_gate.process_matrix 

def Y_pi2_process_matrix(excess_Y_rot: float, dephasing_probability: float):
    """ Process matrix from Lindbladian for Y_pi/2 rotation gate as a fxn of over/under rotation (excess_Y_rot), dephasing rate""" 
    gate_phase = np.pi/2. # for Y rotation gate 
    theta = np.pi/2. + excess_Y_rot
    rabi_rate = theta 

    # Rabi rate is set to theta, so gate_duration is 1. (Omega*t = theta)
    gate_duration = 1. 

    coherent_Z_error = 0.
    #dephasing_probability = phase_std_deviation**2 
    # Build Y_pi/2 Lindbladian from generalized R gate 
    Y_pi2_lindbladian = R_gate_lindbladian_function(rabi_rate, gate_phase, coherent_Z_error, dephasing_probability) 

    Y_pi2_gate = ism.Gate.from_lindbladian(basis, Y_pi2_lindbladian, gate_duration, lindbladian_time_independent=True)
    return Y_pi2_gate.process_matrix 

def t_Xpi2(excess_X_rotation: float, dephasing_probability: float, Z_shuttling_error: float):
    """ Gate model for shuttling and then X_pi2 gate """ 
    shuttle_pm = shuttle_w_dephasing(Z_shuttling_error, dephasing_probability)
    X_pi2_pm = X_pi2_process_matrix(excess_X_rotation, dephasing_probability)
    # Combine process matrices via matrix multiplication
    return X_pi2_pm @ shuttle_pm

def t_Ypi2(excess_Y_rotation: float, dephasing_probability: float, Z_shuttling_error: float):
    """ Gate model for shuttling and then Y_pi2 gate """ 
    shuttle_pm = shuttle_w_dephasing(Z_shuttling_error, dephasing_probability)
    Y_pi2_pm = Y_pi2_process_matrix(excess_Y_rotation, dephasing_probability)
    # Combine process matrices via matrix multiplication
    return Y_pi2_pm @ shuttle_pm

