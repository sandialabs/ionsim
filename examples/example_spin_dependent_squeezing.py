import ionsim as ism

import numpy as np
from scipy.sparse import kron as skron
import sys
import matplotlib
import matplotlib.pyplot as plt
import platform
from scipy.special import factorial 
matplotlib.rcParams['text.usetex']=True 
style_path_data = 'style/plot_style_data.txt'

TPI = 2*np.pi

def four_tone_hamiltonian(basis, spin_basis, modes, spectator_modes, etas, rabi_rate_x, rabi_rate_y, omega_x_red, omega_x_blue, omega_y_red, omega_y_blue, targetIon: AtomicSpin, 
                        index_mode_i:int, index_mode_j:int, phi:float, sparse=False, mod=None): 
    ''' Applies 4-tone MS-like gate to a single ion "k" to produce a spin-dependent motional operation. 
        Consists of 2 sidebands (red and blue) on mode i and another 2 sidebands (red and blue) on mode j:  

        H(t) = hbar Omega_x sigma_x a_i exp(-i Delta t) + h.c. + hbar Omega_y sigma_y a_j exp(-i n Delta t + phi)  + h.c. 

        - currently sets the first MS phase to zero, accepts input for second MS phase as phi 
    ''' 
    # Hamiltonian: H = -eta * Omega/2 sum_{k} [-i sigma_+,k exp(iphi) + i sigma_-,k exp(-iphi)] (a e^i delta t + a† e^{-i delta t} )
    # Hamiltonian: H = -eta * Omega J_y (a e^i delta t + a† e^{-i delta t} ) # for phi = 0; Jy = 1/2 (sigma_y,1 + sigma_y,2)
    motional_basis = ism.StandardBasis([*modes])

    DOFs = basis.degrees_of_freedom

    target_ion_index = DOFs.index(targetIon)

    operators = []

    # build matrices for fundamental raising/lowering spin/Fock operators: 
    # Raising operator for target ion "k" 
    raise_spin_k = spin_basis.enlarge_matrix(ism.Pauli.plus, [targetIon])
    
    fock_dimension = len(modes[index_mode_i].energy_levels)
    assert fock_dimension == len(modes[index_mode_j].energy_levels)

    # First, include modes that are supposed to be included. (non-spectators)
    raise_motion = motional_basis.enlarge_matrix(ism.Fock.raising(fock_dimension), [modes[index_mode_i]])
    lower_motion = motional_basis.enlarge_matrix(ism.Fock.lowering(fock_dimension), [modes[index_mode_j]])

    # Extract Lamb-Dicke parameter for mode i and j, ion k
    eta_ki = etas[target_ion_index, index_mode_i]
    eta_kj = etas[target_ion_index, index_mode_j]

    # spin phase = ( phi_R + phi_B )/ 2 
    #   -------------------
    #   x-spin MS-like gate: 
    # For mode i : 
    # Can achieve sigma_x by setting spin phase of tones 1 and 2 to zero or each as 2pi  
    phase_x = np.pi/2. 

    x_prefactor = 1j*rabi_rate_x * 0.5 * eta_ki * np.exp(1j*phase_x)

    operator_red_mode_i = x_prefactor * skron(raise_spin_k, lower_motion) 
    operator_blue_mode_i = x_prefactor * skron(raise_spin_k, raise_motion)
    # h.c. is already included by ionsim upon providing these coupling operators to the hamiltonian 

    #   -------------------
    #   y-spin MS-like gate: 
    # For mode j : 
    phase_y = 0. 
    y_prefactor = 1j*rabi_rate_y * 0.5 * eta_kj * np.exp(1j*(phase_y + phi))

    operator_red_mode_j = y_prefactor * skron(raise_spin_k, lower_motion) 
    operator_blue_mode_j = y_prefactor * skron(raise_spin_k, raise_motion)

    operators.extend([
        ism.CouplingOperator.from_matrix(basis, operator_red_mode_i, omega_x_red, modulation_function=mod), 
        ism.CouplingOperator.from_matrix(basis, operator_blue_mode_i, omega_x_blue, modulation_function=mod), 
        ism.CouplingOperator.from_matrix(basis, operator_red_mode_j, omega_y_red, modulation_function=mod), 
        ism.CouplingOperator.from_matrix(basis, operator_blue_mode_j, omega_y_blue, modulation_function=mod), 
    ])

    for mode in spectator_modes:
        mode_index = modes.index(mode)
        spectator_raise_motion = motional_basis.enlarge_matrix(ism.Fock.raising(fock_dimension), [mode])
        spectator_lower_motion = motional_basis.enlarge_matrix(ism.Fock.lowering(fock_dimension), [mode])

        # Extract Lamb-Dicke parameter 
        eta_ki = etas[target_ion_index, mode_index]
        eta_kj = etas[target_ion_index, mode_index]

        x_prefactor = 1j*rabi_rate_x * 0.5 * eta_ki * np.exp(1j*phase_x)
    
        operator_red_x_spectator = x_prefactor * skron(raise_spin_k, spectator_lower_motion) 
        operator_blue_x_spectator = x_prefactor * skron(raise_spin_k, spectator_raise_motion)

        y_prefactor = 1j*rabi_rate_y * 0.5 * eta_kj * np.exp(1j*(phase_y + phi))
    
        operator_red_y_spectator = y_prefactor * skron(raise_spin_k, spectator_lower_motion) 
        operator_blue_y_spectator = y_prefactor * skron(raise_spin_k, spectator_raise_motion)

        operators.extend([
            ism.CouplingOperator.from_matrix(basis, operator_red_x_spectator, omega_x_red, modulation_function=mod), 
            ism.CouplingOperator.from_matrix(basis, operator_blue_x_spectator, omega_x_blue, modulation_function=mod), 
            ism.CouplingOperator.from_matrix(basis, operator_red_y_spectator, omega_y_red, modulation_function=mod), 
            ism.CouplingOperator.from_matrix(basis, operator_blue_y_spectator, omega_y_blue, modulation_function=mod), 
        ])

    interaction_frame_energies = [-state.energy for state in basis.states]
    return ism.Hamiltonian(basis, operators, interaction_frame_energies, sparse=sparse)


def thermal_state_populations(nbar: float, fock_dimension): 
    fock_populations = np.zeros(fock_dimension)

    normalization = 1./(1. + nbar) 
    for n in range(fock_dimension):
        fock_populations[n] = (nbar/(1. + nbar))**(n) 

    return fock_populations*normalization 


def dephasing_dissipator(basis, spin_basis, modes, dephasing_rates: list[float]):
    ''' Dissipator for ion motional mode dephasing ''' 

    motional_basis = ism.StandardBasis([*modes])
    spins = spin_basis.degrees_of_freedom
    
    # Build spin basis identity 
    spin_identities = []
    for spin in spins:
        # Identity matrix for mode m in Fock space, enlarged to fit dimensionality of M modes 
        spin_identities.append(spin_basis.enlarge_matrix(ism.Pauli.I, [spin]))

    spin_identity = np.sum(spin_identities, axis=0)

    # Motional mode dephasing is proportional to the phonon number operators 
    # loop over modes:
    lindblad_operators = []
    for mode, dephasing_rate in zip(modes, dephasing_rates):
        fock_dim = len(mode.energy_levels)
        mode_number_matrix = motional_basis.enlarge_matrix(ism.Fock.number(fock_dim), [mode]) 

        # One lindblad operator per mode: L = sqrt(Gamma) * (a†_m a_m)
        dephasing_operator = np.sqrt(dephasing_rate) * skron(spin_identity, mode_number_matrix) 

        # Lindblad operator is diagonal in the Fock occupancy number basis; use EnergyShiftoperator or GeneralOperator 
        lindblad_operators.append(ism.EnergyShiftOperator.from_matrix(basis, dephasing_operator))

    frame_energies = [-state.energy for state in basis.states] 
    return ism.Dissipator(basis, lindblad_operators, frame_energies)



def heating_dissipator(basis, modes, heating_rates: list[float]): 
    ''' Dissipator for ion mode heating ''' 

    motional_basis = ism.StandardBasis([*modes])
    
    # Build spin basis identity 
    spin_identities = []
    for spin in spins:
        # Identity matrix for mode m in Fock space, enlarged to fit dimensionality of M modes 
        spin_identities.append(spin_basis.enlarge_matrix(ism.Pauli.I, [spin]))

    spin_identity = np.sum(spin_identities, axis=0)

    # Phonon raising and lowering operators 
    # loop over modes:
    lindblad_operators = []
    for mode, heating_rate in zip(modes, heating_rates):
        fock_dim = len(mode.energy_levels)
        mode_lowering_matrix = motional_basis.enlarge_matrix(ism.Fock.lowering(fock_dim), [mode]) 
        mode_raising_matrix = motional_basis.enlarge_matrix(ism.Fock.raising(fock_dim), [mode]) 

        # Two lindblad operators per mode: L1 = sqrt(Gamma) * a_m , L2 = sqrt(Gamma) * a†_m 
        lowering_operator = np.sqrt(heating_rate) * skron(spin_identity, mode_lowering_matrix) 
        raising_operator = np.sqrt(heating_rate) * skron(spin_identity, mode_raising_matrix) 
        #lowering_operator = np.sqrt(decay_rate*(N_thermal+1)) * skron(spin_identity, mode_lowering_matrix) 
        #raising_operator = np.sqrt(decay_rate*N_thermal) * skron(spin_identity, mode_raising_matrix) 

        lindblad_operators.append(ism.CouplingOperator.from_matrix(basis, lowering_operator, 0.))
        lindblad_operators.append(ism.CouplingOperator.from_matrix(basis, raising_operator, 0.))

    frame_energies = [-state.energy for state in basis.states] 
    return ism.Dissipator(basis, lindblad_operators, frame_energies)


def carrier_hamiltonian(basis, target_ion, rabi_rate: float, phase: float, omega_carrier: float, sparse: bool=False, mod=None):
    """ Hamiltonian for carrier spin pulse: (hbar * Omega/2) * sigma_phi e^iphi + h.c. """ 
    prefactor = rabi_rate * 0.5 * np.exp(-1j*phase)
    operator_raising = basis.enlarge_matrix(prefactor * ism.Pauli.plus, [target_ion])
    #raise_spin_k = spin_basis.enlarge_matrix(ism.Pauli.plus, [target_ion])

    operators = [ism.CouplingOperator.from_matrix(basis, operator_raising, omega_carrier, modulation_function=mod)] 
    interaction_frame_energies = [-state.energy for state in basis.states]
    return ism.Hamiltonian(basis, operators, interaction_frame_energies, sparse=sparse)


def red_sb_pi_pulse_duration(rabi_rate, fock_state, eta, single_ion: bool=True):
    # Assume either single or 2-ion sysetm
    if single_ion:
        t = np.pi / (rabi_rate * np.sqrt(fock_state) * np.abs(eta))
    else:
        t = np.pi / (rabi_rate * np.sqrt(fock_state * (fock_state - 1)) * np.abs(eta))
    return t  
    


def compute_N(rho_motional, motional_basis):

    #for state in motional_basis.states:
    N_operator = ism.Fock.raising(fock_dimension) @ ism.Fock.lowering(fock_dimension) 
    return np.trace( N_operator.dot(rho_motional.density_matrix))

def estimate_squeezing_magnitude(motional_rho, mode_basis):
    ''' Estimate the squeezing parameter from the X and P variances ''' 
    x, p, x2, p2 = motional_rho.compute_quadratures(True) # includes variance 

    avg_X_tilt = x[0].real
    avg_X2_tilt = x2[0].real
    
    X_variance = avg_X2_tilt - avg_X_tilt**2
    #print(f"variance of X for the tilt mode: {X_variance}")
    
    avg_P_tilt = p[0].real
    avg_P2_tilt = p2[0].real
    
    P_variance = avg_P2_tilt - avg_P_tilt**2
    #print(f"variance of P for the tilt mode: {P_variance}")
    if squeezing_phase == 0.: 
        X_squeezed = False 
    elif squeezing_phase == np.pi:
        X_squeezed = True 

    if X_squeezed:
        r_X = -0.5 * np.log(2. * X_variance)
        r_P = 0.5 * np.log(2. * P_variance)
    else:
        r_X = 0.5 * np.log(2. * X_variance)
        r_P = -0.5 * np.log(2. * P_variance)

    return r_X, r_P

def squeezed_ground_state_coefficients(r_value, phi, fock_dimension):
    wave_function = np.zeros(fock_dimension,dtype=complex) 
    norm = 1./np.sqrt(np.cosh(r_value))
    for n in range(fock_dimension//2):
        wave_function[2*n] = np.exp(1j * n * phi) * np.tanh(r_value)**n 
        #wave_function[2*n] *= norm * np.sqrt(factorial(2 * n))
        wave_function[2*n] *= norm * np.sqrt(factorial(2 * n))/(2**n) 
        wave_function[2*n] /= (factorial(n))
    return wave_function 



def squeezed_vacuum_populations(fock_dim, r_val):
    fock_states = np.arange(fock_dim)
    fock_populations = np.zeros(len(fock_states)) 
    norm = 1./np.cosh(r_val)
    for i in range(fock_dim//2):
        fock_populations[2*i] = norm * (np.tanh(r_val)**(2*i)) * factorial(2 * i)/(2**(2*i) * factorial(i)**2 ) 
    return fock_populations 

    

def main():
    import time
    from matplotlib import pyplot as plt

    print(f"\n --- Simulation of spin-dependent squeezing by 4-tone sideband protocol --- ")
    
    include_spectator_mode = False 
    include_dissipation = False 
    do_reverse_process = False 
    thermal_state_IC = False 
    reverse_phases = False 
    beam_counter_propagating = True

    num_ions = 2

    # Create 171Yb+ qubits 
    spins = [
        ism.AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        for _ in range(num_ions)
    ]

    target_ion = spins[0]    
    target_ion_index = spins.index(target_ion)

    spin_basis = ism.StandardBasis([*spins])

    # Use mode analysis to get motional mode information 
    omega_x = 2.1 * TPI * 1E6 # MHz -> rad/s  
    omega_y = 2.6 * TPI * 1E6 # ""  
    omega_z = 0.50 * TPI * 1E6 # ""  
    trap_analysis = ism.LinearIonChainAnalysis.from_atomic_spin_basis(spin_basis, omega_x, omega_y, omega_z) 
    trap_analysis.solve_ion_trap_equilibrium()
    print()
    trap_analysis.print_chain_summary()

    print(trap_analysis.characteristic_parameters)


    # Truncate the Hilbert space to a single qubit with the two nearby motional modes 
    # Choose to do oprerations on the radial x tilt mode; option to include the COM as a spectator mode  
    #print(trap_analysis.get_mode_properties_by_branch('x', 0))
    #print(trap_analysis.get_mode_properties_by_branch('x', 1))
    tilt_mode_index = 0 
    COM_mode_index = 1 

    fock_dimension = 10
    branch_dir = 'x'
    spin_basis = ism.StandardBasis([target_ion]) # Now reset the spin basis to just include the target ion 
    modes = trap_analysis.build_mode_DOFs_from_branch(branch_dir, [tilt_mode_index, COM_mode_index], fock_dimension)
    if include_spectator_mode:
        basis = ism.StandardBasis([spins[target_ion_index], *modes])
        motional_basis = ism.StandardBasis(modes)
        spectator_modes = [modes[1]]
    else:
        basis = ism.StandardBasis([spins[target_ion_index], modes[tilt_mode_index]])
        motional_basis = ism.StandardBasis([modes[tilt_mode_index]])
        modes = [modes[0]]
        spectator_modes = [] 


    target_mode_index = 0 # for tilt mode in our basis 

    # 4-tone protocol targets 2 modes in general with 2 tones each; for squeezing, we target the same mode with all tones 
    mode_i_index = target_mode_index
    mode_j_index = target_mode_index

    radial_x_tilt_mode_properties = trap_analysis.get_mode_properties_by_branch('x', tilt_mode_index)
    mode_i_freq = radial_x_tilt_mode_properties['frequency'] 
    mode_j_freq = mode_i_freq 

    print(f" - Tilt mode frequency: {mode_i_freq/TPI/1E6} [MHz]\n\n")

    # Laser/driving parameters 
    print(f" --- Laser parameters --- ")
    wavelength = 355 # nm 
    beam_angle = np.pi/4. # relative to axial direction 
    wavevector = TPI / (355. * 1E-9) * np.array([np.cos(beam_angle), np.sin(beam_angle), 0.]) 
    if beam_counter_propagating:
        # Effective wavevector from 
        wavevector *= 2. # Raman process: \delta k is 2x basic wavevector  

    rabi_rate = 175 * 1E3 * TPI  / np.sqrt(2.) #* np.sqrt(0.5) #/ np.sqrt(4.) # kHz -> Hz  
    x_rabi_rate = rabi_rate
    y_rabi_rate = rabi_rate
    print(f" - Carrier rabi rate: {rabi_rate / TPI / 1E3} [kHz]\n")

    detuning = 40.0 * 1E3 * TPI # 40 kHz -> rad/s 
    print(f" - Detuning: {detuning/(TPI * 1E3)} [kHz]\n")
    duration = 60 * 1E-6 
    print(f" - Gate duration: {duration*1E6:.3f} [µs]\n")

    modulate_amplitude = True 

    print(f" - Estimated number of loops: {duration * detuning / TPI}\n")

    detuning_red = detuning
    detuning_blue = detuning

    # Blue and red laser frequencies for 1st-sideband tuning:  
    # For x-spin / operation on mode i, the laser is simply detuned by 1*detuning  
    offset = 0. * TPI * 1E3 # kHz 
    omega_qubit = target_ion.energy_levels[1].energy - target_ion.energy_levels[0].energy
    omega_b_mode_i = (omega_qubit + (mode_i_freq + detuning_blue)) + offset 
    omega_r_mode_i = (omega_qubit - (mode_i_freq + detuning_red)) - offset 
    
    # Second MS pulse 
    # For y-spin / operation on mode j, it is detuned by n*detuning  
    n = -1 # for squeezing 
    omega_b_mode_j = (omega_qubit + (mode_j_freq + n*detuning_blue)) - offset
    omega_r_mode_j = (omega_qubit - (mode_j_freq + n*detuning_red)) + offset

    ## -- Global parameters for script --- 
    frequencies = np.array([omega_r_mode_i, omega_r_mode_j, omega_b_mode_i, omega_b_mode_j]) # MHz
    frequencies_from_carrier = (frequencies - omega_qubit)/TPI/1E6 # MHz
    print(f"  - laser frequencies (minus carrier): {frequencies_from_carrier}\n\n")

 
    def blackman(t: float, magnitude: float = 1.):
        return magnitude*(0.42 - 0.5*np.cos(TPI * t /duration) + 0.08 * np.cos(2.*TPI*t/duration))

    # Retrieve lamb dicke parameters 
    lamb_dicke_parameters = trap_analysis.get_radial_lamb_dicke_parameters(wavevector, branch_dir) # matrix of ion, mode 
    eta = lamb_dicke_parameters[target_ion_index,target_mode_index]
    print(f" - |eta| for target ion, target mode: {np.abs(eta)}\n")
    print(f" - Sideband rabi rate (eta*Omega): {np.abs(eta) * rabi_rate / 1E3 / TPI} [kHz]\n")
    print('\n\n')
    print(f" - Lamb dicke parameters: \n{lamb_dicke_parameters}")
    mod_function = None
    expected_r = duration * ((np.abs(eta) * rabi_rate)**2) / detuning # Square pulse 
    if modulate_amplitude:
        print(f" - Using a Blackman pulse shape\n")
        mod_function = blackman
        # Area of the blackman pulse shape squared is ~0.3046
        expected_r = 0.3046 * duration * ((np.abs(eta) * rabi_rate)**2) / detuning
    else:
        print(f" - Using a constant (square) pulse shape\n")

    r = expected_r
    print(f" - Expecting squeezing of r = {expected_r}\n")

    
    # =============================
    ## Create Hamiltonians: 
    squeezing_phase = 0.
    squeezing_hamiltonian = four_tone_hamiltonian(basis, spin_basis, modes, spectator_modes, lamb_dicke_parameters, x_rabi_rate, y_rabi_rate, omega_r_mode_i, omega_b_mode_i, omega_r_mode_j, 
                                                    omega_b_mode_j, target_ion, mode_i_index, mode_j_index, squeezing_phase, mod=mod_function)

    reverse_squeezing_hamiltonian = four_tone_hamiltonian(basis, spin_basis, modes, spectator_modes, lamb_dicke_parameters, x_rabi_rate, y_rabi_rate, omega_r_mode_i, omega_b_mode_i, omega_r_mode_j, omega_b_mode_j, 
                                                        target_ion, mode_i_index, mode_j_index, squeezing_phase + np.pi, mod=mod_function) 

    pi_pulse_hamiltonian = carrier_hamiltonian(basis, target_ion, rabi_rate, 0., np.abs(omega_qubit), False, None)  
    mode_dephasing_rates = [5.0, 500.] # quanta/second 
    if include_dissipation : 
        # Create dissipator and then full Lindbladian: 
        #ion_heating_dissipator = heating_dissipator(basis, modes, mode_heating_rates) 
        _dephasing_dissipator = dephasing_dissipator(basis, spin_basis, modes, mode_dephasing_rates) 
        lindbladian_4tone = ism.Lindbladian(squeezing_hamiltonian, _dephasing_dissipator)
        #lindbladian_4tone = ism.Lindbladian(squeezing_hamiltonian, ion_heating_dissipator)
        # Create reverse processes 
        lindbladian_4tone_reverse = ism.Lindbladian(reverse_squeezing_hamiltonian, _dephasing_dissipator)

    # Initial fock state, all populated in this state:
    starting_fock_state = 0 
    excited_qubit1_init_state = False 
    if excited_qubit1_init_state:
        qubit1_index = fock_dimension*num_modes 
    else:
        qubit1_index = 0

    nbars = [0.1, 0.5]

    IC_fock_pops = [thermal_state_populations(nbar, fock_dimension) for nbar in nbars] # populations  
    #fock_populations_target_mode_IC = thermal_state(nbar)
    if thermal_state_IC:
        print(f"\n--- Starting with thermal ground state ---\n")
        print(f"Target mode initial populations for nbar = {nbars[0]}, : {IC_fock_pops[0]}")
        coefs = np.kron(np.array([1., 0.]), IC_fock_pops[0])
        for m in range(1, len(modes)):
            print(f"Spectator mode initial populations for nbar = {nbars[m]}, : {IC_fock_pops[m]}")
            coefs = np.kron(coefs, IC_fock_pops[m])
        coefs = np.sqrt(coefs)# populations require square root 
    else:
        print(f"--- Initializing in the ground state of full Hilbert space ---")
        IC_fock_pops[0] = np.zeros(fock_dimension) 
        IC_fock_pops[0][0] = 1. 
        for m in range(1, len(modes)):
            IC_fock_pops[m] = np.zeros(fock_dimension) 
            IC_fock_pops[m][0] = 1.
        coefs = np.zeros(len(basis.states))
        starting_fock_state_mode2 = 0
        coefs[qubit1_index + starting_fock_state + starting_fock_state_mode2] = 1. 

    print_state_names = False 
    if print_state_names:
        for indx, state in enumerate(basis.states):
            print(indx)
            print(state.name) 
    init_state = ism.State.from_coefficients(basis, list(coefs)) 

    times = np.linspace(0, duration, 2000) 

    ground_state_level_index = 0 
    excited_state_level_index = 1 
    ground_state_level_name = spin_basis.degrees_of_freedom[0].energy_levels[ground_state_level_index].name 
    excited_state_level_name = spin_basis.degrees_of_freedom[0].energy_levels[excited_state_level_index].name 
    print("Ground state level name: " + ground_state_level_name)
    print("Excited state level name: " + excited_state_level_name)

    # Now perform cycling to attempt squeezing operation 
    print(f"\n\n\n ------- Squeezing protocol ------- ")
    print(f"Squeezing parameters: r = {r}, phi = {squeezing_phase}")

    # Use rho(t) from previous 
    # Apply 4 tones with n=-1 detuning relationship to get squeezing  
    rho_t = init_state 

    ### Evolve Schrodinger or master equation dynamics for spin-dependent squeezing via 4-tone protocol  
    if include_dissipation:
        rho_t = rho_t.propagate_using_master_equation(lindbladian_4tone, duration, times)
    else:
        rho_t = rho_t.propagate_using_schrodinger_equation(squeezing_hamiltonian, duration, times) # no dissipation  
    
    if do_reverse_process: 
        # Perform pi pulse on qubit:  # Omega * t = pi; t = pi/Omega
        if reverse_phases: 
            if include_dissipation:
                rho_t = rho_t[-1].propagate_using_master_equation(lindbladian_4tone_reverse, duration, times)
            else:
                rho_t = rho_t[-1].propagate_using_schrodinger_equation(reverse_squeezing_hamiltonian, duration, times) # no dissipation  
    
        else:
            pi_pulse_duration = np.pi/rabi_rate
        
            # Perform squeezing again, which should return you back to the ground state     
            if include_dissipation:
                pi_pulse_lindbladian = ism.Lindbladian(hamiltonian = pi_pulse_hamiltonian, dissipator = None)
                rho_t = rho_t[-1].propagate_using_master_equation(pi_pulse_lindbladian, pi_pulse_duration) 
                rho_t = rho_t.propagate_using_master_equation(lindbladian_4tone, duration, times)
            else:
                rho_t = rho_t[-1].propagate_using_schrodinger_equation(pi_pulse_hamiltonian, pi_pulse_duration) # no dissipation  
                rho_t = rho_t.propagate_using_schrodinger_equation(squeezing_hamiltonian, duration, times) # no dissipation  

        # Extract final state 
        final_state = rho_t[-1]

        # Compute state overlap (fidelity of reverse process) 
        ## TODO: uncomment after merging branch 57 
        #reversibility_fidelity = final_state.motional_state.compute_state_fidelity(init_state.motional_state.density_matrix)
        #print(f" - Reversibility fidelity: {reversibility_fidelity}")
            

    if len(times) != len(rho_t):
        rho_t = rho_t[:len(times)]
    print(f"Number of rho(t) timepoints: {len(rho_t)}")
    print(f"Number of timepoints: {len(times)}")
    
    spin_rhos = rho_t.copy()
    motional_rhos = rho_t.copy()
    # 1. Trace out motional modes and plot spin-state populations
    for mode in modes:
        spin_rhos = [rho.trace_out_degree_of_freedom(mode) for rho in spin_rhos] 
    
    new_basis = spin_rhos[0].basis # should be 2-qubit basis 
    
    populations = np.array([rho.compute_basis_state_probabilities() for rho in spin_rhos])
    plt.style.use(style_path_data) 
    plt.figure(figsize=(6,4))
    for i,state in enumerate(new_basis.states):
        plt.plot(times*1E6, populations[:, i], label= state.name)
    plt.ylabel('Population', fontsize = 16)
    plt.xlabel('Gate Duration (µs)', fontsize = 20)
    plt.legend()
    plt.show()

    # 2. Trace out spin DOFs and plot Fock state populations
    X_modes = np.zeros((len(times),len(modes)),dtype=complex)
    P_modes = np.zeros((len(times),len(modes)), dtype=complex)
    X2_modes = np.zeros((len(times),len(modes)), dtype=complex)
    P2_modes = np.zeros((len(times),len(modes)), dtype=complex)
    for spin in [target_ion]:
        motional_rhos = [rho.trace_out_degree_of_freedom(spin) for rho in motional_rhos]
        # For each time, compute quadrature expectations w. variances  
        # TODO: Uncomment after merging Wigner/quadrature helper functions branch 
 #        for i, t in enumerate(times):
 #            x, p, x2, p2 = motional_rhos[i].compute_quadratures(True) # includes variance 
 #            X_modes[i, :] = np.array(x)
 #            P_modes[i, :] = np.array(p)
 #            X2_modes[i, :] = np.array(x2)
 #            P2_modes[i, :] = np.array(p2)

        for mode in spectator_modes: 
            motional_rhos = [rho.trace_out_degree_of_freedom(mode) for rho in motional_rhos]

    new_basis = motional_rhos[0].basis # should be motional basis  

    fock_populations = np.array([rho.compute_basis_state_probabilities() for rho in motional_rhos])
 #    print('\nFock populations at end of protocol: \n')
 #    for state, population in zip(new_basis.states, fock_populations[-1]):
 #        print(f"\nState: n = {state.name}")
 #        print(f"Population: {population}")

    fock_states = np.arange(fock_dimension)
    squeezed_GS_populations = squeezed_vacuum_populations(fock_dimension, r)

    bar_width = 0.35
    plt.figure(figsize=(6,6))
    plt.bar(fock_states - bar_width/2, fock_populations[-1], bar_width, color = 'b', label = 'Simulation')
    if not do_reverse_process: 
        plt.bar(fock_states + bar_width/2, squeezed_GS_populations, bar_width, color = 'r', label = 'Reference: $r_{target} = ' + str(np.round(r,2)) + '$')
    else:
        plt.bar(fock_states + bar_width/2, IC_fock_pops[0], bar_width, color = 'r', label = 'Initial distribution') 
    plt.title('Target mode Fock distribution', fontsize = 14)
    plt.ylabel('Population', fontsize = 16)
    plt.xlabel('Fock state $n$', fontsize = 20)
    plt.legend()
    plt.show()

    sys.exit(0)    



    ############## -------------------------------------------------------------- To include after merging branch 57 --------------------------     ################################# 
    # Compute wigner distributions: 
    limit = 4. # phase space limit of x and p 
    x_grid = np.linspace(-limit, limit, 200)
    p_grid = np.linspace(-limit, limit, 200)

    last_index = len(times)-1 
    # Compute Wigner distributions over a few selected indices 
    sampling_indices = [0, last_index//3, last_index//2, int(2*last_index//3), int(3*last_index//4), int(7*last_index//8), last_index] 
    #sampling_indices = [0, last_index] 

    sampled_states = [rho_t[i] for i in sampling_indices]
    Wigner_distributions = [rho.compute_wigner_distribution(x_grid, p_grid) for rho in sampled_states] 
    #print(len(Wigner_distributions[0]))

    if num_modes == 2:
        mode_labels = ['Target mode', 'Spectator mode 1']
    else:
        mode_labels = ['Target mode']

    for W, t in zip(Wigner_distributions, times[sampling_indices]):
        for i, mode_label in enumerate(mode_labels):
            plt.figure(figsize=(6, 5))
            plt.imshow(W[i], extent=[-limit, limit, -limit, limit], origin='lower', cmap='magma') 
            if do_reverse_process: 
                plt.title(f"$W(x,p)$: " + mode_label + f", t = {t*2.*1E6:.2f} [µs]", fontsize = 12)
            else:
                plt.title(f"$W(x,p)$, t = {t*1E6:.2f} [µs]", fontsize = 16)
                #plt.title(f"$W(x,p)$: " + mode_label + f", t = {t*1E6:.2f} [µs]", fontsize = 12)
            plt.xlabel('Position ($x$)', fontsize = 16)
            plt.ylabel('Momentum ($p$)', fontsize = 16)
            cbar = plt.colorbar()
            cbar.ax.set_ylabel(r'$W(x,p)$', rotation = 0, labelpad = 30, va='center') 
            if excited_qubit1_init_state: 
                squeezing_str = 'anti'
            else:
                squeezing_str = 'fwd'
            plt.savefig(squeezing_str + f"_mode_{i}_Wigner_t_{t*1E6:.2f}.pdf", dpi=300)
            plt.show()

    avg_X_tilt = X_modes[-1,0].real
    avg_X2_tilt = X2_modes[-1,0].real
    
    print(f"<X> for the tilt mode: {avg_X_tilt}")
    print(f"<X^2> for the tilt mode: {avg_X2_tilt}")
    
    X_variance = avg_X2_tilt - avg_X_tilt**2
    print(f"variance of X for the tilt mode: {X_variance}")
    
    avg_P_tilt = P_modes[-1,0].real
    avg_P2_tilt = P2_modes[-1,0].real
    
    print(f"<P> for the tilt mode: {avg_P_tilt}")
    print(f"<P^2> for the tilt mode: {avg_P2_tilt}")
    
    P_variance = avg_P2_tilt - avg_P_tilt**2
    print(f"variance of P for the tilt mode: {P_variance}")

    print(f"\nVerifying r from variances:")    
    # If X is squeezed: 
    if squeezing_phase == 0.:
        X_squeezed = False 
    else:
        X_squeezed = True 

    if X_squeezed:
        r_X = -0.5 * np.log(2. * X_variance)
        r_P = 0.5 * np.log(2. * P_variance)
    else:
        r_X = 0.5 * np.log(2. * X_variance)
        r_P = -0.5 * np.log(2. * P_variance)

    print(f"r estimate from X variance: {r_X}") 
    print(f"r estimate from P variance: {r_P}") 

    target_motional_state = ism.State.from_coefficients(motional_rhos[0].basis, list(squeezed_ground_state_coefficients(r, squeezing_phase, fock_dimension))) 
    squeezed_state_fidelities = np.array([rho_mot.compute_state_fidelity(target_motional_state.density_matrix) for rho_mot in motional_rhos])

    # Check purity: 
    print(f"Purity: {np.trace(motional_rhos[-1].density_matrix @ motional_rhos[-1].density_matrix)}")
    
    print('\n')
    print(f'State fidelity with squeezed ground state: {squeezed_state_fidelities[-1]}')

    r_X = np.zeros_like(times)
    r_P = np.zeros_like(times)
    n_t = np.zeros_like(times)
    for i, t in enumerate(times):
        r_X[i], r_P[i] = estimate_squeezing_magnitude(motional_rhos[i], motional_rhos[i].basis)
        n_t[i] = compute_N(motional_rhos[i], motional_rhos[i].basis)

    plt.style.use(style_path_data) 
    plt.figure(figsize=(5,5))
    plt.plot(times*1E6, r_X, marker = 's', linestyle = 'solid', color = 'k', markersize = 3, linewidth = 1.5, label = r'$r_{X}$')
    plt.plot(times*1E6, r_P, marker = 'o', linestyle = 'solid', color = 'r', markersize = 3, linewidth = 1.5, label = r'$r_{P}$')
    plt.axhline(y = r, color = 'k', linestyle='dashed', linewidth = 1.5, label = r'$r_{target} = ' + str(np.round(r, 4)) + '$')
    plt.xlabel('Time [µs] ', fontsize = 20)
    plt.ylabel(r'$r$', fontsize = 24, rotation = 0., labelpad=15)
    plt.legend()
    plt.show()

    plt.figure(figsize=(5,5))
    plt.plot(times*1E6, squeezed_state_fidelities, marker = 's', linestyle = 'solid', color = 'k', markersize = 3, linewidth = 1.5, label = r'State fidelity')
    plt.xlabel('Time [µs] ', fontsize = 20)
    plt.ylabel(r'$\mathcal{F}$', fontsize = 24, rotation = 0., labelpad=15)
    plt.legend()
    plt.show()

    plt.figure(figsize=(5,5))
    plt.plot(times*1E6, n_t, marker = 's', linestyle = 'solid', color = 'k', markersize = 3, linewidth = 1.5, label = r'Simulation: $N(t)$')
    plt.axhline(y = np.sinh(r)**2, color = 'k', linestyle='dashed', linewidth = 1.5, label = r'$\langle N \rangle = \sinh(r)^2 $')
    plt.xlabel('Time [µs] ', fontsize = 20)
    plt.ylabel(r'$N$', fontsize = 24, rotation = 0., labelpad=15)
    plt.legend()
    plt.show()


    fock_states = np.arange(fock_dimension)
    squeezed_GS_populations = squeezed_vacuum_populations(fock_dimension, r)

    bar_width = 0.35
    plt.figure(figsize=(6,6))
    plt.bar(fock_states - bar_width/2, fock_populations[-1], bar_width, color = 'b', label = 'Simulation')
    if not do_reverse_process: 
        plt.bar(fock_states + bar_width/2, squeezed_GS_populations, bar_width, color = 'r', label = 'Reference: $r_{target} = ' + str(np.round(r,2)) + '$')
    else:
        plt.bar(fock_states + bar_width/2, IC_fock_pops[0], bar_width, color = 'r', label = 'Initial distribution') 
    plt.ylabel('Population', fontsize = 16)
    plt.xlabel('Fock state $n$', fontsize = 20)
    plt.legend()
    plt.show()

    if modulate_amplitude:
        plt.figure(figsize=(5,5))
        plt.plot(times*1E6, mod_function(times), marker = 's', linestyle = 'solid', color = 'k', markersize = 2, linewidth = 1.5) 
        plt.xlabel('Time [µs] ', fontsize = 20)
        plt.ylabel(r'$\Omega$', fontsize = 24, rotation = 0., labelpad=15)
        plt.title('Amplitude Modulation')#: $\sigma = ' + str(variance))
        #plt.legend()
        plt.show()

if __name__ == '__main__':
    main()
