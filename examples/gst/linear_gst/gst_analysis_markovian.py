from pathlib import Path
import numpy as np
import h5py
import sys
import time
from scipy.sparse import kron as skron
from matplotlib import pyplot as plt
from icecream import ic
from typing import Callable
import inspect

import ionsim as sm

""" ################ Single qubit GST Example ################## """ 
def main():
    fname = './simulated_gst_experimental_data.gstdata' 

    # Import outcome data  
    parsed_circuits = sm.parse_gst_circuit_file(fname)

    # Import circuit design 
    design_fname = 'circuit_design.yml'
    gst_circuit_design = sm.GSTCircuitPlanner.load_design(design_fname)

    # Set up basic 1-qubit (1Q) basis  
    num_spins = 1
    
    spins = [
        sm.AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        for _ in range(num_spins)
    ]
    
    basis = sm.StandardBasis([*spins])
    target_spins = [spins[0]]

    qubit_frequency = target_spins[0].energy_levels[1].energy - target_spins[0].energy_levels[0].energy


    ################ Define gate error models: #################### 
    # Requires a basis to be defined 
    # Function to build a Lindbladian for a generalized R gate 
    def idle(theta):
        """ Returns d^2 x d^2 process matrix in standard basis for Z-rotation by theta """  
        # Build identity matrix with Z rotation by theta:
        # TODO: generalize to 2+ qubits  
        assert len(spins) == 1
        I = np.eye(2,dtype=complex)
        I[0,0] = np.exp( - 1j * theta ) 
        I[1,1] = np.exp( 1j * theta ) 
        # Promote to a d^2 x d^2 superoperator 
        return basis.compute_superoperator_from_unitary_operator(I)

    def R_gate_lindbladian_function(rabi_rate: float, phi: float, phase_mean: float, phase_variance: float):
        """ Builds a Lindbladian as a function of R gate rotation angle and phase, as well as phase noise mean and variance """ 
        # Onus is on the user to achieve theta = rabi_rate * gate_duration outside of this function.  
        raise_qubit_matrix = 0.5 * rabi_rate * np.exp(1j*phi)*basis.enlarge_matrix(sm.Pauli.plus, target_spins) 
        laser_freq = qubit_frequency

        # Driving field is on resonance with qubit frequency: 
        coupling_operator = sm.CouplingOperator.from_matrix(basis, raise_qubit_matrix, qubit_frequency) 

        # Include shift from mean of phase noise term  
        coherent_Z_shift = sm.EnergyShiftOperator.from_matrix(basis, sm.Pauli.Z * phase_mean * 0.5) 

        frame_energies = [-state.energy for state in basis.states] 
        H_0 = sm.Hamiltonian(basis, [coupling_operator, coherent_Z_shift], frame_energies) 

        # Dephasing with Lindblad operator L = sqrt(sigma^2) Z = sigma Z, on qubit 1:                 
        dephasing_matrix = np.sqrt(phase_variance) * basis.enlarge_matrix(sm.Pauli.Z , target_spins) 
        lindblad_ops = [sm.EnergyShiftOperator.from_matrix(basis, dephasing_matrix)]               

        dephasing_dissipator = sm.Dissipator(basis, lindblad_ops, frame_energies) 
        return sm.Lindbladian(hamiltonian = H_0, dissipator = dephasing_dissipator)

    # Option 1: We evaluate process matrix functions on the fly as needed in GST by creating a gate, extracting its process matrix, then discarding the created Gate at each eval.  
    # Option 2: We precompute / set up a bunch of Gates on a grid and then build a process matrix interpolating function and give that to GST.  

    # --- Option 1 ---
    # Returns process matrix for noisy X_pi/2 gate from the previous functions.  
    def X_pi2_process_matrix(X_rot: float, phase_mean: float, phase_std_deviation: float):
        """ Process matrix from Lindbladian for X_pi/2 rotation gate as a fxn of over/under rotation (X_rot), phase mean, phase standard deviation""" 
        gate_phase = 0. # for X rotation gate 
        theta = np.pi/2. + X_rot
        rabi_rate = theta 
        # Rabi rate is set to theta, so gate_duration is 1. (Omega*t = theta)
        gate_duration = 1. 
        #gate_duration = theta/rabi_rate 

        phase_variance = phase_std_deviation**2 
        # Build X_pi/2 Lindbladian from generalized R gate 
        X_pi2_lindbladian = R_gate_lindbladian_function(rabi_rate, gate_phase, phase_mean, phase_variance) 

        # Option 1: Creates a gate at every function evaluation. The gate is discarded at the end of this function call.   
        X_pi2_gate = sm.Gate.from_lindbladian(basis, X_pi2_lindbladian, gate_duration, lindbladian_time_independent=True)
        #X_pi2_gate = sm.Gate.from_lindbladian(basis, R_gate_lindbladian_function(rabi_rate, gate_phase, phase_mean, phase_variance), gate_duration)
        return X_pi2_gate.process_matrix 

    # Returns process matrix for noisy Y_pi/2 gate from the previous functions.  
    def Y_pi2_process_matrix(Y_rot: float, phase_mean: float, phase_std_deviation: float):
        """ Process matrix from Lindbladian for Y_pi/2 rotation gate as a fxn of over/under rotation (Y_rot), phase mean, phase variance """ 
        gate_phase = np.pi/2. # for Y rotation gate 
        theta = np.pi/2. + Y_rot
        rabi_rate = theta 
        # Rabi rate is set to theta, so gate_duration is 1. (Omega*t = theta)
        gate_duration = 1. 

        phase_variance = phase_std_deviation**2 
        # Build Y_pi/2 Lindbladian from generalized R gate 
        Y_pi2_lindbladian = R_gate_lindbladian_function(rabi_rate, gate_phase, phase_mean, phase_variance) 

        # Option 1: Creates a gate at every function evaluation. The gate is discarded at the end of this function call.   
        Y_pi2_gate = sm.Gate.from_lindbladian(basis, Y_pi2_lindbladian, gate_duration, lindbladian_time_independent=True)
        return Y_pi2_gate.process_matrix 

    # For GST, define state and measurement parametrizations (models): 
    # Here, we choose deviations from an ideal prep state and ideal POVM effects: 
    ideal_rho_prep = sm.State.from_coefficients(basis, list([1. + 0j, 0. + 0j]))

    ##### Define a parametrization (model) for prep state as a function:  ##### 
    d = len(basis.states)
    def prep_state_function(state_parameters: Vector) -> Vector: 
        """ Model of the prep state as a function of parameters (a vector with d^2 - 1 entries), returns a constrained supervector """ 
        # Here, we parametrize the state as a deviation from a known ideal state
        prep_state = (ideal_rho_prep.supervector).copy()
        prep_state[:-1] += state_parameters # deviations 

        # Enforce Tr[rho] = 1 constraint; Retrieve indices corresponding to diagonal density matrix entries 
        diag_indices = [i * (d + 1) for i in range(d)] # assumes square density matrix 
        prep_state[-1] = 1.0 - np.sum(prep_state[diag_indices[:-1]]) 
        return prep_state  

    ideal_POVM_effects = {} 
    ideal_POVM_effects['0'] = sm.EnergyShiftOperator.from_matrix(basis, sm.Pauli.projector_0) 
    ideal_POVM_effects['1'] = sm.EnergyShiftOperator.from_matrix(basis, sm.Pauli.projector_1) 

    N_effects = len(ideal_POVM_effects)
    assert N_effects == d

    # Set up dictionary of constrained measurement effect (POVM) models 
    POVM_models = {}
    for i, (outcome, ideal_effect) in enumerate(ideal_POVM_effects.items()): 
        if i == (N_effects - 1): 
            # Final POVM model is constrained by completeness / conservation of probability. This is handled in GST class.  
            POVM_models[outcome] = None 
            break
        # Define parametrization (model) for this effect: 
        def effect_function(effect_parameters: Vector, POVM_operator=ideal_effect):
            # Parameters represent deviations from ideal 
            return POVM_operator.superbra + effect_parameters 
        POVM_models[outcome] = effect_function


    # Define dictionary mappings for GST gate name to ionsim gate function 
    ism_gate_dictionary = {}    

    ## Functionality for interpolated gates since using Lindbladians can be slow. 
    use_interpolated_gates = False 
    if use_interpolated_gates:
        print(f" --- Using interpolated gates ---")        


        # Attempt to load process matrix data for X_pi2 and Y_pi2 if they exist  
        X_pi2_fname = './interpolated_Gxpi2.hdf5'
        Y_pi2_fname = './interpolated_Gypi2.hdf5'

        if Path(X_pi2_fname).exists():
            print(f" --- Loading X pi/2 gate interpolant from file --- " )
            # Construct gate interpolant from file 
            Xpi2_gate_interpolant = sm.GateInterpolant.from_file(X_pi2_fname, basis)
        else:
            # Build gate interpolant from process matrix functions specified above 
            Xpi2_model_sig = inspect.signature(X_pi2_process_matrix)
            Xpi2_parameter_names = list(Xpi2_model_sig.parameters.keys())  
            X_rot_domain = np.linspace(-0.1, 0.1, 12)
            mu_domain = np.linspace(-0.5, 0.5, 12)
            sigma_domain = np.linspace(0.0005, 0.1, 20)
            parameter_domains = [X_rot_domain, mu_domain, sigma_domain]
            parameter_grid = dict(zip(Xpi2_parameter_names, parameter_domains))

            Xpi2_gate_interpolant = sm.GateInterpolant.from_process_matrix_function(X_pi2_process_matrix, parameter_grid, basis, 'Gxpi2')
            Xpi2_gate_interpolant.write_to_file(X_pi2_fname)
            # Print to a file for later usage 
            print(f" - Successfully wrote X pi/2 gate to a file --- ")
            
        interpolated_Xpi2 = Xpi2_gate_interpolant.process_matrix_interpolant_function   # returns a process matrix at a grid point 

        # Build similar gate interpolant for Y  
        if Path(Y_pi2_fname).exists():
            print(f" --- Loading Y pi/2 gate interpolant from file --- " )
            # Construct gate interpolant from file 
            Ypi2_gate_interpolant = sm.GateInterpolant.from_file(Y_pi2_fname, basis)
        else:
            # Build gate interpolant from process matrix functions specified above 
            Ypi2_model_sig = inspect.signature(Y_pi2_process_matrix)
            Ypi2_parameter_names = list(Ypi2_model_sig.parameters.keys())  
            Y_rot_domain = np.linspace(-0.1, 0.1, 12)
            mu_domain = np.linspace(-0.5, 0.5, 12)
            sigma_domain = np.linspace(0.0005, 0.1, 20)
            parameter_domains = [Y_rot_domain, mu_domain, sigma_domain]
            parameter_grid = dict(zip(Ypi2_parameter_names, parameter_domains))

            Ypi2_gate_interpolant = sm.GateInterpolant.from_process_matrix_function(Y_pi2_process_matrix, parameter_grid, basis, 'Gypi2')
            Ypi2_gate_interpolant.write_to_file(Y_pi2_fname)
            print(f" - Successfully wrote Y pi/2 gate to a file --- ")

        interpolated_Ypi2 = Ypi2_gate_interpolant.process_matrix_interpolant_function   # returns a process matrix at a grid point 

        ism_gate_dictionary['Gxpi2'] = interpolated_Xpi2  
        ism_gate_dictionary['Gypi2'] = interpolated_Ypi2 
        ism_gate_dictionary['idle'] = idle 
    else:
        ism_gate_dictionary['Gxpi2'] = X_pi2_process_matrix 
        ism_gate_dictionary['Gypi2'] = Y_pi2_process_matrix
        ism_gate_dictionary['idle'] = idle 


    N_params = 14
    #N_params = 13
    parameter_bounds = [(None, None) for i in range(N_params)]
    parameter_bounds[-4] = (0., None)
    parameter_bounds[-1] = (0., None)

    GST_analyzer = sm.GateSetTomography(basis, prep_state_function, POVM_models, parsed_circuits, ism_gate_dictionary, parameter_bounds, gst_circuit_design)
    parameter_indices, N_params = GST_analyzer._build_parameter_organization()
    # 6 parameters in the gate set (3 for each gate); 7 for SPAM 
    theta_guess = np.ones(N_params)*1E-2

    start = time.perf_counter()
    # Specify ideal gate set and target state for linear GST 
    ideal_gate_set = {}
    ideal_gate_set['Gxpi2'] = X_pi2_process_matrix(0., 0., 0.) 
    ideal_gate_set['Gypi2'] = Y_pi2_process_matrix(0., 0., 0.)  
    ideal_gate_set['idle'] = idle(0.) 
    solver_results = GST_analyzer.solve_for_gate_parameters(theta_guess, 'linear', ideal_gate_set, ideal_rho_prep)

    #solver_results = GST_analyzer.solve_for_gate_parameters(theta_guess)
    end = time.perf_counter()
    print(f"\n\nGST analysis took {end - start} [s]\n")
    print(f"Solver results: {solver_results}")
    GST_analyzer.print_parameters()
    GST_analyzer.print_state_and_POVMs()

    # Construct ideal gate set to compute error metric of GST analysis & gate modeling 
    ideal_gate_set = {}
    ideal_gate_set['prep'] = ideal_rho_prep 
    ideal_gate_set['POVM'] = ideal_POVM_effects 
    ideal_gate_set['Gxpi2'] = X_pi2_process_matrix(0.015, 0.025, 0.01) 
    ideal_gate_set['Gypi2'] = Y_pi2_process_matrix(0.0075, 0.125, 0.05)  
    ideal_gate_set['idle'] = idle(0.0035248) 
    #ideal_gate_set['Gxpi2'] = basis.compute_superoperator_from_unitary_operator(sm.Unitary.sqrtX) 
    #ideal_gate_set['Gypi2'] = basis.compute_superoperator_from_unitary_operator(sm.Unitary.sqrtY)
    #ideal_gate_set['idle'] = basis.compute_superoperator_from_unitary_operator(sm.Unitary.I)

    print(f"\n\n")
    print(f" --- Estimating gate set error compared to ideal gate set ---")
    gate_set_error = GST_analyzer.compute_gate_set_error(solver_results, ideal_gate_set)
    print(f"\nGate set error: {gate_set_error}")

    estimate_uncertainties = False 
    write_data_to_file = True 
    if write_data_to_file:
        GST_analyzer.write_results_to_file()

    if estimate_uncertainties:
        print(f"\n\n -------- Estimating parameter uncertainties. ----------")
        uncertainties, covariance = GST_analyzer.estimate_parameter_uncertainties()
        print(f"\nPrinting parameters as one vector: {GST_analyzer.gst_parameters}")
        print(f"\nPrinting uncertainties in the parameters: {uncertainties}")

    return gate_set_error 

if __name__ == '__main__':
    main()
