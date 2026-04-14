from pathlib import Path
import numpy as np
import h5py
import sys
import time
from scipy.sparse import kron as skron
from matplotlib import pyplot as plt
from icecream import ic
from typing import Callable

import ionsim as sm

""" ################ Single qubit GST Example ################## """ 
def main():
    # 1. Import GST sequence data 
    fname = './1Q_gst_sequence.gstdata' 

    # Run the main parsing function:  
    parsed_circuits = sm.parse_gst_circuit_file(fname)

    print_head = True 
    if print_head:
        # Optional print out of first _ lines to check functionality  
        # Print circuit information: 
        head = 64
        for i, circ in enumerate(parsed_circuits):
            print(f"\n--- Experiment {i} ---")
            print(f"    Unparsed circuit line:  {circ.unparsed_data}")
            print(f"    Prep gates:    {circ.fiducial_prep_gates}")
            print(f"    Germ gates:    {circ.germ_gates}")
            print(f"    Germ power:    {circ.germ_power}")
            print(f"    Measure gates:    {circ.fiducial_measurement_gates}")
            print(f"    Measurement outcomes:    {circ.measurement_data.counts}")
            print(f"    Total shots:    {circ.total_counts}")
            print(f"    Circuit depth:    {circ.depth}")
            # Only print the first {head} 
            if i > head:
                break

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
    def R_gate_lindbladian_function(rabi_rate: float, phi: float, phase_mean: float, phase_variance: float):
        """ Builds a Lindbladian as a function of R gate rotation angle and phase, as well as phase noise mean and variance """ 

        # Onus is on the user to achieve theta = rabi_rate * gate_duration 
        raise_qubit_matrix = 0.5 * rabi_rate * basis.enlarge_matrix(sm.Pauli.plus, target_spins) 
        laser_freq = qubit_frequency

        # Driving field is on resonance with qubit frequency: 
        coupling_operator = CouplingOperator.from_matrix(basis, raise_qubit_matrix, qubit_frequency) 

        # Include shift from mean of phase noise term  
        Z_shift = EnergyShiftOperator.from_matrix(basis, sm.Pauli.Z * phase_mean) 

        frame_energies = [-state.energy for state in basis.states] 
        H_0 = sm.Hamiltonian(basis, [coupling_operator, Z_shift], frame_energies) 

        # Dephasing with Lindblad operator L = Z on qubit 1:                 
        dephasing_matrix = np.sqrt(phase_variance) * basis.enlarge_matrix(sm.Pauli.Z , target_spins) 
        lindblad_ops = [EnergyShiftOperator.from_matrix(basis, dephasing_matrix)]               

        dephasing_dissipator = sm.Dissipator(basis, lindblad_ops, frame_energies) 
        return sm.Lindbladian(hamiltonian = H_0, dissipator = dephasing_dissipator)

    # Builds X_pi/2 Lindbladian from generalized R gate 
    def X_pi2_lindbladian_function(X_rot, phase_mean, phase_variance) -> Lindbladian:
        """ Lindbladian for X_pi/2 rotation gate as a fxn of over/under rotation (X_rot), phase mean, phase variance """ 
        gate_phase = 0. # for X rotation gate 
        rabi_rate = 1.

        X_pi2_lindbladian = general_lindbladian_function(rabi_rate, gate_phase, phase_mean, phase_variance) 

        return X_pi2_lindbladian 

    # Builds Y_pi/2 Lindbladian from generalized R gate 
    def Y_pi2_lindbladian_function(Y_rot, phase_mean, phase_variance) -> Lindbladian:
        """ Lindbladian for Y_pi/2 rotation gate as a fxn of over/under rotation (Y_rot), phase mean, phase variance """ 
        gate_phase = np.pi/2. # for Y rotation gate 
        rabi_rate = 1.

        Y_pi2_lindbladian = general_lindbladian_function(rabi_rate, gate_phase, phase_mean, phase_variance) 

        return Y_pi2_lindbladian 


    # Maybe just have a method "compute_process_matrix_from_lindbladian" that avoids creation of gate (if we're sending straight to GST) 
    # But, if we create gates at a set of points, we can do gate interpolation. 

    # Option 1: We evaluate process matrix functions on the fly as needed in GST.  
    # Option 2: We precompute / set up a bunch of Gates on a grid and then build an interpolating function and give to GST.  

    # Returns process matrix for noisy X_pi/2 gate from the previous functions.  
    def X_pi2_process_matrix(X_rot, phase_mean, phase_variance) -> Matrix:
        theta = np.pi/2. + X_rot

        # Rabi rate is set to 1, so gate_duration to theta s.t. Omega*t = theta
        gate_duration = theta 

        # Option 1: Creates a gate at every function evaluation. The gate is discarded at the end of this function call.   
        X_pi2_gate = sm.Gate.from_lindbladian(basis, X_pi2_lindbladian_function(X_rot, phase_mean, phase_variance), gate_duration)
        return X_pi2_gate.process_matrix 

    def Y_pi2_process_matrix(Y_rot, phase_mean, phase_variance) -> Matrix:
        theta = np.pi/2. + Y_rot

        # Rabi rate is set to 1, so gate_duration to theta s.t. Omega*t = theta
        gate_duration = theta 

        # Option 1: Creates a gate at every function evaluation. The gate is discarded at the end of this function call.   
        Y_pi2_gate = sm.Gate.from_lindbladian(basis, Y_pi2_lindbladian_function(Y_rot, phase_mean, phase_variance), gate_duration)
        return Y_pi2_gate.process_matrix 


    # Define dictionary mappings for GST gate name to ionsim gate function 
    ism_gate_dictionary = {}    
    ism_gate_dictionary['Gxpi2']  = X_pi2_process_matrix 
    ism_gate_dictionary['Gypi2'] = Y_pi2_process_matrix
    ism_gate_dictionary['idle'] = idle
    #ism_gate_dictionary['{}'] = null  # no need to specify this; this case is handled by gst class  

    # For GST, define state and measurement parametrizations (models): 
    # Here, we choose deviations from an ideal prep state and ideal POVM effects: 
    ideal_rho_prep = sm.State.from_coefficients(basis, list([1., 0.]))

    ##### Define a parametrization (model) for prep state as a function:  ##### 
    d = len(basis.states)
    def prep_state_function(state_parameters: Vector) -> Vector: 
        """ Model of the prep state as a function of parameters (a vector with d^2 - 1 entries), returns a constrained supervector """ 
        # TODO: Discuss w. Brandon: Should we normalize this here / enforce the constraint here? 

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

    GST_analyzer = sm.GateSetTomography(basis, prep_state_function, POVM_models, parsed_circuits, ism_gate_dictionary)
    solver_results = GST_analyzer.solve_for_gate_parameters()
    print(f"Solver results: {solver_results}")
    GST_analyzer.print_parameters()
    GST_analyzer.print_state_and_POVMs()
    sys.exit(0)

    # TODO: Either save gates evaluated at the parameter values or just the parameter values.  

if __name__ == '__main__':
    main()
