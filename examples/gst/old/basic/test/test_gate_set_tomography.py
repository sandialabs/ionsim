from pathlib import Path
import numpy as np
import h5py
import sys
import time
from scipy.sparse import kron as skron
import matplotlib
from matplotlib import pyplot as plt
from icecream import ic
from typing import Callable
import ionsim as sm
matplotlib.rcParams['text.usetex']=True 
style_path_data = '~/plot_style_data.txt'

""" ################ Single qubit GST Example ################## """ 
def main():
    # 1. Import GST sequence data 
    fname = './1Q_gst_sequence.gstdata' 

    # Run the main parsing function:  
    parsed_circuits = sm.parse_gst_circuit_file(fname)

    print_head = False 
    #head = 780
    head = 20
    if print_head:
        # Optional print out of first _ lines to check functionality  
        # Print circuit information: 
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

    ## Check to see if GST runs quickly and gets decent estimates with not many circuits  
    parsed_circuits = parsed_circuits[:head]
    # Set up basic 1-qubit (1Q) basis  
    num_spins = 1
    
    spins = [
        sm.AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        for _ in range(num_spins)
    ]
    
    basis = sm.StandardBasis([*spins])
    target_spins = [spins[0]]


    ################ Define gate error models: #################### 
    # Requires a basis to be defined 
    # 1. Process matrix functions --> return a process matrix evaluated at the parameter values  
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
   
 
    def X_pi2(X_rot, Z_rot):
        """ Returns Gate from the d^2 x d^2 process matrix function in standard basis 
    
            X_pi2 = exp( -i [ (pi/2 + X_rot) X  + (Z_rot)Z ] )
    
            - X_rot is an additional X_rotation parameter (over/under rotation).
            - Z_rot is a Z_rotation parameter, e.g. from a detuned laser. 
    
        """  
        # TODO: generalize to 2+ qubits  
        assert len(spins) == 1
        x_angle = np.pi/2. + X_rot
        Rxpi2 = sm.Unitary.R_bloch([x_angle/2., 0., Z_rot/2.]) 
        
        # Promote to a d^2 x d^2 superoperator 
        return basis.compute_superoperator_from_unitary_operator(Rxpi2) # superoperator 


    def Y_pi2(Y_rot, Z_rot):
        """ Returns Gate from the d^2 x d^2 process matrix function in standard basis 
    
            Y_pi2 = exp( -i [ (pi/2 + Y_rot) Y  + (Z_rot)Z ] )
    
            - Y_rot is an additional Y_rotation parameter (over/under rotation).
            - Z_rot is a Z_rotation parameter, e.g. from a detuned laser. 
    
        """  
        # TODO: generalize to 2+ qubits  
        assert len(spins) == 1
        y_angle = np.pi/2. + Y_rot
        Rypi2 = sm.Unitary.R_bloch([0., y_angle/2., Z_rot/2.]) 
    
        # Promote to a d^2 x d^2 superoperator 
        return basis.compute_superoperator_from_unitary_operator(Rypi2)


 #    def null():
 #        """ Returns d^2 x d^2 process matrix in standard basis a null ("do nothing for no time") gate"""  
 #        # Build identity matrix with Z rotation by theta:
 #        # TODO: generalize to 2+ qubits  
 #        assert len(spins) == 1
 #        I = np.eye(2,dtype=complex)
 #        # Promote to a d^2 x d^2 superoperator 
 #        return basis.compute_superoperator_from_unitary_operator(I)
 #

    # Define dictionary mappings for GST gate name to ionsim gate function 
    ism_gate_dictionary = {}    
    ism_gate_dictionary['Gxpi2']  = X_pi2 
    ism_gate_dictionary['Gypi2'] = Y_pi2
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

    import time
    start = time.perf_counter()
    GST_analyzer = sm.GateSetTomography(basis, prep_state_function, POVM_models, parsed_circuits, ism_gate_dictionary)
    solver_results = GST_analyzer.solve_for_gate_parameters()
    end = time.perf_counter()
    print(f"Ran GST in {end - start} seconds")
    print(f"Solver results: {solver_results}\n")
    gst_parameters = GST_analyzer.print_parameters()
    GST_analyzer.print_state_and_POVMs()

    # Construct ideal gate set to compute error metric of GST analysis & gate modeling 
    ideal_gate_set = {}
    ideal_gate_set['prep'] = ideal_rho_prep 
    ideal_gate_set['POVM'] = ideal_POVM_effects 
    ideal_gate_set['Gxpi2'] = basis.compute_superoperator_from_unitary_operator(sm.Unitary.sqrtX) 
    ideal_gate_set['Gypi2'] = basis.compute_superoperator_from_unitary_operator(sm.Unitary.sqrtY)
    ideal_gate_set['idle'] = basis.compute_superoperator_from_unitary_operator(sm.Unitary.I)

    print(f"\n\n")
    print(f" --- Estimating gate set error compared to ideal gate set ---")
    gate_set_error = GST_analyzer.compute_gate_set_process_infidelity(gst_parameters, ideal_gate_set)
    print(f"\nGate set error: {gate_set_error}")

    estimate_uncertainties = False 
    write_data_to_file = False 
    if write_data_to_file:
        GST_analyzer.write_results_to_file()

    if estimate_uncertainties:  
        uncertainties, covariance = GST_analyzer.estimate_parameter_uncertainties()
        print(f"\nPrinting parameters as one vector: {GST_analyzer.gst_parameters}")
        print(f"\nPrinting uncertainties in the parameters: {uncertainties}")


 #    print(f"\n\n ---------- Testing staged MLE functionality ------------ \n")
 #    start = time.perf_counter()
 #    solver_results, results_by_stage = GST_analyzer.solve_for_gate_parameters(solver='staged MLE')
 #    end = time.perf_counter()
 #    print(f"Ran staged GST in {end - start} seconds")
 #    print(f"Solver results: {solver_results}\n")
 #    circuit_depths = results_by_stage.keys()
 #    error_metric = {}
 #    for L in circuit_depths:
 #        error_metric[L] = GST_analyzer.compute_gate_set_process_infidelity(results_by_stage[L], ideal_gate_set)    
 #
 #    plt.style.use(style_path_data) 
 #    plt.figure(figsize = (4,4))
 #    plt.plot(circuit_depths, error_metric.values(), marker = 'o', color = 'k', label='GST')
 #    plt.title(r'Gate Set Error vs. Circuit Depth', fontsize = 14)
 #    plt.xlabel(r'Circuit depth $L$', fontsize = 20)
 #    plt.ylabel(r'$\epsilon$', fontsize = 24, rotation = 0, labelpad = 25)
 #    plt.xticks(fontsize = 12)
 #    plt.legend()
 #    plt.xscale('log')
 #    plt.yscale('log')
 #    #plt.savefig(data_directory / f'infidelity_vs_{dx_name}.pdf', bbox_inches='tight')
 #    plt.show()

    #return gate_set_error 

if __name__ == '__main__':
    main()
