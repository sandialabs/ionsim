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
import glob
import re
import ionsim as sm
from gate_models import * 
matplotlib.rcParams['text.usetex']=True 
style_path_data = '~/plot_style_data.txt'

""" ################ Single qubit GST Example ################## """ 
def run_GST(fname: str, include_SPAM_error: bool=False):
    # 1. Import GST sequence data 
    #fname = './simulated_gst_experimental_data.gstdata' 

    # Run the main parsing function:  
    parsed_circuits = sm.parse_gst_circuit_file(fname)
    parsed_circuits = parsed_circuits[:500]

    print_head = False 
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
    num_spins = 2
    
    spins = [
        sm.AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        for _ in range(num_spins)
    ]
    
    basis = sm.StandardBasis([*spins])
    target_spins = [spins[0]]


    ################ Define gate error models: #################### 
    # Requires a basis to be defined 

    # Define dictionary mappings for GST gate name to ionsim gate function 
    ism_gate_dictionary = {}    
    ism_gate_dictionary['Gxpi2']  = X_pi2 
    ism_gate_dictionary['Gypi2'] = Y_pi2
    ism_gate_dictionary['idle'] = idle

    # For GST, define state and measurement parametrizations (models): 
    # Here, we choose deviations from an ideal prep state and ideal POVM effects: 
    ideal_rho_prep = sm.State.from_coefficients(basis, list([1., 0., 0., 0.]))

    ##### Define a parametrization (model) for prep state as a function:  ##### 
    d = len(basis.states)
    assert d == 2**num_spins

    def prep_state_function(state_parameters): 
        """ Model of the prep state as a function of parameters (a vector with d^2 - 1 entries), returns a constrained supervector """ 
        # Here, we parametrize the state as a deviation from a known ideal state
        prep_state = np.zeros(len(ideal_rho_prep.supervector), dtype=complex)
        #prep_state = (ideal_rho_prep.supervector).copy()
        prep_state += ideal_rho_prep.supervector 
        prep_state[:-1] += state_parameters # deviations 

        # Enforce Tr[rho] = 1 constraint; Retrieve indices corresponding to diagonal density matrix entries 
        diag_indices = [i * (d + 1) for i in range(d)] # assumes square density matrix 
        prep_state[-1] = 1.0 - np.sum(prep_state[diag_indices[:-1]]) 
        return prep_state  

    ideal_POVM_effects = {} 
    ideal_POVM_effects['00'] = sm.EnergyShiftOperator.from_matrix(basis, np.kron(sm.Pauli.projector_0, sm.Pauli.projector_0)) 
    ideal_POVM_effects['01'] = sm.EnergyShiftOperator.from_matrix(basis, np.kron(sm.Pauli.projector_0, sm.Pauli.projector_1)) 
    ideal_POVM_effects['10'] = sm.EnergyShiftOperator.from_matrix(basis, np.kron(sm.Pauli.projector_1, sm.Pauli.projector_0)) 
    ideal_POVM_effects['11'] = sm.EnergyShiftOperator.from_matrix(basis, np.kron(sm.Pauli.projector_1, sm.Pauli.projector_1)) 

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

    # 7 parameters in the gate set; 7 for SPAM 
    # Construct ideal gate set to compute error metric of GST analysis & gate modeling 
    ideal_gate_set = {}
    ideal_gate_set['prep'] = ideal_rho_prep 
    ideal_gate_set['POVM'] = ideal_POVM_effects 
    ideal_gate_set['Gxpi2'] = X_pi2(0.045, 0.01) 
    ideal_gate_set['Gypi2'] = Y_pi2(-0.042)
    ideal_gate_set['idle'] = idle(0.0035248)
    design_fname = 'circuit_design.yml'
    gst_circuit_design = sm.GSTCircuitPlanner.load_design(design_fname)
    GST_analyzer = sm.GateSetTomography(basis, prep_state_function, POVM_models, parsed_circuits, ism_gate_dictionary, circuit_design = gst_circuit_design, ideal_gate_set = ideal_gate_set, verbose = False)

    start = time.perf_counter()
    solver_results, results_by_stage = GST_analyzer.solve_for_gate_parameters(None, 'staged MLE')
    #results_by_stage = GST_analyzer.circuit_depth_scaling_analysis()
    end = time.perf_counter()
    print(f"Ran staged GST in {end - start} seconds")
    print(f"Solver results: {solver_results}\n")
    circuit_depths = list(results_by_stage.keys())
    error_metric = {}
    errors_by_gate = {}
    for p in circuit_depths:
        error_metric[p] = GST_analyzer.compute_gate_set_error(results_by_stage[p], ideal_gate_set, include_SPAM_error)  
        errors_by_gate[p] = GST_analyzer.compute_gate_set_error_by_element(results_by_stage[p], ideal_gate_set)
        #error_metric[p] = GST_analyzer.compute_gate_set_process_infidelity(results_by_stage[p], ideal_gate_set, include_SPAM_error=True)  


    # errors_by_gate is a dictionary with keys = depth, and values = dict with keys = gate name, values = error 
    # Organize into a dictionary with key = gate name, value is array of length p with error for each p   
    gate_errors = {}        
    for gate_name in ism_gate_dictionary.keys():
        # Build array of errors, then extract error for each depth 
        errors = np.zeros(len(circuit_depths))
        for i, p in enumerate(circuit_depths):
            errors[i] = errors_by_gate[p][gate_name]
        gate_errors[gate_name] = errors

    # Overlay a plot of slope -1 
    #line_x = np.logspace(np.array(list(error_metric.values()))[0], 
    line_x = np.logspace(0., 2.2, 100) 
    offset = 1E-2
    line_y = offset * (line_x) ** -1

    X = np.log(np.array(circuit_depths))
    infidelities = np.array(list(error_metric.values()))
    Y = np.log(infidelities)
    start_indx = 0
    end_indx = len(X) 
    coefficients = np.polyfit(X[start_indx:end_indx], Y[start_indx:end_indx], 1) 

    print()
    print(coefficients)

    print(f"Slope: {coefficients[0]}")
    print(f"Intercept: {coefficients[1]}")

    plt.style.use(style_path_data) 
    plt.figure(figsize = (4,4))
    plt.plot(circuit_depths, infidelities, marker = 'o', linewidth = 0.5, markersize = 6, color = 'k', label='GST')
    plt.plot(line_x, line_y, linestyle = 'dashed', color='k', linewidth = 2., label= r'$m=-1$')
    plt.plot(np.exp(X), np.exp(coefficients[0]*X + coefficients[1]), linestyle = 'solid', color = 'k', linewidth = 1.5, label = r'Fit: $m = ' + str(np.round(coefficients[0].real, 3)) + '$')
    plt.title(r'Gate Set Error vs. Circuit Depth', fontsize = 14)
    plt.xlabel(r'Circuit depth $L$', fontsize = 20)
    #plt.xlabel(r'Germ power $p$', fontsize = 20)
    plt.ylabel(r'Error', fontsize = 24, rotation = 0, labelpad = 32)
    plt.xticks(fontsize = 12)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('./avg_gate_error_vs_depth.pdf', dpi=300)
    #plt.show()
    np.savetxt('error_vs_p.dat', np.column_stack([np.array(circuit_depths), np.array(list(error_metric.values()))])) 

    colors = ['k', 'r', 'b']
    markers = ['o', 'p', '*']
    plt.figure(figsize = (5,5))
    for i, (gate, errors) in enumerate(gate_errors.items()):
        plt.plot(circuit_depths, errors, marker = markers[i], linewidth = 0.5, markersize = 6, color = colors[i], label=gate)
        np.savetxt(gate + '_error_vs_depth.dat', np.column_stack([np.array(circuit_depths), errors])) 
    ## Organize error by gate in each stage 
    plt.plot(line_x, line_y, linestyle = 'dashed', color='k', linewidth = 2., label= r'$m=-1$')
    #plt.plot(np.exp(X), np.exp(coefficients[0]*X + coefficients[1]), linestyle = 'solid', color = 'k', linewidth = 1.5, label = r'Fit: $m = ' + str(np.round(coefficients[0].real, 3)) + '$')
    plt.title(r'Gate Set Error vs. Circuit Depth', fontsize = 14)
    plt.xlabel(r'Circuit depth $L$', fontsize = 20)
    plt.ylabel(r'Error', fontsize = 24, rotation = 0, labelpad = 32)
    plt.xticks(fontsize = 12)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('./gate_error_vs_depth.pdf', dpi=300)
    #plt.show()



if __name__ == '__main__':
    filename = '20210927_GST_2Qubit_lite_nofpr_003.gstdata'
    run_GST(filename, False)
