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
import os 
matplotlib.rcParams['text.usetex']=True 
style_path_data = '~/plot_style_data.txt'

""" ################ Single qubit GST Example ################## """ 
def run_GST(fname: str, include_SPAM_error: bool=False):
    # 1. Import GST sequence data 
    #fname = './simulated_gst_experimental_data.gstdata' 

    # Run the main parsing function:  
    parsed_circuits = sm.parse_gst_circuit_file(fname)
    parsed_circuits = parsed_circuits
    #parsed_circuits = parsed_circuits[:2500]

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

    num_spins = 2
    
    spins = [
        sm.AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        for _ in range(num_spins)
    ]
    
    basis = sm.StandardBasis([*spins])

    ################ Define gate error models: #################### 
    # Requires a basis to be defined 

    # Define dictionary mappings for GST gate name to ionsim gate function 
    ism_gate_dictionary = {}    
    ism_gate_dictionary['idle'] = idle
    ism_gate_dictionary['Gxpi2:0'] = X_pi2_q0
    ism_gate_dictionary['Gxpi2:1'] = X_pi2_q1
    ism_gate_dictionary['Gypi2:0'] = Y_pi2_q0
    ism_gate_dictionary['Gypi2:1'] = Y_pi2_q1
    ism_gate_dictionary['Gcnot:0:1'] = cnot 

    # For GST, define state and measurement parametrizations (models): 
    # Here, we choose deviations from an ideal prep state and ideal POVM effects: 
    ideal_rho_prep = sm.State.from_coefficients(basis, list([1., 0., 0., 0.]))

    ##### Define a parametrization (model) for prep state as a function:  ##### 
    d = len(basis.states)
    assert d == 2**num_spins

    def prep_state_function(q1_probability_of_wrong_prep: float, q2_probability_of_wrong_prep: float): 
        """ Model of the prep state as a function of parameters (a vector with d^2 - 1 entries), returns a constrained supervector """ 
        # Here, we parametrize the state using the probability of preparing a |1> instead of the intended |0> for a single qubit:
        rho_q1 = np.zeros((2,2), dtype=complex)
        rho_q1[0,0] = (1. - q1_probability_of_wrong_prep)
        rho_q1[1,1] = q1_probability_of_wrong_prep
        rho_q2 = np.zeros((2,2), dtype=complex)
        rho_q2[0,0] = (1. - q2_probability_of_wrong_prep)
        rho_q2[1,1] = q2_probability_of_wrong_prep
        rho_2Q = np.kron(rho_q1, rho_q2)
        state = sm.State.from_density_matrix(basis, rho_2Q)
        #rho_2Q.flatten("F")
        return state.supervector 

    ideal_POVM_effects = {} 
    ideal_POVM_effects['00'] = sm.EnergyShiftOperator.from_matrix(basis, np.kron(sm.Pauli.projector_0, sm.Pauli.projector_0)) 
    ideal_POVM_effects['01'] = sm.EnergyShiftOperator.from_matrix(basis, np.kron(sm.Pauli.projector_0, sm.Pauli.projector_1)) 
    ideal_POVM_effects['10'] = sm.EnergyShiftOperator.from_matrix(basis, np.kron(sm.Pauli.projector_1, sm.Pauli.projector_0)) 
    ideal_POVM_effects['11'] = sm.EnergyShiftOperator.from_matrix(basis, np.kron(sm.Pauli.projector_1, sm.Pauli.projector_1)) 

    N_effects = len(ideal_POVM_effects)
    assert N_effects == d

    # Set up dictionary of constrained measurement effect (POVM) models 
    def E0_1Q(prob_false_bright:float, prob_false_dark: float):
        M = np.zeros((2,2), dtype=complex)
        M[0,0] = (1. - prob_false_bright)
        M[1,1] = prob_false_dark
        return M

    def E1_1Q(prob_false_bright:float, prob_false_dark: float):
        M = np.zeros((2,2), dtype=complex)
        M[0,0] = prob_false_bright
        M[1,1] = (1. - prob_false_dark)
        return M


    POVM_models = {}
    # Simple parametrized measurement effect models 
    def effect_00(prob_false_bright: float, prob_false_dark: float): 
        M0 = E0_1Q(prob_false_bright, prob_false_dark)
        matrix = np.kron(M0,M0)
        operator = sm.EnergyShiftOperator.from_matrix(basis, matrix) 
        return operator.superbra 

    def effect_01(prob_false_bright: float, prob_false_dark: float): 
        M0 = E0_1Q(prob_false_bright, prob_false_dark)
        M1 = E1_1Q(prob_false_bright, prob_false_dark)
        matrix = np.kron(M0,M1)
        operator = sm.EnergyShiftOperator.from_matrix(basis, matrix) 
        return operator.superbra 

    def effect_10(prob_false_bright: float, prob_false_dark: float): 
        M0 = E0_1Q(prob_false_bright, prob_false_dark)
        M1 = E1_1Q(prob_false_bright, prob_false_dark)
        matrix = np.kron(M1,M0)
        operator = sm.EnergyShiftOperator.from_matrix(basis, matrix) 
        return operator.superbra 

    def effect_11(prob_false_bright: float, prob_false_dark: float): 
        M1 = E1_1Q(prob_false_bright, prob_false_dark)
        matrix = np.kron(M1,M1)
        operator = sm.EnergyShiftOperator.from_matrix(basis, matrix) 
        return operator.superbra 

    POVM_models["00"] = effect_00
    POVM_models["01"] = effect_01
    POVM_models["10"] = effect_10
    POVM_models["11"] = effect_11

    # 7 parameters in the gate set; 7 for SPAM 
    # Construct ideal gate set to compute error metric of GST analysis & gate modeling 
    ideal_gate_set = {}
    ideal_gate_set['prep'] = ideal_rho_prep 
    ideal_gate_set['POVM'] = ideal_POVM_effects 
    ideal_gate_set['Gxpi2:0'] = X_pi2_q0(0.0, 0.0) 
    ideal_gate_set['Gxpi2:1'] = X_pi2_q1(0.0, 0.0) 
    ideal_gate_set['Gypi2:0'] = Y_pi2_q0(0.0)
    ideal_gate_set['Gypi2:1'] = Y_pi2_q1(0.0)
    ideal_gate_set['idle'] = idle(0.)
    ideal_gate_set['Gcnot:0:1'] = cnot(0., 0.)

    design_fname = 'circuit_design.yml'
    gst_circuit_design = None 
    if os.path.exists("./" + design_fname):
        gst_circuit_design = sm.GSTCircuitPlanner.load_design(design_fname)

    parameter_bounds = {
        "prep" : {"q1_probability_of_wrong_prep" : (0., 1.), "q2_probability_of_wrong_prep" : (0., 1.)},
        "00" : {"prob_false_bright" : (0., 1.), "prob_false_dark" : (0., 1.)},
        "01" : {"prob_false_bright" : (0., 1.), "prob_false_dark" : (0., 1.)},
        "10" : {"prob_false_bright" : (0., 1.), "prob_false_dark" : (0., 1.)},
        "11" : {"prob_false_bright" : (0., 1.), "prob_false_dark" : (0., 1.)},
    } 
    #parameter_bounds = None 
    GST_analyzer = sm.GateSetTomography(basis, prep_state_function, POVM_models, parsed_circuits, ism_gate_dictionary, circuit_design = gst_circuit_design, 
                                    parameter_bounds = parameter_bounds, ideal_gate_set = ideal_gate_set, verbose = True)
    print(f"Num parameters: {GST_analyzer.num_parameters}")

    parameter_guess = np.ones(GST_analyzer.num_parameters)*1e-2
    #print(f"Number of model parameters: {GST_analyzer.num_parameters}")
    #print(f"Parameter organization: {GST_analyzer.gst_parameter_indices}")
    start = time.perf_counter()
    #solver_options = {"maxfun" : 10000}
    #solver_results = GST_analyzer.solve_for_gate_parameters(None, 'linear') 
    #solver_results = GST_analyzer.solve_for_gate_parameters(None, 'MLE') 
    solver_results = GST_analyzer.solve_for_gate_parameters(parameter_guess, 'staged MLE') 
    #solver_results = GST_analyzer.solve_for_gate_parameters(parameter_guess, 'MLE') 
    #solver_results = GST_analyzer.solve_for_gate_parameters(parameter_guess, 'MLE', options = solver_options)
    #results_by_stage = GST_analyzer.circuit_depth_scaling_analysis()
    end = time.perf_counter()
    print(f"Ran staged GST in {end - start} seconds")
    print(f"Solver results: {solver_results}\n")
    print()
    GST_analyzer.print_parameters()
    GST_analyzer.print_state_and_POVMs()

    print()
    #print(GST_analyzer.lgst_results['estimated_effect'])
    #d2 = 4**2

    #circuit_depths = list(results_by_stage.keys())
    #error_metric = {}
    #errors_by_gate = {}
 #    for p in circuit_depths:
 #        error_metric[p] = GST_analyzer.compute_gate_set_error(results_by_stage[p], ideal_gate_set, include_SPAM_error)  
 #        errors_by_gate[p] = GST_analyzer.compute_gate_set_error_by_element(results_by_stage[p], ideal_gate_set)
        #error_metric[p] = GST_analyzer.compute_gate_set_process_infidelity(results_by_stage[p], ideal_gate_set, include_SPAM_error=True)  


    # errors_by_gate is a dictionary with keys = depth, and values = dict with keys = gate name, values = error 
    # Organize into a dictionary with key = gate name, value is array of length p with error for each p   
 #    gate_errors = {}        
 #    for gate_name in ism_gate_dictionary.keys():
 #        # Build array of errors, then extract error for each depth 
 #        errors = np.zeros(len(circuit_depths))
 #        for i, p in enumerate(circuit_depths):
 #            errors[i] = errors_by_gate[p][gate_name]
 #        gate_errors[gate_name] = errors

    #print(f"Gate set errors: {GST_analyzer.compute_gate_set_error_by_element(solver_results, ideal_gate_set, error_metric = 'process infidelity')}")
    print(f"Gate set errors: {GST_analyzer.compute_gate_set_error_by_element(solver_results, ideal_gate_set, error_metric = 'process infidelity')}")
    #gate_set_error = GST_analyzer.compute_gate_set_error(solver_results.x, ideal_gate_set, include_SPAM_error=include_SPAM_error) 
    #gate_set_error = GST_analyzer.compute_gate_set_error(solver_results.x, ideal_gate_set, include_SPAM_error=include_SPAM_error) 
    return gate_set_error



if __name__ == '__main__':
    filename = '20210927_GST_2Qubit_lite_nofpr_003.gstdata'
    run_GST(filename, False)
