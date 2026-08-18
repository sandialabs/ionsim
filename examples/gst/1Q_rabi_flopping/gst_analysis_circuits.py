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

""" ################ Two qubit GST Example ################## """ 
def run_GST(fname: str, n_circuits: int, include_SPAM_error: bool=False):
    # 1. Import GST sequence data 
    # Run the main parsing function:  
    parsed_circuits = sm.parse_gst_circuit_file(fname)
    parsed_circuits = parsed_circuits[:n_circuits]

    num_spins = 1
    spins = [
        sm.AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        for _ in range(num_spins)
    ]
    basis = sm.StandardBasis([*spins])

    ################ Define gate error models: #################### 
    # Requires a basis to be defined 

    # Define dictionary mappings for GST gate name to ionsim gate function 
    ism_gate_dictionary = {}    
    #ism_gate_dictionary['idle'] = idle
    ism_gate_dictionary['Gxpi8:0'] = X_pi_8_co_prop 

    # For GST, define state and measurement parametrizations (models): 
    # Here, we choose deviations from an ideal prep state and ideal POVM effects: 
    ideal_rho_prep = sm.State.from_coefficients(basis, list([1., 0.]))

    ##### Define a parametrization (model) for prep state as a function:  ##### 
    d = len(basis.states)
    assert d == 2**num_spins

    def prep_state_function(q1_probability_of_wrong_prep: float, q2_probability_of_wrong_prep: float): 
        """ Model of the prep state as a function of parameters (a vector with d^2 - 1 entries), returns a constrained supervector """ 
        # Here, we parametrize the state using the probability of preparing a |1> instead of the intended |0> for a single qubit:
        rho_q1 = np.zeros((2,2), dtype=complex)
        rho_q1[0,0] = (1. - q1_probability_of_wrong_prep)
        rho_q1[1,1] = q1_probability_of_wrong_prep
        state = sm.State.from_density_matrix(basis, rho_q1)
        #rho_2Q.flatten("F")
        return state.supervector 

    ideal_POVM_effects = {} 
    ideal_POVM_effects['0'] = sm.EnergyShiftOperator.from_matrix(basis, sm.Pauli.projector_0)
    ideal_POVM_effects['1'] = sm.EnergyShiftOperator.from_matrix(basis, sm.Pauli.projector_1) 

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

############ Option 1 for POVMs: Give a dictionary of models (callables) ############
 #    POVM_models = {}
 #    # Simple parametrized measurement effect models 
 #    def effect_00(prob_false_bright: float, prob_false_dark: float): 
 #        M0 = E0_1Q(prob_false_bright, prob_false_dark)
 #        matrix = np.kron(M0,M0)
 #        operator = sm.EnergyShiftOperator.from_matrix(basis, matrix) 
 #        return operator.superbra 
 #
 #    def effect_01(prob_false_bright: float, prob_false_dark: float): 
 #        M0 = E0_1Q(prob_false_bright, prob_false_dark)
 #        M1 = E1_1Q(prob_false_bright, prob_false_dark)
 #        matrix = np.kron(M0,M1)
 #        operator = sm.EnergyShiftOperator.from_matrix(basis, matrix) 
 #        return operator.superbra 
 #
 #    def effect_10(prob_false_bright: float, prob_false_dark: float): 
 #        M0 = E0_1Q(prob_false_bright, prob_false_dark)
 #        M1 = E1_1Q(prob_false_bright, prob_false_dark)
 #        matrix = np.kron(M1,M0)
 #        operator = sm.EnergyShiftOperator.from_matrix(basis, matrix) 
 #        return operator.superbra 
 #
 #    def effect_11(prob_false_bright: float, prob_false_dark: float): 
 #        M1 = E1_1Q(prob_false_bright, prob_false_dark)
 #        matrix = np.kron(M1,M1)
 #        operator = sm.EnergyShiftOperator.from_matrix(basis, matrix) 
 #        return operator.superbra 
 #
 #    POVM_models["00"] = effect_00
 #    POVM_models["01"] = effect_01
 #    POVM_models["10"] = effect_10
 #    POVM_models["11"] = effect_11

    ############ Option 2 for POVMs: Give a callable returning a dictionary of model evaluations ############
    # instead of a dict that has outcome; callable pairs, try a POVM models that is a callable that returns a dict 
    def POVM_models(prob_false_bright: float, prob_false_dark: float) -> dict:
        """ Dictionary of POVMs evaluated at the function parameters: """ 
        POVMs = {}

        # 0
        M0 = E0_1Q(prob_false_bright, prob_false_dark)
        operator = sm.EnergyShiftOperator.from_matrix(basis, M0)
        POVMs["0"] = operator.superbra 

        # 1
        M1 = E1_1Q(prob_false_bright, prob_false_dark)
        operator = sm.EnergyShiftOperator.from_matrix(basis, M1) 
        POVMs["1"] = operator.superbra 

        return POVMs ## POVMs["00"] -> row vector 

    # 7 parameters in the gate set; 7 for SPAM 
    # Construct ideal gate set to compute error metric of GST analysis & gate modeling 
    ideal_gate_set = {}
    ideal_gate_set['prep'] = ideal_rho_prep 
    ideal_gate_set['POVM'] = ideal_POVM_effects 
    # set up reference (true) gate 
    rabi_rate = 100e3 * 2*np.pi # rad./s
    pi_time = abs(np.pi)/rabi_rate
    rabi_duration = 16 * pi_time
    spin_flip_rate = 1 / rabi_duration 
    ideal_gate_set['Gxpi8:0'] = X_pi_8_co_prop(spin_flip_rate)

    design_fname = 'circuit_design.yml'
    gst_circuit_design = sm.GSTCircuitPlanner.load_design(design_fname)
    #parameters_guess = {
    #    "prep" : {"q1_probability_of_wrong_prep" : 0.001, "q2_probability_of_wrong_prep" : 0.001},
    #    "POVM" : {"prob_false_bright" : 0.001, "prob_false_dark" : 0.001},
    #    "Gxpi8" : {"spin_flip_rate" : 100}
    #}
    parameters_guess = {"Gxpi8:0" : {"spin_flip_rate" : 20000}}
    # TODO: add parameters guess for prep and measure 

    G = sm.ParsedGate.from_string("Gxpi8:0")
    parameter_bounds = {
        "prep" : {"q1_probability_of_wrong_prep" : (0., 1.), "q2_probability_of_wrong_prep" : (0., 1.)},
        "POVM" : {"prob_false_bright" : (0., 1.), "prob_false_dark" : (0., 1.)},
        G : {"spin_flip_rate" : (0, 200000)}
        #"Gxpi8" : {"spin_flip_rate" : (0, 200000)}
    } 
    GST_analyzer = sm.GateSetTomography(basis, prep_state_function, POVM_models, parsed_circuits, ism_gate_dictionary, circuit_design = gst_circuit_design, 
                                    parameter_bounds = parameter_bounds, ideal_gate_set = ideal_gate_set, verbose = False)
    print(f"Num parameters: {GST_analyzer.num_parameters}")

    #parameter_guess = np.ones(GST_analyzer.num_parameters)*1e-2
    start = time.perf_counter()
    #solver_results = GST_analyzer.solve_for_gate_parameters(parameters_guess, 'linear') 
    solver_results = GST_analyzer.solve_for_gate_parameters(parameters_guess, 'MLE') 
    #solver_results = GST_analyzer.solve_for_gate_parameters(parameter_guess, 'staged MLE') 
    end = time.perf_counter()
    print(f"Ran GST in {end - start} seconds")
    print(f"Solver results: {solver_results}\n")
    print()
    GST_analyzer.print_parameters()
    GST_analyzer.print_state_and_POVMs()

    print()
    #gate_set_error = GST_analyzer.compute_gate_set_error(solver_results.x, ideal_gate_set, include_SPAM_error=include_SPAM_error) 
    gate_set_error = GST_analyzer.compute_gate_set_error_by_element(solver_results.x, ideal_gate_set)
    print(f"Gate set errors: {gate_set_error}")
    print(gate_set_error) 

    print(f"Estimating uncertainty\n")
    means, uncertainties, gate_set_errors, bootstrapped_thetas = GST_analyzer.estimate_parameter_uncertainties(solver_results.x, 'bootstrap') 
    print(f"Means: {means}\n\n")
    print(f"Uncertainties: {uncertainties}\n\n")
    #print(f"Bootstrapped: {bootstrap_thetas}\n")
    N_boot = bootstrapped_thetas.shape[0] 

    errors = np.zeros(N_boot)    
    std_devs = np.zeros(N_boot)    
    for i, err_dict in enumerate(gate_set_errors):
        errors[i] = err_dict[G] 

    error = np.mean(errors)
    std_dev = np.std(errors)
    return error, std_dev
    #return gate_set_error



if __name__ == '__main__':
    fname = "Ncounts_500.gstdata"
    parsed_circuits = sm.parse_gst_circuit_file(fname)
    N_circuits = len(parsed_circuits)
    start = 2
    errors = np.zeros(N_circuits - start)
    std_devs = np.zeros(N_circuits - start)
    for n in range(start, N_circuits):
        errors[n-start], std_devs[n-start] = run_GST("Ncounts_500.gstdata", n, True)

    plt.figure(figsize = (5,5))
    plt.plot(list(range(start, N_circuits)), errors, marker = 'o', linewidth = 1.5, markersize = 7, color = 'k', label=r'$X_{\pi/8}$')
    plt.title(r'Gate Set Error vs. Sample Size', fontsize = 14)
    plt.xlabel(r'Number of circuits', fontsize = 22)
    plt.ylabel(r'Gate estimation error', fontsize = 24, rotation = 0, labelpad = 25)
    #plt.xticks(fontsize = 12)
    plt.legend()
    #plt.xscale('log')
    plt.yscale('log')
    plt.savefig('N_circuits_gate_error.pdf', dpi=300)
    plt.show()
    
    plt.figure(figsize = (5,5))
    plt.plot(list(range(start, N_circuits)), std_devs, marker = 'o', linewidth = 1.5, markersize = 7, color = 'k', label=r'$X_{\pi/8}$')
    plt.title(r'Gate Set Error vs. Sample Size', fontsize = 14)
    plt.xlabel(r'Number of circuits', fontsize = 22)
    plt.ylabel(r'$\sigma$', fontsize = 24, rotation = 0, labelpad = 20)
    #plt.xticks(fontsize = 12)
    plt.legend()
    #plt.xscale('log')
    plt.yscale('log')
    plt.savefig('N_circuits_gate_error.pdf', dpi=300)
    plt.show()



