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
def run_GST(fname: str, include_SPAM_error: bool=False):
    # 1. Import GST sequence data 
    # Run the main parsing function:  
    parsed_circuits = sm.parse_gst_circuit_file(fname)
    parsed_circuits = parsed_circuits

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
    ism_gate_dictionary['idle'] = noisy_idle
    ism_gate_dictionary['Gxpi2:0'] = X_pi_2_co_prop 

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
    energy_shift = 2e3 * 2*np.pi # 2 kHz -> rad/s
    #co_prop_spin_dephasing_rate = 0
    spin_dephasing_rate = 1 / (2*83.3e-3) 
    ideal_gate_set['Gxpi2:0'] = X_pi_2_co_prop(spin_flip_rate)
    ideal_gate_set['idle'] = noisy_idle(energy_shift, spin_dephasing_rate)

    design_fname = 'circuit_design.yml'
    gst_circuit_design = sm.GSTCircuitPlanner.load_design(design_fname)
    #parameters_guess = {
    #    "prep" : {"q1_probability_of_wrong_prep" : 0.001, "q2_probability_of_wrong_prep" : 0.001},
    #    "POVM" : {"prob_false_bright" : 0.001, "prob_false_dark" : 0.001},
    #    "Gxpi8" : {"spin_flip_rate" : 100}
    #}
    #parameters_guess = {"Gxpi2:0" : {"spin_flip_rate" : 20000}, "idle" : {"energy_shift" : 5e3*2.*np.pi, "spin_dephasing_rate" : 3.}}
    parameters_guess = {"Gxpi2:0" : {"spin_flip_rate" : 1.}, "idle" : {"energy_shift" : 5e3*2.*np.pi, "spin_dephasing_rate" : 3.}}
    # TODO: add parameters guess for prep and measure 

    G = sm.ParsedGate.from_string("Gxpi2:0")
    idle_G = sm.ParsedGate("idle", ())
    parameter_bounds = {
        "prep" : {"q1_probability_of_wrong_prep" : (0., 1.), "q2_probability_of_wrong_prep" : (0., 1.)},
        "POVM" : {"prob_false_bright" : (0., 1.), "prob_false_dark" : (0., 1.)},
        G : {"spin_flip_rate" : (0, None)},
        idle_G : {"energy_shift" : (0., 2.*np.pi*100E3), "spin_dephasing_rate" : (0, None)}
        #"Gxpi8" : {"spin_flip_rate" : (0, 200000)}
    } 
    GST_analyzer = sm.GateSetTomography(basis, prep_state_function, POVM_models, parsed_circuits, ism_gate_dictionary, circuit_design = gst_circuit_design, 
                                    parameter_bounds = parameter_bounds, ideal_gate_set = ideal_gate_set, verbose = True)
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
    gate_set_error = GST_analyzer.compute_gate_set_error(solver_results.x, ideal_gate_set, include_SPAM_error=include_SPAM_error) 
    #gate_set_error = GST_analyzer.compute_gate_set_error_by_element(solver_results.x, ideal_gate_set, include_SPAM_error=include_SPAM_error)
    print(gate_set_error) 
    return gate_set_error



if __name__ == '__main__':
    error = run_GST("Ncounts_5000.gstdata", True)
