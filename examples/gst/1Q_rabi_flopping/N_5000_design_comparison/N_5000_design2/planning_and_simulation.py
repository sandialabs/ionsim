from pathlib import Path
import numpy as np
import h5py
import sys
import time
from typing import Callable
import os
import ionsim as ism
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['text.usetex']=True 
style_path_data = '~/plot_style_data.txt'

""" Example script for running GST circuit planner, reading from it, running simulations to ``simulate'' the experiments based on its instructions, 
        and providing "measurement" info for the GST analysis.  """ 

# example: from titus_gate_simulations import noisy_X_pi2
from gate_models import * 

def main():

    # 1. Given the gate set, run the GST circuit planner if it has not been ran yet.  
    gate_names = ['Gxpi8:0'] 
    gates = [] 
    qubit_indices = [0] # index of each qubit  

    for name in gate_names:
        gates.append(ism.ParsedGate.from_string(name))

    Gxpi8_q0 = gates[0]
    #evaluated_gate_models = { Gxpi8_q0 : X_pi_8_co_prop(spin_flip_rate)} 
    # 3. Specify the relationship between GST gate names (e.g. "Gxpi2") and the gate simulation/process matrix model name (e.g. "run_noisy_Xpi2_simulation()")
    rabi_rate = 100e3 * 2*np.pi # rad./s
    pi_time = abs(np.pi)/rabi_rate
    rabi_duration = 16 * pi_time
    spin_flip_rate = 1 / rabi_duration 
    _gate_models = { Gxpi8_q0 : X_pi_8_co_prop} 
    evaluated_gate_models = { Gxpi8_q0 : X_pi_8_co_prop(spin_flip_rate)} 
    print(gates)

    num_qubits = len(qubit_indices)
    gst_circuit_filename = './circuit_planner_example.gstdata' # or .circuitplannerdata
    circuit_design_file = "./circuit_design.yml"
    if Path(circuit_design_file).exists():
        print(f"GST Circuit plan already exists.")
        gst_circuit_planner = ism.GSTCircuitPlanner.load_design(circuit_design_file)
    else:
        print(f"Writing GST Circuit plan.")
        #powers = [1]
        powers = [1, 2, 4, 8, 16, 32]
        #powers = [1, 2, 4, 8, 16, 24, 32, 64, 128]
        #powers = list(range(1, 32+1))
        powers = list(range(1, 32+1))
        #germs = [[Gxpi2_q0], [Gxpi2_q1], [Gypi2_q0], [Gypi2_q1], [cnot_gate], [Gxpi2_q1, cnot_gate,Gxpi2_q0], [Gypi2_q1, cnot_gate, cnot_gate, Gypi2_q0]] 
        #gate_models = {"Gxpi8" : }
        #gate_models = {Gxpi8_q0 : X_pi_8_co_prop} # from the gate_models module 

        fiducials = [[]]
        germs = [[Gxpi8_q0]] 
        gst_circuit_planner = ism.GSTCircuitPlanner(gate_names, qubit_indices, prep_fiducials = fiducials, measure_fiducials = fiducials, germ_powers = powers, germs = germs, gate_models = _gate_models) 
        gst_circuit_planner.write_circuit_plan(gst_circuit_filename, num_qubits) # writes gst circuits to a file  

        design_file = './circuit_design.yml'
        gst_circuit_planner.write_circuit_design(design_file)

    # 2. Using the GST circuit list from a file, read those circuits in.  
    gst_circuits = ism.parse_gst_circuit_file(gst_circuit_filename)


    # 4. Loop over all circuits in the plan and run the corresponding simulations, recording circuit outcomes  
    outcomes = []
    circuit_simulation_output_file = 'simulated_gst_experimental_data.gstdata' # the file you would like to write results to 

    # For the IonSim simulations, set up the 1-qubit (1Q) basis and initial state.  
    spins = [ism.AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0']) for _ in range(num_qubits)]
    basis = ism.StandardBasis([*spins])

    # Construct initial state 
    #rho_0 = ism.State.from_coefficients(basis, np.array([1., 0., 0., 0.])) 
    rho_0 = ism.State.from_coefficients(basis, np.array([1., 0.,])) 
    outcome_labels = ['0', '1']
    outcome_operator0 = ism.EnergyShiftOperator.from_matrix(basis, ism.Pauli.projector_0)
    outcome_operator1 = ism.EnergyShiftOperator.from_matrix(basis, ism.Pauli.projector_1)
    outcome_operators = [outcome_operator0, outcome_operator1]

    #outcome_labels = ['00', '01', '10', '11'] 

    # Simulate each circuit's dynamics on the initial state and ``simulate'' the outcome  
    N_counts = np.array([5000]) 

    #N_counts = np.array([10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 300000, 500000, 1000000])
    prefix = 'Ncounts_'
    postfix = '.gstdata'

    for N in N_counts:
        # Set up the file containing circuit measurement outcomes 
        circuit_outcome_file = prefix + str(N) + postfix 
        gst_circuit_planner.create_circuit_outcomes_file(circuit_outcome_file)
    
        # Option to consider a subset of the circuits: 
        num_circuits_to_simulate = len(gst_circuits) 
        print(f"\n Generating circuit outcomes for N shots = {N}")
        for circuit in gst_circuits:
            #print(f"Running circuit: {circuit}")
            # Reinitialize the state: 
            rho = rho_0 # cp 
    
            # For each gate in the simulator, evolve the state forward according to the gate dynamics         
            for gate in circuit.expanded_gates:
                # Run IonSim simulation of the gate 
                rho = rho.propagate_using_process_matrix(evaluated_gate_models[gate])
    
            # Estimate and record circuit outcomes in a dictionary to create ParsedCircuit object: 
            outcome_probabilities = rho.compute_basis_state_probabilities() 
            N_shots = N 
            estimated_outcome_counts = np.random.multinomial(N_shots, [*outcome_probabilities])
            
            outcome_info = {}
            for label, counts in zip(outcome_labels, estimated_outcome_counts):
                outcome_info[label] = counts
    
            # Update the circuit's attribute directly with the "measurement" outcome information as a CircuitData object  
            circuit_data = ism.CircuitData.from_counts(outcome_info)
            circuit.measurement_data = circuit_data
    
            # e.g. if there's a simulation error (e.g. numerical divergence) on circuit 200's X_pi gate, we should not need to redo the previous 199 circuit simulations. 
            circuit.append_to_file(circuit_outcome_file) 

    circuit_parameters = {"X_pi_8_co_prop__spin_flip_rate" : 12000.}
        #circuit_parameters = {'R__phi' : 0., 'R__theta' : np.pi/2}
    sensitivities = gst_circuit_planner.compute_circuit_sensitivities(gst_circuits[1:], circuit_parameters, rho_0, outcome_operators)
    #for circ, sensitivity in zip(gst_circuits, sensitivities):
    for i, (circ, sensitivity) in enumerate(sensitivities.items()):
        print(f"Sensitivity of circuit number: {i+1}")
        print(f"{sensitivity}\n")

    circuit_indices = np.array(list(range(1, len(sensitivities)+1)))
    #fisher_info = np.zeros(len(circuit_indices)) 
    sensitivity_0 = np.zeros(len(circuit_indices)) 
    sensitivity_1 = np.zeros(len(circuit_indices)) 
    g_powers = np.zeros(len(circuit_indices)) 
    for i, s_circ in enumerate(list(sensitivities.values())):
        germ_power = gst_circuits[i+1].germ_power
        g_powers[i] = germ_power 
        for parameter, sensitivity_vector in s_circ.items():
            #fisher_info[i] = sensitivity_vector
            sensitivity_0[i] = sensitivity_vector[0] 
            sensitivity_1[i] = sensitivity_vector[1] 
            if np.abs(sensitivity_0[i]) < 1E-9:
                print(f"Remove circuit with germ power: {germ_power}")
        
    # TODO: GENERALIZE 
    S = np.array([list(s.values())[0] for s in sensitivities.values()])
    plt.style.use(style_path_data) 
    plt.figure(figsize=(4,4))
    #plt.plot(circuit_indices, S, 'o', label='Spin flip rate')
    plt.title(f"Sensitivity of circuit to Spin flip rate parameter")
    #plt.plot(circuit_indices, fisher_info, marker = 'o', linewidth = 1.5, color = 'r', label='Fisher info')
    plt.plot(circuit_indices, sensitivity_0, 'o', color = 'r', label='$p_0$')
    plt.plot(circuit_indices, sensitivity_1, 'x', color = 'b', label='$p_1$')
    #plt.plot(circuit_indices, 0.5*(np.abs(sensitivity_0) + np.abs(sensitivity_1)), marker = 'p', color = 'g', label='$|S|$')
    plt.xlabel(f'Circuit index', fontsize=16)
    plt.ylabel('Sensitivity', fontsize=16)
    plt.legend()
    #plt.savefig(data_directory / f'infidelity_vs_{dy_name}.pdf', bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    main()
