from pathlib import Path
import numpy as np
import h5py
import sys
import time
from matplotlib import pyplot as plt
from typing import Callable
import os
import ionsim as ism

""" Example script for running GST circuit planner, reading from it, running simulations to ``simulate'' the experiments based on its instructions, 
        and providing "measurement" info for the GST analysis.  """ 

# example: from titus_gate_simulations import noisy_X_pi2
from gate_models import X_pi_2_state_propagator, Y_pi_2_state_propagator, idle_state_propagator 

def main():

    # 1. Given the gate set, run the GST circuit planner if it has not been ran yet.  
    gate_names = ['Gxpi2', 'Gypi2', 'idle']
    qubit_indices = [0] # index of each qubit  
    num_qubits = len(qubit_indices)

    gst_circuit_filename = './circuit_planner_example.gstdata' # or .circuitplannerdata
    if Path(gst_circuit_filename).exists():
        print(f"GST Circuit plan already exists.")
    else:
        print(f"Writing GST Circuit plan.")
        gst_circuit_planner = ism.GSTCircuitPlanner(gate_names, qubit_indices)
        gst_circuit_planner.write_circuit_plan(gst_circuit_filename, num_qubits) # writes gst circuits to a file  

    #sys.exit(0)

    # 2. Using the GST circuit list from a file, read those circuits in.  
    #gst_circuit_filename = 'circuit_planner_example.gstdata'
    gst_circuits = ism.parse_gst_circuit_file(gst_circuit_filename)

    # 3. Specify the relationship between GST gate names (e.g. "Gxpi2") and your simulation name (e.g. "run_noisy_Xpi2_simulation()")
    ## The function should at minimum take in a state and return a state.  
    #gate_mappings = { 'Gxpi2' : noisy_X_pi2, 'Gypi2' : noisy_Y_pi2, 'idle' : idle_gate} # these could be python modules
    gate_mappings = { 'Gxpi2' : X_pi_2_state_propagator, 'Gypi2' : Y_pi_2_state_propagator, 'idle' : idle_state_propagator} # these could be python modules

    # 4. Loop over all circuits in the plan and run the corresponding simulations, recording circuit outcomes  
    outcomes = []
    circuit_simulation_output_file = 'simulated_gst_experimental_data.gstdata' # the file you would like to write results to 

    # For the IonSim simulations, set up the 1-qubit (1Q) basis and initial state.  
    spins = [ism.AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])]
    basis = ism.StandardBasis([*spins])

    # Construct initial state 
    rho_0 = ism.State.from_coefficients(basis, np.array([1., 0.])) # ket{0} state 

    outcome_labels = ['0', '1'] # for 1Q gates 

    # Simulate each circuit's dynamics on the initial state and ``simulate'' the outcome  
    parsed_circuits = []

    # Set up the file containing circuit measurement outcomes 
    ism.GSTCircuitPlanner.create_circuit_outcomes_file(circuit_simulation_output_file, 1)

    # Option to consider a subset of the circuits: 
    num_circuits_to_simulate = 50 
    gst_circuits = gst_circuits[:num_circuits_to_simulate]
    for circuit in gst_circuits:
        print(f"Running circuit: {circuit}")
        # Reinitialize the state: 
        rho = rho_0 # cp 

        # For each gate in the simulator, evolve the state forward according to the gate dynamics         
        for gate in circuit.expanded_gates:
            gate_simulator = gate_mappings[gate.name]

            # Run IonSim simulation of the gate 
            rho = gate_simulator(rho)   # can be noisy or deterministic. If noisy, this is a trajectory-averaged state 

        # Estimate and record circuit outcomes in a dictionary to create ParsedCircuit object: 
        outcome_probabilities = rho.compute_basis_state_probabilities() 
        N_shots = 200    # I think this is just chosen by the experiment, rather than being related to the number of trajectories in a simulation of one gate or circuit. 
        estimated_outcome_counts = np.random.multinomial(N_shots, [outcome_probabilities[0], outcome_probabilities[1]]) 
        
        outcome_info = {}
        for label, counts in zip(outcome_labels, estimated_outcome_counts):
            outcome_info[label] = counts

        # Update the circuit's attribute directly with the "measurement" outcome information as a CircuitData object  
        circuit_data = ism.CircuitData.from_counts(outcome_info)
        circuit.measurement_data = circuit_data

        # e.g. if there's a simulation error (e.g. numerical divergence) on circuit 200's X_pi gate, we should not need to redo the previous 199 circuit simulations. 
        circuit.append_to_file(circuit_simulation_output_file) 
        #sys.exit(0)

    # 5. Or, write the parsed circuit info back to a file or pickle it & output it. The parsed_circuits list would be fed into GST analysis.  
    # Test write_all method:
    #ism.GSTCircuitPlanner.write_all_circuit_outcomes(circuit_simulation_output_file, gst_circuits, 1)
    #sys.exit(0)


if __name__ == '__main__':
    main()
