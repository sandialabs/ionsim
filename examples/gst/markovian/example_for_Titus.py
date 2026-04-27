from pathlib import Path
import numpy as np
import h5py
import sys
import time
from matplotlib import pyplot as plt
from typing import Callable

import ionsim as ism

""" Example script for running GST circuit planner, reading from it, running simulations to ``simulate'' the experiments based on its instructions, 
        and providing "measurement" info for the GST analysis.  """ 


def main():

    # 1. Given the gate set, run the GST circuit planner if it has not been ran yet.  
    #gate_names = ['idle', 'Gxpi2', 'Gypi2']
    gate_names = ['Gxpi2', 'Gypi2', 'idle']
    qubit_indices = [0] # index of each qubit  
    num_qubits = len(qubit_indices)

    gst_circuit_filename = 'circuit_planner_example.gstdata'
    gst_circuit_planner = ism.GSTCircuitPlanner(gate_names, qubit_indices)
    gst_circuit_planner.write_circuit_plan(gst_circuit_filename, num_qubits) # writes gst circuits to a file  

    sys.exit(0)

    # 2. Using the GST circuit list from a file, read those circuits in.  
    #gst_circuit_filename = 'circuit_planner_example.gstdata'
    gst_circuits = ism.parse_gst_circuit_file(gst_circuit_filename)

    # 3. Specify the relationship between GST gate names (e.g. "Gxpi2") and your simulation name (e.g. "run_noisy_Xpi2_simulation()")
    gate_mappings = { 'Gxpi2' : noisy_X_pi2, 'Gypi2' : noisy_Y_pi2, 'Idle' : idle_gate} # these could be python modules

    # 4. Loop over all circuits in the plan and run the corresponding simulations, recording circuit outcomes  
    outcomes = []
    circuit_simulation_output_file = 'simulated_gst_experimental_data.gstdata'

    # For the IonSim simulations, set up the 1-qubit (1Q) basis and initial state.  
    spins = [sm.AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])]
    basis = sm.StandardBasis([*spins])

    # Construct initial state 
    rho_0 = ism.State.from_coefficients(basis, np.array([1., 0.])) # ket{0} state 

    outcome_labels = ['0', '1'] # for 1Q gates 

    # Simulate each circuit's dynamics on the initial state and ``simulate'' the outcome  
    parsed_circuits = []
    for circuit in gst_circuits:
        # Reinitialize the state: 
        rho = rho_0.copy() 
        N_shots = 200 # I think this is just chosen by the experiment, rather than being related to the number of trajectories in a simulation of one gate or circuit. 

        # For each gate in the simulator, evolve the state forward according to the gate dynamics         
        for gate in circuit:
            gate_simulator = gate_mappings[gate.name]

            # Run IonSim simulation of the gate 
            rho = gate_simulator(rho)   # can be noisy or deterministic. If noisy, this is a trajectory-averaged state 

        # Estimate and record circuit outcomes in a dictionary to create ParsedCircuit object: 
        outcome_probabilities = rho.compute_basis_state_probabilities() 
        estimated_outcome_counts = np.random.multinomial(N_shots, [outcome_probabilities[0], outcome_probabilities[1]]) 
        
        outcome_info = {}
        for label, counts in zip(outcome_labels, estimated_outcome_counts):
            outcome_info[label] = counts

        # Create CircuitData object and update the circuit attribute directly 
        circuit_data = ism.CircuitData.from_counts(outcome_info)
        circuit.measurement_data = circuit_data


        # Option to overrwrite the original file at each line so that this information can be written?
        # e.g. if there's a simulation error (e.g. numerical divergence) on circuit 200's X_pi gate, we should not need to redo teh previous 199 circuit simulations. 


    # 5. Write the parsed circuit info back to a file or pickle it & output it 
    sys.exit(0)


if __name__ == '__main__':
    main()
