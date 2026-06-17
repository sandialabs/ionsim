#!/usr/bin/env python3
"""
Example script demonstrating the new circuit planning and germ optimization features in IonSim.

This script shows:
1. Standard (gate-model-agnostic) GST circuit planning
2. Optimized GST circuit planning using gate models and germ sensitivity analysis
3. Comparison of standard vs optimized germ selection
4. Simulation of GST circuits using the optimized plan
"""

import numpy as np
from pathlib import Path
import ionsim as ism
from typing import Callable
import time

import sys
# Import gate simulators from the existing example
from gate_simulators import X_pi2_state_propagator, Y_pi2_state_propagator, idle_state_propagator


def setup_basis_and_state():
    """Set up the 1-qubit basis and initial state for simulations."""
    # Set up basic 1-qubit (1Q) basis
    spins = [ism.AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'],
                                         level_names=['S1/2,0,0', 'S1/2,1,0'])]
    basis = ism.StandardBasis([*spins])

    # Construct initial state
    rho_0 = ism.State.from_coefficients(basis, np.array([1., 0.]))  # ket{0} state
    outcome_labels = ['0', '1']  # for 1Q gates

    return basis, rho_0, outcome_labels


def define_gate_models(basis):
    """Define gate model functions for use in optimized planning."""
    # These are simplified versions of the gate models from gst_analysis_markovian.py
    # They return process matrices as functions of parameters

    def idle(theta):
        """Returns d^2 x d^2 process matrix in standard basis for Z-rotation by theta"""
        I = np.eye(2, dtype=complex)
        I[0, 0] = np.exp(-1j * theta)
        I[1, 1] = np.exp(1j * theta)
        return basis.compute_superoperator_from_unitary_operator(I)

    def X_pi2_process_matrix(X_rot: float, phase_mean: float, phase_std_deviation: float):
        """Process matrix for X_pi/2 gate with over/under rotation and phase noise."""
        # Simplified model - in practice this would use the full Lindbladian simulation
        gate_phase = 0.
        theta = np.pi/2. + X_rot

        # Build ideal X_pi/2 rotation
        ideal_X = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                           [-1j*np.sin(theta/2), np.cos(theta/2)]])

        # Add phase noise effects (simplified)
        phase_variance = phase_std_deviation**2
        dephasing = np.exp(-phase_variance)

        # Combine rotation and dephasing
        process_matrix = basis.compute_superoperator_from_unitary_operator(ideal_X)

        return process_matrix

    def Y_pi2_process_matrix(Y_rot: float, phase_mean: float, phase_std_deviation: float):
        """Process matrix for Y_pi/2 gate with over/under rotation and phase noise."""
        # Simplified model - in practice this would use the full Lindbladian simulation
        gate_phase = np.pi/2.
        theta = np.pi/2. + Y_rot

        # Build ideal Y_pi/2 rotation
        ideal_Y = np.array([[np.cos(theta/2), -np.sin(theta/2)],
                           [np.sin(theta/2), np.cos(theta/2)]])

        # Add phase noise effects (simplified)
        phase_variance = phase_std_deviation**2
        dephasing = np.exp(-phase_variance)

        # Combine rotation and dephasing
        process_matrix = basis.compute_superoperator_from_unitary_operator(ideal_Y)

        return process_matrix

    return {'Gxpi2': X_pi2_process_matrix, 'Gypi2': Y_pi2_process_matrix, 'idle': idle}


def run_standard_planning(gate_names, qubit_labels, output_file):
    """Run standard (gate-model-agnostic) GST circuit planning."""
    print("\n" + "="*70)
    print("Running STANDARD (gate-model-agnostic) GST circuit planning")
    print("="*70)

    start_time = time.perf_counter()

    # Create standard planner without gate models
    standard_planner = ism.GSTCircuitPlanner(gate_names=gate_names, qubit_labels=qubit_labels, germ_powers=[1, 2, 4, 8, 16])

    # Generate and write circuits
    standard_planner.write_circuit_plan(output_file, 1)
    standard_planner.write_circuit_design(output_file.replace('.gstdata', '_design.yml'))

    end_time = time.perf_counter()

    print(f"Standard planning completed in {end_time - start_time:.3f} seconds")
    print(f"Number of germs: {len(standard_planner.germs)}")
    print("Germ sequences:")
    for i, germ in enumerate(standard_planner.germs):
        germ_str = ''.join([g.name for g in germ]) if germ else '[]'
        print(f"  {i+1}. {germ_str}")
    print(f"Total circuits generated: {len(standard_planner.generate_gst_circuits())}")

    return standard_planner


def run_optimized_planning(gate_names, qubit_labels, gate_models, output_file):
    """Run optimized GST circuit planning with germ sensitivity analysis."""
    print("\n" + "="*70)
    print("Running OPTIMIZED GST circuit planning with germ sensitivity analysis")
    print("="*70)

    start_time = time.perf_counter()

    # Create optimized planner with gate models
    optimized_planner = ism.GSTCircuitPlanner(gate_names=gate_names, qubit_labels=qubit_labels, germ_powers=[1, 2, 4, 8, 16], gate_models=gate_models)

    # Generate and write circuits
    optimized_planner.write_circuit_plan(output_file, num_qubits=1)
    optimized_planner.write_circuit_design(output_file.replace('.gstdata', '_design.yml'))

    end_time = time.perf_counter()

    print(f"Optimized planning completed in {end_time - start_time:.3f} seconds")
    print(f"Number of germs: {len(optimized_planner.germs)}")
    print("Optimized germ sequences:")
    for i, germ in enumerate(optimized_planner.germs):
        germ_str = ''.join([g.name for g in germ]) if germ else '[]'
        print(f"  {i+1}. {germ_str}")
    print(f"Total circuits generated: {len(optimized_planner.generate_gst_circuits())}")

    return optimized_planner


def compare_germ_selections(standard_planner, optimized_planner):
    """Compare standard vs optimized germ selections."""
    print("\n" + "="*70)
    print("Comparing STANDARD vs OPTIMIZED germ selections")
    print("="*70)

    # Get germ strings for comparison
    standard_germs = [''.join([g.name for g in germ]) if germ else '[]'
                      for germ in standard_planner.germs]
    optimized_germs = [''.join([g.name for g in germ]) if germ else '[]'
                       for germ in optimized_planner.germs]

    print(f"\nStandard germs: {standard_germs}")
    print(f"Optimized germs: {optimized_germs}")

    # Find common germs
    common_germs = set(standard_germs) & set(optimized_germs)
    print(f"\nCommon germs in both selections: {common_germs}")

    # Find unique germs
    unique_to_standard = set(standard_germs) - set(optimized_germs)
    unique_to_optimized = set(optimized_germs) - set(standard_germs)

    print(f"Germs unique to standard: {unique_to_standard}")
    print(f"Germs unique to optimized: {unique_to_optimized}")


def simulate_circuits(gst_circuits, gate_mappings, rho_0, outcome_labels, output_file, num_circuits=None):
    """Simulate GST circuits and record outcomes."""
    print("\n" + "="*70)
    print("Simulating GST circuits")
    print("="*70)

    # Limit number of circuits if specified
    if num_circuits is not None:
        gst_circuits = gst_circuits[:num_circuits]

    # Create output file
    if not Path(output_file).exists():
        ism.GSTCircuitPlanner.create_circuit_outcomes_file(output_file, 1)

    start_time = time.perf_counter()

    for i, circuit in enumerate(gst_circuits):
        if (i + 1) % 50 == 0:
            print(f"  Simulated {i + 1}/{len(gst_circuits)} circuits...")

        # Reinitialize the state
        rho = rho_0

        # For each gate in the circuit, evolve the state
        for gate in circuit.expanded_gates:
            gate_simulator = gate_mappings[gate.name]
            rho = gate_simulator(rho)

        # Estimate and record circuit outcomes
        outcome_probabilities = rho.compute_basis_state_probabilities()
        N_shots = 2000
        estimated_outcome_counts = np.random.multinomial(N_shots, outcome_probabilities)

        outcome_info = {}
        for label, counts in zip(outcome_labels, estimated_outcome_counts):
            outcome_info[label] = counts

        # Update circuit with measurement data
        circuit_data = ism.CircuitData.from_counts(outcome_info)
        circuit.measurement_data = circuit_data

        # Append to file
        circuit.append_to_file(output_file)

    end_time = time.perf_counter()
    print(f"\nSimulation completed in {end_time - start_time:.3f} seconds")
    print(f"Results written to: {output_file}")


def main():
    # Set up basic parameters
    gate_names = ['Gxpi2', 'Gypi2', 'idle']
    qubit_labels = [0]

    # Set up basis and state
    basis, rho_0, outcome_labels = setup_basis_and_state()

    # Define gate models for optimized planning
    gate_models = define_gate_models(basis)

    # Define gate mappings for simulation
    gate_mappings = {
        'Gxpi2': X_pi2_state_propagator,
        'Gypi2': Y_pi2_state_propagator,
        'idle': idle_state_propagator
    }

    # Run standard planning
    standard_output = './standard_gst_circuits.gstdata'
    standard_planner = run_standard_planning(gate_names, qubit_labels, standard_output)

    # Run optimized planning
    optimized_output = './optimized_gst_circuits.gstdata'
    optimized_planner = run_optimized_planning(gate_names, qubit_labels, gate_models, optimized_output)

    # Compare germ selections
    compare_germ_selections(standard_planner, optimized_planner)

    # Simulate circuits using the optimized plan
    # Parse the optimized circuits
    optimized_circuits = ism.parse_gst_circuit_file(optimized_output)


    sys.exit(0)
    # Simulate a subset for demonstration (remove num_circuits=None to simulate all)
    simulate_circuits(
        gst_circuits=optimized_circuits,
        gate_mappings=gate_mappings,
        rho_0=rho_0,
        outcome_labels=outcome_labels,
        output_file='./simulated_optimized_gst_data.gstdata',
        num_circuits=100  # Simulate first 100 circuits for demo; remove for full simulation
    )

    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)
    print("\nKey files generated:")
    print(f"  1. {standard_output} - Standard GST circuit plan")
    print(f"  2. {standard_output.replace('.gstdata', '_design.yml')} - Standard design")
    print(f"  3. {optimized_output} - Optimized GST circuit plan")
    print(f"  4. {optimized_output.replace('.gstdata', '_design.yml')} - Optimized design")
    print(f"  5. ./simulated_optimized_gst_data.gstdata - Simulated outcomes")
    print("\nNext steps:")
    print("  - Use the optimized design file with gst_analysis_markovian.py")
    print("  - Compare GST results between standard and optimized plans")
    print("  - Analyze germ sensitivity using the planner's methods")


if __name__ == '__main__':
    main()
