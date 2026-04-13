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
    
            X_pi2 = exp( -i [ (pi/2 + X_rot) X  - (Z_rot)Z ] )
    
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
    
            Y_pi2 = exp( -i [ (pi/2 + Y_rot) Y  - (Z_rot)Z ] )
    
            - Y_rot is an additional Y_rotation parameter (over/under rotation).
            - Z_rot is a Z_rotation parameter, e.g. from a detuned laser. 
    
        """  
        # TODO: generalize to 2+ qubits  
        assert len(spins) == 1
        y_angle = np.pi/2. + Y_rot
        Rypi2 = sm.Unitary.R_bloch([0., y_angle/2., Z_rot/2.]) 
    
        # Promote to a d^2 x d^2 superoperator 
        return basis.compute_superoperator_from_unitary_operator(Rypi2)


    def null():
        """ Returns d^2 x d^2 process matrix in standard basis a null ("do nothing for no time") gate"""  
        # Build identity matrix with Z rotation by theta:
        # TODO: generalize to 2+ qubits  
        assert len(spins) == 1
        I = np.eye(2,dtype=complex)
        # Promote to a d^2 x d^2 superoperator 
        return basis.compute_superoperator_from_unitary_operator(I)


    # Define dictionary mappings for GST gate name to ionsim gate function 
    ism_gate_dictionary = {}    
    ism_gate_dictionary['Gxpi2']  = X_pi2 
    ism_gate_dictionary['Gypi2'] = Y_pi2
    ism_gate_dictionary['idle'] = idle
    ism_gate_dictionary['{}'] = null 

    # TODO 's: 
        # Have user specify prep and measure parametrizations (models)  

    # For GST, define state and measurement parametrizations (models): 
    # Here, we choose deviations from an ideal prep state and ideal POVM effects: 
    rho_prep = sm.State.from_coefficients(basis, list([1., 0.]))

    ## Define a parametrization (model) for prep state as a function: 
    def prep_state_function(state_parameters: Vector) -> 
        """ Model of the prep state as a function of parameters """ 
        # Here we choose deviation from the ideal prep state 
        # What should we return? Flattened density matrix? 
        return rho_prep.supervector + state_parameters 


    ideal_POVM_effects = {} 
    ideal_POVM_effects['0'] = sm.EnergyShiftOperator.from_matrix(basis, sm.Pauli.projector_0) 
    ideal_POVM_effects['1'] = sm.EnergyShiftOperator.from_matrix(basis, sm.Pauli.projector_1) 
        
    def measurement_effect_function(POVM_parameters: dict[str, Vector]) -> 
        """ Model of the prep state as a function of parameters """ 
        # Here we choose deviation from the ideal prep state 
        # What should we return? Flattened density matrix? 
        POVM_models = {}
        for outcome, effect in ideal_POVM_effects.items():
            POVM_models[outcome] = effect.superbra 

        return POVM_models 

    #GST_analyzer = sm.GateSetTomography(basis, rho_prep, POVM_effects, parsed_circuits, gate_factory_function)
    #GST_analyzer = sm.GateSetTomography(basis, rho_prep, POVM_effects, parsed_circuits, ism_gate_dictionary)
    GST_analyzer = sm.GateSetTomography(basis, prep_state_function, POVM_effects, parsed_circuits, ism_gate_dictionary)
    solver_results = GST_analyzer.solve_for_gate_parameters()
    print(f"Solver results: {solver_results}")
    GST_analyzer.print_parameters()
    GST_analyzer.print_state_and_POVMs()
    sys.exit(0)

    # TODO: Either save gates evaluated at the parameter values or just the parameter values.  



if __name__ == '__main__':
    main()
