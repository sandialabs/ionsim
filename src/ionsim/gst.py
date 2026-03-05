import numpy as np
import pandas as pd 


from ionsim.process import Gate, Circuit


class GateSetTomography() # or GST() or GST_Base() if we plan to have child classes.
    """ Class for performing quantum gate set tomography with trapped ions or neutral atoms. 

        Member variables include:
            - Basis where the quantum processes (gates), state, and measurement will live. 
            - initial state: rho_0, representing a state prepared natively. 
            - native measurement: M_0, representing a native measurement projector. 

            - parametrized_gates is a boolean. If True, the user should specify a model 
                for the gates as a function of the parameters. If False, the class assumes 
                the gate is modeled as a generic, dense process matrix.  
            
            - gate_parameters 


    """ 
    # Questions: 
        # Should we allow more than 1 initial native state and 1 native measurement? 
            # GST manual suggests that ususally only 1 is available. 
    
    # Initial Thoughts: 
    # - user constructs the class. 
    # - user either specifies models for gates upon construction or uses something like an "add_gate_model()" function 
    # - user solves for parameters post-construction 


    # member variables: 
    # - basis where the quantum state, process, and measurement lives. 
    # -  
    basis: Basis
    initial_state: State
    native_measurement: Operator # TODO: what object should this be? Do we need a "measurement" class?  

    # Either specify fiducial prep/measure circuits, or user must include in gate set/list. 
    gate_set: list[Gate]
    #fiducial_circuit_list: list[str] #e.g. ['g1', 'f1,g1,g2']
        # Each gate has a dictionary of parameters, but this could be none.  
        # include logic of 

    # I think we need a set of experimental data per circuit listed. 
    # - allow for variable number of shots per experiment. 

    parametrized_gates: bool = False 
    gate_parameters: list[NDArray] 
    system_size: int 

    frequencies_data: GST_Data

    def __post_init__(self):
        ''' Check whether the gate set is parametrized '''
        for gate in self.gate_set:
            if gate.parameters is None:
                self.parametrized_gates = False  

    def compute_outcome_probability(circuit: Circuit, outcome):
        """ Returns the probability of outcome "mu" for native measurement "M". """
        outcome_index = frequencies_data[0, :, 0].index(outcome)
        return frequencies_data.get_probability_of_outcome(outcome_index, circuit) 

    @classmethod
    def solve_for_parameters_of_all_gates(cls, solver: str)
        """ Loops over all gates to solve for gate parameters. """ 
        gate_parameters = [] 
        for gate in self.gate_set:
            gate_parameters.append(cls.solve_for_gate_parameters(gate, solver))

        return gate_parameters


    @classmethod
    def solve_for_gate_parameters(cls, gate: Gate, solver: str = 'MLE'):
        """ Function to solve for the parametrization values of a particular gate. 

            Default behavior is a maximum likelihood approach that maximizes
                the likelihood of the gate given the data.  

                max[ L( G | data) ] over parameter set Theta.
        """

        if solver == 'MLE':
            # Maximum likelihood estimation  
            
        elif solver == 'linear':
            # Solve matrix Ax = b problem: Frequencies = A_matrix @ Gate_parameters  
            # Check that gram matrix A_{m,s} = <M | C_{m} C_{s} | rho> is invertible.            

        else:
            raise IonSimError('Invalid solver input.')

        return parameters


class GST_Data():
    """ Class to organize and maintain measurement data for GST """
    # May not be necessary to have this as a separate class 

    # Should these use the same prep state and measurement basis? 
    N_qubits: int  
    N_shots: int    
    
    N_distinct_outcomes: int 
    frequency_data: DataFrame ## TODO: Decide on data structure 
    # Will the data be already shot-averaged?  
    # N_shots x outcome x circuit 


    # TODO: Set up a data structure (e.g. pandas data frame) to maintain data: 
        # Need to access  

    def __post_init__(self):
        """ Check whether the gate set is parametrized """
        assert N_distinct_outcomes == (2**N_qubits)


    def get_probability_of_outcome(outcome_index: int, circuit_name: str): 
        """ Returns probability (shot-averaged frequencies) of outcome "m" in circuit C """
        probability = np.mean(self.frequency_data[outcome_index, circuit_name])
        return probability 



# -- To be deleted later --
''' Example usage: ''' 

my_basis = StandardBasis([spins])
my_gate_set = [Xpi_gate, Xpi_2_gate, idle_gate]
coeffs = np.zeros(2)
coeffs[0] = 1.
rho_0 = State.from_coefficients(coeffs)

single_qubit_GST = GateSetTomography(basis = my_basis, initial_state = rho_0, native_measurement = M, gate_set = my_gate_set) 
gate_parameters = single_qubit_GST.solve_for_all_gate_parameters(solver='MLE')


#model_coeffs = 



# Could build a N-dimensional process matrix by running simulations for each of the N-dimensional parameters, store them. --> hdf5 files
    # - save the raw hdf5 simulation data (process matrix at each error parameter value) 
    # - GST will load the process matrix data, interpolate with it using MLE. cref. one of the examples  

        # - would need to load up one of these for each gate in the gate set

    # Separate data file for X_pi, X_pi/2 , etc. 
        # - each data structure (hdf5 -- file system, e.g. containing folders) contains simulations over many values of the error parameters  


    # TODO: Write function to save parameters / other info to hdf5 file

    # - GST could make calls to other parts of IonSim to run needed simulations. For constructing the required process matrices to do GST. 
        # - if you gave it a gate set, the class could then do those simulations to generate the process matrices needed for GST.   
