import numpy as np
import pandas as pd 
from pathlib import Path

import scipy.stats as stats 
import scipy.optimize as opt 

from ionsim.process import Gate, Circuit
from ionsim.named_operators import Pauli


# Example Workflow: 
# 1. GST_Data Class creation: User creates GST data class from experimental outcome data for various circuits. 
# 2. GST Class creation: User specifies gate set, prep state, measurement operator, and GST measurement data
# 3. User wants gate parameters: GST_Class.solve_for_gate_parameters() is the key method to do so. 
#   3a) gets circuits ran in experiment from the GST_Data object, converts this into IonSim circuit objects 
#   3b) gets frequency value for each experiment and each outcome for that experiment 
#   3c) Do gradient descent / loss minimization procedure: 
    #   - estimate frequency value for each experiment's circuit (and for each possible outcome too) from model/parametrization. 
    #   - compute loss, cost, or likelihood, function. Or do linear inversion.
    #   - evaluate gradient (e.g. difference in exp vs. gate model), take a step in parameter space. 
    #   - repeat until minimizing loss to a tolerance.  
# 4. Return gate model parameters for each gate in the gate set. 


class GateSetTomography() # or GST() or GST_Base() if we plan to have child classes.
    """ Class for performing quantum gate set tomography (GST) with trapped ions or neutral atoms. 

        Member variables include:
            - Basis where the quantum processes (gates), state, and measurement will live. 
            - initial state: rho_0, representing a state prepared natively. 
            - native measurement: M_0, representing a native measurement projector. 
            - gate set: list of Gates, which should include fidicual prep/measurement circuits


        ### How to handle germs for long-form GST?? Does the user specify a germ sequence? Does a user specify "p" the power?  
            -  does the class just iterate over powers or solve at one fixed power? 

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

    basis: Basis
    initial_state: State
    native_measurement: Operator 
    gate_set: dict[str, Gate]    # instead of (list[Gate])

    # experimental data per circuit listed. 
    # - allow for variable number of shots per experiment. 
    gst_experiments_data: GST_Data



    #fiducial_circuit_list: list[str] #e.g. ['g1', 'f1,g1,g2']
        # Each gate has a dictionary of parameters, but this could be none.  
        # include logic of 

    parametrized_gates: bool = False 
    gate_parameters: list[NDArray] 
    system_size: int 


    def __post_init__(self):
        """ Safety checks: 
            1. Gates need to be parametrized.  
            2. Prep and Measurement needs to match between GST data and GST classes.  
        """ 
        for gate in self.gate_set:
            if gate.parameters is None:
                self.parametrized_gates = False  

        # TODO: we need to be able to check for state and operator equality. These compare methods don't exist yet. 
        if not compare_state(gst_experiments_data.prep_state, self.initial_state):
            raise IonSimError("GST Data Class and GST Class should use the same prep states.")

        if not compare_operator(gst_experiments_data.measurement, self.native_measurement):
            raise IonSimError("GST Data Class and GST Class should use the same measurement operator.")


    @classmethod
    def from_GST_Data(cls, gst_data: GST_Data)



        return cls( )


    def compute_circuit_outcome_probability(circuit: Circuit, outcome: int):
        """ Returns the probability of outcome "mu" for native measurement "M". """
        # e.g. outcome = +1 or -1 for single qubit Z measurement 
        outcome_index =  GST_experiments_data[0, :, 0].index(outcome)
        return gst_data.get_experimental_outcome_frequency(outcome_index, circuit) 


    ## I think you need to solve for all parameters simultaneously --> this is the self-consistency of GST  
    ## - evaluate cost function (negative likelihood), which is a sum over all possible measurement outcomes and all parameters 
 #    @classmethod
 #    def solve_for_parameters_of_all_gates(cls, solver: str)
 #        """ Loops over all gates to solve for gate parameters. """ 
 #        gate_parameters = [] 
 #        for gate in self.gate_set:
 #            gate_parameters.append(cls.solve_for_gate_parameters(gate, solver))
 #
 #        return gate_parameters
 #

    def solve_for_gate_parameters(solver: str = 'MLE'):
        """ Function to solve for the parametrization values of a particular gate. 

            - Default behavior is a maximum likelihood approach that finds parameters 
                that maximize the likelihood of the gate given the data, i.e. solving: 

                max[ Likelihood( {G} | data) ] over parameter set Theta.

            - Returns either a dictionary of parameters (name, value) or a 1D array of values.

        """
        # 3. User wants gate parameters: GST_Class.solve_for_gate_parameters() is the key method to do so. 
        #   3a) gets circuits ran in experiment from the GST_Data object, converts this into IonSim circuit objects 
        #   3b) gets frequency value for each experiment and each outcome for that experiment 
        #   3c) Do gradient descent / loss minimization procedure: 
            #   - estimate frequency value for each experiment's circuit (and for each possible outcome too) from model/parametrization. 
            #   - compute loss, cost, or likelihood, function. Or do linear inversion.
            #   - evaluate gradient (e.g. difference in exp vs. gate model), take a step in parameter space. 
            #   - repeat until minimizing loss to a tolerance.  
        # 4. Return gate model parameters for each gate in the gate set  


        # 1. Extract circuits from experiment 
        # Loop over all circuits in the experiment and create IonSim circuits 
        gst_circuits = []
        for circuit_name in GST_data.gst_data_frame['circuit_names']: 
            gst_circuits.append( circuit_from_circuit_name(circuit_name) ) 


        # 2. Extract frequency values for each experiment 
        experimental_frequencies = gst_data.get_frequencies() # 2D array of shape circuit x outcomes -> frequency values 

        assert experimental_frequencies.shape[0] == len(gst_circuits) 

        # TODO: Need to get extract parameters from the gate set somehow  
        # TODO: Need to figure out how to get gate parameters and use them here, from IonSim's Gate objects. 
        # - Compute expected probability of an outcome given a circuit's parametrization. 


        if solver == 'MLE':
            # Maximum likelihood estimation.
            # Specify initial guess. 
            theta_0 = np.zeros_like(gate_parameters)  
            #solver_result = opt.minimize(negative_log_likelihood, theta_0, method = 'L-BFGS-B', bounds = parameter_bounds)
            solver_result = opt.minimize(negative_log_likelihood, theta_0, method = 'L-BFGS-B') 

            
        elif solver == 'linear':
            # Solve matrix Ax = b problem: Frequencies = A_matrix @ Gate_parameters  
            # Check that gram matrix A_{m,s} = <M | C_{m} C_{s} | rho> is invertible.            
            # Compute x = A \ b

        else:
            raise IonSimError('Invalid solver input.')

        return gate_set_parameters 


    def circuit_from_circuit_name(circuit_name: str, noise: Noise) -> Circuit:
        """ Function that returns a circuit object (list of gates) corresponding to the circuit name. 
            - Helps to convert experiment circuit names into IonSim Gate/Circuit objects from the gate set. 
            - e.g. circuit_name = 'idle, X_pi_2' -> gate_list = [idle_gate, X_pi_2_gate] -> Circuit 
        """
        # TODO: Need to decide and state convention for gate ordering: e.g. Left to right: state prep --> gates --> measurement
            # need a look up table for the gates based on name 
        
        # Extract name of each gate from the circuit name  
        gate_names = [name.strip() for name in circuit_names.split(',')]

        # TODO: Add functionality to parse the qubit number when we have 2+ qubits  

        gate_list = []        
        for gate_name in gate_names:
            # Optional: map_experimental_gate_name_to_internal_gate_name(gate_name)
            gate_list.append(self.gate_set[gate_name])
            #if gate_name == 'I' or 'idle':

        return Circuit.from_gates(gate_list, noise) 



    def map_experimental_gate_name_to_internal_gate_name(gate_name: str): -> str
        """ Helper function to convert experimental gate nomenclature to internal
                IonSim gate nomenclature used in this class. """
        # TODO: Maybe the user will specify this mapping / look-up table? 

        return gate_name



class GST_Data():
    """ Class to maintain measurement data for GST and provide data retrieval functionality. """
    # Should these use the same prep state and measurement basis? 
    N_qubits: int  

    # Prep state and measurement may need to become circuit dependent for full generality for 2+ qubits 
    prep_state: State
    measurement: Operator 

    # Can organize data either as a single data structure 
    #   or have each member variable as a different data structure.
    gst_data_frame: pd.DataFrame ## TODO: Decide on data structure 
    # N_shots x outcome x circuit 
    N_possible_outcomes: int


    # TODO: Set up a data structure (e.g. tensor or pandas data frame) to maintain data: 
    def __post_init__(self):
        """ Check whether the gate set is parametrized & operator and state bases match. """
        assert self.N_possible_outcomes == (2**self.N_qubits) # d = 2^N outcomes 

        # TODO need to check that basis equality checking works  
        if prep_state.basis != measurement_operator.basis:
            raise IonSimError("State and Measurement bases must be the same.")





    @classmethod
    def from_gst_sequences(cls, file_string: str, N_qubits: int, prep_state: State, measurement_operator: Operator, time_dependent: bool): 
        """ Helper method for importing gst data from a file of GST circuit sequences with file extension .gstdata, often produced by PyGSTi.

            file_string denotes the file name and location, e.g. "./my_datafile.gst" 

            Organize measurement data into a data frame (df) with arguments: 
            df['circuit_names'] 
            df['gate_start_times'] 
            df['gate_end_times'] 
            df['germ_powers'] 
            df['Measurement_outcomes'] --> {'state' : N_counts}, e.g. {'01' : 1000 counts, '10' : 2000 counts, ... } 
            df['Number of shots'] 

        """ 
        data_frame = {}





        return cls(gst_data_frame = data_frame, prep_state = prep_state, measurement = measurement_operator)  
    
        
    @classmethod
    def import_gst_data(cls, file_string: str, N_qubits: int, prep_state: State, measurement_operator: Operator, shot_averaged: bool):
        """ Helper method for importing gst data from a file.

            file_string denotes the file name and location, e.g. "./my_datafile.xlsx" 

            Organize measurement data into a data frame (df) with arguments: 
            df['circuit_names'] 
            df['gate_start_times'] 
            df['gate_end_times'] 
            df['germ_powers'] 
            df['Measurement_outcomes'] --> {'state' : N_counts}, e.g. {'01' : 1000 counts, '10' : 2000 counts, ... } 
            df['Number of shots'] 

        """ 
        file_type = Path(file_string).suffix
        file_type = file_type[1:]

        outcome_col_label = "Outcome"

        # TODO: Generalize to N qubits 
        if N_qubits == 1:
            possible_outcomes = ['0', '1']
        elif N_qubits == 2:
            possible_outcomes = ['00', '01', '10', '11']

        # Import the data: 
        if file_type in ['xlsx', 'xls', 'xlsm']:
            data_sheet_name = '1Q GST'
            data_frame = pd.read_excel(file_string, sheet_name = data_sheet_name) 
        if file_type == 'hdf5': 
            data_frame = pd.read_hdf5(file_string) 

        # Process and organize data into measurement frequency information: 
        if shot_averaged:
            data_frame['Total shots'] = data_frame.filter(like = outcome_col_label).sum(axis=1) 
            for outcome in possible_outcomes:
                data_frame['Frequency ' + outcome] = data_frame[outcome_col_label + ' ' + outcome] / data_frame['Total shots'] 

        else:
            # Handle case where the circuits are reported with each shot. 
            
        #N_qubits = len(measurement_columns)//2
 #        qubit_measurements = [] # list of size number of qubits 
 #        for j in range(N_qubits): 
 #            qubit_measurements.append(gst_data[''])
        #circuit_names = gst_data['circuit_names']
            
        return cls(gst_data_frame = data_frame, prep_state = prep_state, measurement = measurement_operator)  

    @cachedproperty
    def get_frequencies() -> NDArray: 
        """ Returns 2D array of shape N_circuits x N_outcomes with each value 
             corresponding to a frequency of that outcome for that circuit. """  
        # Calling .values builds a 2D array; therefore, we cache the result to save on cost.  
        return self.gst_data_frame.filter(like = 'Frequency').values 

    def get_experimental_outcome_frequency(measurement_outcome: 'str', circuit_name: str): 
        """ Returns probability (shot-averaged outcomes -> frequency) of outcome "m" in circuit C 

            - measurement_outcome is a string denoting the computational state observed.  
                e.g. outcome = '0' for a single qubit, outcome = '10' for 2-qubits.  

            - There are 2**N possible outcomes for N qubits. 

        """
        assert len(measurement_outcome) == self.N_qubits, 'Measurement outcome does not correspond with number of qubits.'

        #return self.gst_data_frame['Frequency ' + measurement_outcome]
        #shot_averaged_data = True 
        if shot_averaged_data:
            frequency = self.gst_data_frame[outcome_index, circuit_name]
            frequency = self.gst_data_frame[outcome_index, circuit_name]
        else:
            # Sum over all measurement outcomes to get the number of total counts for the circuit 
            total_counts = sum(list(self.gst_data_frame[:, circuit_name].values()))
            frequency = self.gst_data_frame[outcome_index, circuit_name]/total_counts

        return frequency 



# ------ To be deleted later ------
''' Example usage: ''' 

my_basis = StandardBasis([spins])
my_gate_set = [Xpi_gate, Xpi_2_gate, idle_gate]
coeffs = np.zeros(2)
coeffs[0] = 1.
rho_0 = State.from_coefficients(coeffs)
Mz = Pauli.Z 


single_qubit_GST = GateSetTomography(my_basis, rho_0, native_measurement = Mz, gate_set = my_gate_set) 
gate_parameters = single_qubit_GST.solve_for_all_gate_parameters(solver='MLE')



# Put Gate objects into a dictionary to maintain name <--> gate correspondence. 
1Q_gate_set = {'Idle': idle_gate, 'X_pi2' : X_pi_2_gate, 'X_pi' : X_pi}



# Need functionality that generates gate set based on experimental input. 
# Is the gate set just the minimal number of IC gates? like is it G = [I, Xpi, X_pi2, Y_pi, Y_pi2] or something like that? From which we can recover the experimentally executed circuits? 



# Example data set: 
# Gate set: [idle, X_pi/2, X_pi]
# for each gate, we have 
    # 1000 shots of +1 

#model_coeffs = 



## TODO: Transpiler between QSCOUT and our naming gate convention 



# Could build a N-dimensional process matrix by running simulations for each of the N-dimensional parameters, store them. --> hdf5 files
    # - save the raw hdf5 simulation data (process matrix at each error parameter value) 
    # - GST will load the process matrix data, interpolate with it using MLE. cref. one of the examples  

        # - would need to load up one of these for each gate in the gate set

    # Separate data file for X_pi, X_pi/2 , etc. 
        # - each data structure (hdf5 -- file system, e.g. containing folders) contains simulations over many values of the error parameters  


    # TODO: Write function to save parameters / other info to hdf5 file

    # - GST could make calls to other parts of IonSim to run needed simulations. For constructing the required process matrices to do GST. 
        # - if you gave it a gate set, the class could then do those simulations to generate the process matrices needed for GST.   




# Other ideas: 
# - helper functionality for choosing IC fiducial prep/measurement circuits. See Section D.1.1. of PyGST paper. 
