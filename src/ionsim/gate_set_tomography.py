import numpy as np
import pandas as pd 
from pathlib import Path

import scipy.stats as stats 
import scipy.optimize as opt 
from functools import cached_property

from ionsim.process import Gate, Circuit
from ionsim.named_operators import Pauli, Unitary
from ionsim.GST_data_parser  import *
from ionsim.custom_math import matrix_AYB_multiply_to_superoperator 


# Example Workflow: 
# 1. GST_Data Class creation: User creates GST data class from experimental outcome data for various circuits. 
# 2. GST Class creation: User specifies gate set, prep state, measurement operator, and GST measurement data
# 3. GST Class: Solve for model parameters and return  

class GateSetTomography() # or GST() or GST_Base() if we plan to have child classes.
    def __init__(self, basis: StandardBasis, prep_state: State, POVM_measurement_effects: dict[str, list[Operator]], parsed_circuits: list[ParsedCircuit], gate_model_factory: Callable): 
        """ Class for performing quantum gate set tomography (GST) with trapped ions or neutral atoms. 
    
            Member variables include:
                - Basis where the quantum processes (gates), state, and measurement will live. 
                - prep state: rho_0, representing an ideal state prepared natively. 
                - POVM_measurement_effects: is a dictionary of measurement effects: ['0' : E0, '1': E1] or ['00' : E0, '01' : E1, ...] for N = 2 
                - parsed_circuits is a list of Parsed GST Circuits that contain circuit information and measurement information.
                - gate model factory is a function that takes a gate name and qubit tuple and returns an IonSim Gate object, which holds a process matrix (gate) function. 
                - gst_parameters: a 1D numpy array of gate parameters.  
    
        """ 

        # Could alternatively have the user specify this mapping 
        @cached_property
        def gate_dictionary(self):
            ism_gate_dictionary = {}    
            ism_gate_dictionary['Gxpi2'] = 'sqrt_X'
            ism_gate_dictionary['Gxpi'] = 'Xpi'
            ism_gate_dictionary['Gypi2'] = 'sqrt_Y'
            ism_gate_dictionary['Gypi'] = 'Y'
            ism_gate_dictionary['[]'] = 'I'
            ism_gate_dictionary['{}'] = None
            # TODO: add 2Q gates 
            return ism_gate_dictionary

        # Unpack |rho>> and <<E| or <<M| 
        self.ideal_prep_state = prep_state 
        self.measurement_effects = POVM_measurement_effects 

        # Parse circuits list contanining GST circuit sequences and correpsonding data (observations) 
        self.parsed_circuits = parsed_circuits 

        self.gate_model_factory = gate_model_factory # TODO revise 

        # Dimensionality of Hilbert and Hilbert-Schmidt spaces:
        self.d = len(basis.states)
        self.d2 = self.d * self.d

        # 1. Get all unique gates in the gate set 
        self.gate_set = set()  # gate_set contains ParsedGate objects
        for circ in parsed_circuits:
            for g in circ.expanded_gates: 
                gate_set.add(g) 
            

        # 2. Build gate models and possible interpolants during construction. 
        self.gate_models = {} 
        for gate in gate_set:
            ism_name = self.gate_dictionary[gate.name] 
            self.gate_models[gate] = gate_model_factory(ism_name, gate.qubits)


        # 3. Parameters: 
        # Build a parameter look-up dictionary to organizing parameter indices. 
        # Retrieve number of GST parameters (prep + gates + measure) and build & initialize parameter vector  
        self.gst_parameter_indices, self.num_gst_parameters = self._build_parameter_organization()
        self.gst_parameters = np.zeros(self.num_gst_parameters) 



    def _build_parameter_organization(self) -> dict[str, slice], int:
        """ Builds and organizes the parameters for GST. This organizes parameters based on:
            1) Prep state 
            2) Each Gate model, for all gates in the set  
            3) Native measurement

            - Parameter indexing are organized in a dictionary for easy retrieval, e.g. {'prep state' : prep_state_parameter_indexing (as slice)} 
            - Returns the layout dictionary and the total number of parameters 
        """

        # Set up dictionary with appropriate number of independent parameters 
        parameter_indices = {}

        # Index i will incriment as we increase the number of tracked parameters 
        i = 0

        # SPAM has d^3 - 1 parameters 
        N = self.d2 - 1         
        parameter_indices["prep"] = slice(i, i + N) 
        i += N

        N = self.d2
        parameter_indices["measurement"] = slice(i, i + N) 
        i += N

        for gate, gate_model in zip(self.gate_set, self.gate_models):
            N = gate_model.num_parameters
            # TODO: either use ism gate name or ParsedCircuit gate name 
            # Default parametrization is dense (d^2 x d^2) for each gate: 
            parameter_indices[gate.name] = slice(i, i + N)
            i += N  

        return parameter_indices, i 



    def get_prep_state(self, theta) -> Vector:
        """ Returns prep state supervector (d^2 x 1) given the parameter values theta.
            - Enforces the constraint Tr[rho] = 1, eliminating 1 parameter.
        """ 
        # Prep state parameters function as a perturbation away from ideal prep state   
        # TODO: consider a more sophisticated parametrization? 
        prep_params = theta[self.gst_parameter_indices["prep"]]

        assert len(prep_parms) == self.d2

        # Enforce constraint Tr[rho] = 1
        ideal_state = self.ideal_prep_state.supervector

        # Initialize prep state(theta) to the ideal staet
        prep_state = ideal_state.copy()
        # Current prep state is ideal + perturbation(theta); here perturbation(theta) = theta 
        perturbation = prep_params
        prep_state[:-1] += perturbation 
        
        diag_indices = [i * (self.d + 1) for i in range(self.d)] # assumes square density matrix 
        free_diag_indices = diag_indices[:-1]
         
        prep_state[-1] = 1.0 - np.sum(rho_vec[diag_indices[:-1]]) 

        return prep_state 


    def get_measurement_effects(self, theta) -> dict[str, Matrix]:
        """ Returns measurement effects given the parameter values theta. 

            - Effects are stored in a dictionary {'outcome' : Effect_vector with superoperator d^2 x d^2 shape} 
            - e.g. E_0 vector is d^2 x 1 corresponding to |0><0|
            - There is a completeness constraint to enforce: \sum_m E_m = identity
            - By convention, the last effect is constrained. ==> d^2 parameters are constrained.  
        """ 
        # TODO: consider a more sophisticated parametrization? 
        M_effects = {}

        measurement_params = theta[self.parameter_indices["measure"]]
        #assert len(prep_parms) == self.d2

        N_effects = len(self.measurement_effects)
        N_params_per_op = self.d2

        # Parametrize unconstrained (free) effects as ideal + perturbation:
        for i, (label, effect_op) in enumerate(self.measurement_effects.items()):
            if i == (N_effects - 1): # skip last index
                break
            # Use parameters for this operator by index slicing:  
            variation = measurement_params[i * N_params_per_op : (i + 1) * N_params_per_op] 

            # Get superoperator from operator representation 
            ideal_effect_superoperator = matrix_AYB_multiply_to_superoperator(effect_op.static_matrix)

            assert len(variation) == len(ideal_effect_superoperator)
            # Compute resulting effect:  
            M_effects[label] = ideal_effect_superoperator + variation 

        # Final effect is constrained to be E_last = I - sum(E) over all other effects E 
        constrained_effect = matrix_AYB_multiply_to_superoperator(np.eye(self.d)) # identity to superoperator 
        last_label = list(self.measurement_effects.keys())[-1]

        assert last_label is not in list(M_effects.keys()) 

        # Loop over all free effects and subtract them from constrained effect: 
        for i, (label, effect_op) in enumerate(self.measurement_effects.items()): 
            if i == (N_effects - 1): # skip last index
                break
            constrained_effect -= M_effects[label]

        M_effects[last_label] = constrained_effect

        return M_effects 


    def _predict_probabilities(self, circ: ParsedCircuit, theta) -> Vector[float]: 
        """ Predicts outcome probabilities for a GST circuit with gates parametrized by theta """
        #outcome_probabilities = np.zeros

        rho_supervector = self.get_prep_state(theta)
        M_effects = self.get_measurement_effects(theta)

        # Build a composition (chain) of gate process matrices: 
        quantum_map = np.eye(self.d2, dtype=complex)
        # TODO: gate_model is Ionsim object?  
        for gate_model in self.gate_models:
            gate_parameters = theta[self.gst_parameter_indices[gate_model.name]]
            quantum_map = gate_model.process_matrix_function(gate_parameters) @ quantum_map  
            
        mapped_state = quantum_map @ rho_supervector

        outcome_probabilities = {}
        for label, E in M_effects.items()
            outcome_probabilities[label] = np.real(E.conj() @ mapped_state) 

        return outcome_probabilities
        

    def _build_process_matrix_cache(self, theta): 
        """ Evaluate each gate's process matrix function once"""
        process_matrix_cache = {} 
        # TODO: finalize gate_model data structure (DS) and values. 
        for gate_model in self.gate_models.values():
            gate_parameters = theta[self.gst_parameter_indices[gate_model.name]]
            process_matrix_cache[gate_model.name] = gate_model.process_matrix_function(gate_parameters) 
        return process_matrix_cache 



    def log_likelihood(self, theta=None, theta_function=None) -> float:
        """ Computes total log-likelihood of the parameters given the data.

            theta:      parameter vector 
            theta_func:     optional callable(t) -> parameter_vector for time-dependent data.
                            If None, theta is assumed to be t-independent.

            Log likelihood of parameters for each experiment:  
                l_{exp} = sum_{outcomes} N_{outcome} log( p_{outcome} (theta) ) 
             - p_outcome (theta)  is the probability of the outcome using gates modeled by theta. 
             - "outcome" <==> measurement effect. e.g. "0" or "1" for 1Q measurement. 

        """                
        # TODO: make a separate function for t-dependent parameters 
        if theta is None:
            theta = self.gst_parameters

        l_likelihood = 0.

        time_independent_gates = True
        if theta_function is not None:
            t_independent_gates = False 

        self._build_process_matrix_cache()

        probability_TOL = 1E-10

        # Compute log likelihood for each GST circuit, then sum over all GST circuits 
        for circ in self.parsed_circuits:
            probabilities = self.predict_probabilities(circ, theta) # don't need the PM cache? 

            if circ.measurement_data.counts is not None:
                for outcome, count in circ.measurement_data.items()         
                    if count > 0:
                        p = np.clip(probs[outcome], probability_TOL, 1. - probability_TOL)
                        l_likelihood += count * np.log(p)
                 
            else:
                # Time-series data: each shot is equally weighted? 
                for t, outcome in circ.measurement.timestamped_shots:
                    p = np.clip(probs[outcome], probability_TOL, 1. - probability_TOL)
                    l_likelihood += np.log(p)

        return l_likelihood



    def solve_for_gate_parameters(solver: str = 'MLE'):
        """ Function to solve for the parametrization values of a particular gate. 

            - Default behavior is a maximum likelihood approach that finds parameters 
                that maximize the likelihood of the gate given the data, i.e. solving: 

                max[ Likelihood( {G} | data) ] over parameter set Theta.

            - Returns either a dictionary of parameters (name, value) or a 1D array of values.

        """
        # 2. Extract frequency values for each experiment 
        #experimental_frequencies = gst_data.get_frequencies() # 2D array of shape circuit x outcomes -> frequency values 

        #assert experimental_frequencies.shape[0] == len(gst_circuits) 

        # TODO: Need to figure out how to get gate parameters and use them here, from IonSim's Gate objects. 
        # - Compute expected probability of an outcome given a circuit's parametrization. 
        if solver == 'MLE':
            # Maximum likelihood estimation.
            # Specify initial guess. 
            theta_0 = self.gst_parameters.copy() 
            # GST expeirment circuits and outcome data are imbedded in log likelihood function evaluations. 
            solver_result = opt.minimize(fun = lambda params: -self.log_likelihood(params), x0 = theta_0, method = 'L-BFGS-B') # TODO consider adding parameter bounds  
            self.gst_parameters = solver_result.x
            return solver_result
            
        elif solver == 'linear':
            # Solve matrix Ax = b problem: Frequencies = A_matrix @ Gate_parameters  
            # Check that gram matrix A_{m,s} = <M | C_{m} C_{s} | rho> is invertible.            
            # Compute x = A \ b

        else:
            raise IonSimError('Invalid solver input.')


        #return gate_set_parameters 



    # Questions: 
        # Should we allow more than 1 initial native state and 1 native measurement? 
            # GST manual suggests that ususally only 1 is available. 
    
 #    basis: Basis
 #    gate_set: dict[str, Gate]    # instead of (list[Gate])
 #    ideal_gate_set: dict[str, Gate]    
    #gate_set: dict[str, Gate]    # instead of (list[Gate])

    #gate_parameters: Vector  

    # experimental data per circuit listed. 
    # - allow for variable number of shots per experiment. 
    #gst_data: GST_Data
    #circuit_name_dictionary: dict

    #circuit_name_dictionary['sqrt_X']

 #
 #    parametrized_gates: bool = False 
 #    gate_parameters: list[NDArray] 
 #    system_size: int 
 #
 
 #
 #    # +X pi/2
 #    if gatelabel == 'Xpi2':
 #        # theta = _np.pi/4
 #        # phi = 0 # phi = -_np.pi
 #        # time_scale = 0.5
 #        nominal_relative_phase = 0.0
 #
 #    # -X pi/2
 #    elif gatelabel == 'mXpi2':
 #        # theta = _np.pi/4
 #        # phi = _np.pi
 #        # time_scale = 0.5
 #        nominal_relative_phase = -_np.pi
 #
 #    # +X pi
 #    elif gatelabel == 'Xpi':
 #        # theta = _np.pi/2
 #        # phi = 0
 #        nominal_relative_phase = 0.0
 #
 #    # -X pi
 #    elif gatelabel == 'mXpi':   
 #        # theta = _np.pi/2 # theta = _np.pi
 #        # phi = _np.pi
 #        nominal_relative_phase = -_np.pi
 #
 #    # +Y pi
 #    elif gatelabel == 'Ypi':

    #fiducial_circuit_list: list[str] #e.g. ['g1', 'f1,g1,g2']
        # Each gate has a dictionary of parameters, but this could be none.  
        # include logic of 

 #    def __post_init__(self):
 #        """ Safety checks: 
 #            1. Gates need to be parametrized.  
 #            2. Prep and Measurement needs to match between GST data and GST classes.  
 #        """ 
 #        for gate in self.gate_set:
 #            if gate.parameters is None:
 #                self.parametrized_gates = False  
 #
 #        # TODO: we need to be able to check for state and operator equality. These compare methods don't exist yet. 
 #        if not compare_state(gst_experiments_data.prep_state, self.initial_state):
 #            raise IonSimError("GST Data Class and GST Class should use the same prep states.")
 #
 #        if not compare_operator(gst_experiments_data.measurement, self.native_measurement):
 #            raise IonSimError("GST Data Class and GST Class should use the same measurement operator.")

 #    @classmethod
 #    def from_GST_Data(cls, gst_data: GST_Data)
 #
 #
 #
 #        return cls( )


 #    def compute_circuit_outcome_probability(circuit: Circuit, outcome: int):
 #        """ Returns the probability of outcome "mu" for native measurement "M". """
 #        # e.g. outcome = +1 or -1 for single qubit Z measurement 
 #        outcome_index =  GST_experiments_data[0, :, 0].index(outcome)
 #        return gst_data.get_experimental_outcome_frequency(outcome_index, circuit) 
 #

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




 #    def estimate_probability(outcome: str, gst_circuit: ParsedCircuit, gate_parameters: Vector) -> float:
 #        """ Computes the probability of outcome after a GST circuit modeled with various gate parameters. """ 
 #        # Unpack circuits within the GST circuit
 #        prep_circuit = gst_circuit.prep_circuit
 #        measure_circuit = gst_circuit.measure_circuit
 #        germ_circuit = gst_circuit.germ_circuit
 #
 #        # Convert each ParsedGate -> IonSim Gate object 
 #        
 #        
 #
 #        return 0. 


    #def circuit_from_circuit_name(circuit_name: str, noise: Noise) -> Circuit:
 #    def circuit_from_parsed_circuit(parsed_circuit: ParsedCircuit) -> Circuit:
 #        """ Function that returns a circuit object (list of gates) corresponding to the circuit name. 
 #            - Helps to convert experiment circuit names into IonSim Gate/Circuit objects from the gate set. 
 #            - e.g. circuit_name = 'idle, X_pi_2' -> gate_list = [idle_gate, X_pi_2_gate] -> Circuit 
 #        """
 #        # TODO: Need to decide and state convention for gate ordering: e.g. Left to right: state prep --> gates --> measurement
 #            # need a look up table for the gates based on name 
 #        
 #        # Extract name of each gate from the circuit name  
 #        gate_names = [name.strip() for name in circuit_names.split(',')]
 #
 #        # TODO: Add functionality to parse the qubit number when we have 2+ qubits  
 #
 #        gate_list = []        
 #        for gate_name in gate_names:
 #            # Optional: map_experimental_gate_name_to_internal_gate_name(gate_name)
 #            gate_list.append(self.gate_set[gate_name])
 #            #if gate_name == 'I' or 'idle':
 #
 #        return Circuit.from_gates(gate_list, noise) 
 #

 #
 #    def map_experimental_gate_name_to_internal_gate_name(gate_name: str): -> str
 #        """ Helper function to convert experimental gate nomenclature to internal
 #                IonSim gate nomenclature used in this class. """
 #        # TODO: Maybe the user will specify this mapping / look-up table? 
 #        
 #
 #
 #
 #
 #
 #        return gate_name
 #


class GST_Data():
    """ Class to maintain measurement data for GST and provide data retrieval functionality. """
    # Should these use the same prep state and measurement basis? 
    N_qubits: int  

    # Prep state and measurement may need to become circuit dependent for full generality for 2+ qubits 
    ideal_prep_state: State
    ideal_measurement: Operator 

    # Can organize data either as a single data structure 
    #   or have each member variable as a different data structure.
    #gst_data_frame: pd.DataFrame ## TODO: Decide on data structure 
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
    def from_gst_sequence_data(cls, file_string: str, N_qubits: int, prep_state: State, measurement_operator: Operator, time_dependent: bool): 
        """ Helper method for importing gst data from a file of GST circuit sequences with file extension .gstdata, often produced by PyGSTi.

            file_string denotes the file name and location, e.g. "./my_datafile.gst" 

            Organize measurement data into a data frame (df) with arguments: 
            df['circuit_names'] , could be a dict of prep, germ, measure circuits. or these could be separate df entries.  
            df['gate_start_times'] 
            df['gate_end_times'] 
            df['germ_power'] 
            df['Measurement_outcomes'] --> {'state' : N_counts}, e.g. {'01' : 1000 counts, '10' : 2000 counts, ... } 
            df['Number of shots'] 

        """ 
        data_frame = {}


        # Get GST results from data, this contains a list of ParsedCircuit objects  
        gst_results = parse_gst_circuit_file(fname)

        data_frame['Parsed circuits'] = gst_results 

        # Get IonSim Circuit object from this data  


        # Map GST circuit sequences to 


        #num_experiments = len(gst_results)
        



        # Import GST sequences: 
        #imported_data = {} # dictionary with keys = circuit_sequences list, counts (.e.g for 0, 1).
        #circuit_sequences = imported_data['circuit_sequences']





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




### Parser for GST circuit sequence files 
 #import re 
 #from dataclasses import dataclass, field
 #from typing import Optional
 #from pathlib import Path 

#@dataclass
#class  





# ------ To be deleted later ------
 #''' Example usage: ''' 
 #
 #my_basis = StandardBasis([spins])
 #my_gate_set = [Xpi_gate, Xpi_2_gate, idle_gate]
 #coeffs = np.zeros(2)
 #coeffs[0] = 1.
 #rho_0 = State.from_coefficients(coeffs)
 #Mz = Pauli.Z 
 #
 #
 #single_qubit_GST = GateSetTomography(my_basis, rho_0, native_measurement = Mz, gate_set = my_gate_set) 
 #gate_parameters = single_qubit_GST.solve_for_all_gate_parameters(solver='MLE')
 #
 #
 #
 ## Put Gate objects into a dictionary to maintain name <--> gate correspondence. 
 #1Q_gate_set = {'Idle': idle_gate, 'X_pi2' : X_pi_2_gate, 'X_pi' : X_pi}
 #
 #

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
