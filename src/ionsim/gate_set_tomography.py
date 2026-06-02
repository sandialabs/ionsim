import numpy as np
from pathlib import Path

import scipy.stats as stats 
import scipy.optimize as opt 
from functools import cached_property
from typing import Callable
import inspect
import sys
import math 

from ionsim.process import Gate, Circuit
from ionsim.basis import StandardBasis
from ionsim.named_operators import Pauli, Unitary
from ionsim.gst_circuit_parser  import *
from ionsim.custom_math import matrix_AYB_multiply_to_superoperator 
from ionsim.ionsim_error import IonSimError
from ionsim.custom_types import Vector, Matrix
from ionsim.gst_circuit_planner import GSTCircuitPlanner
from ionsim.io import *

def depth_bin(depth):
    """ Bins a circuit depth to the nearest power of 2 """
    if depth <= 1.:
        return 1
    return int(2**(np.ceil(np.log2(depth))))

class GateSetTomography(): # or GST() or GST_Base() if we plan to have child classes.
    def __init__(self, basis: StandardBasis, prep_state_model: Callable, POVM_effect_models: dict[str, Callable], parsed_circuits: list[ParsedCircuit], 
                     gate_mappings: dict[str, Callable], parameter_bounds: list[tuple] | None=None, circuit_design: GSTCircuitPlanner | None=None): 
        """ Class for performing quantum gate set tomography (GST) with trapped ions or neutral atoms. 
    
            Member variables include:
                - Basis where the quantum processes (gates), state, and measurement will live. 
                - prep state: rho_0, representing an ideal state prepared natively. 
                - POVM_measurement_effect models: is a dictionary of constrained measurement effect models :['0' : E0(params)] or ['00' : E0(params), '01' : E1(params), ...] for N = 2 
                - parsed_circuits is a list of Parsed GST Circuits that contain circuit information and measurement information.
                #- gate model factory is a function that takes a gate name and qubit tuple and returns an IonSim Gate object, which holds a process matrix (gate) function. 
                - gate_mappings represents a dictionary that maps GST gate names to IonSim model names, specified by the user.  
                - gst_parameters: a 1D numpy array of gate parameters.  
            
            Optional arguments:
                - circuit design as a circuit planner object. This is not required for doing MLE but is required for linear GST. 
                - parameter bounds on the model parameters, used in MLE.
    
        """ 

        print(f"\n\n --- Constructor for GateSetTomography Class IonSim --- ")

        self.basis = basis 
        # Unpack |rho>> and <<E| or <<M| 
        self.prep_state_model = prep_state_model
        self.POVM_effect_models = POVM_effect_models 

        # Parse circuits list contanining GST circuit sequences and correpsonding data (observations) 
        self.parsed_circuits = parsed_circuits 

        # Dimensionality of Hilbert and Hilbert-Schmidt spaces:
        self.d = len(basis.states)
        self.d2 = self.d * self.d

        # 1. Get all unique gates in the gate set 
        self.gate_set = set()  # gate_set contains ParsedGate objects
        for circ in self.parsed_circuits:
            for g in circ.expanded_gates: 
                self.gate_set.add(g) 

        # 2. Retrieve gate models  
        self.gate_models = {}  # dictionary to map a Parsed Gate (from the gate set) to its model as a process matrix function  
        self.gate_model_factory = self._initialize_gate_model_factory(gate_mappings)
        for gate in self.gate_set:
            #ism_name = gate_dictionary[gate.name] 
            print(f"\n Printing gate set: ")
            print(f"Gate: {gate}")
            print(f"Name: {gate.name}")
            self.gate_models[gate.name] = self.gate_model_factory(gate.name, gate.qubits)

        # 3. Parameters: 
        # Build a parameter look-up dictionary for organizing parameter indices. 
        # Retrieve number of GST parameters (prep + gates + measure) and build & initialize parameter vector  
        self.gst_parameter_indices, self.num_gst_parameters = self._build_parameter_organization()

        # TODO: Allow for user control of initial condition for parameter vector?   
        #self.gst_parameters = np.zeros(self.num_gst_parameters) 
        self.gst_parameters = np.ones(self.num_gst_parameters)*1E-4 

        # 4. Debugging / diagnostics 
        self.LL_eval = 0 
        self.nll_data = []

        self.parameter_bounds = parameter_bounds 

        # Set up cached parameters and process matrices 
        self.cached_theta = None 
        self.process_matrix_cache = None 

        #TODO: Algorithm for optimizing stepwise by circuit depth.  
        # initialize GST results to None 
        if circuit_design :
            # Use a list of tuples instead of list of gates for compatibility with dictionaries 
            self.prep_fiducials = [tuple(prep_fid) for prep_fid in circuit_design.prep_fiducials]
            self.measure_fiducials = [tuple(meas_fid) for meas_fid in circuit_design.measure_fiducials]
        else:
            self.prep_fiducials = None  
            self.measure_fiducials = None 

        self.lgst_results = None   
        self.solver_result = None 

        # Organize a lookup table for fiducial prep/measure circuits; needed for linear GST 
        self._index_fiducials()
        #print(f"\n\n Circuit lookup:")
        #print(self.circuit_lookup)
        # Test: 
        #print(f"\n\n Test :")
        #print(self.circuit_lookup[((), (), 1, ())])


    def _initialize_gate_model_factory(self, gate_mappings: dict) -> Callable:
        """ Sets up a "factory" function that returns a gate model as a function of 
            - gate name (str) 
            - involved qubits as a tuple of qubit indices, ranging from 0 to N_qubits - 1 
        """
        # TODO: Revise/finalize with 2Q examples 

        def gate_factory(gate_name: str, qubits: tuple[int, ...]) -> Callable:
            """ Function to map a gate name & qubit arguments to a gate function """ 
            if gate_name == 'idle':
                return gate_mappings[gate_name]
            elif gate_name == '' or (gate_name is None):
                return gate_mappings['{}']
    
            # TODO: Generalize to 2Q gates 
            #   - for 1Q gates, this is made trivial by the dictionary. For 2Q, it requires functionality 
            assert len(qubits) == 1
            return gate_mappings[gate_name]

        return gate_factory 


    def _build_parameter_organization(self) -> tuple[dict[str, slice], int]:
        """ Builds and organizes the independent parameters for GST. This organizes parameters based on:
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

        # Prep: there are d^2 - 1 independent parameters due to the Trace[rho] = 1 constraint. 
        N = self.d2 - 1 
        parameter_indices["prep"] = slice(i, i + N) 
        i += N

        # Measure: there are d - 1 independent measurement effects from the completeness constraint.
        #   - each effect is a d x d matrix, so there are d^2(d-1) indepenent parameters  
        N = self.d2 * (self.d - 1) 
        parameter_indices["measurement"] = slice(i, i + N) 
        i += N

        for gate, gate_model in zip(self.gate_set, self.gate_models.values()):
            gate_model_sig = inspect.signature(gate_model)
            N = len(gate_model_sig.parameters)

            # Default parametrization is dense (d^2 x d^2) for each gate: 
            parameter_indices[gate.name] = slice(i, i + N)
            i += N  

        return parameter_indices, i 


    def _index_fiducials(self):
        """ Identify unique prep/measure fiducials and build lookup to get observed probabilities. """

 #        prep_fiducials = set()
 #        measure_fiducials = set()
 #
 #        # Fiducial prep/measure circuits are stored as lists of gates. Lists are not hashable and 
 #        #  must be converted to tuples to enable lookup. A "ParsedGate" object is immutable, so tuples of it are hashable.  
 #        for circ in self.parsed_circuits:
 #            prep_fiducials.add(tuple(circ.fiducial_prep_gates))
 #            measure_fiducials.add(tuple(circ.fiducial_measurement_gates))
 #
 #        # Sort by string representation (uses __repr__ in ParsedGate) 
 #        self.prep_fiducials = sorted(prep_fiducials, key=str)
 #        self.measure_fiducials = sorted(measure_fiducials, key=str)

        combined_counts = {}
        # Create keys by full circuit representation and average over duplicates (TODO: Update/change for non-Markovian GST)
        for circ in self.parsed_circuits:
            #key = (tuple(circ.fiducial_prep_gates), tuple(circ.germ_gates), circ.germ_power, tuple(circ.fiducial_measurement_gates))
            #if circ.germ_power != 1:
            #    continue  
            key = tuple(circ.expanded_gates)
            counts = circ.measurement_data.to_counts()

            if key in combined_counts:
                for label, n in counts.items():
                    combined_counts[key][label] = combined_counts[key].get(label, 0) + n
            else:
                combined_counts[key] = counts 
                #combined_counts[key] = dict(counts)

        # Set up circuit -> probability dictionary 
        self.circuit_lookup = {}
        for key, counts in combined_counts.items(): 
            total_counts = sum(counts.values())
            self.circuit_lookup[key] = {outcome: count / total_counts for outcome, count in counts.items()}
            #total_counts = circ.measurement_data.total_counts

    def get_prep_state(self, theta) -> Vector:
        """ Returns prep state supervector (d^2 x 1) given the parameter values theta.
            - Enforces the constraint Tr[rho] = 1, eliminating 1 parameter.
        """ 
        prep_params = theta[self.gst_parameter_indices["prep"]] # d^2 - 1 column vector  

        assert len(prep_params) == (self.d2 - 1)

        prep_state = self.prep_state_model(prep_params)
        return prep_state

    def get_measurement_effects(self, theta) -> dict[str, Vector]:
        """ Returns measurement effects given the parameter values theta. 

            - Effects are stored in a dictionary {'outcome' : Effect_vector with superoperator d^2 x d^2 shape} 
            - e.g. E_0 vector is d^2 x 1 corresponding to |0><0|
            - There is a completeness constraint to enforce: sum_m E_m = identity
            - By convention, the last effect is constrained. ==> d^2 parameters are constrained. 
            - Therefore, there are d^2 (d-1) independent parameters for measurment.  
        """ 
        M_effects = {}
        N_effects = len(self.POVM_effect_models) 
        measurement_params = theta[self.gst_parameter_indices["measurement"]]
        assert N_effects == (self.d) 
        N_params_per_op = self.d2

        # 1. Evaluate unconstrained effect models 
        for i, (label, effect_model) in enumerate(self.POVM_effect_models.items()):
            if i == (N_effects - 1): # skip last index
                break
            # Retrieve model parameters for this effect and plug into effect's model function 
            model_parameters = np.array(measurement_params[i * N_params_per_op : (i + 1) * N_params_per_op]) 
            M_effects[label] = effect_model(model_parameters)

        # 2. Determine Final effect, constrained to be E_last = I - sum(E) over all other effects E 
        last_label = list(self.POVM_effect_models.keys())[-1]
        constrained_effect = np.eye(self.d).flatten()
        assert last_label not in list(M_effects.keys()) 
        assert self.POVM_effect_models[last_label] == None 
        # TODO: Instead of having the last effect be None by arbitrary convention, have this code find which effect is constrained by checking if its function is None.

        # Loop over all independent effects to compute the constrained effect: 
        for i, effect_op in enumerate(M_effects.values()): 
            if i == (N_effects - 1): # skip last index
                break
            constrained_effect -= effect_op 

        M_effects[last_label] = constrained_effect
        return M_effects 

    def _predict_probabilities(self, circ: ParsedCircuit, theta: Vector) -> Vector: 
        """ Predicts outcome probabilities for a GST circuit with gates parametrized by theta """
        rho_supervector = self.get_prep_state(theta)
        M_effects = self.get_measurement_effects(theta)

        # Build a composition (chain) of gate process matrices: 
        quantum_map = np.eye(self.d2, dtype=complex) # handles case for initial gst circuit: do-nothing for no time (null) gate 

        # Retrieve a gate model for each gate and its parameter values  
        for gate in circ.expanded_gates:
            gate_process_matrix = self.process_matrix_cache[gate.name]
            # Accumulate the map:
            quantum_map = gate_process_matrix @ quantum_map 

        mapped_state = quantum_map @ rho_supervector

        outcome_labels = list(M_effects.keys())
        # Compile effects into a matrix to exploit matrix-vector multiplication  
        effect_matrix = np.vstack([
            np.asarray(M_effects[label]).reshape(-1) for label in outcome_labels
        ])
        mapped_state_vector = np.asarray(mapped_state).reshape(-1)

        probability_TOL = 1E-12
        probability_values = np.real(effect_matrix @ mapped_state_vector)
        probability_values = np.clip(probability_values, probability_TOL, 1. - probability_TOL)
        outcome_probabilities = dict(zip(outcome_labels, probability_values))
        return outcome_probabilities
        
    def _refresh_gate_process_matrix_cache(self, theta): 
        """ Evaluate each gate's process matrix function once"""

        if (self.cached_theta is None or self.cached_theta.shape != theta.shape 
            or not np.array_equal(self.cached_theta, theta)):
            process_matrix_cache = {} 
            for gate_name, gate_model in self.gate_models.items():
                # Retrieve parameters for the gate model 
                gate_parameters = theta[self.gst_parameter_indices[gate_name]]
                # Evaluate gate model at those parameter values and store in the PM cache 
                process_matrix_cache[gate_name] = gate_model(*gate_parameters) # gate model returns a process matrix  

            self.cached_theta = theta
            self.process_matrix_cache = process_matrix_cache

        return self.process_matrix_cache 



    def log_likelihood(self, theta: Vector | None=None, theta_function=None) -> float:
        """ Computes total log-likelihood of the parameters given the data.

            theta:      parameter vector 
            theta_func:     optional callable(t) -> parameter_vector for time-dependent data.
                            If None, theta is assumed to be t-independent.

            Log likelihood of parameters for each experiment:  
                l_{exp} = sum_{outcomes} N_{outcome} log( p_{outcome} (theta) ) 
             - p_outcome (theta)  is the probability of the outcome using gates modeled by theta. 
             - "outcome" <==> measurement effect. e.g. "0" or "1" for 1Q measurement. 

        """                
        print(f"\nEvaluating log likelihood")
        self.LL_eval += 1 
        print(f"Evaluation number {self.LL_eval}")
        print(f"\nParameter values: {theta}")

        # TODO: make a separate function for t-dependent parameters 
        if theta is None:
            theta = self.gst_parameters

        l_likelihood = 0.

        time_independent_gates = True
        if theta_function is not None:
            t_independent_gates = False 

        probability_TOL = 1E-12

        # Improve speed by building gate process matrices once 
        #process_matrix_cache = self._build_gate_process_matrix_cache(theta)
        self._refresh_gate_process_matrix_cache(theta)

        # Compute log likelihood for each GST circuit, accumulating over all GST circuits 
        for circ in self.parsed_circuits:
            probabilities = self._predict_probabilities(circ, theta) 

            if circ.measurement_data.counts is not None:
                for outcome, count in circ.measurement_data.counts.items(): 
                    # Only non-zero counts will contribute to the likelihood.  
                    if count > 0:
                        p = np.clip(probabilities[outcome], probability_TOL, 1. - probability_TOL)
                        l_likelihood += count * np.log(p)
            else:
                # Time-series data: each shot is equally weighted? 
                for t, outcome in circ.measurement.timestamped_shots:
                    p = np.clip(probabilities[outcome], probability_TOL, 1. - probability_TOL)
                    l_likelihood += np.log(p)

        print(f"Negative log likelihood: {-l_likelihood}")
        self.nll_data.append(-l_likelihood) 
        return l_likelihood


    
    def chi_squared(self, theta: Vector | None=None, theta_function=None) -> float:
        """ chi^2 estimate for least-squares error between observed frequencies and circuit probabilities. """ 
        chi_squared = 0.

        if theta is None:
            theta = self.gst_parameters

        probability_TOL = 1E-4 # to regularize 

        # Improve speed by building gate process matrices once 
        self._refresh_gate_process_matrix_cache(theta)

        # Compute log likelihood for each GST circuit, accumulating over all GST circuits 
        for circ in self.parsed_circuits:
            #print(f"\nCircuit: {circ.unparsed_data}")
            probabilities = self._predict_probabilities(circ, theta) 

            if circ.measurement_data.counts is not None:
                total_counts = circ.measurement_data.total_counts
                for outcome, count in circ.measurement_data.counts.items(): 
                    frequency = count / total_counts
                    p = np.clip(probabilities[outcome], probability_TOL, 1. - probability_TOL)
                    chi_squared += total_counts * ((p - frequency)**2)/p 
            else:
                raise IonSimError(f"Computing chi squared for time-series data is not yet programmed in IonSim.")

        print(f"Chi squared: {chi_squared}")
        return chi_squared


    def _group_circuits_by_base_depth(self):
        """ Groups the GST circuit by depth, required for staged MLE """ 
        groups = {} # dictionary to store list of circuits at each depth L 
        for circ in self.parsed_circuits:
            germ_length = len(circ.germ_gates)
            base_depth = germ_length*(circ.germ_power)
            if germ_length == 0: 
                L = 1
            else:        
                L = depth_bin(float(base_depth))
            if L not in groups:
                groups[L] = [] 
            groups[L].append(circ)
        return groups

    def _group_circuits_by_depth(self):
        """ Groups the GST circuit by depth, required for staged MLE """ 
        groups = {} # dictionary to store list of circuits at each depth L 
        for circ in self.parsed_circuits:
            L = depth_bin(circ.depth)
            if L not in groups:
                groups[L] = [] 
            groups[L].append(circ)
        return groups

    def _group_circuits_by_germ_power(self):
        """ Groups the GST circuit by depth, required for staged MLE """ 
        groups = {} # dictionary to store list of circuits at each depth L 
        for circ in self.parsed_circuits:
            p = circ.germ_power 
            if p not in groups:
                groups[p] = [] 
            groups[p].append(circ)
        return groups

    def save_nll_data(self):
        print(f"LL evals: {self.LL_eval}")
        print(f"len(nll_data): {len(self.nll_data)})")
        if self.nll_data : 
            np.savetxt('negative_log_likelihood.dat', np.column_stack([np.array(range(0, self.LL_eval)), np.array(self.nll_data)]), header = 'Iteration Neg_Log_Likelihood')
        else:
            raise ValueError(f"No log likelihood data is stored.")


    def get_parameter_value_by_name(self, gate_name: str, parameter_name: str) -> float:
        """ Return the parameter value for a requested parameter in a gate model"""

        gate_model = self.gate_models[gate_name]
        gate_model_sig = inspect.signature(gate_model)
        parameter_names = list(gate_model_sig.parameters.keys())  
        indx = parameter_names.index(parameter_name)
        parameter_values = self.gst_parameters[self.gst_parameter_indices[gate_name]] # names and values share same sorted order  
        return parameter_values[indx]
        
    def get_parameter_values_by_name(self, gate_name: str, parameter_names: list[str]) -> dict:
        """ Return the parameter value for a requested parameter in a gate model"""
        requested_params = {}
        for name in parameter_names:
            requested_params[name] = self.get_parameter_value_by_name(gate_name, name) 
        return requested_params 

    def print_parameters(self):
        # Prep, measure, then gate parameters: 
        print("\n --- Printing parameter values --- ")
        prep_params = self.gst_parameters[self.gst_parameter_indices["prep"]] # d^2 - 1 column vector  
        print(f"Prep state parameters: {prep_params}")

        measurement_params = self.gst_parameters[self.gst_parameter_indices["measurement"]] # d^2 - 1 column vector  
        print(f"\nMeasurement effect parameters: {measurement_params}")

        for gate in self.gate_set:
            gate_model = self.gate_models[gate.name]
            gate_model_sig = inspect.signature(gate_model)
            parameter_names = list(gate_model_sig.parameters.keys())  
            parameter_values = self.gst_parameters[self.gst_parameter_indices[gate.name]] # names and values share same sorted order  

            # Package parameter names, values 
            gate_results = dict(zip(parameter_names, parameter_values)) 
            print(f"\n Gate {gate.name} parameters: {gate_results}")

        return self.gst_parameters 

    def print_state_and_POVMs(self):
        """ Output state supervector and measurement effects """ 
        rho = self.get_prep_state(self.gst_parameters) 
        M_effects = self.get_measurement_effects(self.gst_parameters)

        print(f"\nPrep state supervector: {rho}")
        for label, effect in M_effects.items():
            print(f"\nMeasurement effect {label} vectors: {effect}")

    def solve_for_gate_parameters(self, parameters_guess: Vector | None=None, solver: str = 'MLE'): 
        """ Function to solve for the parametrization values of a particular gate. 

            - Default behavior is a maximum likelihood approach that finds parameters 
                that maximize the likelihood of the gate given the data, i.e. solving: 

                max[ Likelihood( {G} | data) ] over parameter set theta.

            - Returns either a dictionary of parameters (name, value) or a 1D array of values.

        """
        print(f"\n -- Solver for gate parameters in GST using {solver} --- ")
        # Specify initial guess. 
        if parameters_guess is None:
            theta_0 = self.gst_parameters.copy() 
        else:
            theta_0 = parameters_guess
        print(f"Initial parameters: {theta_0}")

        # TODO: Standardize output; This function has heterogeneous output, depending on which case is called. 
        if solver == 'MLE':
            # TODO: Provide bounds for parameters if using interpolated gates 
            # GST expeirment circuits and outcome data are imbedded in log likelihood function evaluations. 
            solver_result = opt.minimize(fun = lambda params: -self.log_likelihood(params), x0 = theta_0, method = 'L-BFGS-B', bounds = self.parameter_bounds) 
            self.solver_result = solver_result
            self.gst_parameters = solver_result.x
            #self.save_nll_data()
            #self.print_parameters()
            return solver_result
            
        elif solver == 'linear':
            #raise IonSimError('Linear GST is not yet programmed into IonSim.')
            self.run_linear_gst()
            self.parameters_from_lgst_results()
            return self.gst_parameters 
        elif solver == 'staged MLE':
            # Do staged MLE --> MLE done in batches of increasing circuit depths. 
            self.solver_result, results_by_stage = self.staged_objective_minimization(method = 'L-BFGS-B', bounds = self.parameter_bounds, suppress_output = False) 
            self.gst_parameters = self.solver_result.x
            return self.solver_result, results_by_stage
        else:
            raise IonSimError('Invalid solver input.')


    def _build_probability_matrix(self, target_gate: ParsedGate | None=None, outcome: str='0'):
        """ Builds the d^2 x d^2 matrix of observed probabilities 
            for a gate or empty gate (corresponding to the Gram Matrix).

            M[i,j] = p(outcome | measure_fid_i x gate x prep_fid_j ) 
        """
        N_prep_circuits = len(self.prep_fiducials)
        N_measure_circuits = len(self.measure_fiducials)

        # Construct matrix using lookup table of circuit outcomes for LGST 
        M = np.zeros((N_measure_circuits, N_prep_circuits))

        target_list = [target_gate] if target_gate else []
 #        if target_gate is None:
 #            gate = tuple()
 #        else:
 #            gate = tuple(target_gate)

        for j, prep_fid in enumerate(self.prep_fiducials):
            for i, measure_fid in enumerate(self.measure_fiducials):
                #key = (prep_fid, gate, 1, measure_fid)
                key = tuple(list(prep_fid) + target_list + list(measure_fid)) 
                if key in self.circuit_lookup:
                    M[i,j] = self.circuit_lookup[key].get(outcome, 0.)
                else:
                    print(f"Attempted key: {key}")
                    raise ValueError(f"Missing LGST circuit: prep = {prep_fid}" + 
                        f", gate = {target_list}, measure = {measure_fid}")
        return M
        
        
    def run_linear_gst(self):
        """ Function to estimate gate set parameters using linear matrix inversion """
        # 1. Build the Gram matrix: <<F_i|F_j>>
        print(f"\n --- Running linear GST ---")
        gram_matrix = self._build_probability_matrix(target_gate = None)

        # TODO: Check that gram matrix A_{m,s} = <M | C_{m} C_{s} | rho> is invertible.            
        # Invert via pseudoinverse for numerical stability 
        print(f"Determinant of gram matrix: {np.linalg.det(gram_matrix)}")
        gram_inv = np.linalg.pinv(gram_matrix)
        
        # #####2. Estimate each gate using the pseudo-inverse
        # 3. Extract SPAM parameter estimates from Gram SVD 
        U, S, Vh = np.linalg.svd(gram_matrix)

        print(f"S shape: {S.shape}")
        print(f"gram matrix shape: {gram_matrix.shape}")

        # Decomposition of Gram matrix = AB, where A is measurement matrix and B is prep matrix 
        # See Section 3. of "Gate Set Tomography" published in Quantum 
        # Gram matrix = U S V^T = U sqrt(S) sqrt(S) V^T = AB 
        sqrt_S = np.diag(np.sqrt(S))
        sqrt_S_inv = np.diag(1./np.sqrt(S))

        # Identify A = U sqrt(S)
        A = U @ sqrt_S
        # Identify B = sqrt(S) V^{T}
        B = sqrt_S @ Vh

        A_inv = np.linalg.pinv(A) # left pseudo-inverse 
        B_inv = np.linalg.pinv(B) # right pseudo-inverse 

        # Get the first "k" singular values           
        #k = self.d2
        #sqrt_S = np.diag(np.sqrt(S[:k]))
        #sqrt_S_inv = np.diag(1./np.sqrt(S[:k]))

        # Find which fiducial index is the empty circuit, corresponding to native prep and measure 
        empty_fid = tuple()
        prep_idx = self.prep_fiducials.index(empty_fid)
        measure_idx = self.measure_fiducials.index(empty_fid)

        # Extract native prep rho_0:
        print(f"root(S) shape: {sqrt_S.shape}")
        print(f"V shape: {Vh.shape}")
        prep_states = sqrt_S @ Vh[:k, :]  # d^2 x d^2 
        estimated_rho = prep_states[:, prep_idx]
        rho_pinv = Vh[:k,:].T @ sqrt_S_inv

        # Extract effects:
        measurement_effects = U[:, :k] @ sqrt_S
        measurement_effects_inv = sqrt_S_inv @ U[:, :k].T
        estimated_effect = measurement_effects[measure_idx, :]  # 1 x d^2

        gate_estimates = {}
        for gate in self.gate_set:
            # Compute gate process matrix by inversion: probabilities P = A G_gate B 
            P_gate = self._build_probability_matrix(target_gate = gate)
            gate_estimates[gate.name] = A_inv @ P_gate @ B_inv 
            #gate_estimates[gate.name] = measurement_effects_inv @ P_gate @ rho_pinv
            #gate_estimates[gate.name] = gram_inv @ P_gate
            print(f"Gate shape: {gate_estimates[gate.name].shape}")

        self.lgst_results = {'gate_estimates' : gate_estimates, 'gram_matrix' : gram_matrix, 
                        'native_prep_state' : estimated_rho, 'estimated_effects' : estimated_effect, 
                        'prep_states' : prep_states, 'measurement_effects' : measurement_effects}
        return self.lgst_results 


    def parameters_from_lgst_results(self):
        """ Extracts the parameters vector from linear GST results """

        if not hasattr(self, 'lgst_results'):
            self.run_linear_gst()

        # Initialize theta 
        theta = self.gst_parameters

        # Extract gate parameters 
        for gate_name, lgst_gate_matrix in self.lgst_results['gate_estimates'].items():
            # Compute the fit parameters for each gate model 
            model = self.gate_models[gate_name]
            fit_parameters = self._fit_gate_model_to_lgst_estimate(model, lgst_gate_matrix)    
            theta[self.gst_parameter_indices[gate_name]] = fit_parameters 

        # Extract SPAM parameters


        
        # Set gst parameters attribute to extracted parameters 
        self.gst_parameters = theta
        return theta


    def _fit_gate_model_to_lgst_estimate(self, gate_model: Callable, target_gate_matrix: Matrix):
        """ Fits a gate model's parameters given process matrix data (target_gate_matrix).

            - gate_model is as Callable that returns a process matrix  
            - uses the Frobenius norm of the process matrix difference as the cost function

        """


        def cost(theta: Vector) -> float:
            # Frobenius norm of the process matrix difference bt. model and LGST-predicted
            M = gate_model(*theta)
            return np.linalg.norm(M - target_gate_matrix, 'fro')**2

        gate_model_sig = inspect.signature(gate_model)
        N_parameters = len(gate_model_sig.parameters)
        p0 = np.zeros(N_parameters,dtype=complex) 

        # TODO: Check that this choice of p0 does not result in singular / ill-posedness  
        result = opt.minimize(cost, p0, method='Nelder-Mead')
        return result.x


    def write_results_to_file(self):
        """ Writes results of GST analysis to disk. Convention is to write HDF5 file per gate. """ 

        # Write results of each gate set to an hdf5 file
        for gate in self.gate_set:
            # Retrieve gate parameter names and values at optimimum; evaluate process matrix  
            gate_model = self.gate_models[gate.name]
            gate_model_sig = inspect.signature(gate_model)
            parameter_names = list(gate_model_sig.parameters.keys())  
            parameter_values = self.gst_parameters[self.gst_parameter_indices[gate.name]] # names and values share same sorted order  

            process_matrix = gate_model(*parameter_values)
            # Write parameter names, values, and process matrix evaluated at those parameter values.
            results_to_write = dict(zip(parameter_names, parameter_values)) 
            results_to_write[gate.name + '_process_matrix'] = process_matrix
            write_results_to_file('gst_optimal_' + gate.name + '.hdf5', results_to_write) 


    def staged_objective_minimization(self, method: str='L-BFGS-B', bounds: list | None=None, suppress_output: bool=True, organize_circuits_by_germ_power: bool=False):
        """ Iterative MLE through batches of data taken at increasing circuit depths """ 
        if organize_circuits_by_germ_power: 
            circuit_groups = self._group_circuits_by_germ_power()
        else:
            circuit_groups = self._group_circuits_by_base_depth()
            #circuit_groups = self._group_circuits_by_depth()

        sorted_depths = sorted(circuit_groups.keys()) # keys are circuit depths 
        solver_results = {} # stores results of parameter estimation at each stage 
        if not suppress_output:
            if organize_circuits_by_germ_power: 
                print(f"--- Staged MLE with bins by germ powers (p): {sorted_depths} ") 
                for p in sorted_depths:
                    print(f"    p={p}: {len(circuit_groups[p])} circuits ")
            else:
                print(f"--- Staged MLE with bins by circuit depth (L): {sorted_depths} ") 
                for L in sorted_depths:
                    print(f"    L={L}: {len(circuit_groups[L])} circuits ")

        #sys.exit(0)
        cumulative_circuits = []
        num_stages = len(sorted_depths)
        for stage, L in enumerate(sorted_depths):
            cumulative_circuits.extend(circuit_groups[L])

            # Store a copy of the circuits so we can re-use internal functions that use parsed_circuits attribute  
            original_circuits = self.parsed_circuits
            self.parsed_circuits = cumulative_circuits 
 #
 #            if stage < (num_stages - 1):
 #                objective_function = self.chi_squared 
 #            else:
 #                objective_function = lambda params: -1. * self.log_likelihood(params) 
            objective_function = lambda params: -1. * self.log_likelihood(params) 

            solver_result = opt.minimize(fun = lambda params: objective_function(params), x0 = np.ones(self.num_gst_parameters)*1E-2, method=method, bounds = bounds)
            #solver_result = opt.minimize(fun = lambda params: objective_function(params),  x0 = self.gst_parameters.copy(), method=method, bounds = bounds)
            self.solver_result = solver_result
            self.gst_parameters = solver_result.x

            # Record solver parameter estimation results at each circuit depth group  
            solver_results[L] = solver_result.x 

            if not suppress_output:
                ll = self.log_likelihood(self.gst_parameters)
                print(f"    Stage {stage + 1} (L <= {L}): ")
                print(f"    {len(cumulative_circuits)} circuits ")
                print(f"    LL = {ll:.3f} ") 
                print(f"    Converged = {solver_result.success} ") 
                        
            # restore circuit information
            self.parsed_circuits = original_circuits

        # return final result, having used all circuits:
        return solver_result, solver_results

        

            
    ### Functions for gate set error metrics ### 
    def compute_gate_set_process_infidelity(self, gst_parameters: Vector, ideal_gate_set: dict, include_SPAM_error: bool=False) -> float:
        """ Estimate process infidelity of each gate in the gate set and compare to ideal, then average (or take a different norm?) over the gate set.

            - takes in an input dictionary "ideal_gate_set" that contains process matrices for each gate in the gate set. 
            - additionally, the ideal_gate_set input contains the ideal prep state and ideal POVM 


        """ 
        if self.solver_result is None:
            self.solve_for_gate_parameters()

        # Estimate process fidelity for each gate 
        gate_errors = {}
        gate_infidelity = 0.
        for gate in self.gate_set:
            ideal_gate = ideal_gate_set[gate.name] # as a process matrix 

            # Get process matrix from gate model at optimum 
            gate_process_matrix_function = self.gate_models[gate.name]
            parameter_values = gst_parameters[self.gst_parameter_indices[gate.name]] # names and values share same sorted order  

            process_matrix = gate_process_matrix_function(*parameter_values)
            gate_model = Gate(self.basis, process_matrix)
            process_infidelity = 1. - gate_model.compute_process_fidelity(ideal_gate)
            gate_errors[gate.name] = process_infidelity
            gate_infidelity += process_infidelity

        gate_infidelity /= len(self.gate_set)   # Average gate error  
        # Compute least-square difference for SPAM
        # prep state: 
        ideal_prep_state = ideal_gate_set['prep'].supervector  
        modeled_prep_state = self.get_prep_state(gst_parameters) 
        # Trace distance: sqrt(sum([rho_ideal[i] - rho_actual[i]]^2))
        prep_error = np.sqrt(np.sum((modeled_prep_state - ideal_prep_state)**2)) 

        # POVMs 
        ideal_POVMs = ideal_gate_set['POVM']  
        POVMs = self.get_measurement_effects(gst_parameters)
        POVM_errors = {}
        measurement_error = 0. 
        for outcome, POVM in ideal_POVMs.items():
            ideal_POVM = POVM.superbra 
            parametrized_POVM = POVMs[outcome] 
            POVM_errors[outcome] = np.sqrt(np.sum((ideal_POVM - parametrized_POVM)**2)) 
            measurement_error += POVM_errors[outcome] 

        print(f"\nGate errors: {gate_errors}")
        print(f"\nPrep error: {prep_error}")
        print(f"\nPOVM errors: {POVM_errors}")
        # TODO: Should we average over the gate infidelities? 
        if include_SPAM_error:
            return gate_infidelity + measurement_error + prep_error 
        else:
            return gate_infidelity 


    def estimate_parameter_uncertainties(self, theta: Vector | None=None, method: str='bootstrap') -> Vector:
        """ Computes uncertainties of each parameter from the Hessian of the log-likelihood at the MLE solution."""
        if self.solver_result is None and theta is None:
            self.solve_for_gate_parameters()

        if theta is None:
            theta = self.gst_parameters 
        else:
            self.gst_parameters = theta

        uncertainties = np.zeros_like(theta)
        if method == 'hessian':
            # L-BFGS-B stores an approximation to the inverse Hessian -- we use this for convariance estimation 
            covariance = np.array(self.solver_result.hess_inv.todense())
            num_parameters = len(theta)
    
            # Uncertainties are taken as diagonals of covariance matrix 
            uncertainties = np.sqrt(np.abs(np.diag(covariance)))
            #return uncertainties, covariance 
        else: 
            uncertainties, bootstrapped_thetas = self.bootstrap_uncertainties()

        # Return a dictionary containing a dictionary for each model (prep, gate 1, gate 2, etc. , measure) 
        uncertainty_results = {}
        # For gates: 
        for gate in self.gate_set:
            gate_model = self.gate_models[gate.name]
            gate_model_sig = inspect.signature(gate_model)
            parameter_names = list(gate_model_sig.parameters.keys())  
            parameter_uncertainties = uncertainties[self.gst_parameter_indices[gate.name]]
            # Package up parameter names and uncertainty values: 
            uncertainty_results[gate.name] = dict(zip(parameter_names, parameter_uncertainties)) 

        # For SPAM: 
        prep_model = self.prep_state_model
        prep_model_sig = inspect.signature(prep_model)
        parameter_names = list(prep_model_sig.parameters.keys())
        prep_param_values = uncertainties[self.gst_parameter_indices['prep']]
        uncertainty_results['Prep'] = dict(zip(parameter_names, prep_param_values)) 

        # Currently only the independent measurement parameters are returned; TODO: generalize as much as possible  
        #model_sig = inspect.signature(model)
        #parameter_names = ['Measurement'] 
        meas_param_values = uncertainties[self.gst_parameter_indices['measurement']]
        #uncertainty_results['Measurement'] = dict(zip(['measure'], meas_param_values)) 
        uncertainty_results['Measurement'] = meas_param_values 

        return uncertainty_results              

    def bootstrap_uncertainties(self, N_bootstrap: int=50):
        """ Bootstrapping for parameter uncertainties: Sample data from the fitted model and re-fit, computing 
                parameter spread. N_bootstrap is the number of resamplings. """

        theta_best = self.gst_parameters.copy()
        bootstrap_thetas = np.zeros((N_bootstrap, len(theta_best)))

        circuit_probabilities = []
        for circ in self.parsed_circuits:
            probs = self._predict_probabilities(circ, theta_best)
            counts = circ.measurement_data.total_counts     # TODO: generalize to t-dependent data 
            circuit_probabilities.append((probs, counts))
        
        for b in range(N_bootstrap):
            for circ, (probs, total_counts) in zip(self.parsed_circuits, circuit_probabilities):
                outcomes = list(probs.keys()) 
                outcome_probs = [probs[outcome] for outcome in outcomes]
                outcome_counts = np.random.multinomial(total_counts, outcome_probs) 
                circ.measurement_data = CircuitData.from_counts(dict(zip(outcomes, outcome_counts)))
                
            # Re-run the MLE analysis to find best fit:
            self.gst_parameters = theta_best.copy()
            self.solve_for_gate_parameters(parameters_guess = theta_best.copy())   # sets self.gst_parameters to optimal  
            bootstrap_thetas[b] = self.gst_parameters

        # Restore original data/fit
        self.gst_parameters = theta_best
            
        # Compute uncertainties as standard deviation of the best fits
        uncertainties = np.std(bootstrap_thetas, axis=0) 
        return uncertainties, bootstrap_thetas
            
