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
from ionsim.custom_types import Vector
from ionsim.io import *

class GateSetTomography(): # or GST() or GST_Base() if we plan to have child classes.
    def __init__(self, basis: StandardBasis, prep_state_model: dict[Callable], POVM_effect_models: dict[str, Callable], parsed_circuits: list[ParsedCircuit], gate_mappings: dict[str, Callable]): 
        """ Class for performing quantum gate set tomography (GST) with trapped ions or neutral atoms. 
    
            Member variables include:
                - Basis where the quantum processes (gates), state, and measurement will live. 
                - prep state: rho_0, representing an ideal state prepared natively. 
                - POVM_measurement_effect models: is a dictionary of constrained measurement effect models :['0' : E0(params)] or ['00' : E0(params), '01' : E1(params), ...] for N = 2 
                - parsed_circuits is a list of Parsed GST Circuits that contain circuit information and measurement information.
                #- gate model factory is a function that takes a gate name and qubit tuple and returns an IonSim Gate object, which holds a process matrix (gate) function. 
                - gate_mappings represents a dictionary that maps GST gate names to IonSim model names, specified by the user.  
                - gst_parameters: a 1D numpy array of gate parameters.  
    
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

        # Test model fxn evaluations 
 #        gate_model = list(self.gate_models.values())[1]
 #
 #        print(list(self.gate_models.keys())[1])
 #        print(gate_model(0.1, 0.25))
 #        print()
 #        print(gate_model(0.33, 0.50))
        #sys.exit(0)

        #TODO: Algorithm for optimizing stepwise by circuit depth.  

        self.solver_result = None # initialize to None  



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


    def _build_parameter_organization(self) -> (dict[str, slice], int):
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
        # Gate model is a callable that takes in the parameter vector and returns a process matrix  

        # Retrieve a gate model for each gate and its parameter values  
        for gate in circ.expanded_gates:
            gate_model = self.gate_models[gate.name]
            gate_parameters = theta[self.gst_parameter_indices[gate.name]]
            # Accumulate the map:
            quantum_map = gate_model(*gate_parameters) @ quantum_map 

        mapped_state = quantum_map @ rho_supervector

        outcome_probabilities = {}
        probability_TOL = 1E-12
        for label, E in M_effects.items():
            outcome_probabilities[label] = np.real(E.dot(mapped_state))
            outcome_probabilities[label] = np.clip(outcome_probabilities[label], probability_TOL, 1. - probability_TOL)             

        return outcome_probabilities
        


 #    def _build_process_matrix_cache(self, theta): 
 #        """ Evaluate each gate's process matrix function once"""
 #        process_matrix_cache = {} 
 #        # TODO: finalize gate_model data structure (DS) and values. 
 #        for gate_model in self.gate_models.values():
 #            gate_parameters = theta[self.gst_parameter_indices[gate_model.name]]
 #            process_matrix_cache[gate_model.name] = gate_model.process_matrix_function(gate_parameters) 
 #        return process_matrix_cache 
 #


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

        #self._build_process_matrix_cache()
        probability_TOL = 1E-12

        # Compute log likelihood for each GST circuit, accumulating over all GST circuits 
        for circ in self.parsed_circuits:
            #print(f"\nCircuit: {circ.unparsed_data}")
            probabilities = self._predict_probabilities(circ, theta) # don't need the PM cache? 

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


    def depth_bin(depth):
        """ Bins a circuit depth to the nearest power of 2 """
        if depth <= 1:
            return 1
        return 2**(math.ceil(math.log2(depth)))

    def _group_circuits_by_depth(self):
        """ Groups the GST circuit by depth, required for staged MLE """ 
        groups = {} # dictionary to store list of circuits at each depth L 
        for circ in self.parsed_circuits:
            L = depth_bin(circ.depth)
            if L not in groups:
                groups[L] = [] 
            groups[L].append(circ)
        return groups


    def save_nll_data(self):
        print(f"LL evals: {self.LL_eval}")
        print(f"len(nll_data): {len(self.nll_data)})")
        if self.nll_data : 
            np.savetxt('negative_log_likelihood.dat', np.column_stack([np.array(range(0, self.LL_eval)), np.array(self.nll_data)]), header = 'Iteration Neg_Log_Likelihood')
        else:
            raise ValueError(f"No log likelihood data is stored.")


    def print_parameters(self):
        # Prep, measure, then gate parameters: 
        print("\n --- Printing parameter values --- ")
        prep_params = self.gst_parameters[self.gst_parameter_indices["prep"]] # d^2 - 1 column vector  
        print(f"Prep state parameters: {prep_params}")

        measurement_params = self.gst_parameters[self.gst_parameter_indices["measurement"]] # d^2 - 1 column vector  
        print(f"\nMeasurement effect parameters: {measurement_params}")

        for gate in self.gate_set:
            gate_parameters = self.gst_parameters[self.gst_parameter_indices[gate.name]]
            print(f"\n Gate {gate.name} parameters: {gate_parameters}")

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

                max[ Likelihood( {G} | data) ] over parameter set Theta.

            - Returns either a dictionary of parameters (name, value) or a 1D array of values.

        """
        print(f"\n -- Solver for gate parameters in GST using {solver} --- ")
        if solver == 'MLE':
            # Maximum likelihood estimation.
            # Specify initial guess. 
            if parameters_guess is None:
                theta_0 = self.gst_parameters.copy() 
            else:
                theta_0 = parameters_guess
            print(f"Initial parameters: {theta_0}")

            # TODO: Provide bounds for parameters if using interpolated gates 
            # GST expeirment circuits and outcome data are imbedded in log likelihood function evaluations. 
            solver_result = opt.minimize(fun = lambda params: -self.log_likelihood(params), x0 = theta_0, method = 'L-BFGS-B') # TODO consider adding parameter bounds in any case  
            self.solver_result = solver_result
            self.gst_parameters = solver_result.x
            #self.save_nll_data()
            #self.print_parameters()
            return solver_result
            
        elif solver == 'linear':
            # Solve matrix Ax = b problem: Frequencies = A_matrix @ Gate_parameters  
            # Check that gram matrix A_{m,s} = <M | C_{m} C_{s} | rho> is invertible.            
            # Compute x = A \ b
            raise IonSimError('Linear GST is not yet programmed into IonSim.')
            return None 
        else:
            raise IonSimError('Invalid solver input.')



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
            write_results_to_file(gate.name + '.hdf5', results_to_write) 

            
    ### Functions for gate set error metrics ### 
    def compute_gate_set_process_infidelity(self, ideal_gate_set: dict) -> float:
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
            parameter_values = self.gst_parameters[self.gst_parameter_indices[gate.name]] # names and values share same sorted order  

            process_matrix = gate_process_matrix_function(*parameter_values)
            gate_model = Gate(self.basis, process_matrix)
            process_infidelity = 1. - gate_model.compute_process_fidelity(ideal_gate)
            gate_errors[gate.name] = process_infidelity
            gate_infidelity += process_infidelity
                        

        # Compute least-square difference for SPAM
        # prep state: 
        ideal_prep_state = ideal_gate_set['prep'].supervector  
        modeled_prep_state = self.get_prep_state(self.gst_parameters) 
        # Trace distance: sqrt(sum([rho_ideal[i] - rho_actual[i]]^2))
        prep_error = np.sqrt(np.sum((modeled_prep_state - ideal_prep_state)**2)) 

        # POVMs 
        ideal_POVMs = ideal_gate_set['POVM']  
        POVMs = self.get_measurement_effects(self.gst_parameters)
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
        return gate_infidelity + measurement_error + prep_error 


    ### Functions for parameter uncertainty estimation ### 
    def estimate_parameter_uncertainties(self, theta: Vector | None=None, method: str='bootstrap') -> Vector:
        """ Computes uncertainties of each parameter from the Hessian of the log-likelihood at the MLE solution."""
        if self.solver_result is None:
            self.solve_for_gate_parameters()

        if theta is None:
            theta = self.gst_parameters 

        if method == 'hessian':
            # L-BFGS-B stores an approximation to the inverse Hessian -- we use this for convariance estimation 
            covariance = np.array(self.solver_result.hess_inv.todense())
            num_parameters = len(theta)
    
            # Uncertainties are taken as diagonals of covariance matrix 
            uncertainties = np.sqrt(np.abs(np.diag(covariance)))
            return uncertainties, covariance 
        else: # bootstrapping
            return self.bootstrap_uncertainties()

    def bootstrap_uncertainties(self, N_bootstrap=100):
        """ Bootstrapping for parameter uncertainties: Sample data from the fitted model and re-fit, computing 
                parameter spread. N_bootstrap is the number of resamplings. """
        if self.solver_result is None:
            self.solve_for_gate_parameters()

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
            self.solve_for_gate_parameters()   # sets self.gst_parameters to optimal  
            bootstrap_thetas[b] = self.gst_parameters

        # Restore original data/fit
        self.gst_parameters = theta_best
            
        # Compute uncertainties as standard deviation of the best fits
        uncertainties = np.std(bootstrap_thetas, axis=0) 
        return uncertainties, bootstrap_thetas
            

# Could build a N-dimensional process matrix by running simulations for each of the N-dimensional parameters, store them. --> hdf5 files
    # - save the raw hdf5 simulation data (process matrix at each error parameter value) 
    # - GST will load the process matrix data, interpolate with it using MLE. cref. one of the examples  

        # - would need to load up one of these for each gate in the gate set

    # - GST could make calls to other parts of IonSim to run needed simulations. For constructing the required process matrices to do GST. 
        # - if you gave it a gate set, the class could then do those simulations to generate the process matrices needed for GST.   
