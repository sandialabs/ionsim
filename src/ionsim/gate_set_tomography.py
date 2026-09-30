import numpy as np
from pathlib import Path

import scipy.stats as stats 
import scipy.optimize as opt 
from typing import Callable
import inspect
import sys
import math 
import warnings
from scipy import stats

from ionsim.process import Gate, Circuit
from ionsim.basis import StandardBasis
from ionsim.named_operators import Pauli, Unitary
from ionsim.gst_circuit_parser import *
from ionsim.gst_circuit_parser import GstGate, GstCircuit, CircuitData, gate_from_label, canonical_gate_label, IDLE_ALIASES
from ionsim.custom_math import matrix_AYB_multiply_to_superoperator 
from ionsim.ionsim_error import IonSimError
from ionsim.custom_types import Vector, Matrix
from ionsim.gst_circuit_planner import GSTCircuitPlanner
from ionsim.gst_parameters import GstModelParameters
from ionsim.state import State
from ionsim.io import *
from ionsim.config import NUMERICAL_EQUIVALENCE_THRESHOLD 

def depth_bin(depth):
    """ Bins a circuit depth to the nearest power of 2 """
    if depth <= 1.:
        return 1
    return int(2**(np.ceil(np.log2(depth))))

class GateSetTomography(): # or GST() or GST_Base() if we plan to have child classes.
    def __init__(self, basis: StandardBasis, prep_state_model: Callable, POVM_effect_models: Callable, parsed_circuits: list[GstCircuit],
                    gate_models: dict[str, Callable], circuit_design: GSTCircuitPlanner | None=None,
                    ideal_gate_set: dict | None=None, verbose: bool=False, *, parameters: GstModelParameters | None=None):
        """ Class for performing quantum gate set tomography (GST) with trapped ions or neutral atoms.

            Arguments:
                - basis: Basis where the quantum processes (gates), state, and measurement will live.
                - prep_state_model: callable returning the prep state supervector rho_0(params).
                - POVM_effect_models: callable returning a dictionary of measurement effects, {'0': E0, '1': E1} or {'00': E00, ...}.
                - parsed_circuits: list of GstCircuits (e.g. from parse_gst_circuit_file) with circuit and measurement information.
                - gate_models: dictionary mapping gate labels to process-matrix functions, e.g. {'Gxpi2:0': model, 'MS:0:1': model, 'idle': model}.
                    Gates are specified by string with their qubit argument(s); the idle gate may be written 'idle' or '[]'.

            Optional arguments:
                - circuit_design: a circuit planner object. This is not required for doing MLE but is required for linear GST.
                - ideal_gate_set: dictionary mapping gate labels, 'prep', and 'POVM' to ideal process matrices, prep State, and POVM effects.

            Model parameters:
                Every argument of the prep, POVM, and gate model functions is a GST parameter. By default, each parameter is independent,
                unbounded, and has an initial guess of zero. Use specify_parameter() to set initial guesses, bounds, and to share
                parameters among models, e.g.

                    gst.specify_parameter("amplitude_noise_strength", model="shared", guess=0.01, bounds=(1e-4, 10.))
                    gst.specify_parameter("phi_error", model="MS:0:1", guess=0., bounds=(0., np.pi/16))

                The parameter vector is organized lazily (when first needed, e.g. by a solver), so parameters may
                be specified in any order after construction.

                Parameters are held in a GstModelParameters object (self.parameters). To share one specification with a circuit
                planner (for Fisher information / sensitivity analysis), build it once and use GateSetTomography.from_model_parameters().
        """

        if verbose:
            print(f"\n\n --- IonSim Gate Set Tomography Analysis --- ")

        self.basis = basis
        # Models and their parameter specification (shared with other GST components, e.g. a circuit planner)
        if parameters is None:
            parameters = GstModelParameters(prep_state_model, POVM_effect_models, gate_models)
        elif not isinstance(parameters, GstModelParameters):
            raise TypeError(f"parameters must be a GstModelParameters object; received {type(parameters).__name__}.")
        elif any(model is not None for model in (prep_state_model, POVM_effect_models, gate_models)):
            raise ValueError("Pass either the model functions or a GstModelParameters object, not both.")
        self.parameters = parameters
        # Unpack |rho>> and <<E| or <<M|
        self.prep_state_model = parameters.prep_state_model
        self.POVM_effect_models = parameters.POVM_effect_models

        # Parse circuits list contanining GST circuit sequences and correpsonding data (observations)
        self.parsed_circuits = parsed_circuits

        # Dimensionality of Hilbert and Hilbert-Schmidt spaces:
        self.d = len(basis.states)
        self.d2 = self.d * self.d

        # 1. Get all unique gates in the gate set
        self.gate_set = set()  # gate_set contains GstGate objects
        for circ in self.parsed_circuits:
            for g in circ.expanded_gates:
                self.gate_set.add(g)

        # 2. Gate models, keyed internally by GstGate (users key them by string label)
        self.gate_models = parameters.gate_models

        missing = self.gate_set - set(self.gate_models.keys())
        if missing:
            missing_strs = sorted(g.label for g in missing)
            raise ValueError(f"Gates found in circuit data but no model, missing models for {missing_strs}")

        if verbose:
            print(f"Gate set tomography on gate set: {sorted(g.label for g in self.gate_set)}")

        # 3. Parameters: the layout (indices, bounds, initial guesses) is organized lazily by self.parameters. The current parameter
        #    vector is reset whenever the specification changes (tracked with the parameter object's version).
        self._model_parameter_names = parameters._model_parameter_names
        self._gst_parameters = None
        self._parameters_version = None

        # 4. Debugging / diagnostics
        self.LL_eval = 0
        self.nll_data = []

        # Set up cached parameters and process matrices
        self.cached_theta = None
        self.process_matrix_cache = None

        # Cache metadata for fast likelihood evaluation.
        # Keep a stable outcome ordering so all vectorized probability operations
        # use consistent indices across circuits and evaluations.
        # TODO: Need to generalize this for time-dep. GST
        self.outcome_labels = tuple(parsed_circuits[0].measurement_data.counts.keys())
        self.outcome_to_index = {label: i for i, label in enumerate(self.outcome_labels)}
        self._likelihood_circuit_cache = {}

        self.ideal_gate_set = None
        if ideal_gate_set is not None:
            self.ideal_gate_set = self._normalize_ideal_gate_set(ideal_gate_set)

        # Verbose logging in objective functions is expensive in iterative solvers.
        self.verbose = verbose

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
        self._initialize_likelihood_circuit_cache()
        self.parameters_guess = None


    ### Model / gate label helpers ###
    @classmethod
    def from_model_parameters(cls, basis: StandardBasis, parsed_circuits: list[GstCircuit], parameters: GstModelParameters,
                    circuit_design: GSTCircuitPlanner | None=None, ideal_gate_set: dict | None=None, verbose: bool=False):
        """ Construct the analysis from a GstModelParameters object (models and parameter specification), e.g. one that is also
            used by a GSTCircuitPlanner for Fisher information / sensitivity analysis. """
        return cls(basis, None, None, parsed_circuits, None, circuit_design=circuit_design, ideal_gate_set=ideal_gate_set,
                   verbose=verbose, parameters=parameters)


    ### Model / gate label helpers ###
    def _model_label(self, model: str) -> str:
        """ Returns the canonical model label for 'prep', 'POVM', or a gate label (e.g. '[]' -> 'idle'). """
        return self.parameters.model_label(model)

    @property
    def model_labels(self) -> list[str]:
        """ Labels of every model with parameters: 'prep', 'POVM', and each gate label. """
        return self.parameters.model_labels

    @property
    def model_parameter_names(self) -> dict[str, list[str]]:
        """ Argument names of each model, keyed by model label """
        return self.parameters.model_parameter_names

    def _normalize_ideal_gate_set(self, ideal_gate_set: dict) -> dict:
        """ Converts a user ideal gate set keyed by gate label strings (plus 'prep' and 'POVM') to internal GstGate keys. """
        if not isinstance(ideal_gate_set, dict):
            raise TypeError(f"Ideal gate set should be specified as a dictionary mapping gates, preps, and POVMs to ideal process matrices, supervectors, and superbra gate set elements. Received type: {type(ideal_gate_set)} ")
        internal = {}
        for key, value in ideal_gate_set.items():
            if key in ('prep', 'POVM'):
                internal[key] = value
            else:
                gate = gate_from_label(key)
                if gate in internal:
                    raise ValueError(f"Gate {key!r} appears more than once in the ideal gate set.")
                internal[gate] = value
        return internal


    ### Parameter specification (delegated to self.parameters, a GstModelParameters object) ###
    def specify_parameter(self, name: str, model: str | list[str], guess: float | None=None, bounds: tuple | None=None):
        """ Specify the initial guess, bounds, and/or sharing of a model parameter; see GstModelParameters.specify_parameter().

            Examples:
                gst.specify_parameter("SPAM_error_probability", model="shared", guess=1e-4, bounds=(0., 1.))
                gst.specify_parameter("phi_error", model="MS:0:1", guess=0., bounds=(0., np.pi/16))
        """
        self.parameters.specify_parameter(name, model, guess=guess, bounds=bounds)

    def _ensure_parameter_layout(self) -> dict:
        """ Returns the parameter layout, resetting the current parameter vector and caches if the specification changed. """
        layout = self.parameters.layout
        if self._parameters_version != self.parameters.version:
            self._parameters_version = self.parameters.version
            self._gst_parameters = layout['initial_guess'].copy()
            self.cached_theta = None
            self.process_matrix_cache = None
        return layout

    @property
    def gst_parameter_indices(self) -> dict[str, list[int]]:
        """ Indices into the parameter vector for each model, keyed by model label ('prep', 'POVM', 'Gxpi2:0', ...). """
        return self._ensure_parameter_layout()['indices_by_model']

    @property
    def shared_indices(self) -> dict[str, int]:
        """ Index into the parameter vector for each shared parameter, keyed by shared parameter name. """
        return self._ensure_parameter_layout()['shared_indices']

    @property
    def num_gst_parameters(self) -> int:
        return len(self._ensure_parameter_layout()['names'])

    @property
    def num_parameters(self) -> int:
        return self.num_gst_parameters

    @property
    def parameter_names(self) -> list[str]:
        """ Returns parameter names in the order of the internal parameter vector theta, e.g. 'shared:SPAM_error_probability', 'MS:0:1.phi_error'. """
        self._ensure_parameter_layout()
        return self.parameters.parameter_names

    @property
    def parameter_bounds(self) -> list[tuple[float | None, float | None]] | None:
        """ (lower, upper) bounds for each parameter in theta order, or None if every parameter is unbounded. """
        self._ensure_parameter_layout()
        return self.parameters.parameter_bounds

    @property
    def parameter_initial_guess(self) -> Vector:
        """ Initial guess for the parameter vector, built from specify_parameter() guesses (0 by default). """
        self._ensure_parameter_layout()
        return self.parameters.initial_guess

    @property
    def gst_parameters(self) -> Vector:
        """ Current parameter vector (the initial guess until a solver has run). """
        self._ensure_parameter_layout()
        return self._gst_parameters

    @gst_parameters.setter
    def gst_parameters(self, theta: Vector):
        self._ensure_parameter_layout()
        theta = np.asarray(theta)
        if theta.shape != (self.num_gst_parameters,):
            raise ValueError(f"Parameter vector must have shape ({self.num_gst_parameters},); received {theta.shape}.")
        self._gst_parameters = theta

    def print_parameter_layout(self):
        """ Prints each entry of the parameter vector with its initial guess and bounds. """
        self.parameters.print_layout()

    def _normalize_parameter_name(self, key: str) -> str:
        return self.parameters.normalize_parameter_name(key)

    def build_theta_from_dict(self, param_values: dict, default_value: float = 0., base: Vector | None=None) -> Vector:
        """ Builds a theta vector from a dictionary of parameter names to values; see GstModelParameters.build_theta_from_dict(). """
        self._ensure_parameter_layout()
        return self.parameters.build_theta_from_dict(param_values, default_value=default_value, base=base)

    def get_parameters(self, theta: Vector, key: str | GstGate):
        """ Retrieve parameters for any model by key ('prep', 'POVM', or a gate label) from theta vector """
        return theta[self.gst_parameter_indices[key.label if isinstance(key, GstGate) else self._model_label(key)]]

    def get_parameter_index(self, name: str, model: str) -> int:
        """ Index in the parameter vector of parameter `name` of `model` ('shared', 'prep', 'POVM', or a gate label). """
        self._ensure_parameter_layout()
        return self.parameters.get_parameter_index(name, model)

    def get_parameter_value(self, name: str, model: str, theta: Vector | None=None) -> float:
        """ Value of parameter `name` of `model` in theta (defaults to the current gst_parameters). """
        if theta is None:
            theta = self.gst_parameters
        return theta[self.get_parameter_index(name, model)]

    def _index_fiducials(self):
        """ Identify unique prep/measure fiducials and build lookup to get observed probabilities. """
        combined_counts = {}
        # Create keys by full circuit representation and average over duplicates (TODO: Update/change for non-Markovian GST)
        for circ in self.parsed_circuits:
            key = tuple(circ.expanded_gates)
            counts = circ.measurement_data.to_counts().copy()

            if key in combined_counts:
                for label, n in counts.items():
                    combined_counts[key][label] = combined_counts[key].get(label, 0) + n
            else:
                combined_counts[key] = counts 

        # Set up circuit -> probability dictionary 
        self.circuit_lookup = {}
        for key, counts in combined_counts.items(): 
            total_counts = sum(counts.values())
            self.circuit_lookup[key] = {outcome: count / total_counts for outcome, count in counts.items()}

    def get_prep_state(self, theta) -> Vector:
        """ Returns prep state supervector (d^2 x 1) given the parameter values theta.
            - Enforces the constraint Tr[rho] = 1.
        """
        return self.parameters.prep_state(theta)

    def get_measurement_effects(self, theta) -> dict[str, Vector]:
        """ Returns measurement effects given the parameter values theta.

            - Effects are stored in a dictionary {'outcome' : Effect_vector with superoperator d^2 x d^2 shape}
            - e.g. E_0 vector is d^2 x 1 corresponding to |0><0| (for d = 2)
            - There is a completeness constraint to enforce: sum_m E_m = identity
        """
        return self.parameters.measurement_effects(theta)

    def _initialize_likelihood_circuit_cache(self):
        """ Build static and measurement-index metadata for each circuit once. """
        # This cache stores per-circuit arrays (outcome indices, counts, shots)
        # so likelihood loops do not repeatedly parse dictionaries/lists.
        self._likelihood_circuit_cache = {}
        for circ in self.parsed_circuits:
            metadata = self._build_likelihood_circuit_metadata(circ)
            self._likelihood_circuit_cache[id(circ)] = metadata

    def _build_likelihood_circuit_metadata(self, circ: GstCircuit) -> dict:
        """ Build cached indexing data used by likelihood and chi-squared loops. """
        # Use gate names instead of GstGate objects so map-composition cache keys
        # are lightweight and hash quickly.
        #gates = tuple(gate for gate in circ.expanded_gates)
        gates = tuple(circ.expanded_gates)

        #gate_names = tuple(gate.name for gate in circ.expanded_gates)
        measurement_data = circ.measurement_data

        metadata = {
            'circ': circ,
            'gates': gates,
            'measurement_data_id': id(measurement_data),
            'has_data': measurement_data is not None,
            'has_counts': False,
            'count_indices': np.empty(0, dtype=np.int64),
            'count_values': np.empty(0, dtype=np.float64),
            'shot_indices': np.empty(0, dtype=np.int64),
            'total_counts': 0.0,
        }

        if measurement_data is None:
            return metadata

        if measurement_data.counts is not None:
            # Pre-extract non-zero counts into aligned index/value arrays for
            # vectorized dot products in the likelihood function.
            count_indices = []
            count_values = []
            for outcome, count in measurement_data.counts.items():
                #if count <= 0:
                #    continue
                if count < 0:
                    continue
                if outcome not in self.outcome_to_index:
                    raise IonSimError(f"Unexpected measurement outcome '{outcome}' in circuit data.")
                count_indices.append(self.outcome_to_index[outcome])
                count_values.append(float(count))

            metadata['has_counts'] = True
            metadata['count_indices'] = np.asarray(count_indices, dtype=np.int64)
            metadata['count_values'] = np.asarray(count_values, dtype=np.float64)
            metadata['total_counts'] = float(np.sum(metadata['count_values']))
            return metadata

        # Time-series branch: store only outcome indices (timestamps are currently
        # not used in the Markovian objective, but preserved in original data).
        shot_indices = []
        for _, outcome in measurement_data.timestamped_shots:
            if outcome not in self.outcome_to_index:
                raise IonSimError(f"Unexpected measurement outcome '{outcome}' in circuit data.")
            shot_indices.append(self.outcome_to_index[outcome])

        metadata['shot_indices'] = np.asarray(shot_indices, dtype=np.int64)
        metadata['total_counts'] = float(len(shot_indices))
        return metadata

    def _get_likelihood_circuit_metadata(self, circ: GstCircuit) -> dict:
        """ Return cached metadata; rebuild if the circuit's measurement object changed. """
        # Bootstrap and other workflows may replace circ.measurement_data, so we
        # detect that and lazily refresh only the affected cache entry.
        key = id(circ)
        measurement_data = circ.measurement_data
        measurement_data_id = id(measurement_data)

        metadata = self._likelihood_circuit_cache.get(key)
        if (metadata is None or metadata['circ'] is not circ
                or metadata['measurement_data_id'] != measurement_data_id):
            metadata = self._build_likelihood_circuit_metadata(circ)
            self._likelihood_circuit_cache[key] = metadata

        return metadata

    def _refresh_prep_and_measure_elements(self, theta: Vector) -> tuple[np.ndarray, np.ndarray]:
        """ Build theta-dependent prep/effect matrices once per objective evaluation. """
        # Prep state and measurement effects do not depend on circuit identity,
        # so compute them once and reuse for all circuits in this theta evaluation.
        rho_supervector = self.get_prep_state(theta)
        measurement_effects = self.get_measurement_effects(theta)
        effect_matrix = np.vstack([np.asarray(measurement_effects[label]) for label in self.outcome_labels])
        return rho_supervector, effect_matrix

    def _compose_quantum_map(self, gates: tuple[GstGate, ...], circuit_map_cache: dict) -> np.ndarray:
        """ Compose the circuit map once for each unique gate sequence in an evaluation. """
        # Many circuits can share the same expanded gate sequence; cache the full
        # composed map for this theta evaluation to avoid repeated matrix chains.
        quantum_map = circuit_map_cache.get(gates)
        if quantum_map is not None:
            return quantum_map

        quantum_map = np.eye(self.d2, dtype=complex)
        for gate in gates:
            quantum_map = self.process_matrix_cache[gate] @ quantum_map

        circuit_map_cache[gates] = quantum_map
        return quantum_map

    def _predict_probability_vector(self, gates: tuple[GstGate, ...], rho_supervector: Vector, effect_matrix: Matrix, 
                                        circuit_map_cache: dict) -> np.ndarray:
        """ Predict clipped outcome probabilities as a dense vector in outcome-label order. """
        # Return dense probabilities in self.outcome_labels order so downstream
        # indexing (counts/shots) is pure NumPy gather/sum math.
        quantum_map = self._compose_quantum_map(gates, circuit_map_cache)
        mapped_state = quantum_map @ rho_supervector
        probability_values = np.real(effect_matrix @ mapped_state)
        return np.clip(probability_values, NUMERICAL_EQUIVALENCE_THRESHOLD, 1. -  NUMERICAL_EQUIVALENCE_THRESHOLD)

    def _predict_probabilities(self, circ: GstCircuit, theta: Vector) -> Vector: 
        """ Predicts outcome probabilities for a GST circuit with gates parametrized by theta """
        # Compatibility helper for existing callers that still expect a dict.
        self._refresh_gate_process_matrix_cache(theta)
        rho_supervector, effect_matrix = self._refresh_prep_and_measure_elements(theta)
        metadata = self._get_likelihood_circuit_metadata(circ)
        probability_values = self._predict_probability_vector(metadata['gates'], rho_supervector, effect_matrix, circuit_map_cache={})
        outcome_probabilities = dict(zip(self.outcome_labels, probability_values))
        return outcome_probabilities
        
    def _refresh_gate_process_matrix_cache(self, theta): 
        """ Evaluate each gate's process matrix function once"""
        if (self.cached_theta is None or self.cached_theta.shape != theta.shape 
            or not np.array_equal(self.cached_theta, theta)):
            process_matrix_cache = {} 
            for gate, gate_model in self.gate_models.items():
                # Retrieve parameters for the gate model 
                gate_parameters = self.get_parameters(theta, gate) 
                # Evaluate gate model at those parameter values and store in the PM cache 
                process_matrix_cache[gate] = gate_model(*gate_parameters) # gate model returns a process matrix  

            self.cached_theta = np.array(theta, copy=True)
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
        if self.verbose:
            print(f"\nEvaluating log likelihood")
        self.LL_eval += 1 
        if self.verbose:
            print(f"Evaluation number {self.LL_eval}")
            print(f"\nParameter values: {theta}")

        # TODO: make a separate function for t-dependent parameters 
        if theta is None:
            theta = self.gst_parameters

        l_likelihood = 0.

        # Improve speed by building gate process matrices once 
        self._refresh_gate_process_matrix_cache(theta)

        # Build theta-dependent context once, then reuse cached circuit metadata
        # and map compositions across the full circuit set.
        rho_supervector, effect_matrix = self._refresh_prep_and_measure_elements(theta)
        circuit_map_cache = {}

        # Compute log likelihood for each GST circuit, accumulating over all GST circuits 
        for circ in self.parsed_circuits:
            metadata = self._get_likelihood_circuit_metadata(circ)
            if not metadata['has_data']:
                raise IonSimError("Cannot evaluate log-likelihood with circuits that have no measurement data.")

            probability_values = self._predict_probability_vector(metadata['gates'], rho_supervector, effect_matrix, circuit_map_cache)
            log_probability_values = np.log(probability_values)

            if metadata['has_counts']:
                if metadata['count_values'].size > 0:
                    assert len(metadata['count_values']) == len(log_probability_values)
                    # Weighted log-likelihood contribution from count data.
                    l_likelihood += np.dot(metadata['count_values'], log_probability_values)
            else:
                raise NotImplementedError(f"Time-dependent GST is not available in this version of IonSim.") 
                # Time-series data: each shot contributes one log-probability term.
                if metadata['shot_indices'].size > 0:
                    l_likelihood += np.sum(log_probability_values[metadata['shot_indices']])

        if self.verbose:
            print(f"Negative log likelihood: {-l_likelihood}")
        self.nll_data.append(-l_likelihood) 
        return l_likelihood


    
    def chi_squared(self, theta: Vector | None=None, theta_function=None) -> float:
        """ chi^2 estimate for least-squares error between observed frequencies and circuit probabilities. """ 
        chi_squared = 0.

        if theta is None:
            theta = self.gst_parameters

        # Improve speed by building gate process matrices once 
        self._refresh_gate_process_matrix_cache(theta)

        # Reuse the same probability context/circuit-map cache strategy as in
        # log_likelihood for consistent performance behavior.
        rho_supervector, effect_matrix = self._refresh_prep_and_measure_elements(theta)
        circuit_map_cache = {}

        # Compute log likelihood for each GST circuit, accumulating over all GST circuits 
        for circ in self.parsed_circuits:
            metadata = self._get_likelihood_circuit_metadata(circ)
            if not metadata['has_data']:
                raise IonSimError("Cannot compute chi squared with circuits that have no measurement data.")

            probability_values = self._predict_probability_vector(metadata['gates'], rho_supervector, effect_matrix,
                                        circuit_map_cache)

            if metadata['has_counts']:
                if metadata['total_counts'] > 0:
                    # Chi-squared between observed frequencies and model probs,
                    # scaled by total shots for that circuit.
                    p_values = probability_values[metadata['count_indices']]
                    frequencies = metadata['count_values'] / metadata['total_counts']
                    chi_squared += metadata['total_counts'] * np.sum(((p_values - frequencies)**2) / p_values)
            else:
                raise IonSimError(f"Computing chi squared for time-series data is not yet programmed in IonSim.")

        if self.verbose:
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
        """ Groups the GST circuit by germ power, required for staged MLE """ 
        groups = {} 
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

    def get_parameter_value_by_name(self, gate: str, parameter_name: str) -> float:
        """ Return the parameter value for a requested parameter in a gate model (gate specified by label, e.g. 'Gxpi2:0') """
        return self.get_parameter_value(parameter_name, gate)

    def get_parameter_values_by_name(self, gate: str, parameter_names: list[str]) -> dict:
        """ Return the parameter values for requested parameters in a gate model (gate specified by label, e.g. 'Gxpi2:0') """
        return {name: self.get_parameter_value(name, gate) for name in parameter_names}

    def print_parameters(self):
        # Prep, measure, then gate parameters:
        print("\n --- Printing parameter values --- ")
        prep_params = self.get_parameters(self.gst_parameters, "prep")
        print(f"Prep state parameters: {dict(zip(self._model_parameter_names['prep'], prep_params))}")

        measure_params = self.get_parameters(self.gst_parameters, "POVM")
        print(f"\nMeasurement model parameters: {dict(zip(self._model_parameter_names['POVM'], measure_params))}")

        for gate in sorted(self.gate_set, key=lambda g: g.label):
            parameter_values = self.get_parameters(self.gst_parameters, gate)
            # Package parameter names, values
            gate_results = dict(zip(self._model_parameter_names[gate.label], parameter_values))
            print(f"\n Gate {gate} parameters: {gate_results}")

        return self.gst_parameters

    def print_state_and_POVMs(self):
        """ Output state supervector and measurement effects """ 
        rho = self.get_prep_state(self.gst_parameters) 
        M_effects = self.get_measurement_effects(self.gst_parameters)

        print(f"\nPrep state supervector: {rho}")
        for label, effect in M_effects.items():
            print(f"\nMeasurement effect {label} vectors: {effect}")


    def _resolve_initial_guess(self, parameters_guess: Vector | dict | str | None) -> Vector:
        """ Interface to parse the initial guess into a vector of initial values for the solvers.

            Used by linear_solve_for_gate_parameters(), mle_solve_for_gate_parameters(), and staged_mle_solve_for_gate_parameters().

            - None: use the guesses given with specify_parameter() (0 for unspecified parameters).
            - 'lgst': fit the gate set models to linear GST estimates and use those parameters (requires a circuit design).
            - dict: parameter names to values (see build_theta_from_dict); unlisted parameters use their specified guesses.
            - list / array: the full parameter vector, in the order of parameter_names.
        """
        if parameters_guess is None:
            theta_0 = self.parameter_initial_guess
        elif isinstance(parameters_guess, str):
            if parameters_guess.lower() not in ('lgst', 'linear'):
                raise ValueError(f"Unknown initial guess option {parameters_guess!r}; use None, 'lgst', a dictionary, or a parameter vector.")
            self.run_linear_gst(self.ideal_gate_set)
            theta_0 = self.parameters_from_lgst_results().copy()
        elif isinstance(parameters_guess, dict):
            theta_0 = self.build_theta_from_dict(parameters_guess, base=self.parameter_initial_guess)
        elif isinstance(parameters_guess, (list, tuple, np.ndarray)):
            theta_0 = np.array(parameters_guess, dtype=float, copy=True)
            if theta_0.shape != (self.num_gst_parameters,):
                raise ValueError(f"Initial parameter vector must have length {self.num_gst_parameters} (see parameter_names); received shape {theta_0.shape}.")
        else:
            raise TypeError(f"Parameter initial guess should be None, 'lgst', a dictionary, or a list/array. Received: {type(parameters_guess)}")

        self.parameters_guess = theta_0
        return theta_0

    def _initial_parameters(self, parameters_guess: Vector | dict | str | None, solver_name: str) -> Vector:
        """ Resolve the initial guess for a solver and report it when verbose """
        theta_0 = self._resolve_initial_guess(parameters_guess)
        if self.verbose:
            print(f"\n -- Solving for gate parameters in GST using {solver_name} --- ")
            print(f"Initial parameters: {dict(zip(self.parameter_names, theta_0))}")
        return theta_0

    def linear_solve_for_gate_parameters(self, parameters_guess: Vector | dict | None=None) -> Vector:
        """ Linear GST (LGST): estimates the gate set from the fiducial circuits by linear inversion (requires a circuit design
            with informationally complete fiducials), then fits the model parameters to those estimates.

            - parameters_guess: starting point for the fit to the LGST estimates when parameters are shared among models
                (vector or dictionary; default: the guesses from specify_parameter()). With no shared parameters, each model is
                fit independently and the guess is not used.

            Returns the parameter vector (also stored in self.gst_parameters); the LGST estimates are in self.lgst_results.
        """
        if isinstance(parameters_guess, str):
            raise ValueError("linear_solve_for_gate_parameters does not accept a string guess; pass None, a dictionary, or a parameter vector.")
        theta_0 = self._initial_parameters(parameters_guess, 'linear GST')
        self.solver_result = self.run_linear_gst(self.ideal_gate_set)
        self.parameters_from_lgst_results(theta_0)
        return self.gst_parameters

    def mle_solve_for_gate_parameters(self, parameters_guess: Vector | dict | str | None=None, **minimize_kwargs):
        """ Maximum likelihood estimation (MLE): finds the parameters that maximize the likelihood of the data over all circuits,

                max[ Likelihood( {G} | data) ] over parameter set theta,

            using scipy.optimize.minimize with method L-BFGS-B and the parameter bounds from specify_parameter().

            - parameters_guess: initial guess. None (default) uses the guesses from specify_parameter(); 'lgst' seeds the solver
                from a linear GST fit; a dictionary or vector gives the values directly.
            - minimize_kwargs: passed to scipy.optimize.minimize, e.g. options = {'maxiter': 500}.

            Returns the scipy OptimizeResult (parameters in result.x, also stored in self.gst_parameters).
        """
        theta_0 = self._initial_parameters(parameters_guess, 'MLE')
        # GST experiment circuits and outcome data are embedded in log likelihood function evaluations.
        solver_result = opt.minimize(fun = lambda params: -self.log_likelihood(params), x0 = theta_0, method = 'L-BFGS-B',
                                     bounds = self.parameter_bounds, **minimize_kwargs)
        self.solver_result = solver_result
        self.gst_parameters = solver_result.x
        return solver_result

    def staged_mle_solve_for_gate_parameters(self, parameters_guess: Vector | dict | str | None=None, organize_circuits_by_germ_power: bool=True,
                                             **minimize_kwargs):
        """ Staged MLE: maximum likelihood estimation on cumulative batches of circuits of increasing depth, each stage starting
            from the previous stage's estimate. This can help avoid local optima for long circuits.

            - parameters_guess: initial guess for the first stage (as for mle_solve_for_gate_parameters, including 'lgst').
            - organize_circuits_by_germ_power: stage by germ power p (default) or by base circuit depth L.
            - minimize_kwargs: passed to scipy.optimize.minimize at every stage, e.g. options = {'maxiter': 500}.

            Returns the final stage's scipy OptimizeResult (parameters in result.x, also stored in self.gst_parameters). The
            estimates from every stage are in self.results_by_stage, keyed by depth.
        """
        theta_0 = self._initial_parameters(parameters_guess, 'staged MLE')
        self.solver_result, self.results_by_stage = self.staged_objective_minimization(theta_0, method = 'L-BFGS-B', bounds = self.parameter_bounds,
                                organize_circuits_by_germ_power = organize_circuits_by_germ_power, **minimize_kwargs)
        self.gst_parameters = self.solver_result.x
        return self.solver_result

    def _build_probability_matrix(self, target_gate: GstGate | None=None, outcome: str | None=None):
        """ Builds the d^2 x d^2 matrix of observed probabilities 
            for a gate or empty gate (corresponding to the Gram Matrix).

            M[i,j] = p(outcome | measure_fid_i x gate x prep_fid_j ) 
        """
        outcomes = list(self.outcome_labels)
        if outcome is None:
            outcome = outcomes[0]

        N_prep_circuits = len(self.prep_fiducials)
        N_measure_circuits = len(self.measure_fiducials)

        # Construct matrix using lookup table of circuit outcomes for LGST 
        M = np.zeros((N_measure_circuits, N_prep_circuits))

        target_list = [target_gate] if target_gate else []
        for j, prep_fid in enumerate(self.prep_fiducials):
            for i, measure_fid in enumerate(self.measure_fiducials):
                #key = (prep_fid, gate, 1, measure_fid)
                key = tuple(list(prep_fid) + target_list + list(measure_fid)) 
                if key in self.circuit_lookup:
                    M[i,j] = self.circuit_lookup[key][outcome]
                else:
                    print(f"Attempted key: {key}")
                    raise ValueError(f"Missing LGST circuit: prep = {prep_fid}" + 
                        f", gate = {target_list}, measure = {measure_fid}")
        return M
        
        
    def run_linear_gst(self, ideal_gate_set: dict | None=None):
        """ Function to estimate gate set parameters using linear matrix inversion """
        # Method follows approach from Neilsen et al. "Gate Set Tomography", Quantum 2021. 
        # 1. Build the Gram matrix: <<F_i|F_j>>
        if self.verbose:
            print(f"\n --- Running linear GST ---")
        gram_matrix = self._build_probability_matrix(target_gate = None)

        gram_matrix_det = np.linalg.det(gram_matrix)
        if np.abs(gram_matrix_det) < 1E-12:
            raise ValueError(f"Gram matrix is not invertible, determinant = {gram_matrix_det}") 
        
        # 2. Compute SVD to get projector to linear-independent subspace  
        U, S, Vh = np.linalg.svd(gram_matrix)

        if len(S) < self.d2:
            raise ValueError(f"Gram matrix is not informationally complete. It has rank {len(S)} instead of {self.d2}.")

        # Projector onto k = d^2 top right singular vectors  
        Pi = Vh[:self.d2, :]

        # Check that the fiducials are informationally complete by checking singular values: 
        TOL = 1E-10
        N_significant_vals = np.sum(S > TOL)
        if N_significant_vals < self.d2:
            raise ValueError(f" Fiducials are not informationally complete. Only {N_significant_vals} singular values instead of {self.d2}.")

        # Check separation between complete and excess subspaces
        if len(S) > self.d2:
            ratio = S[self.d2]/S[self.d2 - 1]  # should be small
            if ratio > 0.1:
                print(f"WARNING: There is weak separation between signal and noise subspace. Ratio = {ratio}")

        # Check condition number of d^2 subspace:
        cond = S[0] / S[self.d2 - 1] 
        if cond > 100:
            print(f"WARNING: Poorly conditioned linear GST. LGST estimates may be noisy. Condition number = {cond}")

        # Decomposition of Gram matrix = AB, where A is measurement matrix and B is prep matrix 
        #   See Section 3. of "Gate Set Tomography" published in Quantum, 2021. 
        #   Gram = AB (fiducial measure @ fiducial prep); decompose B = B_0 Pi, B_0 ideal gauge  
        # TODO: Standardize way for user to specify target/ideal prep state 
        target_state = ideal_gate_set.get('prep') if ideal_gate_set is not None else None
        if target_state is not None:
            N_prep = len(self.prep_fiducials)
            # B_ideal contains all fiducial prep states as its columns 
            B_ideal = np.zeros((self.d2, N_prep), dtype=complex)
            
            for j, prep_fid in enumerate(self.prep_fiducials):
                state = target_state.supervector.copy()
                for gate in prep_fid:
                    state = ideal_gate_set[gate] @ state
                B_ideal[:, j] = state

            # Project onto Pi subspace, Pi Pi^T is identity since rows of Pi are orthonormal 
            B0 = B_ideal @ Pi.conj().T
        else:
            warnings.warn(f"No ideal gate set is specified in the GST analyzer. Linear GST results may not correspond with a desired gauge.")            
            B0 = np.eye(self.d2, dtype=complex)

        # Compute gate process matrix estimates via the following formula (Neilsen, 2021):
        # G_k = B0 (Pi Gram^T Gram Pi^T)^{-1} (Pi Gram^T P_k Pi^T) B0^{-1}
        # Key: "G" = gram matrix, "T" = transpose, "P" = Pi matrix
        PGT = Pi @ gram_matrix.T
        inv_PGTGPT = np.linalg.inv(PGT @ gram_matrix @ Pi.T)
        B0_inv = np.linalg.inv(B0)
        matrix_prefactor = B0 @ inv_PGTGPT @ PGT 
        matrix_postfactor = Pi.T @ B0_inv

        gate_estimates = {}
        # Sorted for a reproducible ordering (set order depends on string hashing, which varies between runs)
        for gate in sorted(self.gate_set, key=lambda g: g.label):
            # Compute gate process matrix by inversion: probabilities P = A G_gate B 
            P_gate = self._build_probability_matrix(target_gate = gate)
            gate_estimates[gate] = matrix_prefactor @ P_gate @ matrix_postfactor 

        # Find which fiducial index is the empty circuit, corresponding to native prep and measure 
        empty_fid = tuple()
        prep_idx = self.prep_fiducials.index(empty_fid)
        measure_idx = self.measure_fiducials.index(empty_fid)

        # Prep state matrix B = B0 Pi 
        prep_states = B0 @ Pi 
        # Extract native prep rho_0:
        estimated_rho = prep_states[:, prep_idx]

        # Extract effects:
        # Measurement effect matrix A = Gram B+ (right pseudoinverse of B)
        measurement_effects = gram_matrix @ np.linalg.pinv(prep_states)
        estimated_effects = {}
        for outcome in self.outcome_labels:
            gram_k = self._build_probability_matrix(outcome = outcome) 
            A_k = gram_k @ Pi.conj().T @ B0_inv
            estimated_effects[outcome] = A_k[measure_idx, :]

        self.lgst_results = {'gate_estimates' : gate_estimates, 'gram_matrix' : gram_matrix, 
                        'native_prep_state' : estimated_rho, 'estimated_effects' : estimated_effects, 
                        'prep_states' : prep_states, 'measurement_effects' : measurement_effects}
        return self.lgst_results 


    def parameters_from_lgst_results(self, theta_0: Vector | None=None):
        """ Extracts the parameters vector from linear GST results.

            If shared parameters exist, fits all models jointly starting from theta_0 (default: the specified initial guess).
            If not, gate set models are fit independently.

        """
        if not hasattr(self, 'lgst_results') or self.lgst_results is None:
            self.run_linear_gst(self.ideal_gate_set)

        if self.shared_indices:
            theta = self._joint_fit_to_lgst(theta_0)
        else:
            theta = self._independent_fit_to_lgst()

        # Set internal gst parameters attribute to extracted parameters
        self.gst_parameters = theta
        return theta

    def _independent_fit_to_lgst(self):
        """ Fits independent gate set models to LGST-estimated elements (no shared model parameters). """
        # Initialize theta
        theta = np.array(self.gst_parameters, dtype=float, copy=True)

        # Extract gate parameters
        for gate, lgst_gate_matrix in self.lgst_results['gate_estimates'].items():
            # Compute the fit parameters for each gate model
            fit_parameters = self._fit_gate_model_to_lgst_estimate(gate, lgst_gate_matrix)
            theta[self.gst_parameter_indices[gate.label]] = fit_parameters.real

        # Extract SPAM parameters:
        # Native prep state         
        prep_fit_parameters = self._fit_prep_model_to_lgst_estimate(self.lgst_results['native_prep_state'])
        theta[self.gst_parameter_indices['prep']] = np.real(prep_fit_parameters)

        # Native measurement effects  
        outcome_parameters = self._fit_measurement_effect_model_to_lgst_estimate(self.lgst_results['estimated_effects'])
        theta[self.gst_parameter_indices["POVM"]] = outcome_parameters.real

        return theta

    def _joint_fit_to_lgst(self, theta_0: Vector | None=None):
        """ Fits all gate set models jointly (if there are shared parameters) """ 
        lgst_gates = self.lgst_results["gate_estimates"]
        lgst_prep = self.lgst_results["native_prep_state"]
        lgst_effects = self.lgst_results["estimated_effects"]

        def cost(theta):
            # Cost function of the gate set to fit gate set parameters from LGST estimates  
            total_cost = 0.
            # Gates
            for gate, lgst_matrix in lgst_gates.items():
                gate_params = self.get_parameters(theta, gate)
                M = self.gate_models[gate](*gate_params)
                total_cost += np.linalg.norm(M - lgst_matrix, 'fro')**2
            
            # Prep 
            prep_params = self.get_parameters(theta, "prep")
            rho = self.prep_state_model(*prep_params)
            total_cost += np.linalg.norm(rho - lgst_prep)**2
                
            # POVM 
            POVM_params = self.get_parameters(theta, "POVM")
            modeled_effects = self.POVM_effect_models(*POVM_params)
            for outcome, effect in lgst_effects.items():
                assert outcome in modeled_effects
                total_cost += np.linalg.norm(modeled_effects[outcome] - effect)**2
            return total_cost.real
            
        if theta_0 is None:
            theta_0 = self.parameter_initial_guess

        ## Note: L-BFGS-B performs significantly better here than Nelder-Mead; Nelder-Mead should not be used in this method.
        result = opt.minimize(cost, theta_0, method='L-BFGS-B', bounds=self.parameter_bounds)#'Nelder-Mead', bounds = self.parameter_bounds)
        return result.x 

        
    ## TODO: Consolidate / factor into a single "fit model to lgst" function if possible  
    def _fit_prep_model_to_lgst_estimate(self, lgst_native_prep: Vector) -> Vector:
        """ Fits a prep state model's parameters given the lgst results for the prep state. """
        prep_indices = self.gst_parameter_indices['prep']
        def cost(theta: Vector) -> float:
            # Frobenius norm of the process matrix difference bt. model and LGST-predicted
            prep_state = self.prep_state_model(*theta)
            return np.linalg.norm(prep_state - lgst_native_prep)**2

        N_parameters = len(prep_indices)
        #N_parameters = len(self.gst_parameters[prep_indices]) 
        p0 = np.zeros(N_parameters) 
        if self.parameter_bounds is not None:
            model_bounds = [self.parameter_bounds[idx] for idx in prep_indices] 
            #model_bounds = self.parameter_bounds[prep_indices]
        else:
            model_bounds = None

        result = opt.minimize(cost, p0, method='Nelder-Mead', bounds = model_bounds) 
        return result.x

    def _fit_measurement_effect_model_to_lgst_estimate(self, lgst_native_measurements: dict[str, Vector]) -> Vector:
        """ Fits a measurement effect's model's parameters given the lgst results for the prep state. """
        def cost(theta: Vector) -> float:
            # Fit measurement effects for an outcome 
            # Cost is Frobenius norm of the process matrix difference bt. model and LGST-predicted
            modeled_effect_matrix = np.vstack([np.asarray(self.get_measurement_effects(theta)[label]) for label in self.outcome_labels])
            lgst_native_measurements_matrix = np.vstack([np.asarray(lgst_native_measurements[outcome]) for outcome in self.outcome_labels])
            return np.linalg.norm(modeled_effect_matrix - lgst_native_measurements_matrix)**2

        measurement_indices = self.gst_parameter_indices["POVM"] 
        N_parameters = len(self.gst_parameters[measurement_indices]) 
        # TODO: Fix the optimization to only vary parameters for this measurement outcome 
        p0 = np.zeros(self.num_gst_parameters, dtype=complex) 

        if self.parameter_bounds is not None:
            model_bounds = self.parameter_bounds
            #model_bounds = self.parameter_bounds[measurement_indices]
        else:
            model_bounds = None

        result = opt.minimize(cost, p0, method='Nelder-Mead', bounds = model_bounds) 
        return result.x[measurement_indices]


    def _fit_gate_model_to_lgst_estimate(self, gate: GstGate, target_gate_matrix: Matrix) -> Vector:
        """ Fits a gate model's parameters given process matrix data (target_gate_matrix).

            - gate_model is as Callable that returns a process matrix  
            - uses the Frobenius norm of the process matrix difference as the cost function

        """
        gate_model = self.gate_models[gate]
        gate_indices = self.gst_parameter_indices[gate.label]
        def cost(theta: Vector) -> float:
            # Frobenius norm of the process matrix difference bt. model and LGST-predicted
            M = gate_model(*theta)
            return np.linalg.norm(M - target_gate_matrix, 'fro')**2

        N_parameters = len(gate_indices)
        p0 = np.zeros(N_parameters, dtype=complex) # zero often corresponds to ideal gate conditions 
        if self.parameter_bounds is not None:
            model_bounds = [self.parameter_bounds[idx] for idx in gate_indices]
        else:
            model_bounds = None

        result = opt.minimize(cost, p0, method='Nelder-Mead', bounds = model_bounds) 
        return result.x


    def write_results_to_file(self):
        """ Writes results of GST analysis to disk. Convention is to write HDF5 file per gate. """

        # Write results of each gate set to an hdf5 file
        for gate in self.gate_set:
            # Retrieve gate parameter names and values at optimimum; evaluate process matrix
            gate_model = self.gate_models[gate]
            parameter_names = self._model_parameter_names[gate.label]
            parameter_values = self.get_parameters(self.gst_parameters, gate) # names and values share same sorted order

            process_matrix = gate_model(*parameter_values)
            # Write parameter names, values, and process matrix evaluated at those parameter values.
            results_to_write = dict(zip(parameter_names, parameter_values))
            results_to_write[gate.label + '_process_matrix'] = process_matrix
            write_results_to_file('gst_optimal_' + gate.label + '.hdf5', results_to_write)

    def staged_objective_minimization(self, parameters_guess: Vector, method: str='L-BFGS-B', bounds: list | None=None, organize_circuits_by_germ_power: bool=True,
                                        **minimize_kwargs):
        """ Iterative MLE through batches of data taken at increasing circuit depths """ 
        print(f" --- Running Maximum likelihood estimation analysis --- ")
        if organize_circuits_by_germ_power: 
            circuit_groups = self._group_circuits_by_germ_power()
        else:
            circuit_groups = self._group_circuits_by_base_depth()
            #circuit_groups = self._group_circuits_by_depth()

        sorted_depths = sorted(circuit_groups.keys()) # keys are circuit depths 
        solver_results = {} # stores results of parameter estimation at each stage 
        if self.verbose: 
            if organize_circuits_by_germ_power: 
                print(f"--- Staged MLE with bins by germ powers (p): {sorted_depths} ") 
                for p in sorted_depths:
                    print(f"    p={p}: {len(circuit_groups[p])} circuits ")
            else:
                print(f"--- Staged MLE with bins by circuit depth (L): {sorted_depths} ") 
                for L in sorted_depths:
                    print(f"    L={L}: {len(circuit_groups[L])} circuits ")

        cumulative_circuits = []
        num_stages = len(sorted_depths)
        for stage, L in enumerate(sorted_depths):
            cumulative_circuits.extend(circuit_groups[L])

            # Store a copy of the circuits so we can re-use internal functions that use parsed_circuits attribute
            original_circuits = self.parsed_circuits
            self.parsed_circuits = cumulative_circuits
            try:
                solver_result = self._minimize_stage(stage, L, parameters_guess, cumulative_circuits, method, bounds, minimize_kwargs)
            finally:
                # restore circuit information
                self.parsed_circuits = original_circuits
            solver_results[L] = solver_result.x

        # return final result, having used all circuits:
        return solver_result, solver_results

    def _minimize_stage(self, stage: int, L: int, parameters_guess: Vector, cumulative_circuits: list, method: str, bounds, minimize_kwargs: dict):
        """ One stage of staged MLE on the cumulative circuits (self.parsed_circuits is set to them by the caller) """
 #        if stage < (num_stages - 1):
 #            objective_function = self.chi_squared
 #        else:
 #            objective_function = lambda params: -1. * self.log_likelihood(params)

        if stage == 0:
            theta_init = parameters_guess
        else:
            theta_init = self.gst_parameters.copy()

        # I found that using log likelihood for all stages gave faster and likely better results
        objective_function = lambda params: -1. * self.log_likelihood(params)

        # TODO: Standardize solve result objects between GST solver methods
        solver_result = opt.minimize(fun = lambda params: objective_function(params),  x0 = theta_init, method=method, bounds = bounds, **minimize_kwargs)
        self.solver_result = solver_result
        self.gst_parameters = solver_result.x


        if self.verbose:
            ll = self.log_likelihood(self.gst_parameters)
            print()
            print(f"    Stage {stage + 1} (L <= {L}): ")
            print(f"    {len(cumulative_circuits)} circuits ")
            print(f"    LL = {ll:.3f} ")
            print(f"    Converged = {solver_result.success} ")
        return solver_result


    ### Functions for gate set error metrics ### 
    def compute_gate_set_error_by_element(self, theta: Vector, ideal_gate_set: dict | None=None, error_metric: str='frobenius norm') -> dict:
        """ Computes an error for each element of the gate set by comparison to the ideal gate set elements.

            - ideal_gate_set: dictionary keyed by gate labels (e.g. 'Gxpi2:0'), 'prep', and 'POVM'.
                Defaults to the ideal gate set given to the constructor.
            - Returns a dictionary keyed by gate label, 'prep', and 'POVM'.

            Current options for gate set error metrics:
                1. Frobenius norm: compares best-fit process matrix vs. reference process matrix
                2. Process infidelity: between best-fit process matrix and reference matrix

        """
        return self._gate_set_error_by_element(theta, self._resolve_ideal_gate_set(ideal_gate_set), error_metric)

    def _resolve_ideal_gate_set(self, ideal_gate_set: dict | None) -> dict:
        """ Returns the internal (GstGate-keyed) ideal gate set from a user dictionary, or the constructor's ideal gate set if None. """
        if ideal_gate_set is None:
            if self.ideal_gate_set is None:
                raise ValueError("No ideal gate set is available; pass one here or to the GateSetTomography constructor.")
            return self.ideal_gate_set
        return self._normalize_ideal_gate_set(ideal_gate_set)

    def _gate_set_error_by_element(self, theta: Vector, internal_ideal_gate_set: dict, error_metric: str='frobenius norm') -> dict:
        """ Gate set errors from an internal (GstGate-keyed) ideal gate set. Results are keyed by gate label. """
        gst_errors = {}
        for gate in self.gate_set:
            if gate not in internal_ideal_gate_set:
                raise ValueError(f"The ideal gate set has no entry for gate {gate.label!r}.")
            ideal_gate = internal_ideal_gate_set[gate] # as a process matrix

            # Get process matrix from gate model at optimum
            gate_process_matrix_function = self.gate_models[gate]
            parameter_values = self.get_parameters(theta, gate)

            process_matrix = gate_process_matrix_function(*parameter_values)
            if error_metric == 'frobenius norm':
                gate_error = np.linalg.norm(process_matrix - ideal_gate, 'fro')
            else: # process infidelity
                gate_model = Gate(self.basis, process_matrix)
                gate_error = 1. - gate_model.compute_process_fidelity(ideal_gate)

            gst_errors[gate.label] = gate_error

        # prep state:
        ideal_prep_state = internal_ideal_gate_set['prep'].supervector
        modeled_prep_state = self.get_prep_state(theta)
        # Trace distance: sqrt(sum([rho_ideal[i] - rho_actual[i]]^2))
        prep_error = np.sqrt(np.sum((modeled_prep_state - ideal_prep_state)**2))
        gst_errors['prep'] = prep_error.real

        # POVMs
        ideal_POVMs = internal_ideal_gate_set['POVM']
        POVMs = self.get_measurement_effects(theta)
        POVM_errors = {}
        for outcome, POVM in ideal_POVMs.items():
            ideal_POVM = POVM.superbra
            modeled_POVM = POVMs[outcome]
            POVM_errors[outcome] = np.sqrt(np.sum((ideal_POVM - modeled_POVM)**2))

        gst_errors['POVM'] = sum(POVM_errors.values())
        if self.verbose:
            print(f"\n GST error by gate set element: {gst_errors}")
        return gst_errors

    def compute_gate_set_error(self, theta: Vector, ideal_gate_set: dict | None=None, include_SPAM_error: bool=False) -> float:
        """ Computes overall error of the gate set tomography parameter estimation by comparing ideal vs. best-fit gate models.

            - takes in an input dictionary "ideal_gate_set" that contains process matrices for each gate in the gate set, keyed by gate label.
            - additionally, the ideal_gate_set input contains the ideal prep state and ideal POVM
            - defaults to the ideal gate set given to the constructor

        """
        gate_set_errors = self._gate_set_error_by_element(theta, self._resolve_ideal_gate_set(ideal_gate_set))

        # Estimate process fidelity for each gate
        return self.average_model_errors(gate_set_errors, include_SPAM_error)

    def compute_average_gate_set_properties(self, N_repetitions: int, theta_true: Vector | dict, N_shots: int, solver: str='MLE', parameters_guess: Vector | dict | str | None=None, **kwargs):
        """ Performs Monte Carlo sampling of the true gate set and then fits each gate set sample with MLE.
            This enables computing gate set parameters and errors averaged over realizations of the true gate set

            - theta_true: true parameter vector, or a dictionary of parameter names to values (see build_theta_from_dict)
            - solver: 'MLE' (mle_solve_for_gate_parameters), 'staged MLE' (staged_mle_solve_for_gate_parameters), or 'linear'
                (linear_solve_for_gate_parameters).
            - parameters_guess: initial guess for each fit; 'lgst' re-seeds MLE solvers from linear GST on each sample
            - kwargs: passed to the solver (e.g. options = {...} for the MLE solvers)
        """
        solvers = {'mle': self.mle_solve_for_gate_parameters, 'staged mle': self.staged_mle_solve_for_gate_parameters,
                   'staged_mle': self.staged_mle_solve_for_gate_parameters, 'linear': self.linear_solve_for_gate_parameters}
        if not isinstance(solver, str) or solver.lower() not in solvers:
            raise ValueError(f"Unknown solver {solver!r}; use 'MLE', 'staged MLE', or 'linear'.")
        solve = solvers[solver.lower()]
        is_linear = solver.lower() == 'linear'

        if isinstance(theta_true, dict):
            theta_true = self.build_theta_from_dict(theta_true)

        # Confirm that this is the true theta:
        test_error = self._gate_set_error_by_element(theta_true, self._resolve_ideal_gate_set(None), error_metric = 'frobenius norm')
        test_error = np.abs(sum(test_error.values()))
        if test_error > NUMERICAL_EQUIVALENCE_THRESHOLD:
            raise IonSimError(f"Specified parameter vector is not the true theta. Gate set error received: {test_error}") 

        circuit_probabilities = []
        # Copy the original circuits 
        original_data = [circ.measurement_data for circ in self.parsed_circuits]

        best_theta_samples = np.zeros((N_repetitions, len(self.gst_parameters))) 
        gate_set_errors = []
        for n in range(N_repetitions):
            self.cached_theta = None
            self.process_matrix_cache = None
            self._likelihood_circuit_cache = {} 
            self._initialize_likelihood_circuit_cache()

            # Sample the true probabilities for each circuit  
            for circ in self.parsed_circuits:
                p = self._predict_probabilities(circ, theta_true)
                p_vals = [p[o] for o in self.outcome_labels]
                #circuit_probabilities.append(p_vals)
                #for circ, (prob_values) in zip(self.parsed_circuits, circuit_probabilities):
                outcome_counts = np.random.multinomial(N_shots, p_vals) 
                circ.measurement_data = CircuitData.from_counts(dict(zip(self.outcome_labels, outcome_counts)))

            self.lgst_results = None   
            self.solver_result = None 
            self._index_fiducials() # Reindex and organize fiducial information for linear GST if needed

            # For each repetition, perform the fit (the initial guess is re-resolved, e.g. a new LGST seed for each sample)
            results = solve(parameters_guess = parameters_guess, **kwargs)
            if is_linear:
                best_theta_samples[n, :] = results
            else:
                best_theta_samples[n, :] = results.x
            # Then compute the gate set errors:
            if is_linear:
                gate_set_errors.append(self._gate_set_error_by_element(results, self.ideal_gate_set, 'frobenius norm'))
            else:
                gate_set_errors.append(self._gate_set_error_by_element(results.x, self.ideal_gate_set, 'frobenius norm'))

            print(f"Finished repetition {n} with parameters: {best_theta_samples[n, :]}.")

        # Restore original parsed circuits from user 
        for circ, data in zip(self.parsed_circuits, original_data):
            circ.measurement_data = data

        theta_avg = np.mean(best_theta_samples, axis=0)
        theta_std_err = stats.sem(best_theta_samples, axis=0)

        # Compute average gate set error 
        avg_gate_set_error = {}
        gate_set_error_standard_error = {}

        for model in gate_set_errors[0].keys():
            errors = np.zeros(N_repetitions)    
            #std_devs = np.zeros(N_repetitions)    
            for i, err_dict in enumerate(gate_set_errors):
                errors[i] = err_dict[model] 
            avg_gate_set_error[model] = np.mean(errors)
            #avg_gate_set_error[model] = np.median(errors)
            # Standard error = standard deviation / sqrt(N)
            #gate_set_error_standard_error[model] = np.std(errors, axis=0)/np.sqrt(N_repetitions) 
            gate_set_error_standard_error[model] = stats.sem(errors, axis=0)

        return theta_avg, theta_std_err, avg_gate_set_error, gate_set_error_standard_error 


    def average_model_errors(self, gate_set_error: dict, include_SPAM_error: bool=True) -> float: 
        """ Average over gate model errors and include SPAM model errors """  
        gst_error = 0.
        gst_error = sum([gate_set_error[gate.label] for gate in self.gate_set]) / len(self.gate_set)

        if include_SPAM_error:
            #POVM_errors = np.array(list(gate_set_errors['POVM'].values())).real
            #SPAM_error = gate_set_errors['prep'] + sum(POVM_errors)
            SPAM_error = gate_set_error['prep'] + gate_set_error["POVM"]
            return gst_error + SPAM_error 
        else:
            return gst_error 

    def propagate_errors(self, gate_set_std_errs: dict, include_SPAM_error: bool=True) -> float: 
        """ Propagate errors to get an overall gate set standard derviation for the gate set error """   
        gst_error = 0.
        gst_error = sum([gate_set_std_errs[gate.label]**2 for gate in self.gate_set]) / (len(self.gate_set)**2)
        if include_SPAM_error:
            SPAM_error = gate_set_std_errs['prep'] + gate_set_std_errs["POVM"]
            gst_error += SPAM_error**2 
        return np.sqrt(gst_error)


    #def estimate_parameter_uncertainties(self, theta: Vector | None=None, method: str='bootstrap') -> Vector:
    ## TODO: Consider deleting this; we probably don't need bootstrapping 
 #    def bootstrapping_analysis(self, theta: Vector | None=None,  N_bootstrap: int=50):
 #        """ Computes uncertainties of each parameter from the Hessian of the log-likelihood at the MLE solution."""
 #        if self.solver_result is None and theta is None:
 #            self.solve_for_gate_parameters()
 #
 #        if theta is None:
 #            theta = self.gst_parameters 
 #        else:
 #            self.gst_parameters = theta
 #
 #        uncertainties = np.zeros_like(self.gst_parameters)
 #        uncertainties, means, bootstrapped_thetas = self.bootstrap_parameters(N_bootstrap)
 #
 #        # Return a dictionary containing a dictionary for each model (prep, gate 1, gate 2, etc. , measure) 
 #        mean_results = {}
 #        uncertainty_results = {}
 #        # For gates: 
 #        for gate in self.gate_set:
 #            gate_model = self.gate_models[gate]
 #            gate_model_sig = inspect.signature(gate_model)
 #            parameter_names = list(gate_model_sig.parameters.keys())  
 #            parameter_means = means[self.gst_parameter_indices[gate]]
 #            parameter_uncertainties = uncertainties[self.gst_parameter_indices[gate]]
 #            # Package up parameter names and uncertainty values: 
 #            mean_results[gate] = dict(zip(parameter_names, parameter_means))
 #            uncertainty_results[gate] = dict(zip(parameter_names, parameter_uncertainties)) 
 #
 #        # For SPAM: 
 #        prep_model = self.prep_state_model
 #        prep_model_sig = inspect.signature(prep_model)
 #        parameter_names = list(prep_model_sig.parameters.keys())
 #        prep_param_means = means[self.gst_parameter_indices['prep']]
 #        prep_param_unc = uncertainties[self.gst_parameter_indices['prep']]
 #        mean_results["prep"] = dict(zip(parameter_names, prep_param_means))
 #        uncertainty_results['prep'] = dict(zip(parameter_names, prep_param_unc)) 
 #
 #        # Currently only the independent measurement parameters are returned; TODO: generalize as much as possible  
 #        measure_model = self.POVM_effect_models
 #        measure_model_sig = inspect.signature(measure_model) 
 #        parameter_names = list(measure_model_sig.parameters.keys())
 #        parameter_means = means[self.gst_parameter_indices["POVM"]]
 #        parameter_uncertainties = uncertainties[self.gst_parameter_indices["POVM"]]
 #        mean_results["POVM"] = dict(zip(parameter_names, prep_param_means))
 #        uncertainty_results["POVM"] = dict(zip(parameter_names, parameter_uncertainties))
 #
 #        # Compute gate set errors for each bootstrapped sample  
 #        gate_set_errors = []
 #        N_bootstrap = bootstrapped_thetas.shape[0]
 #        for i in range(N_bootstrap):
 #            sampled_theta = bootstrapped_thetas[i,:] 
 #            gate_set_errors.append(self.compute_gate_set_error_by_element(sampled_theta, self.ideal_gate_set, 'frobenius norm')) 
 #        
 #        return mean_results, uncertainty_results, gate_set_errors, bootstrapped_thetas 
 #
 #    def bootstrap_parameters(self, N_bootstrap: int=50):
 #        """ Bootstrapping for parameter uncertainties: Sample data from the fitted model and re-fit, computing 
 #                parameter spread. N_bootstrap is the number of resamplings. """
 #        if self.verbose:
 #            print(f"Bootstrapping the uncertainties")
 #        theta_best = self.gst_parameters.copy()
 #        bootstrap_thetas = np.zeros((N_bootstrap, len(theta_best)))
 #
 #        circuit_probabilities = []
 #        for circ in self.parsed_circuits:
 #            probs = self._predict_probabilities(circ, theta_best)
 #            counts = circ.measurement_data.total_counts     # TODO: generalize to t-dependent data 
 #            circuit_probabilities.append((probs, counts))
 #        
 #        for b in range(N_bootstrap):
 #            for circ, (probs, total_counts) in zip(self.parsed_circuits, circuit_probabilities):
 #                outcomes = list(probs.keys()) 
 #                outcome_probs = [probs[outcome] for outcome in outcomes]
 #                outcome_counts = np.random.multinomial(total_counts, outcome_probs) 
 #                circ.measurement_data = CircuitData.from_counts(dict(zip(outcomes, outcome_counts)))
 #                
 #            # Re-run the MLE analysis to find best fit:
 #            #self.gst_parameters = theta_best.copy()
 #            #self.solve_for_gate_parameters(parameters_guess = theta_best.copy(), solver = 'MLE')   # sets self.gst_parameters to optimal  
 #            self.solve_for_gate_parameters(parameters_guess = self.parameters_guess, solver = 'MLE') 
 #            bootstrap_thetas[b] = self.gst_parameters
 #
 #        # Restore original data/fit
 #        self.gst_parameters = theta_best
 #            
 #        # Compute uncertainties as standard deviation of the best fits
 #        means = np.mean(bootstrap_thetas, axis=0)
 #        uncertainties = np.std(bootstrap_thetas, axis=0) 
 #        return uncertainties, means, bootstrap_thetas
