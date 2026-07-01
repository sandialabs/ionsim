import numpy as np
import re
import yaml 
from pathlib import Path 
from itertools import product
import inspect
import matplotlib.pyplot as plt

from ionsim.gst_circuit_parser import ParsedCircuit, ParsedGate


""" Circuit planner has 2 modes: 1) Gate model agnostic, 2) optimized planner based on gate models and germ sensitivies. """ 
class GSTCircuitPlanner:
    def __init__(self, gate_names: list[str], qubit_labels: list[int], prep_fiducials = None, measure_fiducials = None, germs = None, germ_powers: list[int]=[1,2,4,8,16], 
                    gate_models: dict | None=None, long_sequence_GST:bool = True):
        """ Constructor for GST Circuit Planner class. The user passes in the gate names and qubit labels at a minimum.

            - Sets up list of prep gates, measure gates, and germ gates. The class organizes GST circuits based on those gates requested germ powers.
            - Can write the GST circuit sequences to a file.
            - Optional arguments to provide a dictionary of gate process matrix models, which should match the gate names
            - long GST: 'True' will use germs to do long-gst circuits, 'false' will use only linear gst circuits  
            - mode: 'standard' for agnostic planning, 'optimized' for gate-model-aware planning

        """ 
        self.qubit_labels = qubit_labels
        self.gate_names = gate_names
        if long_sequence_GST:
            self.germ_powers = germ_powers
        else:
            self.germ_powers = [1]

        # Build Parsed Gate objects from gate names and store them in a dictionary  
        self._construct_gate_name_to_object_mapping(gate_names, qubit_labels) 

        # Set up prep/measure/germ circuits depending on user input. A default is used if none is supplied.  
        if prep_fiducials is None and measure_fiducials is None and len(qubit_labels) == 1:
            # Use standard 1Q GST fiducial choices 
            prep_fiducials, measure_fiducials = self.standard_1Q_fiducials()
        elif prep_fiducials is None and measure_fiducials is None and len(qubit_labels) > 1:
            raise IonSimError(f"2-qubit GST circuit planning default options are currently not implemented in IonSim. Please specify a choice of fiducial prep circuits.")

        if germs is None and len(qubit_labels) == 1: 
            germs = self.standard_1Q_germs(gate_names)

        # If optimized mode, optimize germ selection
        # Set mode --> either standard (gate model agnostic) or gate-model optimized 
        if gate_models is None:
            self.mode = 'standard' 
        else:
            self.mode = 'optimized' 

        # Check that gate models correspond with gate names if gate models are provided  
        self.gate_models = None
        if gate_models is not None:
            gate_model_names = gate_models.keys()
            if gate_model_names != gate_names:
                ValueError(f"The gate models is missing one of the gates. Expected gate models for {gate_model_names} and received models for {gate_model_names}")
            self.gate_models = gate_models

        self.long_GST = long_sequence_GST  

        # Ensure consistency in inputs: 
        # Convert all string-based fiducials/germs to ParsedGate objects
        self.prep_fiducials = [self.to_parsed_seq(fid) for fid in prep_fiducials]
        self.measure_fiducials = [self.to_parsed_seq(fid) for fid in measure_fiducials]
        self.germs = [self.to_parsed_seq(germ) for germ in germs]
        
        if self.mode == 'optimized':
            assert self.long_GST
            if self.germs is None or not self.germs:
                # Generate candidate germs for optimization
                candidate_germs = self._generate_candidate_germs_1Q(gate_names)
                optimized_germs = self.optimize_germs(candidate_germs)
                germs = optimized_germs
            else:
                # Optimize from provided germs
                optimized_germs = self.optimize_germs(germs)
                germs = optimized_germs
            # Set the germs list according to optimization  
            self.germs = germs 



    def _construct_gate_name_to_object_mapping(self, gate_names: list[str], qubit_labels: list[str]): 
        """ Set up the gate name -> ParsedGate look up dictionary """ 
        self.gate_lookup = {}
        for name in gate_names:
            if name == 'idle': # use empty qubit arguments 
                self.gate_lookup[name] = ParsedGate(name, ())
            else:
                self.gate_lookup[name] = ParsedGate(name, tuple(qubit_labels))

    def generate_gst_circuits(self) -> list:
        """Generate GST circuits. Convert string gates to ParsedGate and avoid duplicates."""

        gst_circuits = []
        unique = set()

        if self.long_GST:
            circuits = self._linear_gst_circuits() + self._long_gst_circuits()
        else:
            circuits = self._linear_gst_circuits() 

        for circ in circuits: 
            key = circ.build_circuit_string()

            if key not in unique:
                unique.add(key)
                gst_circuits.append(circ)

        self.gst_circuits = gst_circuits
        return gst_circuits

    def _linear_gst_circuits(self) -> list:
        """ Linear GST circuits (no germ powers). Consists of two circuit sets:

            1. Fiducial prep & measure 
            2. Fidcuial prep, gate, then measure. 

        """ 
        circuits = []

        # Group 1: Fiducial prep & measure 
        for prep_fiducial in self.prep_fiducials:
            for measure_fiducial in self.measure_fiducials:
                circuits.append( ParsedCircuit.plan(prep_fiducial, [], 1, measure_fiducial, self.qubit_labels)) 

        # Group 2: Fiducial prep, gate, and measure. For each gate, run the prep & measure circuits. 
        for gate_name in self.gate_names:
            gate = self.gate_lookup[gate_name] 
            for prep_fiducial in self.prep_fiducials:
                for measure_fiducial in self.measure_fiducials:
                    circuits.append( ParsedCircuit.plan(prep_fiducial, [gate], 1, measure_fiducial, self.qubit_labels)) 

        return circuits 

    def _long_gst_circuits(self) -> list:
        """ Long-form GST circuits: fiducial_prep + prep^{germ} + fiducial_measure """ 
        assert self.long_GST
        circuits = []
        for germ in self.germs:
            for power in self.germ_powers:
                for prep_fiducial in self.prep_fiducials:
                    for measure_fiducial in self.measure_fiducials:
                        circuits.append( ParsedCircuit.plan(prep_fiducial, germ, power, measure_fiducial, self.qubit_labels)) 

        return circuits 

    def write_circuit_plan(self, filepath: str | Path, N_qubits: int = 1):
        """ Writes a gst data file compatible with the parser """ 
        if not hasattr(self, 'circuits'):
            self.generate_gst_circuits() 

        d = 2**N_qubits # Hilbert space dimensionality 
        outcome_labels = [''.join(bits) for bits in product('01', repeat=N_qubits)] 

        with open(filepath, 'w') as f:
            # Write the header 
            columns = ", ".join(f"{outcome} count" for outcome in outcome_labels)
            f.write(f"## Columns = {columns}\n")

            for circ in self.gst_circuits:
                f.write(f"{circ.build_circuit_string()}\n")
                       
    @staticmethod
    def standard_1Q_fiducials() -> list:
        """ For 1Q gates, the fiducial circuits are standardized for {X_pi/2, Y_pi/2} gates. 

            - returns the prep and measure fiducials as lists of lists containing ParsedGate objects

        """  
        qubits = (0, )
        X_pi2 = ParsedGate('Gxpi2', qubits)
        Y_pi2 = ParsedGate('Gypi2', qubits)
        #idle = ParsedGate('idle', ())

        # include empty list for "do nothing for no time" initial sequence 
        fiducials = [[], [X_pi2], [Y_pi2], [X_pi2, X_pi2], [Y_pi2, Y_pi2], [X_pi2, X_pi2, X_pi2], [Y_pi2, Y_pi2, Y_pi2] ]
        return fiducials, fiducials 

    @staticmethod
    def standard_1Q_germs(gate_names: list[str]) -> list:
        """ For 1Q gates, the germs are the gates themselves and specific combinations of them. 

            - returns the list of germs; each germ is a list of ParsedGate objects 

        """  
        qubits = (0, )
        X_pi2 = ParsedGate('Gxpi2', qubits)
        Y_pi2 = ParsedGate('Gypi2', qubits)
        idle = ParsedGate('[]', ()) # should it be qubits? 

        if 'idle' in gate_names:
            germs = [ [X_pi2], [Y_pi2], [idle], [X_pi2, Y_pi2], [X_pi2, X_pi2, Y_pi2] ]
        else:
            germs = [ [X_pi2], [Y_pi2], [X_pi2, Y_pi2], [X_pi2, X_pi2, Y_pi2] ]

        return germs

    def _generate_candidate_germs_1Q(self, gate_names: list[str]) -> list:
        """Generate a comprehensive set of candidate germs for 1Q optimization."""
        qubits = (0, )
        X_pi2 = ParsedGate('Gxpi2', qubits)
        Y_pi2 = ParsedGate('Gypi2', qubits)
        idle = ParsedGate('[]', ())

        # Generate comprehensive candidate set
        candidates = []

        # Single gates
        if 'Gxpi2' in gate_names:
            candidates.append([X_pi2])
        if 'Gypi2' in gate_names:
            candidates.append([Y_pi2])
        if 'idle' in gate_names:
            candidates.append([idle])

        # Two-gate sequences
        if 'Gxpi2' in gate_names and 'Gypi2' in gate_names:
            candidates.extend([
                [X_pi2, Y_pi2],
                [Y_pi2, X_pi2],
                [X_pi2, X_pi2],
                [Y_pi2, Y_pi2]
            ])

        # Three-gate sequences
        if 'Gxpi2' in gate_names and 'Gypi2' in gate_names:
            candidates.extend([
                [X_pi2, X_pi2, Y_pi2],
                [Y_pi2, Y_pi2, X_pi2],
                [X_pi2, Y_pi2, X_pi2],
                [Y_pi2, X_pi2, Y_pi2],
                [X_pi2, X_pi2, X_pi2],
                [Y_pi2, Y_pi2, Y_pi2]
            ])

        # Four-gate sequences (for more comprehensive coverage)
        if 'Gxpi2' in gate_names and 'Gypi2' in gate_names:
            candidates.extend([
                [X_pi2, Y_pi2, X_pi2, Y_pi2],
                [X_pi2, X_pi2, Y_pi2, Y_pi2],
                [X_pi2, Y_pi2, Y_pi2, X_pi2]
            ])

        return candidates


    @staticmethod
    def write_all_circuit_outcomes(filename: str, circuits: list[ParsedCircuit], N_qubits:int=1):
        """ Writes all circuit information to a file """
        d = 2**N_qubits # Hilbert space dimensionality 
        outcome_labels = [''.join(bits) for bits in product('01', repeat=N_qubits)] 

        with open(filename, 'w') as f:
            # Write the header 
            columns = ", ".join(f"{outcome} count" for outcome in outcome_labels)
            f.write(f"## Columns = {columns}\n")

            for circ in circuits:
                f.write(circ._format_circuit_line() + "\n")

    @staticmethod
    def create_circuit_outcomes_file(filename: str, N_qubits:int=1):
        """ Creates a GST circuit file with appropriate header """ 
        d = 2**N_qubits # Hilbert space dimensionality 
        outcome_labels = [''.join(bits) for bits in product('01', repeat=N_qubits)] 

        with open(filename, 'w') as f:
            # Write the header 
            columns = ", ".join(f"{outcome} count" for outcome in outcome_labels)
            f.write(f"## Columns = {columns}\n")

    def to_parsed_gate(self, g):
            if isinstance(g, ParsedGate):
                return g
            if isinstance(g, str):
                # Handle special case for idle gate represented as '[]'
                if g == '[]':
                    return ParsedGate('[]', ())
                if g in self.gate_lookup:
                    return self.gate_lookup[g]
                raise ValueError(f"Unknown gate name: {g}")
            raise TypeError(f"Bad gate type: {type(g)} -> {g}")

    def to_parsed_seq(self, seq):
        return [self.to_parsed_gate(g) for g in seq]


    def _compute_germ_process_matrix(self, germ, theta_dict):
        """Compute the process matrix for a germ given parameter values for each gate model.

        Args:
            germ: List of ParsedGate objects representing the germ
            theta_dict: Dictionary mapping gate names to their parameter arrays

        Returns:
            Process matrix for the germ sequence
        """
        d = 2**len(self.qubit_labels)
        d2 = d**2

        germ_process_matrix = np.eye(d2, dtype=complex)

        for gate in germ:
            # Get the gate model function for this gate
            # Convention is for idle gate to be named '[]'; however, gate models generally use "idle" instead.
            gate_name = 'idle' if gate.name == '[]' else gate.name
            gate_func = self.gate_models[gate_name]

            # Get parameters for this specific gate model
            theta = theta_dict[gate_name]

            # Evaluate at current parameters
            gate_matrix = gate_func(*theta)
            germ_process_matrix = gate_matrix @ germ_process_matrix

        return germ_process_matrix

    #def compute_germ_sensitivities(self, germs: list[list[ParsedGate]], max_power: int=16):
    def compute_germ_sensitivities(self, germs: list[list[ParsedGate]]): 
        """Compute sensitivity of all gate model parameters to germ sequences.

        This method computes the sensitivity of each parameter from each gate model
        used in a germ to the germ's process matrix and its powers. This provides
        a comprehensive view of how each parameter affects the germ behavior.

        Args:
            germs: List of germs (each germ is a list of ParsedGate objects)

        Returns:
            Dictionary: {germ_name: sensitivity_data} where sensitivity_data is a
            dictionary mapping gate names to their sensitivity matrices. Each
            sensitivity matrix has shape (n_params, max_power) where
            sensitivity_matrix[param_idx, power-1] represents the sensitivity
            of parameter param_idx to germ^power.
        """
        import inspect

        if self.gate_models is None:
            raise ValueError("Gate models must be provided for sensitivity analysis.")

        d = 2**len(self.qubit_labels)
        d2 = d**2

        sensitivity_results = {}
        max_power = self.germ_powers[-1]

        for germ in germs:
            # Create a descriptive name for the germ
            germ_name = ''.join([gate.name for gate in germ])
            print(f" - Computing sensitivities for germ {germ_name}")

            # Collect all unique gate models in this germ and their parameter information
            germ_gate_models = {}
            theta_dict_nominal = {}

            for gate in germ:
                gate_name = 'idle' if gate.name == '[]' else gate.name
                if gate_name not in germ_gate_models:
                    gate_func = self.gate_models[gate_name]
                    sig = inspect.signature(gate_func)
                    param_names = list(sig.parameters.keys())
                    n_params = len(param_names)

                    germ_gate_models[gate_name] = {
                        'function': gate_func,
                        'param_names': param_names,
                        'n_params': n_params
                    }
                    # Store nominal parameters (zeros) for this gate model
                    theta_dict_nominal[gate_name] = np.zeros(n_params)

            # Initialize sensitivity data structure for this germ
            germ_sensitivity_data = {}

            for gate_name, gate_info in germ_gate_models.items():
                n_params = gate_info['n_params']
                # Initialize sensitivity matrix: params x powers
                germ_sensitivity_data[gate_name] = np.zeros((n_params, len(self.germ_powers)))

            # Compute nominal germ process matrix
            germ_process_matrix_nominal = self._compute_germ_process_matrix(germ, theta_dict_nominal)

            for i, power in enumerate(self.germ_powers):
                # Compute germ^power process matrix at nominal parameters
                G_power_nominal = np.linalg.matrix_power(germ_process_matrix_nominal, power)

                # Compute sensitivity via finite differences for each parameter of each gate model
                for gate_name, gate_info in germ_gate_models.items():
                    n_params = gate_info['n_params']

                    for param_idx in range(n_params):
                        # Perturb this specific parameter
                        theta_dict_perturbed = {gn: params.copy() for gn, params in theta_dict_nominal.items()}
                        #epsilon = 1e-6
                        epsilon = 1e-3
                        theta_dict_perturbed[gate_name][param_idx] += epsilon

                        # Compute perturbed germ process matrix
                        germ_process_matrix_perturbed = self._compute_germ_process_matrix(germ, theta_dict_perturbed)
                        G_power_perturbed = np.linalg.matrix_power(germ_process_matrix_perturbed, power)

                        # Compute Frobenius norm of difference
                        diff = np.linalg.norm(G_power_perturbed - G_power_nominal, 'fro')
                        sensitivity = diff / epsilon

                        germ_sensitivity_data[gate_name][param_idx, i] = sensitivity

            sensitivity_results[germ_name] = germ_sensitivity_data

        return sensitivity_results

    def _select_germs_based_on_sensitivity(self, sensitivity_data, candidate_germs):
        """ Select germs that provide good coverage of parameter sensitivity.

            sensitivity_data: Dictionary of {germ_name: {gate_name: sensitivity_matrix}}
            candidate_germs: List of candidate germs (lists of ParsedGate objects)

            Returns a list of selected germs (lists of ParsedGate objects)
        """
        # Simple selection strategy: choose germs with highest average sensitivity
        # across all parameters and powers

        germ_scores = {}
        germ_name_to_germ = {}

        # Map germ names back to actual germ objects
        for germ in candidate_germs:
            germ_name = ''.join([gate.name for gate in germ])
            germ_name_to_germ[germ_name] = germ

        for germ_name, gate_sensitivities in sensitivity_data.items():
            total_sensitivity = 0
            total_params = 0

            # Sum sensitivity across all gate models in this germ
            for gate_name, sensitivity_matrix in gate_sensitivities.items():
                # Compute average sensitivity for this gate model
                avg_sensitivity = np.mean(sensitivity_matrix)
                # Get number of parameters for weighting
                n_params = sensitivity_matrix.shape[0]

                total_sensitivity += avg_sensitivity * n_params
                total_params += n_params

            # Compute weighted average sensitivity for this germ
            if total_params > 0:
                germ_scores[germ_name] = total_sensitivity / total_params
            else:
                germ_scores[germ_name] = 0

        # Sort germs by score (highest first)
        sorted_germs = sorted(germ_scores.items(), key=lambda x: x[1], reverse=True)

        # Return the actual germ objects sorted by sensitivity score
        return [germ_name_to_germ[germ] for germ, score in sorted_germs]

    def optimize_germs(self, candidate_germs=None, n_germs_to_select=None):
        """ Select optimal germs based on sensitivity analysis.

            candidate_germs: List of candidate germs to consider (if None, use current germs)
            n_germs_to_select: Number of germs to select (if None, select all)

            Returns a list of selected germs that maximize parameter sensitivity.
        """
        if self.gate_models is None:
            raise ValueError("Gate models must be provided for germ optimization.")

        assert self.long_GST
        # Use candidate germs if provided, otherwise use current germs
        germs_to_consider = candidate_germs if candidate_germs is not None else self.germs

        # Compute sensitivity for all germs (new method that handles multiple gate models per germ)
        sensitivity_data = self.compute_germ_sensitivities(germs_to_consider)

        # Select germs based on sensitivity
        selected_germs = self._select_germs_based_on_sensitivity(sensitivity_data, germs_to_consider)

        # Limit number of germs if requested
        if n_germs_to_select and len(selected_germs) > n_germs_to_select:
            selected_germs = selected_germs[:n_germs_to_select]

        return selected_germs

    def analyze_germ_amplification_completeness(self, germs=None, sensitivity_threshold=1e-3):
        """ Analyze whether the germ set provides amplification for all gate model parameters.

            This method checks if each parameter in each gate model is sufficiently amplified
            by at least one germ in the germ set. It provides diagnostic information about
            which parameters are well-amplified and which are not.

            Args:
                germs: List of germs to analyze (if None, use current germs)
                sensitivity_threshold: Minimum sensitivity value to consider a parameter amplified

            Returns:
                Dictionary containing:
                - 'amplification_status': Overall status ('complete', 'incomplete', or 'no_gate_models')
                - 'amplified_parameters': Dictionary mapping gate names to lists of amplified parameter names
                - 'unamplified_parameters': Dictionary mapping gate names to lists of unamplified parameter names
                - 'parameter_sensitivities': Detailed sensitivity information for each parameter
                - 'warnings': List of warning messages
        """
        if self.gate_models is None:
            return {
                'amplification_status': 'no_gate_models',
                'amplified_parameters': {},
                'unamplified_parameters': {},
                'parameter_sensitivities': {},
                'warnings': ['No gate models provided - cannot analyze amplification completeness']
            }

        assert self.long_GST
        # Use current germs if none provided
        germs_to_analyze = germs if germs is not None else self.germs

        if not germs_to_analyze:
            return { 'amplification_status': 'no_germs', 'amplified_parameters': {}, 'unamplified_parameters': {},
                'parameter_sensitivities': {}, 'warnings': ['No germs provided - cannot analyze amplification completeness'] }

        # Sensitivity for the germs: {germ name, {gate model : array of shape params x powers}} 
        # i.e. dictionary of key = germ name, value = dictionary with key = gate model, matrix of d[germ]/dtheta of shape parameters x germ powers
        sensitivity_data = self.compute_germ_sensitivities(germs_to_analyze)

        # Collect all parameters across all gate models
        all_parameters = {}
        for gate_name, gate_func in self.gate_models.items():
            import inspect
            sig = inspect.signature(gate_func)
            param_names = list(sig.parameters.keys())
            all_parameters[gate_name] = param_names

        # Analyze amplification for each parameter
        amplified_parameters = {gate_name: [] for gate_name in self.gate_models.keys()}
        unamplified_parameters = {gate_name: [] for gate_name in self.gate_models.keys()}
        parameter_sensitivities = {}

        warnings = []

        # For each gate model, extract parameter sensitivities for each germ 
        for gate_name, param_names in all_parameters.items():
            parameter_sensitivities[gate_name] = {}

            for param_idx, param_name in enumerate(param_names):
                max_sensitivity = 0
                best_germ = None
                best_power = None
                max_sensitivity_in_high_power_range = 0
                best_high_power_germ = None
                best_high_power = None

                # Check sensitivity across all germs and powers
                for germ_name, gate_sensitivities in sensitivity_data.items():
                    if gate_name in gate_sensitivities:
                        sensitivity_matrix = gate_sensitivities[gate_name]

                        # Find maximum sensitivity for this parameter in this germ for this gate model across all powers
                        param_sensitivities = sensitivity_matrix[param_idx, :]
                        germ_max_sensitivity = np.max(param_sensitivities) # over all powers 

                        if germ_max_sensitivity > max_sensitivity:
                            max_sensitivity = germ_max_sensitivity
                            # Find the power that gives maximum sensitivity
                            best_power_idx = np.argmax(param_sensitivities)
                            best_power = self.germ_powers[best_power_idx] 
                            best_germ = germ_name

                        # Check sensitivity at high germ powers (last half of power range)
                        high_power_indices = range(len(param_sensitivities) // 2, len(param_sensitivities))
                        high_power_sensitivities = param_sensitivities[high_power_indices]
                        if len(high_power_sensitivities) > 0:
                            germ_high_power_max = np.max(high_power_sensitivities)
                            if germ_high_power_max > max_sensitivity_in_high_power_range:
                                max_sensitivity_in_high_power_range = germ_high_power_max
                                # Find the high power that gives maximum sensitivity
                                high_power_idx = np.argmax(high_power_sensitivities)
                                #best_high_power = high_power_indices[high_power_idx] + 1  # Convert to 1-indexed
                                best_high_power = self.germ_powers[high_power_idx] 
                                best_high_power_germ = germ_name

                # Determine amplification status
                is_amplified_overall = max_sensitivity >= sensitivity_threshold
                is_amplified_at_high_powers = max_sensitivity_in_high_power_range >= sensitivity_threshold

                # Check if sensitivity increases with germ power (amplification)
                shows_amplification = False
                overall_max_at_high_power = False

                if best_germ is not None and best_high_power_germ is not None:
                    # Get full sensitivity curve for the best germ
                    sensitivity_matrix = sensitivity_data[best_germ][gate_name]
                    param_sensitivities = sensitivity_matrix[param_idx, :]

                    # Check if sensitivity at highest power is significantly greater than at lowest power
                    if len(param_sensitivities) > 1:
                        lowest_power_sensitivity = param_sensitivities[0]
                        highest_power_sensitivity = param_sensitivities[-1]
                        amplification_factor = highest_power_sensitivity / lowest_power_sensitivity if lowest_power_sensitivity > 0 else float('inf')
                        shows_amplification = amplification_factor > 2.0  # Arbitrary threshold for "significant amplification"

                        # Check if the overall maximum sensitivity occurs at high powers
                        overall_max_power = np.argmax(param_sensitivities) + 1  # Convert to 1-indexed
                        # Consider "high power" as powers in the upper half of the range
                        high_power_threshold = len(param_sensitivities) // 2
                        overall_max_at_high_power = overall_max_power > high_power_threshold

                parameter_sensitivities[gate_name][param_name] = {
                    'max_sensitivity': max_sensitivity,
                    'best_germ': best_germ,
                    'best_power': best_power,
                    'max_sensitivity_in_high_power_range': max_sensitivity_in_high_power_range,
                    'best_high_power_germ': best_high_power_germ,
                    'best_high_power': best_high_power,
                    'is_amplified': is_amplified_overall,
                    'is_amplified_at_high_powers': is_amplified_at_high_powers,
                    'shows_amplification': shows_amplification,
                    'overall_max_at_high_power': overall_max_at_high_power,
                }

                # Categorize parameters based on amplification quality
                if is_amplified_at_high_powers and shows_amplification and overall_max_at_high_power:
                    # Ideal case: sensitive at high powers, shows amplification, and max occurs at high powers
                    amplified_parameters[gate_name].append(param_name)
                elif is_amplified_at_high_powers and overall_max_at_high_power:
                    # Sensitive at high powers and max occurs at high powers, but doesn't show strong amplification
                    amplified_parameters[gate_name].append(param_name)
                    warnings.append(f"Parameter '{param_name}' in gate '{gate_name}' is amplified at high powers (sensitivity: {max_sensitivity_in_high_power_range:.2e}) but shows limited amplification growth.")
                elif is_amplified_at_high_powers and not overall_max_at_high_power:
                    # CRITICAL: Sensitive at high powers but maximum occurs at LOW powers
                    unamplified_parameters[gate_name].append(param_name)
                    warnings.append(f"Parameter '{param_name}' in gate '{gate_name}' has sensitivity at high powers ({max_sensitivity_in_high_power_range:.2e}) but MAXIMUM sensitivity occurs at LOW power ({best_power}) with value {max_sensitivity:.2e} - NOT properly amplified for long-sequence GST!")
                elif is_amplified_overall:
                    # Amplified overall but not at high powers - problematic for long-sequence GST
                    unamplified_parameters[gate_name].append(param_name)
                    warnings.append(f"Parameter '{param_name}' in gate '{gate_name}' has sensitivity at low powers but NOT at high powers (overall max: {max_sensitivity:.2e} at power {best_power}, high-power max: {max_sensitivity_in_high_power_range:.2e}) - NOT amplificationally complete!")
                else:
                    # Not amplified at all
                    unamplified_parameters[gate_name].append(param_name)
                    warnings.append(f"Parameter '{param_name}' in gate '{gate_name}' has low sensitivity overall (max: {max_sensitivity:.2e}) - may not be well amplified by any germ")

        # Determine overall amplification status
        total_unamplified = sum(len(params) for params in unamplified_parameters.values())

        # Check if all parameters are amplified at high powers
        total_params = sum(len(params) for params in all_parameters.values())
        amplified_at_high_powers_count = 0

        for gate_name, param_names in all_parameters.items():
            for param_name in param_names:
                sensitivity_info = parameter_sensitivities[gate_name][param_name]
                if sensitivity_info['is_amplified_at_high_powers']:
                    amplified_at_high_powers_count += 1

        if total_unamplified == 0:
            if amplified_at_high_powers_count == total_params:
                amplification_status = 'complete'
            else:
                amplification_status = 'partial'  # All parameters have some sensitivity, but not all amplify well
        else:
            amplification_status = 'incomplete'

        return {
            'amplification_status': amplification_status,
            'amplified_parameters': amplified_parameters,
            'unamplified_parameters': unamplified_parameters,
            'parameter_sensitivities': parameter_sensitivities,
            'warnings': warnings
        }

    def print_amplification_diagnostics(self, germs=None, sensitivity_threshold=1e-4):
        """ Print diagnostic information about germ amplification completeness.

            This method provides a human-readable summary of which parameters are
            well-amplified by the germ set and which are not.

            Args:
                germs: List of germs to analyze (if None, use current germs)
                sensitivity_threshold: Minimum sensitivity value to consider a parameter amplified
        """
        assert self.long_GST
        diagnostics = self.analyze_germ_amplification_completeness(germs, sensitivity_threshold)

        print("\n" + "="*80)
        print("GERM AMPLIFICATION COMPLETENESS DIAGNOSTICS")
        print("="*80)

        if diagnostics['amplification_status'] == 'no_gate_models':
            print("No gate models provided - cannot analyze amplification completeness")
            return

        if diagnostics['amplification_status'] == 'no_germs':
            print("No germs provided - cannot analyze amplification completeness")
            return

        print(f"Amplification Status: {diagnostics['amplification_status'].upper()}")
        print(f"Sensitivity Threshold: {sensitivity_threshold:.2e}")
        print(f"Number of Germs Analyzed: {len(self.germs if germs is None else germs)}")

        if diagnostics['amplification_status'] == 'complete':
            print("\n✓ ALL PARAMETERS ARE WELL-AMPLIFIED")
            print("The germ set provides good sensitivity to all gate model parameters at high powers.")
            print("This is ideal for long-sequence GST where error should scale as 1/L.")
        elif diagnostics['amplification_status'] == 'partial':
            print("\n⚠ PARTIAL AMPLIFICATION DETECTED")
            print("All parameters have some sensitivity, but not all show strong amplification at high powers.")
            print("Some parameters may not benefit fully from longer circuits.")
        else:
            print("\n❌ INCOMPLETE AMPLIFICATION DETECTED")
            print("Some gate model parameters are NOT well-amplified by the germ set.")
            print("Parameters without high-power sensitivity will show flat error scaling in staged MLE.")

        print("\n" + "-"*80)
        print("PARAMETER AMPLIFICATION SUMMARY")
        print("-"*80)

        for gate_name in self.gate_models.keys():
            print(f"\nGate: {gate_name}")

            amplified = diagnostics['amplified_parameters'][gate_name]
            unamplified = diagnostics['unamplified_parameters'][gate_name]

            if amplified:
                print(f"  ✓ Amplified parameters ({len(amplified)}):")
                for param_name in amplified:
                    sensitivity_info = diagnostics['parameter_sensitivities'][gate_name][param_name]
                    if sensitivity_info['shows_amplification'] and sensitivity_info['overall_max_at_high_power']:
                        amplification_status = "✓ Strong amplification (max at high power)"
                    elif sensitivity_info['shows_amplification']:
                        amplification_status = "⚠ Weak amplification (max not at high power)"
                    elif sensitivity_info['overall_max_at_high_power']:
                        amplification_status = "⚠ Limited amplification (max at high power but no growth)"
                    else:
                        amplification_status = "❌ Poor amplification (max at low power)"

                    print(f"    - {param_name}: {amplification_status}")
                    print(f"      Best overall: germ={sensitivity_info['best_germ']}, power={sensitivity_info['best_power']} (sensitivity={sensitivity_info['max_sensitivity']:.2e})")

            if unamplified:
                print(f"  ⚠ Problematic parameters ({len(unamplified)}):")
                for param_name in unamplified:
                    sensitivity_info = diagnostics['parameter_sensitivities'][gate_name][param_name]
                    print(f"    - {param_name}:")
                    print(f"      Best overall: germ={sensitivity_info['best_germ']}, power={sensitivity_info['best_power']} (sensitivity={sensitivity_info['max_sensitivity']:.2e})")

                    if sensitivity_info['is_amplified'] and not sensitivity_info['is_amplified_at_high_powers']:
                        print(f"      ❌ CRITICAL: Sensitive at low powers but NOT at high powers!")
                        print(f"      This parameter will NOT benefit from longer circuits.")
                    elif sensitivity_info['is_amplified_at_high_powers'] and not sensitivity_info.get('overall_max_at_high_power', False):
                        print(f"      ❌ CRITICAL: Maximum sensitivity at LOW power ({sensitivity_info['best_power']})!")
                        print(f"      This parameter will NOT benefit from longer circuits.")
                    else:
                        print(f"      ❌ Not sensitive at any power level.")

        if diagnostics['warnings']:
            print("\n" + "-"*80)
            print("WARNINGS")
            print("-"*80)
            for warning in diagnostics['warnings']:
                print(f"  ⚠ {warning}")

        print("\n" + "="*80)

        return diagnostics

    def plot_parameter_sensitivity_curves(self, germs=None, sensitivity_threshold=1e-4, filename=None, include_all_germs:bool=False):
        """ Plot sensitivity vs. germ power for each parameter to visualize amplification.

            This creates diagnostic plots showing how sensitivity changes with germ power,
            which is essential for verifying amplification completeness.

            Args:
                germs: List of germs to analyze (if None, use current germs)
                sensitivity_threshold: Threshold for considering a parameter amplified
                filename: If provided, save the plot to this file
        """

        assert self.long_GST
        if self.gate_models is None:
            print("No gate models provided - cannot plot sensitivity curves")
            return

        # Compute sensitivity data
        germs_to_analyze = germs if germs is not None else self.germs
        sensitivity_data = self.compute_germ_sensitivities(germs_to_analyze)

        # Collect all parameters
        all_parameters = {}
        for gate_name, gate_func in self.gate_models.items():
            sig = inspect.signature(gate_func)
            param_names = list(sig.parameters.keys())
            all_parameters[gate_name] = param_names

        # Create plots
        n_gates = len(all_parameters)
        fig, axes = plt.subplots(n_gates, 1, figsize=(12, 6 * n_gates))
        if n_gates == 1:
            axes = [axes]  # Ensure axes is iterable

        # Make a subplot for each gate model 
        for ax, (gate_name, param_names) in zip(axes, all_parameters.items()):
            ax.set_title(f"Gate: {gate_name}", fontsize=14, fontweight='bold')

            for param_idx, param_name in enumerate(param_names):
                # Find the best germ for this parameter
                max_sensitivity = 0
                best_germ_name = None

                # Build sensitivity matrix for each germ 
                for germ_name, gate_sensitivities in sensitivity_data.items():
                    if gate_name in gate_sensitivities:
                        sensitivity_matrix = gate_sensitivities[gate_name]
                        param_sensitivities = sensitivity_matrix[param_idx, :]
                        germ_max = np.max(param_sensitivities)

                        if germ_max > max_sensitivity:
                            max_sensitivity = germ_max
                            best_germ_name = germ_name

                if best_germ_name is not None and not include_all_germs:
                    sensitivity_matrix = sensitivity_data[best_germ_name][gate_name]
                    # sensitivities for a germ has shape (n_params, powers) 
                    sensitivities_at_powers = []
                    for i, power in enumerate(self.germ_powers):
                        sensitivities_at_powers.append(sensitivity_matrix[param_idx, i])

                    sensitivities_at_powers = np.array(sensitivities_at_powers)

                    # Plot the sensitivity curve
                    ax.semilogy(self.germ_powers, sensitivities_at_powers, 'o-', label=f"{param_name} (best germ: {best_germ_name})")

                    # Add threshold line
                   # ax.axhline(sensitivity_threshold, color='red', linestyle='--', alpha=0.5, label='Threshold' if param_idx == 0 else "")

                    # Annotate max sensitivity
                    max_idx = np.argmax(sensitivities_at_powers)
                    ax.annotate(f"{sensitivities_at_powers[max_idx]:.1e}",
                               (self.germ_powers[max_idx], sensitivities_at_powers[max_idx]),
                               textcoords="offset points", xytext=(10,10), ha='center')
                elif include_all_germs:
                    for germ_name, gate_sensitivities in sensitivity_data.items():
                        if gate_name in gate_sensitivities:
                            sensitivity_matrix = gate_sensitivities[gate_name] # matrix of parameters x powers 

                            # sensitivities for a germ has shape (n_params, powers) 
                            sensitivities_at_powers = []
                            for i, power in enumerate(self.germ_powers):
                                sensitivities_at_powers.append(sensitivity_matrix[param_idx, i])
    
                            # Plot the sensitivity curve
                            ax.semilogy(self.germ_powers, np.array(sensitivities_at_powers), 'o-', label=f"{param_name} (germ: {germ_name})")
    
            # Add threshold line
            ax.axhline(sensitivity_threshold, color='red', linestyle='--', alpha=0.5, label='Threshold' if param_idx == 0 else "")

            ax.set_xlabel("Germ Power", fontsize=12)
            ax.set_ylabel("Sensitivity (Frobenius norm)", fontsize=12)
            ax.grid(True, which="both", ls="--")

            # Only add legend if there are parameters with data
            if param_names:
                ax.legend(fontsize=10)

            # Add amplification guidance
            if param_names:  # If there are parameters for this gate
                ax.text(0.02, 0.95,
                       "✓ Good amplification: curve rises with germ power\n"
                       "❌ No amplification: flat or falling curve",
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

        #plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=300)
            #plt.savefig(filename, bbox_inches='tight', dpi=300)
            print(f"Saved sensitivity plots to {filename}")
        else:
            plt.show()

        return fig, axes

    def write_circuit_design(self, filepath):
        """ Writes a design yaml file with circuit design information """
        #filename = 'GST_circuit_design.yaml'  

        def gate_list_to_dict(gate_list):
            """ Convert list of Gate objects to a dictionary format """ 
            return [{'name' : g.name, 'qubits' : list(g.qubits)} for g in gate_list]


        def fiducials_to_dict(fiducials):            
            """ Convert list of fiducial sequences (list of ParsedGates) to dictionary."""
            return [gate_list_to_dict(fid) for fid in fiducials]


        design = {
            'gate_names' : self.gate_names,
            'qubit_labels' : self.qubit_labels,
            'prep_fiducials' : fiducials_to_dict(self.prep_fiducials), 
            'measure_fiducials' : fiducials_to_dict(self.measure_fiducials),
            'germs': fiducials_to_dict(self.germs),
            'germ_powers' : self.germ_powers 
        }

        with open(filepath, 'w') as f:
            yaml.dump(design, f, default_flow_style=False, sort_keys=False) 

    
    @classmethod
    def load_design(cls, filepath):
        """ Load an experimental design from a YAML file, returns the planner class instance """ 

        def dict_to_gate_list(dict_list):
            """ Converts dictionary list of gates to a list of ParsedGates """ 
            return [ParsedGate(name=g['name'], qubits = tuple(g['qubits']))
                for g in dict_list]
        

        def dict_to_fiducials(fid_list):
            """ Converts dictionary list of fiducials to list of ParsedGates """ 
            return [dict_to_gate_list(fid) for fid in fid_list]
            

        with open(filepath, 'r') as f:
            design = yaml.safe_load(f)

        planner = cls(gate_names = design['gate_names'], qubit_labels = design['qubit_labels'],
                    prep_fiducials = dict_to_fiducials(design['prep_fiducials']), 
                    measure_fiducials = dict_to_fiducials(design['measure_fiducials']), 
                    germs = dict_to_fiducials(design['germs']), germ_powers = design['germ_powers'] )

        return planner 

