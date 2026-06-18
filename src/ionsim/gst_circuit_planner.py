import numpy as np
import re
import yaml 
from pathlib import Path 
from itertools import product

from ionsim.gst_circuit_parser import ParsedCircuit, ParsedGate


""" Circuit planner has 2 modes: 1) Gate model agnostic, 2) optimized planner based on gate models and germ sensitivies. """ 
class GSTCircuitPlanner:
    def __init__(self, gate_names: list[str], qubit_labels: list[int], prep_fiducials = None, measure_fiducials = None, germs = None, germ_powers: list[int]=[1,2,4,8,16], gate_models: dict | None=None):
        """ Constructor for GST Circuit Planner class. The user passes in the gate names and qubit labels at a minimum.

            - Sets up list of prep gates, measure gates, and germ gates. The class organizes GST circuits based on those gates requested germ powers.
            - Can write the GST circuit sequences to a file.
            - Optional arguments to provide a dictionary of gate process matrix models, which should match the gate names
            - mode: 'standard' for agnostic planning, 'optimized' for gate-model-aware planning

        """ 
        self.qubit_labels = qubit_labels
        self.gate_names = gate_names
        self.germ_powers = germ_powers

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

        if self.mode == 'optimized':
            if germs is None:
                # Generate candidate germs for optimization
                candidate_germs = self._generate_candidate_germs_1Q(gate_names)
                optimized_germs = self.optimize_germs(candidate_germs)
                germs = optimized_germs
            else:
                # Optimize from provided germs
                optimized_germs = self.optimize_germs(germs)
                germs = optimized_germs

        # Ensure consistency in inputs: 
        # Convert all string-based fiducials/germs to ParsedGate objects
        self.prep_fiducials = [self.to_parsed_seq(fid) for fid in prep_fiducials]
        self.measure_fiducials = [self.to_parsed_seq(fid) for fid in measure_fiducials]
        self.germs = [self.to_parsed_seq(germ) for germ in germs]


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

        for circ in self._linear_gst_circuits() + self._long_gst_circuits():
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

    def compute_germ_sensitivities(self, germs: list[list[ParsedGate]], max_power: int=16):
        """Compute sensitivity of all gate model parameters to germ sequences.

        This method computes the sensitivity of each parameter from each gate model
        used in a germ to the germ's process matrix and its powers. This provides
        a comprehensive view of how each parameter affects the germ behavior.

        Args:
            germs: List of germs (each germ is a list of ParsedGate objects)
            max_power: Maximum germ power to consider

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
                germ_sensitivity_data[gate_name] = np.zeros((n_params, max_power))

            # Compute nominal germ process matrix
            germ_process_matrix_nominal = self._compute_germ_process_matrix(germ, theta_dict_nominal)

            for power in range(1, max_power + 1):
                # Compute germ^power process matrix at nominal parameters
                G_power_nominal = np.linalg.matrix_power(germ_process_matrix_nominal, power)

                # Compute sensitivity via finite differences for each parameter of each gate model
                for gate_name, gate_info in germ_gate_models.items():
                    n_params = gate_info['n_params']

                    for param_idx in range(n_params):
                        # Perturb this specific parameter
                        theta_dict_perturbed = {gn: params.copy() for gn, params in theta_dict_nominal.items()}
                        epsilon = 1e-6
                        theta_dict_perturbed[gate_name][param_idx] += epsilon

                        # Compute perturbed germ process matrix
                        germ_process_matrix_perturbed = self._compute_germ_process_matrix(germ, theta_dict_perturbed)
                        G_power_perturbed = np.linalg.matrix_power(germ_process_matrix_perturbed, power)

                        # Compute Frobenius norm of difference
                        diff = np.linalg.norm(G_power_perturbed - G_power_nominal, 'fro')
                        sensitivity = diff / epsilon

                        germ_sensitivity_data[gate_name][param_idx, power-1] = sensitivity

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

    def analyze_germ_amplification_completeness(self, germs=None, sensitivity_threshold=1e-6):
        """Analyze whether the germ set provides amplification for all gate model parameters.

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

        # Use current germs if none provided
        germs_to_analyze = germs if germs is not None else self.germs

        if not germs_to_analyze:
            return {
                'amplification_status': 'no_germs',
                'amplified_parameters': {},
                'unamplified_parameters': {},
                'parameter_sensitivities': {},
                'warnings': ['No germs provided - cannot analyze amplification completeness']
            }

        # Compute sensitivity for the germs
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

        for gate_name, param_names in all_parameters.items():
            parameter_sensitivities[gate_name] = {}

            for param_idx, param_name in enumerate(param_names):
                max_sensitivity = 0
                best_germ = None
                best_power = None

                # Check sensitivity across all germs and powers
                for germ_name, gate_sensitivities in sensitivity_data.items():
                    if gate_name in gate_sensitivities:
                        sensitivity_matrix = gate_sensitivities[gate_name]

                        # Find maximum sensitivity for this parameter across all powers
                        param_sensitivities = sensitivity_matrix[param_idx, :]
                        germ_max_sensitivity = np.max(param_sensitivities)

                        if germ_max_sensitivity > max_sensitivity:
                            max_sensitivity = germ_max_sensitivity
                            # Find the power that gives maximum sensitivity
                            best_power_idx = np.argmax(param_sensitivities)
                            best_power = best_power_idx + 1  # Convert from 0-indexed to 1-indexed
                            best_germ = germ_name

                parameter_sensitivities[gate_name][param_name] = {
                    'max_sensitivity': max_sensitivity,
                    'best_germ': best_germ,
                    'best_power': best_power,
                    'is_amplified': max_sensitivity >= sensitivity_threshold
                }

                if max_sensitivity >= sensitivity_threshold:
                    amplified_parameters[gate_name].append(param_name)
                else:
                    unamplified_parameters[gate_name].append(param_name)
                    warnings.append(f"Parameter '{param_name}' in gate '{gate_name}' has low sensitivity (max: {max_sensitivity:.2e}) - may not be well amplified by any germ")

        # Determine overall amplification status
        total_unamplified = sum(len(params) for params in unamplified_parameters.values())
        if total_unamplified == 0:
            amplification_status = 'complete'
        else:
            amplification_status = 'incomplete'

        return {
            'amplification_status': amplification_status,
            'amplified_parameters': amplified_parameters,
            'unamplified_parameters': unamplified_parameters,
            'parameter_sensitivities': parameter_sensitivities,
            'warnings': warnings
        }

    def print_amplification_diagnostics(self, germs=None, sensitivity_threshold=1e-6):
        """Print diagnostic information about germ amplification completeness.

        This method provides a human-readable summary of which parameters are
        well-amplified by the germ set and which are not.

        Args:
            germs: List of germs to analyze (if None, use current germs)
            sensitivity_threshold: Minimum sensitivity value to consider a parameter amplified
        """
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
            print("The germ set provides good sensitivity to all gate model parameters.")
        else:
            print("\n⚠ INCOMPLETE AMPLIFICATION DETECTED")
            print("Some gate model parameters have low sensitivity to the germ set.")

        print("\n" + "-"*80)
        print("PARAMETER AMPLIFICATION SUMMARY")
        print("-"*80)

        for gate_name in self.gate_models.keys():
            print(f"\nGate: {gate_name}")

            amplified = diagnostics['amplified_parameters'][gate_name]
            unamplified = diagnostics['unamplified_parameters'][gate_name]

            if amplified:
                print(f"  ✓ Well-amplified parameters ({len(amplified)}):")
                for param_name in amplified:
                    sensitivity_info = diagnostics['parameter_sensitivities'][gate_name][param_name]
                    print(f"    - {param_name}: max sensitivity = {sensitivity_info['max_sensitivity']:.2e}")
                    print(f"      (best germ: {sensitivity_info['best_germ']}, power: {sensitivity_info['best_power']})")

            if unamplified:
                print(f"  ⚠ Low-sensitivity parameters ({len(unamplified)}):")
                for param_name in unamplified:
                    sensitivity_info = diagnostics['parameter_sensitivities'][gate_name][param_name]
                    print(f"    - {param_name}: max sensitivity = {sensitivity_info['max_sensitivity']:.2e}")
                    print(f"      (best germ: {sensitivity_info['best_germ']}, power: {sensitivity_info['best_power']})")

        if diagnostics['warnings']:
            print("\n" + "-"*80)
            print("WARNINGS")
            print("-"*80)
            for warning in diagnostics['warnings']:
                print(f"  ⚠ {warning}")

        print("\n" + "="*80)

        return diagnostics

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

