import numpy as np
import re
import yaml 
from pathlib import Path 
from itertools import product
import inspect
import matplotlib.pyplot as plt
import warnings

from ionsim.state import State 
from ionsim.operator import Operator  
from ionsim.process import Circuit, Gate
from ionsim.custom_types import Matrix 
from ionsim.gst_circuit_parser import GstCircuit, GstGate, gate_from_label

""" Circuit planner has 2 modes: 1) Gate model agnostic, 2) optimized planner based on gate models and germ sensitivies. """ 
class GSTCircuitPlanner:
    def __init__(self, gate_names: list[str], qubit_labels: list[int], prep_fiducials: list[list[str]] | None=None,
                    measure_fiducials: list[list[str]] | None=None, germs: list[list[str]] | None=None, germ_powers: list[int]=[1,2,4,8,16],
                    gate_models: dict[str, callable] | None=None, long_sequence_GST: bool=True):
        """ Constructor for GST Circuit Planner class. The user passes in the gate names and qubit labels at a minimum.

            - All gates are specified by string label with their qubit argument(s), e.g. 'Gxpi2:0', 'MS:0:1'.
              The global idle gate may be written 'idle' or '[]'.
            - gate_names: the gate set, e.g. ['Gxpi2:0', 'Gypi2:0', 'idle'].
            - prep_fiducials / measure_fiducials / germs: lists of gate-label sequences, e.g. [[], ['Gxpi2:0'], ['Gxpi2:0', 'Gxpi2:0']].
              Defaults are used for any that are not supplied.
            - Sets up list of prep gates, measure gates, and germ gates. The class organizes GST circuits based on those gates requested germ powers.
            - Can write the GST circuit sequences to a file.
            - Optional arguments to provide a dictionary of gate process matrix models keyed by gate label (e.g. {'Gxpi2:0': model}).
            - long GST: 'True' will use germs to do long-gst circuits, 'false' will use only linear gst circuits

        """
        self.qubit_labels = list(qubit_labels)
        if long_sequence_GST:
            self.germ_powers = germ_powers
        else:
            self.germ_powers = [1]

        # Build GstGate objects from gate names and store them in a dictionary keyed by canonical label
        self._construct_gate_name_to_object_mapping(gate_names)

        # Set up prep/measure/germ circuits depending on user input. A default is used if none is supplied.
        if prep_fiducials is None and measure_fiducials is None:
            if len(self.qubit_labels) == 1:
                # Use standard 1Q GST fiducial choices
                prep_fiducials, measure_fiducials = self.standard_1Q_fiducials(self.qubit_labels[0])
            else:
                prep_fiducials, measure_fiducials = self.standard_nQ_fiducials()
        elif prep_fiducials is None or measure_fiducials is None:
            raise ValueError("Specify both prep_fiducials and measure_fiducials, or neither (to use the defaults).")

        if germs is None:
            if len(self.qubit_labels) == 1:
                germs = self.standard_1Q_germs(self.gate_names, self.qubit_labels[0])
            elif long_sequence_GST:
                raise ValueError("Default germs are only available for 1-qubit GST. Please specify germs as lists of gate labels, e.g. [['Gxpi2:0'], ['MS:0:1']].")
            else:
                germs = []

        # Set mode --> either standard (gate model agnostic) or gate-model optimized
        self.mode = 'standard'

        # Gate models (optional, used for sensitivity / Fisher information analysis) are keyed by gate label
        self.gate_models = None
        self.ism_gate_cache = {}
        self.num_model_parameters = None
        if gate_models is not None:
            self.gate_models = {self.to_gst_gate(label): model for label, model in gate_models.items()}

        self.long_GST = long_sequence_GST

        # Convert string-based fiducials/germs to internal GstGate sequences
        self.prep_fiducials = [self.to_gst_sequence(fid, 'prep fiducial') for fid in prep_fiducials]
        self.measure_fiducials = [self.to_gst_sequence(fid, 'measure fiducial') for fid in measure_fiducials]
        self.germs = [self.to_gst_sequence(germ, 'germ') for germ in germs]

        if self.mode == 'optimized':
            raise NotImplementedError("Optimized germ selection is not currently available.")


    def _construct_gate_name_to_object_mapping(self, gate_names: list[str]):
        """ Set up the canonical gate label -> GstGate look up dictionary """
        if isinstance(gate_names, str):
            raise TypeError(f"gate_names must be a list of gate labels, e.g. ['Gxpi2:0', 'Gypi2:0']; received the string {gate_names!r}.")
        self.gate_lookup = {}
        for name in gate_names:
            gate = gate_from_label(name)
            if gate.label in self.gate_lookup:
                raise ValueError(f"Gate {name!r} appears more than once in gate_names.")
            for q in gate.qubits:
                if q not in self.qubit_labels:
                    raise ValueError(f"Gate {name!r} acts on qubit {q}, which is not in qubit_labels {self.qubit_labels}.")
            self.gate_lookup[gate.label] = gate
        # Store canonical labels (e.g. '[]' -> 'idle')
        self.gate_names = list(self.gate_lookup.keys())

    def generate_gst_circuits(self) -> list:
        """Generate GST circuits. Convert string gates to GstGate and avoid duplicates."""

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
                circuits.append( GstCircuit._from_gates(prep_fiducial, [], 1, measure_fiducial, self.qubit_labels)) 

        # Group 2: Fiducial prep, gate, and measure. For each gate, run the prep & measure circuits. 
        for gate in self.gate_lookup.values():
            for prep_fiducial in self.prep_fiducials:
                for measure_fiducial in self.measure_fiducials:
                    circuits.append( GstCircuit._from_gates(prep_fiducial, [gate], 1, measure_fiducial, self.qubit_labels)) 

        do_nothing_circuit = GstCircuit._from_gates([], [], 1, [], self.qubit_labels)
        if do_nothing_circuit not in circuits:
            circuits.insert(0, do_nothing_circuit)

        return circuits 

    def _long_gst_circuits(self) -> list:
        """ Long-form GST circuits: fiducial_prep + prep^{germ} + fiducial_measure """ 
        assert self.long_GST
        circuits = []
        for germ in self.germs:
            for power in self.germ_powers:
                for prep_fiducial in self.prep_fiducials:
                    for measure_fiducial in self.measure_fiducials:
                        circuits.append( GstCircuit._from_gates(prep_fiducial, germ, power, measure_fiducial, self.qubit_labels)) 

        return circuits 

    def write_circuit_plan(self, filepath: str | Path, N_qubits: int = 1):
        """ Writes a gst data file compatible with the parser """ 
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
    def standard_1Q_fiducials(qubit: int=0) -> tuple[list[list[str]], list[list[str]]]:
        """ For 1Q gates, the fiducial circuits are standardized for {X_pi/2, Y_pi/2} gates.

            - returns the prep and measure fiducials as lists of gate-label sequences

        """
        X_pi2 = f'Gxpi2:{qubit}'
        Y_pi2 = f'Gypi2:{qubit}'

        # include empty list for "do nothing for no time" initial sequence
        # We should only need 4 fiducials for informational completeness
        fiducials = [[], [X_pi2], [Y_pi2], [X_pi2, X_pi2]]
        return fiducials, [list(fid) for fid in fiducials]

    def standard_nQ_fiducials(self) -> tuple[list[list[str]], list[list[str]]]:
        """N-qubit fiducial from tensor product of 1Q fiducial sets """
        from itertools import product as iter_product
        single_qubit_fids = {}
        for q in self.qubit_labels:
            gx = f'Gxpi2:{q}'
            gy = f'Gypi2:{q}'
            single_qubit_fids[q] = [
                [],
                [gx],
                [gy],
                [gx, gx],
            ]

        fiducials = []
        # Cartesian product across all qubits for N qubits
        for combo in iter_product(*(single_qubit_fids[q] for q in self.qubit_labels)):
            # Make gate lists from each qubit
            fid = []
            for gate_list in combo:
                fid.extend(gate_list)
            fiducials.append(fid)

        return fiducials, [list(fid) for fid in fiducials]


    @staticmethod
    def standard_1Q_germs(gate_names: list[str], qubit: int=0) -> list[list[str]]:
        """ For 1Q gates, the germs are the gates themselves and specific combinations of them.

            - returns the list of germs; each germ is a list of gate labels

        """
        X_pi2 = f'Gxpi2:{qubit}'
        Y_pi2 = f'Gypi2:{qubit}'

        has_idle = any(gate_from_label(name).is_idle for name in gate_names)
        if has_idle:
            germs = [ [X_pi2], [Y_pi2], ['idle'], [X_pi2, Y_pi2], [X_pi2, X_pi2, Y_pi2] ]
        else:
            germs = [ [X_pi2], [Y_pi2], [X_pi2, Y_pi2], [X_pi2, X_pi2, Y_pi2] ]

        return germs

    def _generate_candidate_germs_1Q(self, gate_names: list[str]) -> list:
        """Generate a comprehensive set of candidate germs for 1Q optimization."""
        X_pi2 = 'Gxpi2:0'
        Y_pi2 = 'Gypi2:0'
        idle = 'idle'
        gate_names = [gate_from_label(name).label for name in gate_names]

        # Generate comprehensive candidate set
        candidates = []

        # Single gates
        if X_pi2 in gate_names:
            candidates.append([X_pi2])
        if Y_pi2 in gate_names:
            candidates.append([Y_pi2])
        if 'idle' in gate_names:
            candidates.append([idle])

        # Two-gate sequences
        if X_pi2 in gate_names and Y_pi2 in gate_names:
            candidates.extend([
                [X_pi2, Y_pi2],
                [Y_pi2, X_pi2],
                [X_pi2, X_pi2],
                [Y_pi2, Y_pi2]
            ])

        # Three-gate sequences
        if X_pi2 in gate_names and Y_pi2 in gate_names:
            candidates.extend([
                [X_pi2, X_pi2, Y_pi2],
                [Y_pi2, Y_pi2, X_pi2],
                [X_pi2, Y_pi2, X_pi2],
                [Y_pi2, X_pi2, Y_pi2],
                [X_pi2, X_pi2, X_pi2],
                [Y_pi2, Y_pi2, Y_pi2]
            ])

        # Four-gate sequences (for more comprehensive coverage)
        if X_pi2 in gate_names and Y_pi2 in gate_names:
            candidates.extend([
                [X_pi2, Y_pi2, X_pi2, Y_pi2],
                [X_pi2, X_pi2, Y_pi2, Y_pi2],
                [X_pi2, Y_pi2, Y_pi2, X_pi2]
            ])

        return candidates


    @staticmethod
    def write_all_circuit_outcomes(filename: str, circuits: list[GstCircuit]): 
        """ Writes all circuit information to a file """
        N_qubits = circuits[0].num_qubits 
        d = 2**N_qubits # Hilbert space dimensionality 
        outcome_labels = [''.join(bits) for bits in product('01', repeat=N_qubits)] 

        with open(filename, 'w') as f:
            # Write the header 
            columns = ", ".join(f"{outcome} count" for outcome in outcome_labels)
            f.write(f"## Columns = {columns}\n")

            for circ in circuits:
                f.write(circ._format_circuit_line() + "\n")

    def create_circuit_outcomes_file(self, filename: str): 
        """ Creates a GST circuit file with appropriate header """ 
        N_qubits = len(self.qubit_labels)
        d = 2**N_qubits # Hilbert space dimensionality 
        outcome_labels = [''.join(bits) for bits in product('01', repeat=N_qubits)] 

        with open(filename, 'w') as f:
            # Write the header 
            columns = ", ".join(f"{outcome} count" for outcome in outcome_labels)
            f.write(f"## Columns = {columns}\n")

    def to_gst_gate(self, label: str) -> GstGate:
        """ Converts a gate label (string) to a GstGate, checking it belongs to the planner's gate set.

            The idle gate ('idle' or '[]') is always permitted.
        """
        gate = gate_from_label(label)
        if gate.is_idle or gate.label in self.gate_lookup:
            return self.gate_lookup.get(gate.label, gate)
        raise ValueError(f"Gate {label!r} is not in the planner's gate set {self.gate_names}.")

    def to_gst_sequence(self, seq: list[str], role: str='gate sequence') -> list[GstGate]:
        """ Converts a list of gate labels (e.g. ['Gxpi2:0', 'Gypi2:0']) to a list of GstGates. """
        if isinstance(seq, str):
            raise TypeError(f"Each {role} must be a list of gate labels, e.g. ['Gxpi2:0', 'Gypi2:0']; received the string {seq!r}. "
                            f"Wrap single gates in a list, e.g. [{seq!r}].")
        return [self.to_gst_gate(g) for g in seq]


    def _compute_germ_process_matrix(self, germ, theta_dict):
        """Compute the process matrix for a germ given parameter values for each gate model.

        Args:
            germ: List of GstGate objects representing the germ
            theta_dict: Dictionary mapping gate labels (e.g. 'Gxpi2:0') to their parameter arrays

        Returns:
            Process matrix for the germ sequence
        """
        d = 2**len(self.qubit_labels)
        d2 = d**2

        germ_process_matrix = np.eye(d2, dtype=complex)

        for gate in germ:
            # Get the gate model function for this gate (gate models are keyed by GstGate internally)
            gate_func = self.gate_models[gate]

            # Get parameters for this specific gate model (theta_dict is keyed by gate label)
            theta = theta_dict[gate.label]

            # Evaluate at current parameters
            gate_matrix = gate_func(*theta)
            germ_process_matrix = gate_matrix @ germ_process_matrix

        return germ_process_matrix

    def compute_circuit_sensitivities(self, gst_circuits: list[GstCircuit], circuit_parameters, initial_state: State, outcome_operators: list[Operator]):
        """ Computes sensitivites of each circuit to gate model parameters """ 
        sensitivities = {}
        # remove do nothing (empty) circuit(s), which carry no information about gate model parameters
        circuits = [circ for circ in gst_circuits if circ.depth > 0]
        #for circ in self.gst_circuits:
        for circ in circuits:
            sensitivities[tuple(circ.expanded_gates)] = self.compute_circuit_sensitivity(circ, circuit_parameters, initial_state, outcome_operators)
        return sensitivities


    def compute_design_fisher_information(self, gst_circuits: list[GstCircuit], circuit_parameters, initial_state: State, outcome_operators: list[Operator]):
        """ Computes sensitivites of each circuit to gate model parameters """ 
        fisher_information = {}
        fisher_information_matrices = {}
        # remove do nothing (empty) circuit(s), which carry no information about gate model parameters
        circuits = [circ for circ in gst_circuits if circ.depth > 0]

        if self.gate_models is None:
            raise ValueError("Gate models must be provided for sensitivity analysis.")

        # Refresh gate model cache if necessary :
        if not self.ism_gate_cache:
            self.refresh_ism_gate_cache(circuit_parameters, initial_state)

        for circ in circuits:
            fisher_information[tuple(circ.expanded_gates)], fisher_information_matrices[tuple(circ.expanded_gates)] = self.compute_circuit_fisher_information(circ, circuit_parameters, initial_state, outcome_operators)
        return fisher_information, fisher_information_matrices 


    def refresh_ism_gate_cache(self, circuit_parameters: dict, initial_state: State):
        """ Rebuilds the caache of IonSim (ism) Gate objects """ 
        self.ism_gate_cache = {}
        for gate in self.gate_models: 
            pm_function = self.gate_models[gate]
            parameters = (inspect.signature(pm_function)).parameters.keys()
            fxn_name = pm_function.__name__
            parameters = [fxn_name + "__" + param for param in parameters]
            values = []
            for p in parameters:
                if p in circuit_parameters.keys():
                    values.append(circuit_parameters[p])                    
            parameters_values = dict(zip(parameters, values))  
            self.ism_gate_cache[gate] = Gate.from_process_matrix_function(initial_state.basis, pm_function, parameters_values)

    def compute_circuit_sensitivity(self, circuit: GstCircuit, circuit_parameters: dict, initial_state: State, outcome_operators: list[Operator]):
        """ Computes sensitivty of a circuit to gate model parameters """ 
        outcomes = circuit.measurement_data.counts
        N = circuit.measurement_data.total_counts

        # Get list of unique parameters 
        if self.gate_models is None:
            raise ValueError("Gate models must be provided for sensitivity analysis.")

        # Build gate model cache:
        if not self.ism_gate_cache:
            self.refresh_ism_gate_cache(circuit_parameters, initial_state)

        # Generate ionsim circuit model 
        ism_gates = []
        for gate in circuit.expanded_gates:
            ism_gates.append(self.ism_gate_cache[gate])
            
        ism_circuit = Circuit.from_gates(ism_gates)
        circuit_pm_function = ism_circuit.process_matrix_function 

        # Test outcome probability function  
        if len(outcome_operators) == 1:
            prob_function = ism_circuit.build_outcome_probabilities_function(initial_state, outcome_operators[0])
            prob, prob_gradients = circuit_pm_function.gradient(prob_function, wrt = list(circuit_parameters.keys()), **circuit_parameters) 
            return prob_gradients
        else:
            if len(outcome_operators) == 0:
                raise IonSimError(f"You must provide at least one outcome operator. Received {len(outcome_operators)}.")
            probs_function = ism_circuit.build_outcome_probabilities_function(initial_state, outcome_operators)
            prob, prob_gradients = circuit_pm_function.jacobian(probs_function, wrt = list(circuit_parameters.keys()), **circuit_parameters) 
            return prob_gradients

    def compute_circuit_fisher_information(self, circuit: GstCircuit, circuit_parameters: dict, initial_state: State, outcome_operators: dict[str, Operator]):
        """ Computes sensitivty of a circuit to gate model parameters """ 
        outcomes = circuit.measurement_data.counts
        N = circuit.measurement_data.total_counts

        # Get list of unique parameters 
        if self.gate_models is None:
            raise ValueError("Gate models must be provided for sensitivity analysis.")

        # Generate ionsim circuit model 
        if not self.ism_gate_cache:
            self.refresh_ism_gate_cache(circuit_parameters, initial_state)

        ism_gates = []
        for gate in circuit.expanded_gates:
            ism_gates.append(self.ism_gate_cache[gate])
            
        ism_circuit = Circuit.from_gates(ism_gates)
        circuit_pm_function = ism_circuit.process_matrix_function 
        # Parse args for circuit process matrix function and ensure matching:
        possible_args = list(inspect.signature(circuit_pm_function).parameters.keys())
        input_args = {}
        for p in circuit_parameters.keys():
            if p in possible_args:
                input_args[p] = circuit_parameters[p]  

        if len(outcome_operators) == 1:
            prob_function = ism_circuit.build_outcome_probabilities_function(initial_state, list(outcome_operators.values())[0])
            prob, prob_gradients = circuit_pm_function.gradient(prob_function, wrt = list(circuit_parameters.keys()), **circuit_parameters) 
            hessian = circuit_pm_function.hessian(prob_function, wrt = list(circuit_parameters.keys()), **circuit_parameters) 
            fisher_info = self.compute_fisher_information(prob, prob_gradients, N)
            return fisher_info
        else:
            if len(outcome_operators) == 0:
                raise IonSimError(f"You must provide at least one outcome operator. Received {len(outcome_operators)}.")
            probs_function = ism_circuit.build_outcome_probabilities_function(initial_state, outcome_operators.values())
            prob, prob_gradients = circuit_pm_function.jacobian(probs_function, wrt = list(input_args.keys()), **input_args) 
            hessian = circuit_pm_function.hessian_per_outcome(probs_function, wrt = list(input_args.keys()), outcome_labels = outcome_operators.keys(), **input_args) 
            fisher_dict, fisher_info_matrix = self.compute_fisher_information(prob, prob_gradients, hessian, N)
            return fisher_dict, fisher_info_matrix

    def compute_fisher_information(self, prob, prob_gradients: dict, hessian: dict, N: int, include_hessian:bool=False) -> (dict, Matrix):
        """ returns fisher information matrix from the parameters """ 
        FI_dict = {}
        parameters = list(prob_gradients.keys())
        self.num_model_parameters = len(parameters)
        size = len(parameters)
        FI_matrix = np.zeros((size, size)) 
        for param1, gradient1 in prob_gradients.items():
            for param2, gradient2 in prob_gradients.items():
                hessians = list(hessian[param1][param2].values())
                if include_hessian:
                    FI_contribution = N*sum([((grad1*grad2)/p - H) for grad1, grad2, p, H in zip(gradient1, gradient2, prob, hessians)])
                else:
                    FI_contribution = N*sum([((grad1*grad2)/p) for grad1, grad2, p, H in zip(gradient1, gradient2, prob, hessians)])
                key = (param1, param2)
                FI_dict[key] = FI_contribution
                i = parameters.index(param1) 
                j = parameters.index(param2) 
                FI_matrix[i, j] = FI_contribution

        return FI_dict, FI_matrix 


    def compute_design_eigenvalues_and_inverse_fisher_matrix(self, fisher_information_matrices: dict) -> tuple[dict, dict]:
        """ Compute eigenvalues of accumulated fisher information matrix and matrix inverse across the design """  
        # Accumulate the fisher information matrix across the design 
        circuit_indices = np.array(list(range(1, len(fisher_information_matrices)+1)))
        num_gate_parameters = self.num_model_parameters
        accumulated_I_matrix = np.zeros((len(circuit_indices), num_gate_parameters, num_gate_parameters))

        eigenvalues = {}   
        inverse_Fisher_infos = {} # np.zeros_like(accumulated_I_matrix)
        for i, (circ, fisher_info) in enumerate(fisher_information_matrices.items()):
            I_accumulated = sum(list(fisher_information_matrices.values())[:(i+1)])
            #eigenvalues[i, :], _ = np.linalg.eigh(I_accumulated)
            eigenvalues[circ] = np.zeros(num_gate_parameters)
            eigenvalues[circ][:], _ = np.linalg.eigh(I_accumulated)
            inverse_Fisher_infos[circ] = np.zeros_like(accumulated_I_matrix) 
            try:
                inverse_Fisher_infos[circ] = np.linalg.inv(I_accumulated)
            except:
                warnings.warn(f"Failed to invert on circuit: {circ}")

        return eigenvalues, inverse_Fisher_infos


    def write_circuit_design(self, filepath):
        """ Writes a design yaml file with circuit design information """
        #filename = 'GST_circuit_design.yaml'  

        def gate_list_to_dict(gate_list):
            """ Convert list of Gate objects to a dictionary format """ 
            return [{'name' : g.name, 'qubits' : list(g.qubits)} for g in gate_list]


        def fiducials_to_dict(fiducials):            
            """ Convert list of fiducial sequences (list of GstGates) to dictionary."""
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
            """ Converts dictionary list of gates to a list of gate labels (strings) """
            return [GstGate(name=g['name'], qubits = tuple(g['qubits'] or ())).label
                for g in dict_list]
        

        def dict_to_fiducials(fid_list):
            """ Converts dictionary list of fiducials to lists of gate labels """
            return [dict_to_gate_list(fid) for fid in fid_list]
            

        with open(filepath, 'r') as f:
            design = yaml.safe_load(f)

        planner = cls(gate_names = design['gate_names'], qubit_labels = design['qubit_labels'],
                    prep_fiducials = dict_to_fiducials(design['prep_fiducials']), 
                    measure_fiducials = dict_to_fiducials(design['measure_fiducials']), 
                    germs = dict_to_fiducials(design['germs']), germ_powers = design['germ_powers'] )

        return planner 

