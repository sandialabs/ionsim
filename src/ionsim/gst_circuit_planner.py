import numpy as np
import re
import yaml 
from pathlib import Path 
from itertools import product
import inspect
import matplotlib.pyplot as plt

from ionsim.state import State 
from ionsim.operator import Operator  
from ionsim.process import Circuit, Gate
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
        self._construct_gate_name_to_object_mapping(gate_names) 

        # Set up prep/measure/germ circuits depending on user input. A default is used if none is supplied.  
        if prep_fiducials is None and measure_fiducials is None and len(qubit_labels) == 1:
            # Use standard 1Q GST fiducial choices 
            prep_fiducials, measure_fiducials = self.standard_1Q_fiducials()
        elif prep_fiducials is None and measure_fiducials is None and len(qubit_labels) > 1:
            prep_fiducials, measure_fiducials = self.standard_nQ_fiducials()
            #raise IonSimError(f"2-qubit GST circuit planning default options are currently not implemented in IonSim. Please specify a choice of fiducial prep circuits.")
        elif prep_fiducials is not None and len(prep_fiducials) == 0:
            prep_fiducials = []

        if measure_fiducials is not None and len(measure_fiducials) == 0:
            measure_fiducials = []

        # Include empty in prep/measure fiducials to ensure "do nothing for no time" circuit is included 
        if germs is None and len(qubit_labels) == 1: 
            germs = self.standard_1Q_germs(gate_names)

        # If optimized mode, optimize germ selection
        # Set mode --> either standard (gate model agnostic) or gate-model optimized 
        self.mode = 'standard' 

        # Check that gate models correspond with gate names if gate models are provided  
        self.gate_models = None
        if gate_models is not None:
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



    def _construct_gate_name_to_object_mapping(self, gate_names: list[str]): 
        """ Set up the gate name -> ParsedGate look up dictionary """ 
        self.gate_lookup = {}
        for name in gate_names:
            gate = ParsedGate.from_string(name)
            self.gate_lookup[name] = gate
 #            if name == 'idle': # use empty qubit arguments 
 #                self.gate_lookup[name] = ParsedGate(name, ())
 #            else:
 #                self.gate_lookup[name] = ParsedGate(name, tuple(qubit_labels))

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

        do_nothing_circuit = ParsedCircuit.plan([], [], 1, [], self.qubit_labels)
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
        # include empty list for "do nothing for no time" initial sequence 
        # We should only need 4 fiducials for informational completeness 
        fiducials = [[], [X_pi2], [Y_pi2], [X_pi2, X_pi2]] 
        return fiducials, fiducials 

    def standard_nQ_fiducials(self) -> list:
        """N-qubit fiducial from tensor product of 1Q fiducial sets """
        from itertools import product as iter_product
        single_qubit_fids = {}
        for q in self.qubit_labels:
            gx = ParsedGate('Gxpi2', (q,)) 
            gy = ParsedGate('Gypi2', (q,)) 
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

    @staticmethod
    def write_all_circuit_outcomes(filename: str, circuits: list[ParsedCircuit]): 
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

    def compute_circuit_sensitivities(self, gst_circuits: list[ParsedCircuit], circuit_parameters, initial_state: State, outcome_operators: list[Operator]):
        """ Computes sensitivites of each circuit to gate model parameters """ 
        sensitivities = {}
        # remove do nothing circuit 
        do_nothing_circuit = ParsedCircuit.plan([], [], 1, [], self.qubit_labels)
        circuits = gst_circuits.copy()
        if do_nothing_circuit in circuits:
            circuits = circuits.remove(do_nothing_circuit) 
        for circ in circuits:
            sensitivities[tuple(circ.expanded_gates)] = self.compute_circuit_sensitivity(circ, circuit_parameters, initial_state, outcome_operators)
        return sensitivities


    def compute_design_fisher_information(self, gst_circuits: list[ParsedCircuit], circuit_parameters, initial_state: State, outcome_operators: list[Operator]):
        """ Computes sensitivites of each circuit to gate model parameters """ 
        fisher_information = {}
        # Remove do nothing circuit 
        if gst_circuits[0].expanded_gates == []: 
            circuits = gst_circuits.copy()
            circuits = circuits[1:] 

        for circ in circuits:
            fisher_information[tuple(circ.expanded_gates)] = self.compute_circuit_fisher_information(circ, circuit_parameters, initial_state, outcome_operators)
        return fisher_information 


    def compute_circuit_sensitivity(self, circuit: ParsedCircuit, circuit_parameters: dict, initial_state: State, outcome_operators: list[Operator]):
        """ Computes sensitivty of a circuit to gate model parameters """ 
        outcomes = circuit.measurement_data.counts
        N = circuit.measurement_data.total_counts

        # Get list of unique parameters 
        if self.gate_models is None:
            raise ValueError("Gate models must be provided for sensitivity analysis.")

        # Generate ionsim circuit model 
        ism_gates = []
        for gate in circuit.expanded_gates:
            pm_function = self.gate_models[gate]
            parameters = (inspect.signature(pm_function)).parameters.keys()
            fxn_name = pm_function.__name__
            parameters = [fxn_name + "__" + param for param in parameters]
            values = []
            for p in parameters:
                if p in circuit_parameters.keys():
                    values.append(circuit_parameters[p])                    
            parameters_values = dict(zip(parameters, values))  
            gate = Gate.from_process_matrix_function(initial_state.basis, pm_function, parameters_values)
            ism_gates.append(gate)
            
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

    def compute_circuit_fisher_information(self, circuit: ParsedCircuit, circuit_parameters: dict, initial_state: State, outcome_operators: list[Operator]):
        """ Computes sensitivty of a circuit to gate model parameters """ 
        outcomes = circuit.measurement_data.counts
        N = circuit.measurement_data.total_counts

        # Get list of unique parameters 
        if self.gate_models is None:
            raise ValueError("Gate models must be provided for sensitivity analysis.")

        # Generate ionsim circuit model for sensitivity calculation 
        ism_gates = []
        for gate in circuit.expanded_gates:
            pm_function = self.gate_models[gate]
            parameters = (inspect.signature(pm_function)).parameters.keys()
            fxn_name = pm_function.__name__
            parameters = [fxn_name + "__" + param for param in parameters]
            values = []
            for p in parameters:
                if p in circuit_parameters.keys():
                    values.append(circuit_parameters[p])                    
            parameters_values = dict(zip(parameters, values))  
            gate = Gate.from_process_matrix_function(initial_state.basis, pm_function, parameters_values)
            ism_gates.append(gate)
            
        ism_circuit = Circuit.from_gates(ism_gates)
        circuit_pm_function = ism_circuit.process_matrix_function 

        if len(outcome_operators) == 1:
            prob_function = ism_circuit.build_outcome_probabilities_function(initial_state, outcome_operators[0])
            prob, prob_gradients = circuit_pm_function.gradient(prob_function, wrt = list(circuit_parameters.keys()), **circuit_parameters) 
            fisher_info = self.compute_fisher_information(prob, prob_gradients, N)
            return fisher_info
        else:
            if len(outcome_operators) == 0:
                raise IonSimError(f"You must provide at least one outcome operator. Received {len(outcome_operators)}.")
            probs_function = ism_circuit.build_outcome_probabilities_function(initial_state, outcome_operators)
            prob, prob_gradients = circuit_pm_function.jacobian(probs_function, wrt = list(circuit_parameters.keys()), **circuit_parameters) 
            fisher_info = self.compute_fisher_information(prob, prob_gradients, N)
            return fisher_info

    def compute_fisher_information(self, prob, prob_gradients: dict, N: int) -> dict:
        """ returns fisher information matrix from the parameters """ 
        FI = {}
        for param1, gradient1 in prob_gradients.items():
            for param2, gradient2 in prob_gradients.items():
                FI[(param1, param2)] = N*sum([(grad1*grad2)/p for grad1, grad2, p in zip(gradient1, gradient2, prob)])
        return FI 


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

