import numpy as np
import re
import yaml 
from pathlib import Path 
from itertools import product

from ionsim.gst_circuit_parser import ParsedCircuit, ParsedGate

class GSTCircuitPlanner:
    def __init__(self, gate_names: list[str], qubit_labels: list[int], prep_fiducials = None, measure_fiducials = None, germs = None, germ_powers=[1,2,4,8,16]):
        """ Constructor for GST Circuit Planner class. The user passes in the gate names and qubit labels at a minimum. 

            - Sets up list of prep gates, measure gates, and germ gates. The class organizes GST circuits based on those gates requested germ powers. 
            - Can write the GST circuit sequences to a file.  

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

    def write_circuit_plan(self, filepath: str | Path, N_qubits:int = 1):
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
                if g in self.gate_lookup:
                    return self.gate_lookup[g]
                raise ValueError(f"Unknown gate name: {g}")
            raise TypeError(f"Bad gate type: {type(g)} -> {g}")

    def to_parsed_seq(self, seq):
        return [self.to_parsed_gate(g) for g in seq]

    
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


