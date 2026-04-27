import numpy as np
from pathlib import Path 
import re
from ionsim.GST_data_parser import ParsedCircuit, ParsedGate
from itertools import product

class GSTCircuitPlanner:
    def __init__(self, gate_names: list[str], qubit_labels, prep_fiducials = None, measure_fiducials = None, germs = None, germ_powers=[1,2,4,8,16]):
        """ Constructor for GST Circuit Planner class """ 

        self.qubit_labels = qubit_labels
        self.gate_names = gate_names
        self.germ_powers = germ_powers
        
        # Build Parsed Gate objects from gate names and store them in a dictionary  
        self._construct_gate_name_to_object_mapping(gate_names, qubit_labels) 

        # Set up prep/measure/germ circuits depending on user input. A default is used if none is supplied.  
        if prep_fiducials is None and len(qubit_labels) == 1:
            # Use standard 1Q GST fiducial choices 
            prep_fiducials, measure_fiducials = self.standard_1Q_fiducials()
        elif prep_fiducials is None and len(qubit_labels) > 1:
            raise IonSimError(f"2-qubit GST circuit planning default options are currently not implemented in IonSim. Please specify a choice of fiducial prep circuits.")
        
        if germs is None and len(qubit_labels) == 1:
            germs = self.standard_1Q_germs()
        
        self.prep_fiducials = prep_fiducials
        self.germs = germs 
        self.measure_fiducials = measure_fiducials


    def _construct_gate_name_to_object_mapping(self, gate_names: list[str], qubit_labels: list[str]): 
        """ Set up the gate name -> ParsedGate look up dictionary """ 
        self.gate_lookup = {}
        for name in gate_names:
            if name == 'idle': # use empty qubit arguments 
                self.gate_lookup[name] = ParsedGate(name, ())
            else:
                self.gate_lookup[name] = ParsedGate(name, tuple(qubit_labels))

    def generate_gst_circuits(self):
        """ Generate the GST circuits to be ran in experiments. Avoid duplicates """ 
        gst_circuits = []
        unique = set() 

        # Combine circuits from linear GST (no germ power) with long-form GST:     
        for circ in (self._linear_gst_circuits() + self._long_gst_circuits()):
            raw_circuit_name = circ.build_circuit_string()
            if raw_circuit_name not in unique:
                unique.add(raw_circuit_name)
                gst_circuits.append(circ)
            
        self.gst_circuits = gst_circuits
        return gst_circuits 

    def _linear_gst_circuits(self):
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

    def _long_gst_circuits(self):
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


        fiducials = [ [], [X_pi2], [Y_pi2], [X_pi2, X_pi2], [Y_pi2, Y_pi2] ]
        return fiducials, fiducials 

    @staticmethod
    def standard_1Q_germs() -> list:
        """ For 1Q gates, the germs are the gates themselves and specific combinations of them. 

            - returns the list of germs; each germ is a list of ParsedGate objects 

        """  
        qubits = (0, )
        X_pi2 = ParsedGate('Gxpi2', qubits)
        Y_pi2 = ParsedGate('Gypi2', qubits)
        idle = ParsedGate('[]', ()) # should it be qubits? 

        germs = [ [X_pi2], [Y_pi2], [idle], [X_pi2, Y_pi2], [X_pi2, X_pi2, Y_pi2] ]
        return germs 

