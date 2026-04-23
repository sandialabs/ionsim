import numpy as np
from dataclasses import dataclass, field
from pathlib import Path 
import re


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
        


    def _construct_gate_name_to_object_mapping(self, gate_names: list[str], qubit_labels: list[str]) -> list[]
        """ Set up the gate name -> ParsedGate look up dictionary """ 
        self.gate_lookup = {}
        for name in gate_names:
            self.gate_lookup[name] = ParsedGate(name, tuple(qubit_labels))


    def generate_circuits(eslf):
        """ Build the GST circuits 


 
