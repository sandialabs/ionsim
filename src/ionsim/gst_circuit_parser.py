import numpy as np
from dataclasses import dataclass, field
from pathlib import Path 
import re

@dataclass()
class CircuitData:
    """ Circuit experiment data either in the form of counts or single-shots with timestamps. 

        - counts: {'0': 100, '1' : 100}  
        - (single) shots: [ (t0, '0') , (t1, '0'), (t2, '1'), ... (t_i, 'outcome') , ... ]
            where t_i are floats representing the time of the measurement. 
    """

    counts: dict[str, int] | None=None
    timestamped_shots: list[tuple[float, str]] | None=None


    @staticmethod
    def from_counts(counts):
        return CircuitData(counts=counts, timestamped_shots = None)
        

    @staticmethod
    def from_timestamped_shots(cls, single_shot_data):
        return CircuitData(timestamped_shots = single_shot_data) 


    def to_counts(self) -> dict:
        """ Time-average single shots into counts (discards time information). """
        if self.counts is not None:
            return self.counts

        c = {}
        # Loop through times and incriment the count of each outcome
        for _, outcome in self.timestamped_shots:
            c[outcome] += c.get(outcome, 0) + 1 
        return c

    def time_binned(self, bin_edges: list[float]):
        """ Bin the single-shot data into windows of time. Bin edges is a list of time points defining the N-1 bins. 

            Returns list of count dictionaries, 1 per bin.

        """
        N_bins = len(bin_edges) - 1

        bins = [{} for _ in range(N_bins) ]

        # loop over all outcomes, binning as the loop proceeds
        for t, outcome in self.timestamped_shots:
            for i in range(N_bins):
                if bin_edges[i] <= t < bin_edges[i + 1]:
                    bins[i][outcome] = bins[i].get(outcome, 0) + 1
                    break
        return bins
                    

    @property
    def total_counts(self):
        if self.counts is not None:
            return sum(self.counts.values()) 
        return len(self.timestamped_shots)



@dataclass(frozen=True) 
class ParsedGate:
    """ Parsed gate from GST file with information on the gate and involved qubits """

    name: str 
    qubits: tuple[int, ...] # qubits are indexed by integers starting at 0 

    def __repr__(self):
        if (self.name == "idle") or (self.name == "I"):
            return "[]"
        if not self.qubits:
            return self.name
        q = ",".join(str(q) for q in self.qubits)
        return f"{self.name}:{q}"


@dataclass
class ParsedCircuit:
    """ Parsed circuit from GST file, optionally with measurement outcomes 

        - follows convention of Prep gates --> {(Germ_gates)^germ_power} --> measure gates  
        - Stores the file string contents

    """
    # ParsedCircuit class should remain unfrozen so its measurement_data attribute can be modified by an experiment. 
    unparsed_data: str
    fiducial_prep_gates: list[ParsedGate]
    germ_gates: list[ParsedGate]
    fiducial_measurement_gates: list[ParsedGate]
    germ_power: int 

    line_labels: list[int]   # not as important, TODO: delete?   
    measurement_data: CircuitData | None


    @property
    def expanded_gates(self) -> list[ParsedGate]:
        """ List of gates, expanded (no germ power included) """
        return self.fiducial_prep_gates + self.germ_gates * self.germ_power + self.fiducial_measurement_gates


    @property
    def total_counts(self) -> int:
        """ Number of measurement counts """
        return self.measurement_data.total_counts 
        #return sum(self.measurement_counts.values())


    @property
    def depth(self) -> int:
        """ Number of total gates in the circuit """
        return len(self.expanded_gates)



    def __repr__(self):
        gates_readable = " ".join(repr(gate) for gate in self.expanded_gates) or "(empty)"
        return f"ParsedCircuit({gates_readable}, data={self.measurement_data})"


    def build_circuit_string(self) -> str: 
        """ Build string representation, useful for writing circuit instructions. """
        # Ex] Gxpi2:0(Gxpi2:0)^{2}Gypi2:0@(0)

        # Helper function for chaining gate names into a single string 
        def _gates_to_str(gates: list[ParsedGate]):
            return "".join(repr(g) for g in gates)

        prep = _gates_to_str(self.fiducial_prep_gates)
        measure = _gates_to_str(self.fiducial_measurement_gates)

        if self.germ_gates:
            germ = _gates_to_str(self.germ_gates)
            if self.germ_power > 1:
                germ_block = f"({germ})^{self.germ_power}"
            else:
                germ_block = f"({germ})"
            circuit = f"{prep}{germ_block}{measure}"
        elif not prep and not measure:
            circuit = "{}"
        else:
            circuit = f"{prep}{measure}"

        labels = ",".join(str(q) for q in self.line_labels)
        return f"{circuit}@({labels})"

    def _format_circuit_line(self):
        """ Formats the circuit string with measurement information. """
        # TODO: Handle case where data is time-dependent 
        circuit_str = self.build_circuit_string()
        if self.measurement_data is None or self.measurement_data.counts is None: 
            return circuit_str

        # Check spacings to align with gstdata formatting from pygsti 
        counts_str = "  ".join(str(self.measurement_data.counts[k]) for k in sorted(self.measurement_data.counts.keys()))
        return f"{circuit_str}  {counts_str}"


    @staticmethod
    def plan(prep_gates: list[ParsedGate], germ_gates: list[ParsedGate], germ_power: int, measure_gates: list[ParsedGate], line_labels: list[int]):
        """ Constructs and returns a circuit that is planned - no measurement data exists yet. """ 
        planned_circ = ParsedCircuit("", prep_gates, germ_gates, measure_gates, germ_power, line_labels, measurement_data = None)
        planned_circ.unparsed_data = planned_circ.build_circuit_string()
        return planned_circ  


    def append_to_file(self, filename):
        """ Appends circuit information to a gstdata type file"""
        with open(filename, 'a') as f:
            f.write(self._format_circuit_line() + "\n")


def parse_circuit_string(circ: str) -> list[ParsedGate]:
    """ Extract the gate sequence from the circuit string """

    # Extracts from patterns like '', '[]', '{}', 'Gxpi2:0'

    # Check that we have a valid circuit string 
    if not circ or not circ.strip():
        return []

    gates = []

    pattern = r"([A-Za-z]\w*):(\d+(?::\d+)*)|\[\]"

    # Find matches for the pattern and build a ParsedGate object for each match 
    for m in re.finditer(pattern, circ):
        if m.group(0) == "[]":
            gates.append(ParsedGate("idle", ()))
        else:
            name = m.group(1)
            qubits = tuple(int(qubit) for qubit in m.group(2).split(":"))
            gates.append(ParsedGate(name, qubits))

    return gates 



def parse_measurement_outcome_labels(header: str) -> list[str]:
    """ Extract measurement outcome labels """ 
    match = re.search(r"Columns\s*=\s*(.+)", header)

    if not match:
        raise ValueError(f"Cannot parse header: {header!r} from file.")

    columns = match.group(1)

    labels = [col.strip().split()[0] for col in columns.split(",")]
    return labels 



def parse_circuit_line(line: str, outcome_labels: list[str]) -> ParsedCircuit:
    """ Parse a GST circuit line, containing a sequence of gates and possibly measurement count outcomes. """ 
    ## TODO: Add parsing functionality for t-dependent data. This is currently not handled 
    # For GST data files, this is of the format circuit list then measurement counts 

    # Strip the line if it's not already stripped 
    line = line.strip()

    # Separate the circuit from the measurement outcomes
    #match = re.match(r"^(.+?)@\(([^)]+)\)\s+(.+)$", line) 
    match = re.match(r"^(.+?)@\(([^)]+)\)(?:\s+(.+))?$", line) 
    if not match:
        raise ValueError(f"Cannot parse line: {line!r}")

    # Match group 1 is the circuit sequence  
    # Match group 2 is the @(0) directive so it should be ignored  
    # Match group 3 is the measurement counts  
    circuit_sequence = match.group(1).strip() 
    line_labels = [int(q) for q in match.group(2).split(",")]
    count_info = match.group(3) # None if there's no measurement information  

    # Handle cases where there is / isn't measurement data: 
    if count_info is not None:
        count_values = [int(x) for x in count_info.split()]

        # Build CircuitData if there is measurement data  
        if len(count_values) != len(outcome_labels):
            raise ValueError(f"Expected {len(outcome_labels)} measurement outcomes but received {len(count_values)} on line: {line!r}")
   
        # Create a dictionary with measurement outcomes and corresponding counts for this line  
        measurement_counts = dict(zip(outcome_labels, count_values)) 

        parsed_measurement_data = CircuitData.from_counts(measurement_counts)
    else:
        parsed_measurement_data = None 
    
    # Parse circuit sequence, starting with empty (do nothing -- prep then measure) string 
    if circuit_sequence == "{}":
        return ParsedCircuit(unparsed_data = line, fiducial_prep_gates=[], germ_gates = [], fiducial_measurement_gates = [],
                            germ_power = 1, line_labels = line_labels, measurement_data = parsed_measurement_data) 

    # Find the germ block if it exists  
    germ_match = re.search(r"\(([^)]*)\)(?:\^(\d+))?", circuit_sequence)

    # Parse the germ content vs. the prep and measure  
    if germ_match:
        prep = circuit_sequence[: germ_match.start()]
        measure = circuit_sequence[germ_match.end() :]
        germ = germ_match.group(1)
        germ_power = int(germ_match.group(2)) if germ_match.group(2) else 1 
        
        prep_gates = parse_circuit_string(prep) 
        measure_gates = parse_circuit_string(measure) 
        germ_gates = parse_circuit_string(germ)
    else:
        # No germ, everything is either a prep or measure gate (convention would choose one)
        prep_gates = parse_circuit_string(circuit_sequence)
        germ_gates = []
        measure_gates = []
        germ_power = 1
        
    return ParsedCircuit(line, prep_gates, germ_gates, measure_gates, germ_power, line_labels, parsed_measurement_data) 


def parse_gst_circuit_file(filepath: str | Path) -> list[ParsedCircuit]:
    """ Parse a GST circuit results file, containing circuits and outcomes on each line. """
    filepath = Path(filepath)
    results: list[ParsedCircuit] = []
    outcome_labels: list[str] | None = None

    # Open file and parse each line: 
    with open(filepath, "r") as f:
        for line in f:
            # Strip converts line into a string
            stripped_line = line.strip() 
            if not stripped_line: 
                continue

            # Parse header vs. circuit lines     
            if stripped_line.startswith("#"):
                if "Columns" in stripped_line:
                    outcome_labels = parse_measurement_outcome_labels(stripped_line)
                continue 

            if outcome_labels is None: 
                raise ValueError("Encountered circuit data before a header.")

            results.append(parse_circuit_line(stripped_line, outcome_labels))

    return results 
