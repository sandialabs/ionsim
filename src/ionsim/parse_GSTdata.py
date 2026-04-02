import numpy as np
from dataclasses import dataclass, field
from pathlib import Path 
import re


@dataclass(frozen=True) 
class ParsedGate:
    """ Parsed gate from GST file with information on the gate and involved qubits """

    name: str 
    qubits: tuple[int, ...] # qubits are indexed by integers starting at 0 


    def __repr__(self):
        if (self.name == "idle") or (self.name == "I"):
            return "idle"
        q = ",".join(str(q) for q in self.qubits)
        return f"{self.name}:{q}"


@dataclass
class ParsedCircuit:
    """ Parsed circuit from GST file with measurement outcome in counts 

        - follows convention of Prep gates --> {(Germ_gates)^germ_power} --> measure gates  
        
        - Stores the file string contents, 


    """

    unparsed_data: str
    prep_gates: list[ParsedGate]
    germ_gates: list[ParsedGate]
    measurement_gates: list[ParsedGate]
    germ_power: int 

    line_labels: list[int]   # not as important, TODO: delete?   
    measurement_counts: dict[str, int]

    
    @property
    def expanded_gates(self) -> list[Gate]:
        """ List of gates, expanded (no germ power included) """
        return self.prep_gates + self.germ_gates * self.germ_power + self.measurement_gates


    @property
    def total_counts(self) -> int:
        """ Number of measurement counts """
        return sum(self.measurement_counts.values())


    @property
    def depth(self) -> int:
        """ Number of total gates in the circuit """
        return len(self.expanded_gates)



    def __repr__(self):
        gates_readable = " ".join(repr(gate) for gate in self.expanded_gates) or "(empty)"
        return f"ParsedCircuit({gates_readable}, counts={self.counts})"



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
    """ Parse a GST circuit line, containing a sequence of gates and the measurement count outcomes. """ 
    # For GST data files, this is of the format circuit list then measurement counts 

    # Strip the line if it's not already stripped 
    line = line.strip()

    # Separate the circuit from the measurement outcomes
    match = re.match(r"^(.+?)@\(([^)]+)\)\s+(.+)$", line) 
    
    if not match:
        raise ValueError(f"Cannot parse line: {line!r}")

    # Match group 1 is the circuit sequence  
    # Match group 2 is the @(0) directive so it should be ignored  
    # Match group 3 is the measurement counts  
    circuit_sequence = match.group(1).strip() 
    line_labels = [int(q) for q in match.group(2).split(",")]
    count_values = [int(x) for x in match.group(3).split()]
    
    if len(count_values) != len(outcome_labels):
        raise ValueError(f"Expected {len(outcome_labels)} measurement outcomes but received {len(count_values)} on line: {line!r}")
   
    # Create a dictionary with measurement outcomes and corresponding counts for this line  
    measurement_counts = dict(zip(outcome_labels, count_values)) 

    # Parse circuit sequence, starting with empty (do nothing -- prep then measure) string 
    if circuit_sequence == "{}":
        return ParsedCircuit(unparsed_data = line, prep_gates=[], germ_gates = [], measurement_gates = [],
                            germ_power = 1, line_labels = line_labels, measurement_counts = measurement_counts) 


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
        
    
    return ParsedCircuit(line, prep_gates, germ_gates, measure_gates, 1, line_labels, measurement_counts) 



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




if __name__=="__main__":
    # Example usage 

    fname = './gst_m12_m52_011.gstdata' 

    # Run the main parsing function:  
    results = parse_gst_circuit_file(fname)

    head = 64
    # Print circuit information: 
    for i, circ in enumerate(results):
        print(f"\n--- Experiment {i} ---")
        print(f"    Unparsed circuit line:  {circ.unparsed_data}")
        print(f"    Prep gates:    {circ.prep_gates}")
        print(f"    Germ gates:    {circ.germ_gates}")
        print(f"    Germ power:    {circ.germ_power}")
        print(f"    Measure gates:    {circ.measurement_gates}")
        print(f"    Measurement outcomes:    {circ.measurement_counts}")
        print(f"    Total shots:    {circ.total_counts}")
        print(f"    Circuit depth:    {circ.depth}")
        # Only print the first {head} 
        if i > head:
            break
