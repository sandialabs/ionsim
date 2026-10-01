import numpy as np
from dataclasses import dataclass, field
from pathlib import Path 
import re

__all__ = ['CircuitData', 'GstGate', 'GstCircuit', 'IDLE_LABEL', 'IDLE_ALIASES', 'gate_from_label', 'canonical_gate_label',
           'parse_circuit_string', 'parse_measurement_outcome_labels', 'parse_circuit_line', 'parse_gst_circuit_file']

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
    def from_timestamped_shots(single_shot_data):
        return CircuitData(timestamped_shots = single_shot_data) 


    def to_counts(self) -> dict:
        """ Time-average single shots into counts (discards time information). """
        if self.counts is not None:
            return self.counts

        c = {}
        # Loop through times and incriment the count of each outcome
        for _, outcome in self.timestamped_shots:
            c[outcome] = c.get(outcome, 0) + 1
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



# Accepted spellings of the global idle gate (acts on all qubits, takes no qubit arguments).
# '[]' is the pyGSTi-style token used in .gstdata files; 'idle' is the canonical label in IonSim.
IDLE_LABEL = 'idle'
IDLE_ALIASES = ('idle', '[]')

# Gate names: start with a letter; may contain letters, digits, '_', '+', '-'. (':' separates qubits.)
_GATE_NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_+\-]*")


@dataclass(frozen=True)
class GstGate:
    """ A GST gate: a gate name acting on a tuple of qubit indices (indexed from 0).

        Users specify gates by string label throughout the GST modules, e.g.
            'Gxpi2:0'   -> GstGate('Gxpi2', (0,))
            'MS:0:1'    -> GstGate('MS', (0, 1))
            'idle' or '[]' -> the global idle gate GstGate('idle', ())

        Every gate other than idle must specify its qubit(s) with a colon. Use GstGate.from_string()
        to convert a label; str(gate) (or gate.label) converts back to the canonical label.
    """

    name: str
    qubits: tuple[int, ...] = ()

    def __post_init__(self):
        # Canonicalize so that 'idle' and '[]' always produce the same (equal, same-hash) object.
        name = IDLE_LABEL if self.name in IDLE_ALIASES else self.name
        qubits = tuple(int(q) for q in self.qubits)

        if name == IDLE_LABEL:
            if qubits:
                raise ValueError(f"The '{IDLE_LABEL}' gate is a global idle and takes no qubit arguments; received qubits {qubits}. "
                                 f"For a qubit-specific idle, use a distinct gate name, e.g. 'Gi:{qubits[0]}'.")
        else:
            if not _GATE_NAME_PATTERN.fullmatch(name):
                raise ValueError(f"Invalid gate name {name!r}. Gate names must start with a letter and contain only letters, digits, '_', '+', or '-'.")
            if not qubits:
                raise ValueError(f"Gate {name!r} must specify the qubit(s) it acts on, e.g. '{name}:0' or '{name}:0:1'.")
            if any(q < 0 for q in qubits):
                raise ValueError(f"Qubit indices must be non-negative; received {qubits} for gate {name!r}.")
            if len(set(qubits)) != len(qubits):
                raise ValueError(f"Repeated qubit index in gate {name!r}: {qubits}.")

        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'qubits', qubits)

    @classmethod
    def from_string(cls, gate_str: str) -> "GstGate":
        """ Creates an instance from a gate label, e.g. 'Gxpi2:0', 'MS:0:1', 'idle', or '[]'.

            The qubit argument(s) are required (colon-separated) for every gate except idle.
        """
        if not isinstance(gate_str, str):
            raise TypeError(f"Gates must be specified by string label (e.g. 'Gxpi2:0', 'MS:0:1', 'idle'); received {type(gate_str).__name__}: {gate_str!r}.")
        label = gate_str.strip()
        if label in IDLE_ALIASES:
            return cls.idle()

        name, *qubit_strs = label.split(':')
        if not qubit_strs:
            raise ValueError(f"Gate {label!r} must specify the qubit(s) it acts on with a colon, e.g. '{label}:0' or '{label}:0:1'.")
        if not all(q.strip().isdigit() for q in qubit_strs):
            raise ValueError(f"Could not parse qubit indices in gate label {label!r}; expected the form 'name:q' or 'name:q0:q1' with integer q.")
        return cls(name.strip(), tuple(int(q) for q in qubit_strs))

    @classmethod
    def idle(cls) -> "GstGate":
        """ The global idle gate. """
        return cls(IDLE_LABEL, ())

    @property
    def is_idle(self) -> bool:
        return self.name == IDLE_LABEL

    @property
    def label(self) -> str:
        """ Canonical string label, e.g. 'Gxpi2:0', 'MS:0:1', or 'idle'. """
        if self.is_idle:
            return IDLE_LABEL
        return self.name + ":" + ":".join(str(q) for q in self.qubits)

    def to_circuit_token(self) -> str:
        """ Token used when writing circuit strings to .gstdata files ('[]' for idle). """
        return "[]" if self.is_idle else self.label

    def __str__(self):
        return self.label

    def __repr__(self):
        return self.label


def gate_from_label(gate: str) -> GstGate:
    """ Converts a user-specified gate label (string) into a GstGate. Raises TypeError for non-strings. """
    return GstGate.from_string(gate)


def canonical_gate_label(gate: str) -> str:
    """ Returns the canonical label of a user gate string, e.g. '[]' -> 'idle', ' Gxpi2:0 ' -> 'Gxpi2:0'. """
    return GstGate.from_string(gate).label


@dataclass
class GstCircuit:
    """ GST circuit (read from a GST file or planned), optionally with measurement outcomes 

        - follows convention of Prep gates --> {(Germ_gates)^germ_power} --> measure gates  
        - Stores the file string contents

    """
    # GstCircuit class should remain unfrozen so its measurement_data attribute can be modified by an experiment. 
    unparsed_data: str
    fiducial_prep_gates: list[GstGate]
    germ_gates: list[GstGate]
    fiducial_measurement_gates: list[GstGate]
    germ_power: int 

    line_labels: list[int]   # not as important, TODO: delete?   
    measurement_data: CircuitData | None


    @property
    def expanded_gates(self) -> list[GstGate]:
        """ List of gates, expanded (no germ power included) """
        return self.fiducial_prep_gates + self.germ_gates * self.germ_power + self.fiducial_measurement_gates


    @property
    def expanded_gate_labels(self) -> list[str]:
        """ Canonical string labels of the expanded gate sequence, e.g. ['Gxpi2:0', 'idle', 'MS:0:1'] """
        return [gate.label for gate in self.expanded_gates]

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
        return f"GstCircuit({gates_readable}, data={self.measurement_data})"

    @property
    def num_qubits(self):
        return len(self.line_labels)


    def build_circuit_string(self) -> str: 
        """ Build string representation, useful for writing circuit instructions. """
        # Ex] Gxpi2:0(Gxpi2:0)^{2}Gypi2:0@(0)

        # Helper function for chaining gate names into a single string 
        def _gates_to_str(gates: list[GstGate]):
            return "".join(g.to_circuit_token() for g in gates)

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
    def plan(prep_gates: list[str], germ_gates: list[str], germ_power: int, measure_gates: list[str], line_labels: list[int]) -> "GstCircuit":
        """ Constructs and returns a circuit that is planned - no measurement data exists yet.

            Gates are specified by string label, e.g. GstCircuit.plan(['Gxpi2:0'], ['Gypi2:0'], 4, [], [0]).
        """
        def _to_gates(labels, role):
            if isinstance(labels, str):
                raise TypeError(f"{role} gates must be a list of gate labels, e.g. ['Gxpi2:0']; received the string {labels!r}.")
            return [gate_from_label(g) for g in labels]

        return GstCircuit._from_gates(_to_gates(prep_gates, 'Prep'), _to_gates(germ_gates, 'Germ'), germ_power,
                                      _to_gates(measure_gates, 'Measure'), line_labels)

    @staticmethod
    def _from_gates(prep_gates: list[GstGate], germ_gates: list[GstGate], germ_power: int, measure_gates: list[GstGate], line_labels: list[int]) -> "GstCircuit":
        """ Internal constructor for a planned circuit from GstGate objects (used by the circuit planner). """
        planned_circ = GstCircuit("", list(prep_gates), list(germ_gates), list(measure_gates), germ_power, list(line_labels), measurement_data = None)
        planned_circ.unparsed_data = planned_circ.build_circuit_string()
        return planned_circ

    def append_to_file(self, filename):
        """ Appends circuit information to a gstdata type file"""
        with open(filename, 'a') as f:
            f.write(self._format_circuit_line() + "\n")


def parse_circuit_string(circ: str) -> list[GstGate]:
    """ Extract the gate sequence from a circuit string, e.g. 'Gxpi2:0[]Gypi2:1' -> [Gxpi2:0, idle, Gypi2:1].

        Recognized tokens are 'name:q' / 'name:q0:q1' gates and the idle token '[]' (a standalone 'idle' is also accepted).
        Raises ValueError on any unrecognized content, e.g. a gate missing its qubit argument.
    """
    # Check that we have a valid circuit string
    if not circ or not circ.strip():
        return []

    gates = []
    # 'idle' must not be part of a longer gate name (e.g. 'idleGx:0' is a gate named 'idleGx') and takes no qubits
    pattern = r"\[\]|(?<![A-Za-z_])idle(?![A-Za-z0-9_:+\-])|([A-Za-z][A-Za-z0-9_+\-]*):(\d+(?::\d+)*)"

    # Find matches for the pattern and build a GstGate object for each match, tracking unmatched text
    unmatched = []
    position = 0
    for m in re.finditer(pattern, circ):
        unmatched.append(circ[position:m.start()])
        position = m.end()
        if m.group(1) is None:
            gates.append(GstGate.idle())
        else:
            qubits = tuple(int(qubit) for qubit in m.group(2).split(":"))
            gates.append(GstGate(m.group(1), qubits))
    unmatched.append(circ[position:])

    leftover = "".join(unmatched).strip()
    if leftover:
        raise ValueError(f"Could not parse {leftover!r} in circuit string {circ!r}. Gates must have the form 'name:q' "
                         f"(e.g. 'Gxpi2:0', 'MS:0:1'), and idle is written as '[]'.")
    return gates


def parse_measurement_outcome_labels(header: str) -> list[str]:
    """ Extract measurement outcome labels """ 
    match = re.search(r"Columns\s*=\s*(.+)", header)

    if not match:
        raise ValueError(f"Cannot parse header: {header!r} from file.")

    columns = match.group(1)

    labels = [col.strip().split()[0] for col in columns.split(",")]
    return labels 



def parse_circuit_line(line: str, outcome_labels: list[str]) -> GstCircuit:
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

        measurement_data = CircuitData.from_counts(measurement_counts)
    else:
        measurement_data = None 
    
    # Parse circuit sequence, starting with empty (do nothing -- prep then measure) string 
    if circuit_sequence == "{}":
        return GstCircuit(unparsed_data = line, fiducial_prep_gates=[], germ_gates = [], fiducial_measurement_gates = [],
                            germ_power = 1, line_labels = line_labels, measurement_data = measurement_data) 

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
        
    return GstCircuit(line, prep_gates, germ_gates, measure_gates, germ_power, line_labels, measurement_data) 


def parse_gst_circuit_file(filepath: str | Path) -> list[GstCircuit]:
    """ Parse a GST circuit results file, containing circuits and outcomes on each line. """
    filepath = Path(filepath)
    results: list[GstCircuit] = []
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
