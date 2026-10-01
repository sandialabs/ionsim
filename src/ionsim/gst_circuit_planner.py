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
from ionsim.process import Circuit, Gate, Circuit_Process_Matrix_Function_Helper, GateProcessMatrixCache
from ionsim.custom_types import Matrix, Vector
from ionsim.custom_math import finite_difference_derivatives
from ionsim.config import NUMERICAL_EQUIVALENCE_THRESHOLD
from ionsim.gate_set_model import GateSetModel
from ionsim.gst_circuit_parser import GstCircuit, GstGate, gate_from_label

class GSTCircuitPlanner:
    def __init__(self, gate_names: list[str], qubit_labels: list[int], prep_fiducials: list[list[str]] | None=None,
                    measure_fiducials: list[list[str]] | None=None, germs: list[list[str]] | None=None, germ_powers: list[int]=[1,2,4,8,16],
                    gate_set_model: GateSetModel | None=None, long_sequence_GST: bool=True, evaluator_tolerance: float | None=None,
                    gate_models=None):
        """ Constructor for GST Circuit Planner class. The user passes in the gate names and qubit labels at a minimum.

            - All gates are specified by string label with their qubit argument(s), e.g. 'Gxpi2:0', 'MS:0:1'.
              The global idle gate may be written 'idle' or '[]'.
            - gate_names: the gate set, e.g. ['Gxpi2:0', 'Gypi2:0', 'idle'].
            - prep_fiducials / measure_fiducials / germs: lists of gate-label sequences, e.g. [[], ['Gxpi2:0'], ['Gxpi2:0', 'Gxpi2:0']].
              Defaults are used for any that are not supplied.
            - Sets up list of prep gates, measure gates, and germ gates. The class organizes GST circuits based on those gates requested germ powers.
            - Can write the GST circuit sequences to a file.
            - gate_set_model: optional GateSetModel (the models of the prep, POVM, and gates and their parameters, including
              shared parameters), required for sensitivity and Fisher information analysis. The same object can be given to
              the GST solvers (gate_set_tomography.py) so the analysis uses identical parameters.
            - evaluator_tolerance: relative accuracy of the model functions, used to size finite-difference steps (default:
              machine precision, appropriate for matrix exponentials; use the ODE solver tolerance for solver-based models).
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

        # Models and parameters (optional, used for sensitivity / Fisher information analysis)
        if gate_models is not None:
            raise TypeError("gate_models was replaced by gate_set_model = GateSetModel(prep_state_model, POVM_effect_models, gate_models), "
                            "which includes the SPAM models and shared-parameter specification.")
        if gate_set_model is not None and not isinstance(gate_set_model, GateSetModel):
            raise TypeError(f"gate_set_model must be a GateSetModel object; received {type(gate_set_model).__name__}.")
        self.gate_set_model = gate_set_model
        self.evaluator_tolerance = evaluator_tolerance
        # Gate process matrices are cached across every circuit of the design
        self.gate_cache = GateProcessMatrixCache()

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
        """ For 1Q gates, the fiducial circuits are the standard choices for {X_pi/2, Y_pi/2} gates.

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


    ### Circuit sensitivity and Fisher information (requires model parameters) ###
    def _require_gate_set_model(self) -> GateSetModel:
        if self.gate_set_model is None:
            raise ValueError("Sensitivity and Fisher information analysis requires the gate-set models: construct the planner with "
                             "gate_set_model = GateSetModel(prep_state_model, POVM_effect_models, gate_models), or set planner.gate_set_model.")
        return self.gate_set_model

    def _compute_germ_process_matrix(self, germ, theta):
        """Compute the process matrix for a germ at the parameter vector theta (see self.gate_set_model.parameter_names).

        Args:
            germ: List of GstGate objects representing the germ
            theta: parameter vector, or a dictionary of parameter names to values

        Returns:
            Process matrix for the germ sequence
        """
        gate_set_model = self._require_gate_set_model()
        theta = gate_set_model.parse_theta(theta)
        d2 = gate_set_model.prep_state(theta).size
        germ_process_matrix = np.eye(d2, dtype=complex)
        for gate in germ:
            germ_process_matrix = gate_set_model.gate_process_matrix(gate, theta) @ germ_process_matrix
        return germ_process_matrix

    def build_circuit_process_matrix_function(self, circuit: GstCircuit) -> Circuit_Process_Matrix_Function_Helper | None:
        """ Circuit process matrix function (Circuit_Process_Matrix_Function_Helper) whose keyword arguments are the global
            parameter names of self.gate_set_model, e.g. 'shared:amplitude_noise_strength' or 'MS:0:1.phi_error', so shared
            parameters are a single argument. Gate process matrices are cached across all circuits of the planner.
            Returns None for an empty circuit.
        """
        gate_set_model = self._require_gate_set_model()
        gates = list(circuit.expanded_gates)
        if not gates:
            return None
        missing = sorted({gate.label for gate in gates if gate not in gate_set_model.gate_models})
        if missing:
            raise ValueError(f"No gate models for {missing} in the planner's gate set model (models exist for {[g.label for g in gate_set_model.gate_models]}).")

        # Map each gate argument, namespaced by gate label, to its global parameter name (shared arguments map to one name)
        parameter_names = {}
        for gate in dict.fromkeys(gates):
            argument_names = inspect.signature(gate_set_model.evaluation_gate_models[gate]).parameters.keys()
            for argument, global_name in zip(argument_names, gate_set_model.argument_names(gate)):
                parameter_names[f"{gate.label}.{argument}"] = global_name

        # The helper composes its gate sequence left to right as matrices, i.e. last-applied gate first
        evaluation_models = gate_set_model.evaluation_gate_models   # interpolated where requested
        return Circuit_Process_Matrix_Function_Helper([evaluation_models[g] for g in gates[::-1]], separator='.',
                    gate_labels=[g.label for g in gates[::-1]], parameter_names=parameter_names,
                    gate_cache=self.gate_cache, evaluator_tolerance=self.evaluator_tolerance)

    def _circuit_probability_derivatives(self, circuit: GstCircuit, parameter_values, order: int):
        """ Outcome probabilities of a circuit (prep and POVM models included) and their derivatives with respect to every
            parameter the circuit depends on.

            Returns (outcome labels, parameter indices, probabilities, jacobian [param, outcome], hessian [param, param, outcome] or None)
        """
        gate_set_model = self._require_gate_set_model()
        theta_0 = gate_set_model.parse_theta(parameter_values)
        names = gate_set_model.parameter_names
        circuit_function = self.build_circuit_process_matrix_function(circuit)

        # Parameters the probabilities depend on: prep, POVM, and the circuit's gates
        wrt = gate_set_model.parameter_indices_of_models(['prep', 'POVM'] + list(dict.fromkeys(circuit.expanded_gates)))
        outcome_labels = list(gate_set_model.measurement_effects(theta_0).keys())
        circuit_arguments = [] if circuit_function is None else [(name, names.index(name)) for name in circuit_function.parameter_names]

        def probabilities(x):
            theta = theta_0.copy()
            theta[wrt] = x
            rho = gate_set_model.prep_state(theta)
            effects = gate_set_model.measurement_effects(theta)
            effect_matrix = np.vstack([np.asarray(effects[label]) for label in outcome_labels])
            if circuit_function is not None:
                rho = circuit_function(**{name: theta[i] for name, i in circuit_arguments}) @ rho
            return np.real(effect_matrix @ rho)

        bounds = [gate_set_model.layout['bounds'][i] for i in wrt]
        probs, jacobian, hessian = finite_difference_derivatives(probabilities, theta_0[wrt], order=order,
                                        evaluator_tolerance=self.evaluator_tolerance, bounds=bounds)
        return outcome_labels, wrt, probs, jacobian, hessian

    def compute_circuit_sensitivity(self, circuit: GstCircuit, parameter_values: Vector | dict | None=None) -> dict:
        """ Derivatives of a circuit's outcome probabilities with respect to the gate-set parameters it depends on
            (prep, POVM, and the circuit's gates).

            - parameter_values: point of evaluation, as a parameter vector (order of self.gate_set_model.parameter_names) or a
                dictionary of parameter names to values; unlisted parameters use their specified initial guesses.

            Returns {parameter name: {outcome label: dp_outcome/dparameter}}
        """
        names = self._require_gate_set_model().parameter_names
        outcome_labels, wrt, _, jacobian, _ = self._circuit_probability_derivatives(circuit, parameter_values, order=1)
        return {names[i]: dict(zip(outcome_labels, jacobian[a])) for a, i in enumerate(wrt)}

    def compute_circuit_sensitivities(self, gst_circuits: list[GstCircuit], parameter_values: Vector | dict | None=None) -> dict:
        """ compute_circuit_sensitivity() for each circuit, keyed by the circuit's expanded gate sequence """
        return {tuple(circ.expanded_gates): self.compute_circuit_sensitivity(circ, parameter_values) for circ in gst_circuits}

    def _circuit_shots(self, circuit: GstCircuit, N_shots: int | None) -> int:
        if N_shots is not None:
            return N_shots
        if circuit.measurement_data is None:
            raise ValueError(f"Circuit {circuit.build_circuit_string()} has no measurement data; pass N_shots for planned circuits.")
        return circuit.measurement_data.total_counts

    def compute_circuit_fisher_information(self, circuit: GstCircuit, parameter_values: Vector | dict | None=None, N_shots: int | None=None,
                                            include_hessian: bool=False) -> tuple[dict, Matrix]:
        """ Fisher information of a circuit's outcomes about the gate-set parameters, including the prep and POVM models.

            I_ij = N sum_k (dp_k/dtheta_i)(dp_k/dtheta_j) / p_k   [ - N sum_k d2p_k/dtheta_i dtheta_j, if include_hessian ]

            The Hessian term sums to zero for a complete POVM (sum_k p_k = 1), so it is excluded by default.

            - parameter_values: point of evaluation (vector or dictionary; unlisted parameters use their initial guesses).
            - N_shots: number of shots; defaults to the circuit's measurement data counts.

            Returns (dictionary {(name_i, name_j): I_ij} over the parameters the circuit depends on, full matrix in the order of
            self.gate_set_model.parameter_names)
        """
        gate_set_model = self._require_gate_set_model()
        names = gate_set_model.parameter_names
        N = self._circuit_shots(circuit, N_shots)
        _, wrt, probs, jacobian, hessian = self._circuit_probability_derivatives(circuit, parameter_values, order=2 if include_hessian else 1)
        p = np.clip(probs, NUMERICAL_EQUIVALENCE_THRESHOLD, 1. - NUMERICAL_EQUIVALENCE_THRESHOLD)

        block = N * np.einsum('ik,jk->ij', jacobian / p, jacobian)
        if include_hessian:
            block -= N * hessian.sum(axis=-1)

        FI_matrix = np.zeros((len(names), len(names)))
        FI_matrix[np.ix_(wrt, wrt)] = block
        FI_dict = {(names[i], names[j]): block[a, b] for a, i in enumerate(wrt) for b, j in enumerate(wrt)}
        return FI_dict, FI_matrix

    def compute_design_fisher_information(self, gst_circuits: list[GstCircuit], parameter_values: Vector | dict | None=None,
                                            N_shots: int | None=None, include_hessian: bool=False) -> tuple[dict, dict]:
        """ compute_circuit_fisher_information() for each circuit (including the empty circuit, which is informative about SPAM),
            keyed by the circuit's expanded gate sequence.

            Returns (Fisher information dictionaries, Fisher information matrices)
        """
        fisher_information = {}
        fisher_information_matrices = {}
        for circ in gst_circuits:
            key = tuple(circ.expanded_gates)
            fisher_information[key], fisher_information_matrices[key] = self.compute_circuit_fisher_information(
                circ, parameter_values, N_shots=N_shots, include_hessian=include_hessian)
        return fisher_information, fisher_information_matrices

    def compute_design_eigenvalues_and_inverse_fisher_matrix(self, fisher_information_matrices: dict) -> tuple[dict, dict]:
        """ Compute eigenvalues of the Fisher information matrix accumulated over the design, circuit by circuit, and its inverse
            (the Cramer-Rao bound on the covariance of the parameter estimates). The inverse is None while the accumulated
            matrix is singular (not yet every parameter is informed by the circuits so far). """
        eigenvalues = {}
        inverse_Fisher_infos = {}
        I_accumulated = None
        singular = 0
        for circ, fisher_info in fisher_information_matrices.items():
            I_accumulated = fisher_info.copy() if I_accumulated is None else I_accumulated + fisher_info
            eigenvalues[circ] = np.linalg.eigvalsh(I_accumulated)
            if eigenvalues[circ][0] <= 1e-12 * max(eigenvalues[circ][-1], 1e-300):
                inverse_Fisher_infos[circ] = None
                singular += 1
            else:
                inverse_Fisher_infos[circ] = np.linalg.inv(I_accumulated)
        if singular == len(fisher_information_matrices) and singular > 0:
            warnings.warn("The accumulated Fisher information matrix is singular for the whole design: some parameters are not "
                          "determined by these circuits.")
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

