""" Gate set tomography (GST) analysis.

    The analysis is a set of functions. Each takes the GST circuits (with measurement data), a GateSetModel (the models of
    the prep, POVM, and gates, and their parameters), and any options explicitly, and returns its results:

        lin  = linear_solve_for_gate_parameters(gst_circuits, gate_set_model, circuit_design, target=ideal_values)
        fit  = mle_solve_for_gate_parameters(gst_circuits, gate_set_model, initial_guess=lin.theta)
        errs = gate_set_errors(gate_set_model, fit.theta, reference=true_values)

    Studies (e.g. of the number of shots or of circuit depth) loop over these with simulate_gst_data():

        data = simulate_gst_data(gst_circuits, gate_set_model, true_values, N_shots, rng)
        fit  = mle_solve_for_gate_parameters(data, gate_set_model)

    A gate set is evaluated from a gate set model and its parameter values, theta. Parameter values (theta, targets,
    references, initial guesses) are given as a vector in the order of gate_set_model.parameter_names, or as a dictionary of
    parameter names to values in which unlisted parameters take their specified initial guesses.

    Expensive gate models can be replaced in the solvers (and the circuit planner) by interpolations on a parameter grid; see
    GateSetModel.interpolate_gate_model.
"""
import numpy as np
from itertools import groupby
from dataclasses import dataclass, field, replace

import scipy.optimize as opt
from scipy import stats
from typing import Callable
import warnings

from ionsim.process import Gate
from ionsim.basis import StandardBasis
from ionsim.gst_circuit_parser import GstGate, GstCircuit, CircuitData, gate_from_label
from ionsim.gate_set_model import GateSetModel
from ionsim.ionsim_error import IonSimError
from ionsim.custom_types import Vector, Matrix
from ionsim.io import write_results_to_file
from ionsim.config import NUMERICAL_EQUIVALENCE_THRESHOLD


__all__ = ['GstResult', 'linear_solve_for_gate_parameters', 'mle_solve_for_gate_parameters', 'staged_mle_solve_for_gate_parameters',
           'log_likelihood', 'chi_squared', 'simulate_gst_data', 'evaluate_gate_set', 'gate_set_errors', 'average_gate_set_error',
           'average_gate_set_error_uncertainty', 'write_gate_set_results']


def depth_bin(depth):
    """ Bins a circuit depth to the nearest power of 2 """
    if depth <= 1.:
        return 1
    return int(2**(np.ceil(np.log2(depth))))


### Results ###
@dataclass
class GstResult:
    """ Result of a GST solve.

        - theta: estimated parameter vector, in the order of parameter_names
        - method: 'linear', 'MLE', or 'staged MLE'
        - log_likelihood: log-likelihood of theta for the circuits' data
        - initial_guess: parameter vector the solver started from
        - success / message: solver convergence (always True for linear GST)
        - optimizer_result: scipy OptimizeResult (MLE solvers; final stage for staged MLE)
        - stage_estimates: {depth: theta} after each stage (staged MLE)
        - lgst_estimates: linear GST estimates, e.g. 'gate_estimates' (keyed by gate label), 'gram_matrix' (linear GST)
        - gauge_iterations: number of re-gauging passes after the first linear GST fit (linear GST)
        - negative_log_likelihood_history: objective value at each evaluation (MLE solvers)
    """
    theta: np.ndarray
    parameter_names: list
    method: str
    log_likelihood: float
    initial_guess: np.ndarray
    success: bool = True
    message: str = ''
    optimizer_result: object = None
    stage_estimates: dict | None = None
    lgst_estimates: dict | None = None
    gauge_iterations: int | None = None
    negative_log_likelihood_history: list = field(default_factory=list)

    @property
    def parameter_values(self) -> dict:
        """ {parameter name: estimated value} """
        return dict(zip(self.parameter_names, self.theta))

    def __repr__(self):
        values = ", ".join(f"{name}={value:.6g}" for name, value in self.parameter_values.items())
        return f"GstResult(method={self.method!r}, success={self.success}, log_likelihood={self.log_likelihood:.6g}, {values})"


### Circuit data ###
@dataclass
class _CircuitData:
    """ Measurement data of a list of circuits, tabulated from the circuits' current data.

        Built at the start of every GST operation, so it always reflects the data as it is then; the solvers build it once and
        reuse it for every objective evaluation.
    """
    circuits: list                  # the circuits, in order
    sequences: list                 # unique expanded gate sequences (tuples of GstGate)
    sequence_index: np.ndarray      # (n_circuits,) index into sequences for each circuit
    counts: np.ndarray              # (n_circuits, n_outcomes) counts aligned with the outcome labels
    has_data: np.ndarray            # (n_circuits,) whether each circuit has measurement data

    def frequencies_by_sequence(self, outcome_labels) -> dict:
        """ Observed outcome frequencies for each unique gate sequence, pooling repeated circuits """
        pooled = np.zeros((len(self.sequences), self.counts.shape[1]))
        np.add.at(pooled, self.sequence_index[self.has_data], self.counts[self.has_data])
        frequencies = {}
        for k, sequence in enumerate(self.sequences):
            total = pooled[k].sum()
            if total > 0:
                frequencies[sequence] = dict(zip(outcome_labels, pooled[k] / total))
        return frequencies


### Gate sets (models evaluated at parameter values, or given explicitly) ###
def evaluate_gate_set(gate_set_model: GateSetModel, theta: Vector | dict | None=None) -> dict:
    """ Evaluates the gate set model at the specified parameter values (theta defaults to the specified initial guesses). """
    theta = gate_set_model.parse_theta(theta)
    gate_set = {'prep': gate_set_model.prep_state(theta), 'POVM': gate_set_model.measurement_effects(theta)}
    for gate in gate_set_model.gate_models:
        gate_set[gate.label] = gate_set_model.gate_process_matrix(gate, theta)
    return gate_set

def _as_supervector(element) -> np.ndarray:
    """ Supervector of a prep state given as an ionsim State or an array """
    return np.asarray(element.supervector if hasattr(element, 'supervector') else element)

def _as_superbra(element) -> np.ndarray:
    """ Superbra of a measurement effect given as an ionsim operator or an array """
    return np.asarray(element.superbra if hasattr(element, 'superbra') else element)

def _internal_gate_set(gate_set_model: GateSetModel, values: Vector | dict | None, gate_set: dict | None, role: str) -> dict:
    """ A gate set keyed internally by GstGate (plus 'prep', 'POVM'), from parameter values or an explicit gate-set dictionary.
        Exactly one of values / gate_set must be given. """
    if (values is None) == (gate_set is None):
        raise ValueError(f"Specify the {role} as either parameter values ({role}=...) or an explicit gate set ({role}_gate_set=...), not both or neither.")
    if gate_set is None:
        gate_set = evaluate_gate_set(gate_set_model, values)
    if not isinstance(gate_set, dict):
        raise TypeError(f"The {role} gate set must be a dictionary of gate labels, 'prep', and 'POVM'; received {type(gate_set).__name__}.")
    internal = {}
    for key, value in gate_set.items():
        if key == 'prep':
            internal['prep'] = _as_supervector(value)
        elif key == 'POVM':
            internal['POVM'] = {outcome: _as_superbra(effect) for outcome, effect in value.items()}
        else:
            gate = gate_from_label(key)
            if gate in internal:
                raise ValueError(f"Gate {key!r} appears more than once in the {role} gate set.")
            internal[gate] = np.asarray(value)
    for required in ('prep', 'POVM'):
        if required not in internal:
            raise ValueError(f"The {role} gate set has no {required!r} entry.")
    return internal


### Internal: one GST problem (circuits + models) ###
class _GstProblem:
    """ The circuits and models of one GST calculation, with quantities derived from them. Created by each public function;
        holds no results between calls. """

    def __init__(self, gst_circuits: list[GstCircuit], gate_set_model: GateSetModel, verbose: bool=False, exact_models: bool=False):
        if not isinstance(gate_set_model, GateSetModel):
            raise TypeError(f"gate_set_model must be a GateSetModel object; received {type(gate_set_model).__name__}.")
        if isinstance(gst_circuits, GstCircuit) or not all(isinstance(c, GstCircuit) for c in gst_circuits):
            raise TypeError("gst_circuits must be a list of GstCircuit objects (e.g. from parse_gst_circuit_file).")
        if len(gst_circuits) == 0:
            raise ValueError("gst_circuits is empty.")
        self.gst_circuits = list(gst_circuits)
        self.gate_set_model = gate_set_model
        # Repeated evaluations use the evaluation models (interpolated where requested); exact_models forces the exact models
        self.gate_models = gate_set_model.gate_models if exact_models else gate_set_model.evaluation_gate_models
        self.verbose = verbose

        # Gates appearing in the circuits must have models
        self.gate_set = {g for circ in self.gst_circuits for g in circ.expanded_gates}
        missing = self.gate_set - set(self.gate_models.keys())
        if missing:
            raise ValueError(f"Gates found in circuit data but no model, missing models for {sorted(g.label for g in missing)}")

        # Outcome labels: from the first circuit with data (stable order for all vectorized probability operations)
        with_counts = [c for c in self.gst_circuits if c.measurement_data is not None and c.measurement_data.counts is not None]
        if with_counts:
            self.outcome_labels = tuple(with_counts[0].measurement_data.counts.keys())
        else:
            self.outcome_labels = tuple(gate_set_model.measurement_effects(gate_set_model.initial_guess).keys())
        self.outcome_to_index = {label: i for i, label in enumerate(self.outcome_labels)}

        self.d2 = gate_set_model.prep_state(gate_set_model.initial_guess).size
        self.nll_history = []

    ### Data ###
    def tabulate(self, circuits: list[GstCircuit], require_data: bool=True) -> _CircuitData:
        """ Tabulate the current measurement data of `circuits`. Counts are aligned with the outcome labels by label. """
        sequences = {}
        sequence_index = np.zeros(len(circuits), dtype=np.int64)
        counts = np.zeros((len(circuits), len(self.outcome_labels)))
        has_data = np.zeros(len(circuits), dtype=bool)
        for i, circ in enumerate(circuits):
            sequence_index[i] = sequences.setdefault(tuple(circ.expanded_gates), len(sequences))
            data = circ.measurement_data
            if data is None:
                if require_data:
                    raise IonSimError(f"Circuit {circ.build_circuit_string()} has no measurement data.")
                continue
            if data.counts is None:
                raise NotImplementedError(f"Time-dependent GST is not available in this version of IonSim.")
            for outcome, count in data.counts.items():
                if outcome not in self.outcome_to_index:
                    raise IonSimError(f"Unexpected measurement outcome '{outcome}' in circuit data.")
                if count < 0:
                    raise IonSimError(f"Negative count {count} for outcome '{outcome}' in circuit {circ.build_circuit_string()}.")
                counts[i, self.outcome_to_index[outcome]] = count
            has_data[i] = True
        return _CircuitData(list(circuits), list(sequences.keys()), sequence_index, counts, has_data)

    ### Model predictions ###
    def get_parameters(self, theta: Vector, key):
        return self.gate_set_model.model_parameters(theta, key)

    def prep_and_effects(self, theta: Vector) -> tuple[np.ndarray, np.ndarray]:
        """ Prep supervector and effect matrix (rows in outcome-label order) at theta """
        rho_supervector = self.gate_set_model.prep_state(theta)
        measurement_effects = self.gate_set_model.measurement_effects(theta)
        effect_matrix = np.vstack([np.asarray(measurement_effects[label]) for label in self.outcome_labels])
        return rho_supervector, effect_matrix

    def gate_process_matrices(self, theta: Vector) -> dict:
        """ Evaluate each gate's process matrix function once at theta """
        return {gate: gate_model(*self.get_parameters(theta, gate)) for gate, gate_model in self.gate_models.items()}

    def compose_quantum_map(self, gates: tuple, gate_matrices: dict, circuit_map_cache: dict) -> np.ndarray:
        """ Compose the circuit map once for each unique gate sequence in an evaluation.

            A run of n identical consecutive gates (e.g. a single-gate germ raised to a power) is composed as the matrix power
            G^n by repeated squaring, which takes ~log2(n) matrix products instead of n.
        """
        quantum_map = circuit_map_cache.get(gates)
        if quantum_map is not None:
            return quantum_map
        quantum_map = np.eye(self.d2, dtype=complex)
        for gate, run in groupby(gates):
            count = sum(1 for _ in run)
            matrix = gate_matrices[gate]
            quantum_map = (matrix if count == 1 else np.linalg.matrix_power(matrix, count)) @ quantum_map
        circuit_map_cache[gates] = quantum_map
        return quantum_map

    def predict_sequence_probabilities(self, theta: Vector, sequences: list) -> np.ndarray:
        """ Clipped outcome probabilities (n_sequences x n_outcomes, outcome-label order) for gate sequences at theta """
        gate_matrices = self.gate_process_matrices(theta)
        rho_supervector, effect_matrix = self.prep_and_effects(theta)
        circuit_map_cache = {}
        probabilities = []
        for gates in sequences:
            mapped_state = self.compose_quantum_map(gates, gate_matrices, circuit_map_cache) @ rho_supervector
            probability_values = np.real(effect_matrix @ mapped_state)
            probabilities.append(np.clip(probability_values, NUMERICAL_EQUIVALENCE_THRESHOLD, 1. - NUMERICAL_EQUIVALENCE_THRESHOLD))
        return np.array(probabilities)

    ### Objectives ###
    def log_likelihood(self, theta: Vector, data: _CircuitData) -> float:
        """ Log-likelihood of theta for tabulated data: sum over circuits and outcomes of N_outcome log p_outcome(theta) """
        log_probabilities = np.log(self.predict_sequence_probabilities(theta, data.sequences))
        l_likelihood = 0.
        for counts, k in zip(data.counts, data.sequence_index):
            l_likelihood += np.dot(counts, log_probabilities[k])
        self.nll_history.append(-l_likelihood)
        if self.verbose:
            print(f"Evaluation {len(self.nll_history)}: negative log likelihood {-l_likelihood} at {theta}")
        return l_likelihood

    def chi_squared(self, theta: Vector, data: _CircuitData) -> float:
        """ chi^2 between observed frequencies and model probabilities, scaled by the shots of each circuit """
        probabilities = self.predict_sequence_probabilities(theta, data.sequences)
        chi_squared = 0.
        for counts, k in zip(data.counts, data.sequence_index):
            total_counts = counts.sum()
            if total_counts > 0:
                chi_squared += total_counts * np.sum(((probabilities[k] - counts / total_counts)**2) / probabilities[k])
        return chi_squared

    def parse_initial_guess(self, initial_guess) -> Vector:
        """ Initial parameter vector from None (specified guesses), a dictionary, or a vector """
        if isinstance(initial_guess, str):
            raise ValueError(f"String initial guesses (e.g. {initial_guess!r}) are not supported. To start from linear GST, run "
                             f"linear_solve_for_gate_parameters() and pass its result's theta as initial_guess.")
        if initial_guess is not None and not isinstance(initial_guess, (dict, list, tuple, np.ndarray)):
            raise TypeError(f"initial_guess should be None, a dictionary, or a list/array. Received: {type(initial_guess)}")
        return self.gate_set_model.parse_theta(initial_guess)

    ### Linear GST ###
    def build_probability_matrix(self, frequencies: dict, prep_fiducials, measure_fiducials, target_gate=None, outcome=None):
        """ Builds the matrix of observed probabilities for a gate or empty gate (corresponding to the Gram Matrix).

            M[i,j] = p(outcome | measure_fid_i x gate x prep_fid_j )
        """
        if outcome is None:
            outcome = self.outcome_labels[0]
        M = np.zeros((len(measure_fiducials), len(prep_fiducials)))
        target_list = [target_gate] if target_gate else []
        for j, prep_fid in enumerate(prep_fiducials):
            for i, measure_fid in enumerate(measure_fiducials):
                key = tuple(list(prep_fid) + target_list + list(measure_fid))
                if key in frequencies:
                    M[i,j] = frequencies[key][outcome]
                else:
                    raise ValueError(f"Missing LGST circuit: prep = {prep_fid}, gate = {target_list}, measure = {measure_fid}")
        return M

    def run_linear_gst(self, prep_fiducials, measure_fiducials, target_gate_set: dict) -> dict:
        """ Estimate the gate set by linear inversion (Nielsen et al., "Gate Set Tomography", Quantum 2021), in the gauge of the
            target gate set. Returns the LGST estimates. """
        if self.verbose:
            print(f"\n --- Running linear GST ---")
        # 1. Build the Gram matrix: <<F_i|F_j>> from the circuits' current data
        frequencies = self.tabulate(self.gst_circuits).frequencies_by_sequence(self.outcome_labels)
        gram_matrix = self.build_probability_matrix(frequencies, prep_fiducials, measure_fiducials)
        gram_matrix_det = np.linalg.det(gram_matrix)
        if np.abs(gram_matrix_det) < 1E-12:
            raise ValueError(f"Gram matrix is not invertible, determinant = {gram_matrix_det}")

        # 2. Compute SVD to get projector to linear-independent subspace
        U, S, Vh = np.linalg.svd(gram_matrix)
        if len(S) < self.d2:
            raise ValueError(f"Gram matrix is not informationally complete. It has rank {len(S)} instead of {self.d2}.")

        # Projector onto k = d^2 top right singular vectors
        Pi = Vh[:self.d2, :]

        # Check that the fiducials are informationally complete by checking singular values:
        TOL = 1E-10
        N_significant_vals = np.sum(S > TOL)
        if N_significant_vals < self.d2:
            raise ValueError(f" Fiducials are not informationally complete. Only {N_significant_vals} singular values instead of {self.d2}.")

        # Check separation between complete and excess subspaces
        if len(S) > self.d2:
            ratio = S[self.d2]/S[self.d2 - 1]  # should be small
            if ratio > 0.1:
                print(f"WARNING: There is weak separation between signal and noise subspace. Ratio = {ratio}")

        # Check condition number of d^2 subspace:
        cond = S[0] / S[self.d2 - 1]
        if cond > 100:
            print(f"WARNING: Poorly conditioned linear GST. LGST estimates may be noisy. Condition number = {cond}")

        # Decomposition of Gram matrix = AB, where A is measurement matrix and B is prep matrix
        #   See Section 3. of "Gate Set Tomography" published in Quantum, 2021.
        #   Gram = AB (fiducial measure @ fiducial prep); decompose B = B_0 Pi, B_0 in the gauge of the target gate set
        B_target = np.zeros((self.d2, len(prep_fiducials)), dtype=complex)
        for j, prep_fid in enumerate(prep_fiducials):
            state = target_gate_set['prep'].copy()
            for gate in prep_fid:
                state = target_gate_set[gate] @ state
            B_target[:, j] = state
        # Project onto Pi subspace, Pi Pi^T is identity since rows of Pi are orthonormal
        B0 = B_target @ Pi.conj().T

        # Compute gate process matrix estimates via the following formula (Nielsen, 2021):
        # G_k = B0 (Pi Gram^T Gram Pi^T)^{-1} (Pi Gram^T P_k Pi^T) B0^{-1}
        PGT = Pi @ gram_matrix.T
        inv_PGTGPT = np.linalg.inv(PGT @ gram_matrix @ Pi.T)
        B0_inv = np.linalg.inv(B0)
        matrix_prefactor = B0 @ inv_PGTGPT @ PGT
        matrix_postfactor = Pi.T @ B0_inv

        gate_estimates = {}
        # Sorted for a reproducible ordering (set order depends on string hashing, which varies between runs)
        for gate in sorted(self.gate_set, key=lambda g: g.label):
            P_gate = self.build_probability_matrix(frequencies, prep_fiducials, measure_fiducials, target_gate = gate)
            gate_estimates[gate] = matrix_prefactor @ P_gate @ matrix_postfactor

        # Find which fiducial index is the empty circuit, corresponding to native prep and measure
        empty_fid = tuple()
        prep_idx = prep_fiducials.index(empty_fid)
        measure_idx = measure_fiducials.index(empty_fid)

        # Prep state matrix B = B0 Pi; native prep rho_0:
        prep_states = B0 @ Pi
        estimated_rho = prep_states[:, prep_idx]

        # Measurement effect matrix A = Gram B+ (right pseudoinverse of B)
        measurement_effects = gram_matrix @ np.linalg.pinv(prep_states)
        estimated_effects = {}
        for outcome in self.outcome_labels:
            gram_k = self.build_probability_matrix(frequencies, prep_fiducials, measure_fiducials, outcome = outcome)
            A_k = gram_k @ Pi.conj().T @ B0_inv
            estimated_effects[outcome] = A_k[measure_idx, :]

        return {'gate_estimates' : gate_estimates, 'gram_matrix' : gram_matrix,
                'native_prep_state' : estimated_rho, 'estimated_effects' : estimated_effects,
                'prep_states' : prep_states, 'measurement_effects' : measurement_effects}

    def parameters_from_lgst(self, lgst: dict, theta_0: Vector, minimize_kwargs: dict | None=None) -> Vector:
        """ Fit the models' parameters to LGST estimates, starting from theta_0.

            Minimizes the summed squared differences between the modeled and LGST-estimated gate set elements (Frobenius norm
            for process matrices, Euclidean norm for the prep supervector and effects) with L-BFGS-B within the parameter
            bounds. With shared parameters, all models are fit together; otherwise each model is fit on its own parameters.
        """
        minimize_kwargs = minimize_kwargs or {}
        theta = np.array(theta_0, dtype=float, copy=True)
        model_labels = ['prep', 'POVM'] + [gate.label for gate in lgst['gate_estimates']]
        if self.gate_set_model.shared_indices:
            return self._fit_models_to_lgst(lgst, theta, model_labels, minimize_kwargs)
        for label in model_labels:
            theta = self._fit_models_to_lgst(lgst, theta, [label], minimize_kwargs)
        return theta

    def _lgst_cost_terms(self, lgst: dict, label: str, theta: Vector) -> float:
        """ Squared difference between a model ('prep', 'POVM', or a gate label) at theta and its LGST estimate """
        if label == 'prep':
            rho = self.gate_set_model.prep_state_model(*self.get_parameters(theta, 'prep'))
            return np.linalg.norm(rho - lgst['native_prep_state'])**2
        if label == 'POVM':
            modeled_effects = self.gate_set_model.POVM_effect_models(*self.get_parameters(theta, 'POVM'))
            return sum(np.linalg.norm(np.asarray(modeled_effects[outcome]) - effect)**2 for outcome, effect in lgst['estimated_effects'].items())
        gate = gate_from_label(label)
        M = self.gate_models[gate](*self.get_parameters(theta, gate))
        return np.linalg.norm(M - lgst['gate_estimates'][gate], 'fro')**2

    def _fit_models_to_lgst(self, lgst: dict, theta: Vector, model_labels: list[str], minimize_kwargs: dict) -> Vector:
        """ Fit the parameters of the given models to their LGST estimates; other parameters stay at their values in theta """
        indices = self.gate_set_model.parameter_indices_of_models(model_labels)
        if not indices:
            return theta

        def cost(x):
            trial = theta.copy()
            trial[indices] = x
            return float(np.real(sum(self._lgst_cost_terms(lgst, label, trial) for label in model_labels)))

        bounds = self.gate_set_model.parameter_bounds
        model_bounds = None if bounds is None else [bounds[i] for i in indices]
        ## Note: L-BFGS-B performs significantly better here than Nelder-Mead
        result = opt.minimize(cost, theta[indices], method='L-BFGS-B', bounds=model_bounds, **minimize_kwargs)
        fitted = theta.copy()
        fitted[indices] = result.x
        return fitted

    def group_circuits(self, organize_circuits_by_germ_power: bool) -> dict:
        """ Circuits grouped by germ power p, or by base depth L (germ length x germ power, binned to powers of 2) """
        groups = {}
        for circ in self.gst_circuits:
            if organize_circuits_by_germ_power:
                key = circ.germ_power
            else:
                germ_length = len(circ.germ_gates)
                key = 1 if germ_length == 0 else depth_bin(float(germ_length*circ.germ_power))
            groups.setdefault(key, []).append(circ)
        return groups


def _gst_result(problem: _GstProblem, theta: Vector, method: str, theta_0: Vector, data: _CircuitData, **fields) -> GstResult:
    """ Package a GstResult (log-likelihood of theta on the data, history of objective values) """
    history = list(problem.nll_history)
    log_likelihood = problem.log_likelihood(theta, data)
    return GstResult(theta=np.array(theta, dtype=float), parameter_names=problem.gate_set_model.parameter_names, method=method,
                     log_likelihood=log_likelihood, initial_guess=np.array(theta_0, dtype=float),
                     negative_log_likelihood_history=history, **fields)


### Solvers ###
def linear_solve_for_gate_parameters(gst_circuits: list[GstCircuit], gate_set_model: GateSetModel, circuit_design,
                                     target: Vector | dict | None=None, target_gate_set: dict | None=None,
                                     initial_guess: Vector | dict | None=None, max_gauge_iterations: int=100,
                                     gauge_tolerance: float=1e-6, verbose: bool=False, **minimize_kwargs) -> GstResult:
    """ Linear GST (LGST): estimates the gate set from the fiducial circuits by linear inversion, then fits the model parameters
        to those estimates. Uses only the measurement data and the models, so it applies to experimental data.

        LGST estimates are determined up to a gauge, which is fixed by a reference gate set (the target). The model fit is only
        unbiased when that reference is in the gauge of the true gate set, which is unknown in an experiment. So the fit is
        re-gauged: LGST is repeated in the gauge of the fitted model, and the model refit, until the parameters converge. The
        true gate set is a fixed point of this iteration; starting from an ideal target, it removes the gauge bias.

        - gst_circuits: circuits with measurement data, including the fiducial (LGST) circuits of the design.
        - gate_set_model: GateSetModel (the models of the gate set elements and their parameters).
        - circuit_design: the circuit design (e.g. a GSTCircuitPlanner) whose prep_fiducials and measure_fiducials were used.
        - target / target_gate_set: the initial gauge reference, a gate set close to ideal (e.g. prep |0> and ideal rotations).
            Give either parameter values at which the models are ideal (target=...), or an explicit gate set (target_gate_set=
            {'prep': ..., 'POVM': {...}, gate label: process matrix}, entries as arrays or ionsim State/operator objects).
        - initial_guess: starting point for the first fit to the LGST estimates (default: the specified guesses).
        - max_gauge_iterations: maximum number of re-gauging passes after the first fit (0 disables re-gauging).
        - gauge_tolerance: re-gauging stops when no parameter changes by more than gauge_tolerance * max(1, |parameter|).
        - verbose: print progress.
        - minimize_kwargs: passed to scipy.optimize.minimize (L-BFGS-B) for the model fits. The default options are tight
            (ftol = 1e-15, gtol = 1e-12) so that the fits are precise enough for re-gauging to converge.

        Returns a GstResult; result.lgst_estimates holds the final pass's LGST estimates (gate estimates keyed by gate label),
        result.gauge_iterations the number of re-gauging passes, and result.success whether re-gauging converged.
    """
    problem = _GstProblem(gst_circuits, gate_set_model, verbose)
    prep_fiducials = [tuple(fid) for fid in circuit_design.prep_fiducials]
    measure_fiducials = [tuple(fid) for fid in circuit_design.measure_fiducials]
    gauge_target = _internal_gate_set(gate_set_model, target, target_gate_set, 'target')
    missing = sorted({g.label for fid in prep_fiducials for g in fid} - {k.label for k in gauge_target if isinstance(k, GstGate)})
    if missing:
        raise ValueError(f"The target gate set has no process matrices for the prep fiducial gates {missing}.")
    if max_gauge_iterations < 0:
        raise ValueError(f"max_gauge_iterations must be non-negative; received {max_gauge_iterations}.")

    # Model fits must be more precise than the re-gauging tolerance, so the default L-BFGS-B options are tight
    minimize_kwargs = dict(minimize_kwargs)
    minimize_kwargs.setdefault('options', {'ftol': 1e-15, 'gtol': 1e-12, 'maxiter': 20000})

    theta_0 = problem.parse_initial_guess(initial_guess)
    lgst = problem.run_linear_gst(prep_fiducials, measure_fiducials, gauge_target)
    theta = problem.parameters_from_lgst(lgst, theta_0, minimize_kwargs)

    # Re-gauge: repeat LGST in the gauge of the fitted model and refit, until the parameters converge
    converged = max_gauge_iterations == 0
    iterations = 0
    while iterations < max_gauge_iterations:
        regauge_target = _internal_gate_set(gate_set_model, theta, None, 'target')
        lgst = problem.run_linear_gst(prep_fiducials, measure_fiducials, regauge_target)
        new_theta = problem.parameters_from_lgst(lgst, theta, minimize_kwargs)
        iterations += 1
        change = np.max(np.abs(new_theta - theta) / np.maximum(1., np.abs(theta)))
        theta = new_theta
        if verbose:
            print(f"Re-gauging pass {iterations}: largest relative parameter change {change:.3e}")
        if change <= gauge_tolerance:
            converged = True
            break

    message = "re-gauging disabled" if max_gauge_iterations == 0 else (
        f"re-gauging converged in {iterations} passes" if converged else
        f"re-gauging did not converge in {max_gauge_iterations} passes (last relative change {change:.3e})")
    if not converged:
        warnings.warn(f"Linear GST: {message}. Increase max_gauge_iterations or gauge_tolerance.")
    lgst = dict(lgst, gate_estimates={gate.label: matrix for gate, matrix in lgst['gate_estimates'].items()})
    return _gst_result(problem, theta, 'linear', theta_0, problem.tabulate(problem.gst_circuits), lgst_estimates=lgst,
                       gauge_iterations=iterations, success=converged, message=message)


def mle_solve_for_gate_parameters(gst_circuits: list[GstCircuit], gate_set_model: GateSetModel,
                                  initial_guess: Vector | dict | None=None, verbose: bool=False, **minimize_kwargs) -> GstResult:
    """ Maximum likelihood estimation (MLE): finds the parameters that maximize the likelihood of the data over all circuits,

            max[ Likelihood( {G} | data) ] over parameter set theta,

        using scipy.optimize.minimize with method L-BFGS-B and the parameter bounds from the parameter specification.

        - gst_circuits: circuits with measurement data.
        - gate_set_model: GateSetModel (the models of the gate set elements and their parameters).
        - initial_guess: None (the specified guesses), a dictionary of parameter values, or a vector, e.g. the theta of a
            linear GST result.
        - verbose: print each objective evaluation.
        - minimize_kwargs: passed to scipy.optimize.minimize, e.g. options = {'maxiter': 500}.

        Returns a GstResult (the scipy result is in result.optimizer_result).
    """
    problem = _GstProblem(gst_circuits, gate_set_model, verbose)
    theta_0 = problem.parse_initial_guess(initial_guess)
    # Tabulate the circuits' current data once; every objective evaluation reuses it
    data = problem.tabulate(problem.gst_circuits)
    solver_result = opt.minimize(fun = lambda params: -problem.log_likelihood(params, data), x0 = theta_0, method = 'L-BFGS-B',
                                 bounds = gate_set_model.parameter_bounds, **minimize_kwargs)
    return _gst_result(problem, solver_result.x, 'MLE', theta_0, data, success=bool(solver_result.success),
                       message=str(solver_result.message), optimizer_result=solver_result)


def staged_mle_solve_for_gate_parameters(gst_circuits: list[GstCircuit], gate_set_model: GateSetModel,
                                         initial_guess: Vector | dict | None=None, organize_circuits_by_germ_power: bool=True,
                                         verbose: bool=False, **minimize_kwargs) -> GstResult:
    """ Staged MLE: maximum likelihood estimation on cumulative batches of circuits of increasing depth, each stage starting
        from the previous stage's estimate. This can help avoid local optima for long circuits.

        - initial_guess: initial guess for the first stage (as for mle_solve_for_gate_parameters).
        - organize_circuits_by_germ_power: stage by germ power p (default) or by base circuit depth L.
        - minimize_kwargs: passed to scipy.optimize.minimize at every stage, e.g. options = {'maxiter': 500}.

        Returns a GstResult from the final stage; result.stage_estimates holds the estimate after each stage, keyed by depth.
    """
    problem = _GstProblem(gst_circuits, gate_set_model, verbose)
    theta_0 = problem.parse_initial_guess(initial_guess)
    circuit_groups = problem.group_circuits(organize_circuits_by_germ_power)
    sorted_depths = sorted(circuit_groups.keys())
    if verbose:
        print(f"--- Staged MLE with bins by {'germ powers (p)' if organize_circuits_by_germ_power else 'circuit depth (L)'}: {sorted_depths}")

    stage_estimates = {}
    cumulative_circuits = []
    theta = theta_0
    for stage, L in enumerate(sorted_depths):
        cumulative_circuits.extend(circuit_groups[L])
        data = problem.tabulate(cumulative_circuits)
        # I found that using log likelihood for all stages gave faster and likely better results
        solver_result = opt.minimize(fun = lambda params: -problem.log_likelihood(params, data), x0 = theta, method = 'L-BFGS-B',
                                     bounds = gate_set_model.parameter_bounds, **minimize_kwargs)
        theta = solver_result.x
        stage_estimates[L] = solver_result.x
        if verbose:
            print(f"    Stage {stage + 1} (L <= {L}): {len(cumulative_circuits)} circuits, "
                  f"LL = {-solver_result.fun:.3f}, converged = {solver_result.success}")

    return _gst_result(problem, theta, 'staged MLE', theta_0, data, success=bool(solver_result.success),
                       message=str(solver_result.message), optimizer_result=solver_result, stage_estimates=stage_estimates)


### Objectives ###
def log_likelihood(gst_circuits: list[GstCircuit], gate_set_model: GateSetModel, theta: Vector | dict) -> float:
    """ Total log-likelihood of parameter values theta given the circuits' current measurement data:
            sum over circuits and outcomes of N_outcome log( p_outcome(theta) ) """
    problem = _GstProblem(gst_circuits, gate_set_model)
    return problem.log_likelihood(gate_set_model.parse_theta(theta), problem.tabulate(problem.gst_circuits))


def chi_squared(gst_circuits: list[GstCircuit], gate_set_model: GateSetModel, theta: Vector | dict) -> float:
    """ chi^2 between the circuits' observed outcome frequencies and the model probabilities at theta """
    problem = _GstProblem(gst_circuits, gate_set_model)
    return problem.chi_squared(gate_set_model.parse_theta(theta), problem.tabulate(problem.gst_circuits))


### Simulation ###
def simulate_gst_data(gst_circuits: list[GstCircuit], gate_set_model: GateSetModel, theta: Vector | dict, N_shots: int,
                      rng: np.random.Generator | int | None=None) -> list[GstCircuit]:
    """ Sample measurement outcomes for each circuit from the models at parameter values theta.

        Returns new circuits (copies with the sampled counts); the given circuits are not modified. The exact gate models are
        used, also for gates with interpolated models.
        - rng: a numpy Generator or seed, for reproducibility (default: a new Generator).
    """
    rng = np.random.default_rng(rng)
    problem = _GstProblem(gst_circuits, gate_set_model, exact_models=True)   # simulate from the exact models
    data = problem.tabulate(problem.gst_circuits, require_data=False)
    probabilities = problem.predict_sequence_probabilities(gate_set_model.parse_theta(theta), data.sequences)
    simulated = []
    for circ, k in zip(problem.gst_circuits, data.sequence_index):
        p = probabilities[k] / probabilities[k].sum()
        counts = rng.multinomial(N_shots, p)
        simulated.append(replace(circ, measurement_data = CircuitData.from_counts(dict(zip(problem.outcome_labels, counts)))))
    return simulated


### Gate set error metrics ###
def gate_set_errors(gate_set_model: GateSetModel, theta: Vector | dict, reference: Vector | dict | None=None,
                    reference_gate_set: dict | None=None, error_metric: str='frobenius norm',
                    basis: StandardBasis | None=None) -> dict:
    """ Error of each gate set element at theta, compared to a reference gate set.

        - reference: parameter values of the reference (e.g. the true values in a simulation study), or
        - reference_gate_set: an explicit gate set {'prep': ..., 'POVM': {...}, gate label: process matrix}.
        - error_metric: 'frobenius norm' (process matrix difference) or 'process infidelity' (requires basis).

        Returns {gate label: error, 'prep': error, 'POVM': error} for every gate model. Prep and POVM errors are Euclidean norms
        of the supervector / summed superbra differences.
    """
    if error_metric not in ('frobenius norm', 'process infidelity'):
        raise ValueError(f"Unknown error metric {error_metric!r}; use 'frobenius norm' or 'process infidelity'.")
    if error_metric == 'process infidelity' and basis is None:
        raise ValueError("The 'process infidelity' metric requires the basis.")
    reference_set = _internal_gate_set(gate_set_model, reference, reference_gate_set, 'reference')
    theta = gate_set_model.parse_theta(theta)

    errors = {}
    for gate in gate_set_model.gate_models:
        if gate not in reference_set:
            raise ValueError(f"The reference gate set has no entry for gate {gate.label!r}.")
        process_matrix = gate_set_model.gate_process_matrix(gate, theta)
        if error_metric == 'frobenius norm':
            errors[gate.label] = np.linalg.norm(process_matrix - reference_set[gate], 'fro')
        else:
            errors[gate.label] = 1. - Gate(basis, process_matrix).compute_process_fidelity(reference_set[gate])

    errors['prep'] = np.linalg.norm(gate_set_model.prep_state(theta) - reference_set['prep'])
    modeled_effects = gate_set_model.measurement_effects(theta)
    errors['POVM'] = sum(np.linalg.norm(np.asarray(modeled_effects[outcome]) - effect) for outcome, effect in reference_set['POVM'].items())
    return errors


def average_gate_set_error(errors: dict, include_SPAM_error: bool=True) -> float:
    """ Average of the gate errors in a gate_set_errors() dictionary, plus the prep and POVM errors if include_SPAM_error """
    gate_errors = [value for key, value in errors.items() if key not in ('prep', 'POVM')]
    gst_error = sum(gate_errors) / len(gate_errors)
    if include_SPAM_error:
        return gst_error + errors['prep'] + errors['POVM']
    return gst_error


def average_gate_set_error_uncertainty(error_uncertainties: dict, include_SPAM_error: bool=True) -> float:
    """ Uncertainty of average_gate_set_error() from the uncertainties (e.g. standard errors) of each element's error """
    gate_uncertainties = [value for key, value in error_uncertainties.items() if key not in ('prep', 'POVM')]
    gst_variance = sum(u**2 for u in gate_uncertainties) / len(gate_uncertainties)**2
    if include_SPAM_error:
        gst_variance += (error_uncertainties['prep'] + error_uncertainties['POVM'])**2
    return np.sqrt(gst_variance)


### Output ###
def write_gate_set_results(gate_set_model: GateSetModel, theta: Vector | dict, prefix: str='gst_optimal_'):
    """ Writes each gate's parameter values and process matrix at theta to an HDF5 file, '<prefix><gate label>.hdf5' """
    theta = gate_set_model.parse_theta(theta)
    for gate, gate_model in gate_set_model.gate_models.items():
        parameter_values = gate_set_model.model_parameters(theta, gate)
        results_to_write = dict(zip(gate_set_model.model_parameter_names[gate.label], parameter_values))
        results_to_write[gate.label + '_process_matrix'] = gate_model(*parameter_values)
        write_results_to_file(prefix + gate.label + '.hdf5', results_to_write)
