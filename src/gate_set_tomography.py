""" Gate set tomography (GST) analysis.

    The analysis is called by a set of "solve" functions. Each takes the GST circuits (with measurement data), a GateSetModel (the models of
    the prep, POVM, and gates, and their parameters), and any options explicitly, and returns its results:

    For example:
        lin_results = linear_solve_for_gate_parameters(gst_circuits, gate_set_model, circuit_design, target=ideal_values)
        mle_results = mle_solve_for_gate_parameters(gst_circuits, gate_set_model, initial_guess=lin_results.theta)
        errs = gate_set_errors(gate_set_model, mle_results.theta, reference=true_values)

    Studies (e.g. of the number of shots or of circuit depth) loop over these with simulate_gst_data():

        data = simulate_gst_data(gst_circuits, gate_set_model, true_values, N_shots, rng)
        fit  = mle_solve_for_gate_parameters(data, gate_set_model)

    A gate set is evaluated from a gate set model and its parameter values, theta. Parameter values (theta, targets,
    references, initial guesses) are given as a vector in the order of gate_set_model.parameter_names, or as a dictionary of
    parameter names to values in which unlisted parameters take their specified initial guesses.

    Expensive gate models can be replaced in the solvers (and the circuit planner) by interpolations on a parameter grid; see
    GateSetModel.interpolate_gate_model.

    Circuit-level (quasistatic) noise (GateSetModel.set_circuit_noise): a gate argument that fluctuates from shot to shot but is
    the same in every gate of a circuit in a shot. The MLE solvers, log_likelihood, chi_squared, and simulate_gst_data average
    each circuit's outcome probabilities over the noise realizations, p = sum_k w_k p(circuit | x_k), with the noise
    distribution's parameters fitted like any other parameter. The noisy gates are evaluated once per realization and each
    circuit is composed once with them stacked over the realizations, so the arithmetic grows linearly with the number of
    realizations. Linear GST does not support circuit noise. Gate set errors compare the nominal gates.

    Dimensionless fitting (dimensionless_fit=True in the solvers; off by default): the L-BFGS-B optimizations run on
    dimensionless variables x = theta / s, one scale s per parameter, rather than on theta itself. Physical parameters often
    span many orders of magnitude (e.g. an error probability ~1e-3 and a rate ~1e3 1/s); on theta, L-BFGS-B's absolute
    finite-difference step, gradient tolerance, and quasi-Newton updates treat every coordinate alike, so large-valued
    parameters can stop near their initial guesses while the solver reports success. The scaling is internal, so every theta
    passed in or returned (initial guesses, results, stage estimates, optimizer_result.x) is in physical units.

    A dimensionless fit combines:
    - Adaptive scales: the first L-BFGS-B run uses the initial scales (parse_parameter_scalings: by default |initial guess|,
      with a warned fallback for guesses of 0);
      each run is then restarted from its result with s = |theta| (the initial scale for parameters below SCALE_FLOOR of
      it, e.g. at 0) until a restart no longer improves the objective (max_restarts). This makes the scaling insensitive to
      initial guesses that are off by orders of magnitude, and clears L-BFGS-B's curvature memory, which otherwise can stop
      runs early.
    - Gradients by finite differences on x with relative steps h = r * max(|x|, 1) (scipy's jac='2-point' for
      finite_differences='forward', '3-point' for 'central'; one-sided at the bounds), rather than L-BFGS-B's default
      absolute step of 1e-8, which is lost in round-off for large |x|.
    - For the MLE solvers, the objective log L_sat - log L(theta) >= 0 (the log-likelihood ratio to the saturated model):
      the same minimizer as -log L, with a value of order the number of circuits, so that L-BFGS-B's relative stopping
      tolerance (ftol) refers to meaningful changes rather than to the large absolute value of log L.
"""
import numpy as np
from itertools import groupby
from dataclasses import dataclass, field, replace

import scipy.optimize as opt
from typing import Callable
import warnings

from ionsim.process import Gate
from ionsim.basis import StandardBasis
from ionsim.gst_circuit_parser import GstGate, GstCircuit, CircuitData, gst_gate_from_label
from ionsim.gate_set_model import GateSetModel
from ionsim.ionsim_error import IonSimError
from ionsim.custom_types import Vector
from ionsim.io import write_results_to_file
from ionsim.config import NUMERICAL_EQUIVALENCE_THRESHOLD

__all__ = ['GstResult', 'parse_parameter_scalings', 'linear_solve_for_gate_parameters', 'mle_solve_for_gate_parameters',
           'staged_mle_solve_for_gate_parameters',
           'log_likelihood', 'chi_squared', 'simulate_gst_data', 'gate_set_errors', 'average_gate_set_error',
           'average_gate_set_error_uncertainty', 'parameter_squared_errors', 'write_gate_set_results']


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
        - parameter_scales: with dimensionless_fit, the scale s of each parameter in the optimizer's last run (x = theta / s);
            None otherwise. In optimizer_result, x and jac are in physical units; hess_inv refers to the scaled variables x.
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
    parameter_scales: np.ndarray | None = None

    @property
    def parameter_values(self) -> dict:
        """ {parameter name: estimated value} """
        return dict(zip(self.parameter_names, self.theta))

    def __repr__(self):
        values = ", ".join(f"{name}={value:.6g}" for name, value in self.parameter_values.items())
        return f"GstResult(method={self.method!r}, success={self.success}, log_likelihood={self.log_likelihood:.6g}, {values})"


### Parameter scaling ###
SCALE_FLOOR = 1e-2           # adaptive scaling uses |theta| only where it is at least this fraction of the initial scale
RESTART_TOLERANCE = 1e-10    # relative objective improvement below which L-BFGS-B is not restarted (without an absolute
                             # objective tolerance, e.g. for the linear GST model fits)
LIKELIHOOD_TOLERANCE = 1e-5  # default absolute tolerance on the log-likelihood for dimensionless MLE (a change of 0.5 is ~1 sigma)


def parse_parameter_scalings(gate_set_model: GateSetModel, parameter_scales: Vector | dict | None=None) -> Vector:
    """ Parsing of user input for the initial scale s of each parameter (theta order) for dimensionless fits, which run on x = theta / s.
        Later L-BFGS-B runs use |theta| at their start (s where |theta| < SCALE_FLOOR * s); see the module docstring.

        - None (default): s = |initial guess| (the guesses given to set_parameter, not a per-solve starting point, so the
            scale does not depend on e.g. a linear GST estimate that lands near zero).
        - a vector (one positive value per parameter, theta order), or a dictionary {parameter name: scale} in which unlisted
            parameters take their default scales.

        A parameter whose initial guess is 0 has no magnitude to scale by. Unless parameter_scales gives its scale, s = the
        width of its bounds if both are finite, and 1 (in the parameter's units) otherwise, with a warning: these fallbacks
        can be off by orders of magnitude (e.g. bounds of +-0.2 on an offset of order 1e-3), so give such parameters a scale.

        A good scale is the magnitude a parameter is expected to have; it need not be precise (within ~10x is fine).
    """
    names = gate_set_model.parameter_names
    guess = np.abs(np.asarray(gate_set_model.initial_guess, dtype=float))
    bounds = gate_set_model.parameter_bounds or [(None, None)] * len(names)
    if parameter_scales is None:
        given = {}
    elif isinstance(parameter_scales, dict):
        unknown = set(parameter_scales) - set(names)
        if unknown:
            raise ValueError(f"parameter_scales has unknown parameters {sorted(unknown)}. Parameters: {names}.")
        given = parameter_scales
    elif isinstance(parameter_scales, (list, tuple, np.ndarray)):
        vector = np.asarray(parameter_scales, dtype=float)
        if vector.shape != (len(names),):
            raise ValueError(f"parameter_scales must have one entry per parameter ({len(names)}); received shape {vector.shape}.")
        given = dict(zip(names, vector))
    else:
        raise TypeError(f"parameter_scales must be None, a vector, or a dictionary; received {type(parameter_scales).__name__}.")

    scale = np.empty(len(names))
    parameter_warnings = {}
    for i, (name, (lo, hi)) in enumerate(zip(names, bounds)):
        if name in given:
            scale[i] = given[name]
        elif guess[i] != 0.:
            scale[i] = guess[i]
        else:
            finite = lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi)
            # Scale given by width of user-specified bounds 
            scale[i] = (hi - lo) if finite and hi > lo else 1.
            if not (finite and hi == lo):          # a parameter fixed by its bounds does not move; no need to warn
                parameter_warnings[name] = scale[i]
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.):
        raise ValueError(f"Parameter scales must be finite and positive; received {dict(zip(names, scale))}.")
    if parameter_warnings:
        listed = ", ".join(f"{name!r}: {value:g}" for name, value in parameter_warnings.items())
        warnings.warn(f"Dimensionless fit: parameters with initial guess 0 have no magnitude to scale by, so their scales fall "
                      f"back to the width of their bounds (or 1): {{{listed}}}. If these are far from the parameters' expected "
                      f"magnitudes, give them with parameter_scales={{name: scale, ...}}.", stacklevel=2)
    return scale


### Circuit data ###
@dataclass(frozen=True)
class _CircuitOutcomes:
    """ Internal book-keeping of the measurement outcomes (counts) of a list of circuits, stored as arrays and built once per GST operation (from_circuits, in
        the _GstProblem constructor) and reused for every objective evaluation; subset_from_rows() selects circuits, e.g. for the stages
        of staged MLE. Circuits that share an expanded gate sequence share one row of predictions to save computation effort.
    """
    outcome_labels: tuple           # column order of counts
    sequences: list                 # unique expanded gate sequences (tuples of GstGate)
    sequence_index: np.ndarray      # (n_circuits,) index into sequences for each circuit
    counts: np.ndarray              # counts in shape (n_circuits, n_outcomes); zeros for a circuit without data

    @classmethod
    def from_circuits(cls, circuits: list[GstCircuit], outcome_labels: tuple, require_data: bool=True) -> '_CircuitOutcomes':
        """ Record the circuits' current counts, aligned with outcome_labels by label, and check them. A circuit without
            data raises an error, unless require_data=False (e.g. to simulate data), in which case its counts are zero. """
        column = {label: i for i, label in enumerate(outcome_labels)}
        sequences = {}
        sequence_index = np.zeros(len(circuits), dtype=np.int64)
        counts = np.zeros((len(circuits), len(outcome_labels)))
        for i, circ in enumerate(circuits):
            sequence_index[i] = sequences.setdefault(tuple(circ.expanded_gates), len(sequences))
            data = circ.measurement_data
            # Check that data exists and its format 
            if data is None:
                if require_data:
                    raise IonSimError(f"Circuit {circ.build_circuit_string()} has no measurement data.")
                continue
            if data.counts is None:
                raise NotImplementedError("Time-dependent GST is not available in this version of IonSim.")
            for outcome, count in data.counts.items():
                if outcome not in column:
                    raise IonSimError(f"Unexpected measurement outcome '{outcome}' in circuit data.")
                if count < 0:
                    raise IonSimError(f"Negative count {count} for outcome '{outcome}' in circuit {circ.build_circuit_string()}.")
                counts[i, column[outcome]] = count
        return cls(tuple(outcome_labels), list(sequences), sequence_index, counts)

    def subset_from_rows(self, rows: list[int]) -> '_CircuitOutcomes':
        """ The outcomes of the circuits at `rows` (indices into the circuits, in that order), keeping only the sequences they
            use, so that predictions are not computed for the other circuits' sequences """
        rows = np.asarray(rows, dtype=np.int64)
        used, sequence_index = np.unique(self.sequence_index[rows], return_inverse=True)
        return _CircuitOutcomes(self.outcome_labels, [self.sequences[k] for k in used], sequence_index.astype(np.int64),
                                self.counts[rows])

    def log_likelihood(self, probabilities: np.ndarray) -> float:
        """ sum over circuits and outcomes of N log p, for probabilities (n_sequences x n_outcomes) of the unique sequences """
        return float(np.sum(self.counts * np.log(probabilities[self.sequence_index])))

    def saturated_log_likelihood(self) -> float:
        """ Log-likelihood of the saturated model (each circuit's probabilities equal to its observed frequencies), the maximum
            attainable for the data: sum over circuits and outcomes of N log(N / N_circuit) """
        totals = self.counts.sum(axis=1, keepdims=True)
        observed = self.counts > 0
        return float(np.sum(self.counts[observed] * np.log((self.counts / np.where(totals > 0, totals, 1.))[observed])))

    def chi_squared(self, probabilities: np.ndarray) -> float:
        """ sum over circuits with data of N_circuit * sum over outcomes of (p - f)^2 / p, with f the observed frequencies """
        totals = self.counts.sum(axis=1)
        with_data = totals > 0
        p = probabilities[self.sequence_index[with_data]]
        f = self.counts[with_data] / totals[with_data, None]
        return float(np.sum(totals[with_data] * np.sum((p - f)**2 / p, axis=1)))

    def pooled_frequencies(self) -> dict:
        """ Observed outcome frequencies {sequence: {outcome: frequency}} of each unique gate sequence with data, pooling the
            counts of circuits that share it """
        pooled = np.zeros((len(self.sequences), len(self.outcome_labels)))
        np.add.at(pooled, self.sequence_index, self.counts)
        totals = pooled.sum(axis=1)
        return {sequence: dict(zip(self.outcome_labels, pooled[k] / totals[k]))
                for k, sequence in enumerate(self.sequences) if totals[k] > 0}

# Wrapper methods for supervector/superbra 
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
        gate_set = gate_set_model(values)
    if not isinstance(gate_set, dict):
        raise TypeError(f"The {role} gate set must be a dictionary of gate labels, 'prep', and 'POVM'; received {type(gate_set).__name__}.")
    internal = {}
    for key, value in gate_set.items():
        if key == 'prep':
            internal['prep'] = _as_supervector(value)
        elif key == 'POVM':
            internal['POVM'] = {outcome: _as_superbra(effect) for outcome, effect in value.items()}
        else:
            gate = gst_gate_from_label(key)
            if gate in internal:
                raise ValueError(f"Gate {key!r} appears more than once in the {role} gate set.")
            internal[gate] = np.asarray(value)
    for required in ('prep', 'POVM'):
        if required not in internal:
            raise ValueError(f"The {role} gate set has no {required!r} entry.")
    return internal


### Internal: one GST problem (circuits + models) ###
class _GstProblem:
    """ The circuits and models of one GST calculation, with quantities derived from them. Created by each public function
        (one problem per call; nothing is kept between calls). A solver's result is stored in self.solution
        (build_solution_from_theta); None until then. """

    def __init__(self, gst_circuits: list[GstCircuit], gate_set_model: GateSetModel, verbose: bool=False, exact_models: bool=False,
                 require_data: bool=True, dimensionless_fit: bool=False, parameter_scales: Vector | dict | None=None, finite_differences: str='forward',
                 max_restarts: int=5, likelihood_tolerance: float=LIKELIHOOD_TOLERANCE):
        if not isinstance(gate_set_model, GateSetModel):
            raise TypeError(f"gate_set_model must be a GateSetModel object; received {type(gate_set_model).__name__}.")
        if isinstance(gst_circuits, GstCircuit) or not all(isinstance(c, GstCircuit) for c in gst_circuits):
            raise TypeError("gst_circuits must be a list of GstCircuit objects (e.g. from parse_gst_circuit_file).")
        if len(gst_circuits) == 0:
            raise ValueError("gst_circuits is empty.")

        self.gst_circuits = list(gst_circuits)
        self.gate_set_model = gate_set_model
        self.d2 = gate_set_model.prep_state(gate_set_model.initial_guess).size

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

        # Tabulate the circuits' measurement counts and check them once (require_data=False allows circuits without data, e.g. to build outcome references via simulation)
        self.outcomes = _CircuitOutcomes.from_circuits(self.gst_circuits, self.outcome_labels, require_data)


        # Gates displaced by circuit-level noise (GateSetModel.set_circuit_noise), among the gates of these circuits
        self.noisy_gates = set(gate_set_model.circuit_noise_gates) & self.gate_set
        self.nll_history = []
        self.solution = None                                    # GstResult of the solve (build_solution_from_theta)
        # Dimensionless fitting (see the module docstring); its options must keep their defaults without it
        self.dimensionless_fit = bool(dimensionless_fit)
        if not self.dimensionless_fit:
            changed = [name for name, is_default in [
                ('parameter_scales', parameter_scales is None),
                ('finite_differences', finite_differences == 'forward'),
                ('max_restarts', isinstance(max_restarts, (int, np.integer)) and max_restarts == 5),
                ('likelihood_tolerance', likelihood_tolerance == LIKELIHOOD_TOLERANCE)] if not is_default]
            if changed:
                raise ValueError(f"{changed} only apply with dimensionless_fit=True.")
            self.parameter_scales = None
            self.last_scale = None
            return
        self.parameter_scales = parse_parameter_scalings(gate_set_model, parameter_scales)
        if finite_differences not in ('forward', 'central'):
            raise ValueError(f"finite_differences must be 'forward' or 'central'; received {finite_differences!r}.")
        self.finite_differences = finite_differences
        if not isinstance(max_restarts, (int, np.integer)) or max_restarts < 0:
            raise ValueError(f"max_restarts must be a non-negative integer; received {max_restarts!r}.")
        self.max_restarts = int(max_restarts)
        if not likelihood_tolerance > 0.:
            raise ValueError(f"likelihood_tolerance must be positive; received {likelihood_tolerance!r}.")
        self.likelihood_tolerance = float(likelihood_tolerance)
        self.scale_floor = SCALE_FLOOR * self.parameter_scales
        self.last_scale = self.parameter_scales.copy()          # scales of the most recent L-BFGS-B run (in GstResult)

    ### Parameter scaling (dimensionless fits) ###
    def parameter_scales_from_theta(self, theta: Vector) -> Vector: 
        """ Scales for a dimensionless L-BFGS-B run starting at theta: |theta| where |theta| >= SCALE_FLOOR times the initial
            scale, and the initial scale for parameters at or near 0 """
        magnitude = np.abs(np.asarray(theta, dtype=float))
        return np.where(magnitude >= self.scale_floor, magnitude, self.parameter_scales)

    @staticmethod
    def internal_bounds(bounds: list | None, scale: np.ndarray, indices: list[int]) -> list | None:
        """ L-BFGS-B bounds on x = theta / scale for the parameters at indices (None if every parameter is unbounded) """
        if bounds is None:
            return None
        return [(None if bounds[i][0] is None else bounds[i][0] / scale[i],
                 None if bounds[i][1] is None else bounds[i][1] / scale[i]) for i in indices]

    def minimize(self, objective: Callable[[np.ndarray], float], theta_start: Vector, subset_indices: list[int] | None=None,
                 objective_tolerance: float | None=None, **minimize_kwargs) -> tuple[np.ndarray, opt.OptimizeResult]:
        """ Minimize objective(theta) with L-BFGS-B within the parameter bounds: on theta itself with L-BFGS-B's built-in
            finite-difference gradients (default), or with dimensionless_fit on scaled variables x = theta / s.

            - objective: a function of the full physical parameter vector theta.
            - theta_start: the starting point (full theta). 
            - subset_indices: only those parameters are optimized; the others stay
                at their values in theta_start.
            - objective_tolerance: with dimensionless_fit, an absolute tolerance on the objective: L-BFGS-B stops when an
                iteration improves the objective by less than it (its ftol is set to objective_tolerance / |objective at the
                start of the run| unless minimize_kwargs gives options['ftol']), and runs are not restarted when a run improves
                it by less. None uses scipy's relative ftol and a relative restart tolerance (RESTART_TOLERANCE).
            - minimize_kwargs: passed to scipy.optimize.minimize; with dimensionless_fit, options (e.g. gtol) apply to the
                scaled variables.

            Dimensionless fit: each run uses the scales s = self.parameter_scales_from_theta(start), and is restarted from its result (with the
            scales updated to it) until a restart improves the objective by less than objective_tolerance (or, without one,
            RESTART_TOLERANCE times max(1, |objective|)), at most self.max_restarts times. Gradients are scipy's finite
            differences with relative steps on x (jac='2-point' for 'forward', n + 1 objective evaluations per gradient;
            '3-point' for 'central', 2n + 1); options['finite_diff_rel_step'] in minimize_kwargs overrides the step.

            Returns (theta, result): the optimized full theta in physical units, and the scipy result of the run that produced
            theta (a final restart that does not improve the objective only confirms it), with x and jac in physical units.
            With dimensionless_fit, result.x_internal holds the scaled optimum, result.scale the scales, result.hess_inv
            refers to the scaled variables, result.nit / nfev are summed over the runs, and result.restarts counts the
            restarts.
        """
        theta = np.array(theta_start, dtype=float, copy=True)
        free = list(range(len(theta))) if subset_indices is None else list(subset_indices)
        bounds = self.gate_set_model.parameter_bounds

        if not self.dimensionless_fit:
            # L-BFGS-B on theta (the free parameters), with its built-in finite-difference gradients
            def cost(x):
                trial = theta.copy()
                trial[free] = x
                return objective(trial)

            model_bounds = None if bounds is None else [bounds[i] for i in free]
            result = opt.minimize(cost, theta[free], method='L-BFGS-B', bounds=model_bounds, **minimize_kwargs)
            fitted = theta.copy()
            fitted[free] = result.x
            return fitted, result

        f_previous, total_nit, total_nfev = objective(theta), 0, 1
        reported, reported_scale = None, None
        for restart in range(self.max_restarts + 1):
            scale = self.parameter_scales_from_theta(theta)

            def full_theta(x, scale=scale, base=theta):
                trial = base.copy()
                trial[free] = np.asarray(x, dtype=float) * scale[free]
                return trial

            x_bounds = self.internal_bounds(bounds, scale, free)
            f_x = lambda x, full_theta=full_theta: objective(full_theta(x))
            x0 = theta[free] / scale[free]
            # scipy's relative-step finite differences, h = r * max(|x|, 1), one-sided at the bounds
            scheme = '3-point' if self.finite_differences == 'central' else '2-point'
            run_kwargs = minimize_kwargs
            if objective_tolerance is not None and 'ftol' not in minimize_kwargs.get('options', {}):
                # L-BFGS-B's ftol is relative to max(|f|, 1); |f| only decreases during the run, so this bounds the absolute
                # change at which it stops by objective_tolerance
                run_kwargs = dict(minimize_kwargs)
                run_kwargs['options'] = dict(minimize_kwargs.get('options', {}),
                                             ftol=objective_tolerance / max(1., abs(f_previous)))
            result = opt.minimize(f_x, x0, jac=scheme, method='L-BFGS-B', bounds=x_bounds, **run_kwargs)
            total_nit += result.nit
            total_nfev += result.nfev
            improvement = f_previous - result.fun
            threshold = (RESTART_TOLERANCE * max(1., abs(result.fun)) if objective_tolerance is None else objective_tolerance)
            improved = improvement >= threshold
            if improvement >= 0.:                       # keep the best point (a run never returns a worse one in practice)
                theta = full_theta(result.x)
                f_previous = result.fun
            # Report the run that produced theta: a restart that does not improve only confirms the previous run (its
            # status, e.g. an abnormal line search on finite-difference noise at the optimum, says nothing about theta)
            if reported is None or (improved and improvement >= 0.):
                reported, reported_scale = result, scale
            if not improved:
                break

        self.last_scale = reported_scale
        reported.x = theta[free]
        reported.x_internal = theta[free] / reported_scale[free]
        reported.scale = reported_scale[free]
        if getattr(reported, 'jac', None) is not None:
            reported.jac = np.asarray(reported.jac) / reported_scale[free]      # d objective / d theta
        reported.nit, reported.nfev, reported.restarts = total_nit, total_nfev, restart
        return theta, reported

    ### Model predictions ###
    def prep_and_effects(self, theta: Vector) -> tuple[np.ndarray, np.ndarray]:
        """ Prep supervector and effect matrix (rows in outcome-label order) at theta """
        rho_supervector = self.gate_set_model.prep_state(theta)
        measurement_effects = self.gate_set_model.measurement_effects(theta)
        effect_matrix = np.vstack([np.asarray(measurement_effects[label]) for label in self.outcome_labels])
        return rho_supervector, effect_matrix

    def gate_process_matrices(self, theta: Vector, displacement: float=0., gates=None) -> dict:
        """ Evaluate each gate's process matrix function once at theta (only `gates`, if given), with the circuit-noise argument
            of the noisy gates displaced by `displacement` """
        gates = self.gate_models.keys() if gates is None else gates
        return {gate: self.gate_models[gate](*self.gate_set_model.displaced_model_parameters(
                    theta, gate, displacement if gate in self.noisy_gates else 0.)) for gate in gates}

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
        """ Clipped outcome probabilities (n_sequences x n_outcomes, outcome-label order) for gate sequences at theta.

            With circuit-level noise, the probabilities of each sequence are averaged over the noise realizations, with every
            noisy gate of the sequence displaced by the same realization; sequences without noisy gates are evaluated once.
        """
        rho_supervector, effect_matrix = self.prep_and_effects(theta)
        gate_matrices = self.gate_process_matrices(theta)
        probabilities = np.zeros((len(sequences), len(effect_matrix)))
        circuit_map_cache = {}
        noisy = [k for k, gates in enumerate(sequences) if self.noisy_gates.intersection(gates)]
        noisy_set = set(noisy)
        for k in (k for k in range(len(sequences)) if k not in noisy_set):
            mapped_state = self.compose_quantum_map(sequences[k], gate_matrices, circuit_map_cache) @ rho_supervector
            probabilities[k] = np.real(effect_matrix @ mapped_state)
        if noisy:
            # Noisy gates as stacks over the realizations (K, d2, d2); composition broadcasts over the stack, so each
            # sequence is composed once for all realizations
            displacements, weights = self.gate_set_model.circuit_noise_realizations(theta)
            displaced = [self.gate_process_matrices(theta, x, self.noisy_gates) for x in displacements]
            stacked = dict(gate_matrices)
            for gate in self.noisy_gates:
                stacked[gate] = np.stack([matrices[gate] for matrices in displaced])
            circuit_map_cache = {}
            for k in noisy:
                mapped_states = self.compose_quantum_map(sequences[k], stacked, circuit_map_cache) @ rho_supervector   # (K, d2)
                probabilities[k] = weights @ np.real(mapped_states @ effect_matrix.T)
        return np.clip(probabilities, NUMERICAL_EQUIVALENCE_THRESHOLD, 1. - NUMERICAL_EQUIVALENCE_THRESHOLD)

    ### Objectives ###
    def log_likelihood(self, theta: Vector, outcomes: _CircuitOutcomes) -> float:
        """ Log-likelihood of theta for tabulated outcomes: sum over circuits and outcomes of N_outcome log p_outcome(theta) """
        return outcomes.log_likelihood(self.predict_sequence_probabilities(theta, outcomes.sequences))

    def chi_squared(self, theta: Vector, outcomes: _CircuitOutcomes) -> float:
        """ chi^2 between observed frequencies and model probabilities at theta, scaled by the shots of each circuit """
        return outcomes.chi_squared(self.predict_sequence_probabilities(theta, outcomes.sequences))

    @property
    def mle_tolerance(self) -> float | None:
        """ Absolute log-likelihood tolerance of the MLE solvers' optimizations (dimensionless fits only) """
        return self.likelihood_tolerance if self.dimensionless_fit else None

    def mle_objective(self, outcomes: _CircuitOutcomes) -> Callable[[np.ndarray], float]:
        """ The MLE solvers' objective: -log L(theta); with dimensionless_fit, log L_sat - log L(theta) >= 0 (the
            log-likelihood ratio to the saturated model), which has the same minimizer and a value of order the number of
            circuits, so that relative tolerances are meaningful. Each evaluation's -log L is recorded in nll_history. """
        offset = outcomes.saturated_log_likelihood() if self.dimensionless_fit else 0.

        def objective(theta):
            l_likelihood = self.log_likelihood(theta, outcomes)
            self.nll_history.append(-l_likelihood)
            if self.verbose:
                print(f"Evaluation {len(self.nll_history)}: negative log likelihood {-l_likelihood} at {theta}")
            return offset - l_likelihood
        return objective

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

    def run_linear_gst(self, frequencies: dict, prep_fiducials, measure_fiducials, target_gate_set: dict) -> dict:
        """ Estimate the gate set by linear inversion (Nielsen et al., "Gate Set Tomography", Quantum 2021), in the gauge of the
            target gate set, from the observed frequencies of each sequence (self.outcomes.pooled_frequencies()). Returns the
            LGST estimates. """
        if self.verbose:
            print(f"\n --- Running linear GST ---")
        # 1. Build the Gram matrix: <<F_i|F_j>> from the observed frequencies
        gram_matrix = self.build_probability_matrix(frequencies, prep_fiducials, measure_fiducials)

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
                warnings.warn(f"There is weak separation between signal and noise subspace. Ratio = {ratio}")

        # Check condition number of d^2 subspace:
        cond = S[0] / S[self.d2 - 1]
        if cond > 100:
            warnings.warn(f"Poorly conditioned linear GST. LGST estimates may be noisy. Condition number = {cond}")

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
            rho = self.gate_set_model.prep_state_model(*self.gate_set_model.model_parameters(theta, 'prep'))
            return np.linalg.norm(rho - lgst['native_prep_state'])**2
        if label == 'POVM':
            modeled_effects = self.gate_set_model.POVM_effect_models(*self.gate_set_model.model_parameters(theta, 'POVM'))
            return sum(np.linalg.norm(np.asarray(modeled_effects[outcome]) - effect)**2 for outcome, effect in lgst['estimated_effects'].items())
        gate = gst_gate_from_label(label)
        M = self.gate_models[gate](*self.gate_set_model.model_parameters(theta, gate))
        return np.linalg.norm(M - lgst['gate_estimates'][gate], 'fro')**2

    def _fit_models_to_lgst(self, lgst: dict, theta: Vector, model_labels: list[str], minimize_kwargs: dict) -> Vector:
        """ Fit the parameters of the given models to their LGST estimates; other parameters stay at their values in theta """
        indices = self.gate_set_model.parameter_indices_of_models(model_labels)
        if not indices:
            return theta

        def cost(trial):
            return float(np.real(sum(self._lgst_cost_terms(lgst, label, trial) for label in model_labels)))

        ## Note: L-BFGS-B performs significantly better here than Nelder-Mead
        fitted, _ = self.minimize(cost, theta, indices, **minimize_kwargs)
        return fitted

    def group_circuits(self, organize_circuits_by_germ_power: bool) -> dict:
        """ Indices of the circuits (rows of self.outcomes) grouped by germ power p, or by base depth L (germ length x germ
            power, binned to powers of 2) """
        groups = {}
        for row, circ in enumerate(self.gst_circuits):
            if organize_circuits_by_germ_power:
                key = circ.germ_power
            else:
                depth = len(circ.germ_gates) * circ.germ_power
                key = 1 if depth <= 1 else int(2**np.ceil(np.log2(depth)))      # binned up to a power of 2
            groups.setdefault(key, []).append(row)
        return groups

    ### Solution ###
    def build_solution_from_theta(self, theta: Vector, method: str, theta_0: Vector, **fields) -> GstResult:
        """ GstResult of a solve, stored in self.solution and returned: theta and its log-likelihood on all of the problem's
            circuits, the starting point theta_0, the history of objective evaluations, and (dimensionless fits) the scales of
            the last run; `fields` are the solver-specific fields (e.g. success, message, optimizer_result) """
        self.solution = GstResult(theta=np.array(theta, dtype=float), parameter_names=self.gate_set_model.parameter_names, method=method,
                         log_likelihood=self.log_likelihood(theta, self.outcomes), initial_guess=np.array(theta_0, dtype=float),
                         negative_log_likelihood_history=list(self.nll_history),
                         parameter_scales=None if self.last_scale is None else self.last_scale.copy(), **fields)
        return self.solution


### Solvers ###
def linear_solve_for_gate_parameters(gst_circuits: list[GstCircuit], gate_set_model: GateSetModel, circuit_design,
                                     target: Vector | dict | None=None, target_gate_set: dict | None=None,
                                     initial_guess: Vector | dict | None=None, max_gauge_iterations: int=100,
                                     gauge_tolerance: float=1e-6, verbose: bool=False, dimensionless_fit: bool=False,
                                     parameter_scales: Vector | dict | None=None, finite_differences: str='forward',
                                     max_restarts: int=5, **minimize_kwargs) -> GstResult:
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
        - dimensionless_fit: fit dimensionless parameters theta / scale with adaptive scales, relative-step gradients, and (for
            MLE) the log-likelihood ratio objective; see the module docstring. Recommended when parameters span orders of
            magnitude. Results are in physical units either way. The following options apply only with dimensionless_fit:
            - parameter_scales: initial scales: None (|initial guess|), a vector, or a {parameter name: scale} dictionary
                (unlisted parameters take |initial guess|). Give scales for parameters whose initial guess is 0; see
                parse_parameter_scalings.
            - finite_differences: 'forward' (n + 1 objective evaluations per gradient) or 'central' (2n + 1, more accurate).
            - max_restarts: maximum number of L-BFGS-B restarts with scales updated to the current estimate.
        - minimize_kwargs: passed to scipy.optimize.minimize (L-BFGS-B) for the model fits (with dimensionless_fit, options
            apply to the scaled variables). The default options are tight (ftol = 1e-15, gtol = 1e-12) so that the fits are
            precise enough for re-gauging to converge.

        Returns a GstResult; result.lgst_estimates holds the final pass's LGST estimates (gate estimates keyed by gate label),
        result.gauge_iterations the number of re-gauging passes, and result.success whether re-gauging converged.
    """
    if gate_set_model.circuit_noise is not None:
        raise NotImplementedError("Linear GST does not support circuit-level noise (GateSetModel.set_circuit_noise): LGST "
                                  "estimates noise-averaged gates. Use mle_solve_for_gate_parameters or "
                                  "staged_mle_solve_for_gate_parameters.")
    problem = _GstProblem(gst_circuits, gate_set_model, verbose, parameter_scales=parameter_scales,
                          dimensionless_fit=dimensionless_fit, finite_differences=finite_differences,
                          max_restarts=max_restarts)
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
    frequencies = problem.outcomes.pooled_frequencies()
    lgst = problem.run_linear_gst(frequencies, prep_fiducials, measure_fiducials, gauge_target)
    theta = problem.parameters_from_lgst(lgst, theta_0, minimize_kwargs)

    # Re-gauge: repeat LGST in the gauge of the fitted model and refit, until the parameters converge
    converged = max_gauge_iterations == 0
    iterations = 0
    while iterations < max_gauge_iterations:
        regauge_target = _internal_gate_set(gate_set_model, theta, None, 'target')
        lgst = problem.run_linear_gst(frequencies, prep_fiducials, measure_fiducials, regauge_target)
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
    return problem.build_solution_from_theta(theta, 'linear', theta_0, lgst_estimates=lgst, gauge_iterations=iterations, success=converged, message=message)


def mle_solve_for_gate_parameters(gst_circuits: list[GstCircuit], gate_set_model: GateSetModel,
                                  initial_guess: Vector | dict | None=None, verbose: bool=False, dimensionless_fit: bool=False,
                                  parameter_scales: Vector | dict | None=None, finite_differences: str='forward',
                                  max_restarts: int=5, likelihood_tolerance: float=LIKELIHOOD_TOLERANCE,
                                  **minimize_kwargs) -> GstResult:
    """ Maximum likelihood estimation (MLE): finds the parameters that maximize the likelihood of the data over all circuits,

            max[ Likelihood( {G} | data) ] over parameter set theta,

        using scipy.optimize.minimize with method L-BFGS-B and the parameter bounds from the parameter specification.

        - gst_circuits: circuits with measurement data.
        - gate_set_model: GateSetModel (the models of the gate set elements and their parameters).
        - initial_guess: None (uses the guesses specified in gate_set_model), or an override in the form of a dictionary of parameter values, or a vector
        - verbose: print each objective evaluation.
        - dimensionless_fit: fit dimensionless parameters theta / scale with adaptive scales, relative-step gradients, and (for
            MLE) the log-likelihood ratio objective; see the module docstring. Recommended when parameters span orders of
            magnitude. Results are in physical units either way. The following options apply only with dimensionless_fit:
            - parameter_scales: initial scales: None (|initial guess|), a vector, or a {parameter name: scale} dictionary
                (unlisted parameters take |initial guess|). Give scales for parameters whose initial guess is 0; see
                parse_parameter_scalings.
            - finite_differences: 'forward' (n + 1 objective evaluations per gradient) or 'central' (2n + 1, more accurate).
            - max_restarts: maximum number of L-BFGS-B restarts with scales updated to the current estimate.
            - likelihood_tolerance: absolute log-likelihood tolerance (MLE solvers): stop when an iteration, or a restart,
                improves the log-likelihood by less (default 1e-5; a change of 0.5 is ~1 standard deviation).
        - minimize_kwargs: passed to scipy.optimize.minimize, e.g. options = {'maxiter': 500} (with dimensionless_fit,
            options apply to the scaled variables, and options['ftol'] overrides likelihood_tolerance).

        Returns a GstResult (the scipy result is in result.optimizer_result).
    """
    problem = _GstProblem(gst_circuits, gate_set_model, verbose, parameter_scales=parameter_scales,
                          dimensionless_fit=dimensionless_fit, finite_differences=finite_differences,
                          max_restarts=max_restarts, likelihood_tolerance=likelihood_tolerance)
    theta_0 = problem.parse_initial_guess(initial_guess)
    theta, solver_result = problem.minimize(problem.mle_objective(problem.outcomes), theta_0, objective_tolerance=problem.mle_tolerance,
                                            **minimize_kwargs)
    return problem.build_solution_from_theta(theta, 'MLE', theta_0, success=bool(solver_result.success), message=str(solver_result.message), optimizer_result=solver_result)


def staged_mle_solve_for_gate_parameters(gst_circuits: list[GstCircuit], gate_set_model: GateSetModel,
                                         initial_guess: Vector | dict | None=None, organize_circuits_by_germ_power: bool=True,
                                         verbose: bool=False, dimensionless_fit: bool=False,
                                         parameter_scales: Vector | dict | None=None, finite_differences: str='forward',
                                         max_restarts: int=5, likelihood_tolerance: float=LIKELIHOOD_TOLERANCE,
                                         **minimize_kwargs) -> GstResult:
    """ Staged MLE: maximum likelihood estimation on cumulative batches of circuits of increasing depth, each stage starting
        from the previous stage's estimate. This can help avoid local optima for long circuits.

        - initial_guess: initial guess for the first stage (as for mle_solve_for_gate_parameters).
        - organize_circuits_by_germ_power: stage by germ power p (default) or by base circuit depth L.
        - dimensionless_fit: fit dimensionless parameters theta / scale with adaptive scales, relative-step gradients, and (for
            MLE) the log-likelihood ratio objective; see the module docstring. Recommended when parameters span orders of
            magnitude. Results are in physical units either way. The following options apply only with dimensionless_fit:
            - parameter_scales: initial scales: None (|initial guess|), a vector, or a {parameter name: scale} dictionary
                (unlisted parameters take |initial guess|). Give scales for parameters whose initial guess is 0; see
                parse_parameter_scalings.
            - finite_differences: 'forward' (n + 1 objective evaluations per gradient) or 'central' (2n + 1, more accurate).
            - max_restarts: maximum number of L-BFGS-B restarts with scales updated to the current estimate.
            - likelihood_tolerance: absolute log-likelihood tolerance (MLE solvers): stop when an iteration, or a restart,
                improves the log-likelihood by less (default 1e-5; a change of 0.5 is ~1 standard deviation).
        - minimize_kwargs: passed to scipy.optimize.minimize at every stage, e.g. options = {'maxiter': 500} (with
            dimensionless_fit, options apply to the scaled variables, and options['ftol'] overrides likelihood_tolerance).

        Returns a GstResult from the final stage; result.stage_estimates holds the estimate after each stage, keyed by depth.
    """
    problem = _GstProblem(gst_circuits, gate_set_model, verbose, parameter_scales=parameter_scales,
                          dimensionless_fit=dimensionless_fit, finite_differences=finite_differences,
                          max_restarts=max_restarts, likelihood_tolerance=likelihood_tolerance)
    theta_0 = problem.parse_initial_guess(initial_guess)
    circuit_groups = problem.group_circuits(organize_circuits_by_germ_power)
    sorted_depths = sorted(circuit_groups.keys())
    if verbose:
        print(f"--- Staged MLE with bins by {'germ powers (p)' if organize_circuits_by_germ_power else 'circuit depth (L)'}: {sorted_depths}")

    stage_estimates = {}
    cumulative_rows = []
    theta = theta_0
    for stage, L in enumerate(sorted_depths):
        cumulative_rows.extend(circuit_groups[L])
        circuit_outcomes = problem.outcomes.subset_from_rows(cumulative_rows)
        # I found that using log likelihood for all stages gave faster and likely better results
        theta, solver_result = problem.minimize(problem.mle_objective(circuit_outcomes), theta, objective_tolerance=problem.mle_tolerance,
                                                **minimize_kwargs)
        stage_estimates[L] = theta.copy()
        if verbose:
            print(f"    Stage {stage + 1} (L <= {L}): {len(cumulative_rows)} circuits, "
                  f"LL = {problem.log_likelihood(theta, circuit_outcomes):.3f}, converged = {solver_result.success}")

    # The last stage includes every circuit, so the result's log-likelihood (on all circuits) is that stage's
    return problem.build_solution_from_theta(theta, 'staged MLE', theta_0, success=bool(solver_result.success),
                                             message=str(solver_result.message), optimizer_result=solver_result,
                                             stage_estimates=stage_estimates)


### Likelihood and chi^2 ###
def log_likelihood(gst_circuits: list[GstCircuit], gate_set_model: GateSetModel, theta: Vector | dict) -> float:
    """ Total log-likelihood of parameter values theta given the circuits' current measurement data:
            sum over circuits and outcomes of N_outcome log( p_outcome(theta) ) """
    problem = _GstProblem(gst_circuits, gate_set_model)
    return problem.log_likelihood(gate_set_model.parse_theta(theta), problem.outcomes)


def chi_squared(gst_circuits: list[GstCircuit], gate_set_model: GateSetModel, theta: Vector | dict) -> float:
    """ chi^2 between the circuits' observed outcome frequencies and the model probabilities at theta """
    problem = _GstProblem(gst_circuits, gate_set_model)
    return problem.chi_squared(gate_set_model.parse_theta(theta), problem.outcomes)


### Simulation ###
def simulate_gst_data(gst_circuits: list[GstCircuit], gate_set_model: GateSetModel, theta: Vector | dict, N_shots: int,
                      rng: np.random.Generator | int | None=None) -> list[GstCircuit]:
    """ Sample measurement outcomes for each circuit from the models at parameter values theta.

        Returns new circuits (copies with the sampled counts); the given circuits are not modified. The exact gate models are
        used, also for gates with interpolated models.
        - rng: a numpy Generator or seed, for reproducibility (default: a new Generator).
    """
    # Simulate each circuit, generating probabilities taken to be ground truth.  
    rng = np.random.default_rng(rng)
    # Simulate from the exact models; the circuits need no data
    problem = _GstProblem(gst_circuits, gate_set_model, exact_models=True, require_data=False)
    probabilities = problem.predict_sequence_probabilities(gate_set_model.parse_theta(theta), problem.outcomes.sequences)
    simulated = []
    # Sample outcomes for each circuit  
    for circ, k in zip(problem.gst_circuits, problem.outcomes.sequence_index):
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
        With circuit-level noise (GateSetModel.set_circuit_noise), the nominal gates (no noise displacement) are compared: the
        noise distribution is assessed through its parameters (e.g. parameter_squared_errors), since noise-averaged gates would
        describe it as Markovian.
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


def parameter_squared_errors(gate_set_model: GateSetModel, theta: Vector | dict, reference: Vector | dict, relative_error: bool=False) -> dict:
    """ Squared error of each parameter estimate, {parameter name: (theta_i - reference_i)^2}, in the order of
        gate_set_model.parameter_names (e.g. reference = the true values in a simulation study). Averaged over replicates, these
        are the mean squared errors to compare with the Cramer-Rao bounds (squared).

        - theta, reference: parameter vectors or dictionaries of parameter names to values (unlisted parameters take their
            initial guesses).
        - relative_error: if True, ((theta_i - reference_i) / reference_i)^2 (nan where reference_i = 0).
    """
    parameter_estimate = gate_set_model.parse_theta(theta)
    reference = gate_set_model.parse_theta(reference)
    squared = (parameter_estimate - reference)**2
    if relative_error:
        with np.errstate(divide='ignore', invalid='ignore'):
            squared = np.where(reference != 0., squared / reference**2, np.nan)
    return dict(zip(gate_set_model.parameter_names, squared.tolist()))


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
