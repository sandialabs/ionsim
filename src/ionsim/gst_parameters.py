#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

""" Parameter specification for GST gate-set models, shared by the GST analysis (gate_set_tomography.py) and circuit design
    / sensitivity analysis (gst_circuit_planner.py) so that parameters, their sharing among models, bounds, and initial guesses
    are specified once and named consistently everywhere.
"""

import inspect
import warnings
from typing import Callable

import numpy as np

from ionsim.custom_types import Vector
from ionsim.gst_circuit_parser import GstGate, gate_from_label, canonical_gate_label, IDLE_ALIASES
from ionsim.ionsim_error import IonSimError


class GstModelParameters:
    """ The models of a GST gate set and the organization of their parameters into a single parameter vector theta.

        Models:
            - 'prep': prep state model, a callable returning the prep state supervector rho_0(params).
            - 'POVM': measurement model, a callable returning a dictionary of effects, {'0': E0, '1': E1} or {'00': E00, ...}.
            - gate models, keyed by gate label (e.g. 'Gxpi2:0', 'MS:0:1', 'idle'), callables returning process matrices.

        Every argument of every model function is a parameter. By default each parameter is independent, unbounded, and has an
        initial guess of zero. specify_parameter() sets initial guesses and bounds and shares parameters among models:

            parameters = GstModelParameters(prep_state_function, POVM_models, {'Gxpi2:0': X_pi2_q0, 'MS:0:1': MS_pm})
            parameters.specify_parameter("SPAM_error_probability", model="shared", guess=1e-4, bounds=(0., 1.))
            parameters.specify_parameter("phi_error", model="MS:0:1", guess=0., bounds=(0., np.pi/16))

        Parameter names (e.g. 'shared:SPAM_error_probability', 'MS:0:1.phi_error', 'prep.p') and the layout of theta are
        organized lazily when first needed, so parameters may be specified in any order.

        The same object can be given to GateSetTomography and GSTCircuitPlanner so that the analysis and the circuit design /
        Fisher information use identical parameters.
    """

    def __init__(self, prep_state_model: Callable, POVM_effect_models: Callable, gate_models: dict[str, Callable]):
        if not callable(prep_state_model):
            raise TypeError(f"prep_state_model must be callable; received {type(prep_state_model).__name__}.")
        if not callable(POVM_effect_models):
            raise TypeError(f"POVM_effect_models must be callable (returning a dictionary of effects); received {type(POVM_effect_models).__name__}.")
        if not isinstance(gate_models, dict):
            raise TypeError(f"gate_models must be a dictionary mapping gate labels (e.g. 'Gxpi2:0') to model functions; received {type(gate_models).__name__}.")

        self.prep_state_model = prep_state_model
        self.POVM_effect_models = POVM_effect_models

        # Gate models keyed internally by GstGate (users key them by string label)
        self.gate_models = {}
        for key, model in gate_models.items():
            gate = gate_from_label(key)
            if gate in self.gate_models:
                raise ValueError(f"Gate {key!r} has more than one model (labels {IDLE_ALIASES} all refer to the idle gate).")
            if not callable(model):
                raise TypeError(f"The model for gate {key!r} must be callable; received {type(model).__name__}.")
            self.gate_models[gate] = model

        self._model_functions = {'prep': self.prep_state_model, 'POVM': self.POVM_effect_models}
        for gate, model in self.gate_models.items():
            self._model_functions[gate.label] = model
        self._model_parameter_names = {label: list(inspect.signature(fn).parameters.keys()) for label, fn in self._model_functions.items()}

        self._shared_parameter_specs = {}   # shared name -> {'models': tuple[str] | None, 'guess': float | None, 'bounds': tuple | None}
        self._model_parameter_specs = {}    # (model label, parameter name) -> {'guess': float | None, 'bounds': tuple | None}
        self._layout = None
        # Incremented whenever the specification changes; users of this object compare it to detect re-organization
        self.version = 0


    ### Models ###
    def model_label(self, model: str | GstGate) -> str:
        """ Returns the canonical model label for 'prep', 'POVM', or a gate label (e.g. '[]' -> 'idle'). """
        if model in ('prep', 'POVM'):
            return model
        if isinstance(model, GstGate):
            raise TypeError(f"Specify gates by string label (e.g. {model.label!r}) rather than GstGate objects.")
        if not isinstance(model, str):
            raise TypeError(f"Model must be 'prep', 'POVM', or a gate label string (e.g. 'Gxpi2:0'); received {type(model).__name__}.")
        try:
            label = canonical_gate_label(model)
        except ValueError as err:
            raise ValueError(f"Unknown model {model!r}. Available models: {self.model_labels}. ({err})") from None
        if label not in self._model_functions:
            raise ValueError(f"No model for gate {model!r}. Available models: {self.model_labels}.")
        return label

    @property
    def model_labels(self) -> list[str]:
        """ Labels of every model with parameters: 'prep', 'POVM', and each gate label. """
        return list(self._model_functions.keys())

    @property
    def model_parameter_names(self) -> dict[str, list[str]]:
        """ Argument names of each model, keyed by model label, e.g. {'prep': ['SPAM_error_probability'], 'Gxpi2:0': [...], ...} """
        return {label: list(names) for label, names in self._model_parameter_names.items()}

    def model_function(self, model: str | GstGate) -> Callable:
        """ Model function for 'prep', 'POVM', or a gate (label or GstGate) """
        label = model.label if isinstance(model, GstGate) else self.model_label(model)
        return self._model_functions[label]


    ### Parameter specification ###
    @staticmethod
    def _validate_bounds(bounds, context: str) -> tuple | None:
        if bounds is None:
            return None
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError(f"Bounds for {context} must be a (lower, upper) pair; use None for an open side. Received {bounds!r}.")
        lower, upper = bounds
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"Lower bound exceeds upper bound for {context}: {bounds!r}.")
        return (lower, upper)

    @staticmethod
    def _check_guess_within_bounds(guess, bounds, context: str):
        if guess is None or bounds is None:
            return
        lower, upper = bounds
        if (lower is not None and guess < lower) or (upper is not None and guess > upper):
            raise ValueError(f"Initial guess {guess} for {context} lies outside its bounds {bounds}.")

    def specify_parameter(self, name: str, model: str | list[str], guess: float | None=None, bounds: tuple | None=None):
        """ Specify the initial guess, bounds, and/or sharing of a model parameter. Parameters are identified by the argument
            names of the model functions.

            - name: the model argument name, e.g. "amplitude_noise_strength".
            - model: which model(s) the parameter belongs to:
                * a single model label: 'prep', 'POVM', or a gate label such as 'Gxpi2:0', 'MS:0:1', 'idle'.
                    The parameter is independent to that model.
                * "shared": one parameter shared by every model that has an argument called `name`.
                * a list of model labels, e.g. ['Gxpi2:0', 'Gypi2:0']: one parameter shared among only those models.
            - guess: initial value used by the solvers (default 0, moved inside the bounds if necessary).
            - bounds: (lower, upper) pair; use None for an open side. Parameters are unbounded by default.

            Calling this again for the same parameter updates it; arguments left as None keep their previous values.

            Examples:
                parameters.specify_parameter("SPAM_error_probability", model="shared", guess=1e-4, bounds=(0., 1.))
                parameters.specify_parameter("amplitude_noise_strength", model="shared", guess=0.01, bounds=(1e-4, 10.))
                parameters.specify_parameter("phi_error", model="MS:0:1", guess=0., bounds=(0., np.pi/16))
        """
        if not isinstance(name, str):
            raise TypeError(f"Parameter name must be a string; received {type(name).__name__}.")
        if guess is not None:
            guess = float(guess)

        if isinstance(model, str) and model == 'shared' or isinstance(model, (list, tuple)):
            # Shared parameter: determine member models now so that errors are reported immediately.
            if isinstance(model, (list, tuple)):
                if len(model) == 0:
                    raise ValueError(f"Specify at least one model to share parameter {name!r} among.")
                member_models = tuple(self.model_label(m) for m in model)
                if len(set(member_models)) != len(member_models):
                    raise ValueError(f"Repeated model in the list of models sharing parameter {name!r}: {list(model)}.")
                lacking = [m for m in member_models if name not in self._model_parameter_names[m]]
                if lacking:
                    raise ValueError(f"Model(s) {lacking} have no parameter named {name!r}. "
                                     f"Their parameters are: { {m: self._model_parameter_names[m] for m in lacking} }.")
            else:
                member_models = None
                if not any(name in names for names in self._model_parameter_names.values()):
                    raise ValueError(f"No model has a parameter named {name!r}. Model parameters are: {self._model_parameter_names}.")

            context = f"shared parameter {name!r}"
            spec = self._shared_parameter_specs.get(name, {'models': None, 'guess': None, 'bounds': None})
            spec = dict(spec, models=member_models)
        else:
            label = self.model_label(model)
            if name not in self._model_parameter_names[label]:
                raise ValueError(f"Model {label!r} has no parameter named {name!r}. Its parameters are: {self._model_parameter_names[label]}.")
            context = f"parameter {name!r} of model {label!r}"
            spec = dict(self._model_parameter_specs.get((label, name), {'guess': None, 'bounds': None}))

        if bounds is not None:
            spec['bounds'] = self._validate_bounds(bounds, context)
        if guess is not None:
            spec['guess'] = guess
        self._check_guess_within_bounds(spec['guess'], spec['bounds'], context)

        if 'models' in spec:
            self._shared_parameter_specs[name] = spec
        else:
            self._model_parameter_specs[(label, name)] = spec
        self._invalidate_layout()

    def _invalidate_layout(self):
        """ Discard the parameter layout; it is rebuilt on next use. """
        if self._layout is not None:
            warnings.warn("The GST parameter specification changed after the parameter vector was organized. The parameter vector "
                          "is being re-organized; parameter vectors built before this point (e.g. from build_theta_from_dict or a "
                          "previous solve) may no longer correspond to the new ordering.")
        self._layout = None
        self.version += 1

    @property
    def layout(self) -> dict:
        """ The parameter layout (built on first use): per-model index lists, shared indices, names, bounds, initial guess. """
        if self._layout is None:
            self._layout = self._build_layout()
        return self._layout

    def _build_layout(self) -> dict:
        """ Builds and organizes the independent parameters for GST. This organizes parameters as:
            1) Shared parameters, in the order they were specified
            2) Prep state model parameters
            3) Native measurement (POVM) model parameters
            4) Each gate model's parameters, for all gates in the set
        """
        # Which (model, parameter name) pairs are tied to each shared parameter
        shared_lookup = {}
        for shared_name, spec in self._shared_parameter_specs.items():
            members = spec['models']
            if members is None:
                members = [label for label, names in self._model_parameter_names.items() if shared_name in names]
            for label in members:
                shared_lookup[(label, shared_name)] = shared_name

        conflicts = [f"{label}.{name}" for (label, name) in self._model_parameter_specs if (label, name) in shared_lookup]
        if conflicts:
            raise ValueError(f"Parameter(s) {conflicts} were specified as independent but are also shared (specify_parameter(..., model='shared')). "
                             f"Either share the parameter among a list of models that excludes these, or drop the independent specification.")

        names, bounds, guesses = [], [], []

        def _add_parameter(label, spec):
            spec = spec or {}
            bound = spec.get('bounds') or (None, None)
            guess = spec.get('guess')
            if guess is None:
                # Default guess of 0, moved inside the bounds if needed
                lower, upper = bound
                guess = 0.
                if lower is not None:
                    guess = max(guess, lower)
                if upper is not None:
                    guess = min(guess, upper)
            names.append(label)
            bounds.append(bound)
            guesses.append(guess)
            return len(names) - 1

        # Allocate shared parameters first
        shared_indices = {}
        for shared_name, spec in self._shared_parameter_specs.items():
            shared_indices[shared_name] = _add_parameter(f"shared:{shared_name}", spec)

        # Build per-model mapping that maps model label -> [theta_idx, ...]
        indices_by_model = {}
        for label, parameter_names in self._model_parameter_names.items():
            theta_indices = []
            for pname in parameter_names:
                if (label, pname) in shared_lookup:
                    # Shared parameter, point to shared slot in theta
                    theta_indices.append(shared_indices[shared_lookup[(label, pname)]])
                else:
                    theta_indices.append(_add_parameter(f"{label}.{pname}", self._model_parameter_specs.get((label, pname))))
            indices_by_model[label] = theta_indices

        return {'indices_by_model': indices_by_model, 'shared_indices': shared_indices, 'names': names,
                'bounds': bounds, 'initial_guess': np.array(guesses, dtype=float)}

    @property
    def indices_by_model(self) -> dict[str, list[int]]:
        """ Indices into the parameter vector for each model, keyed by model label ('prep', 'POVM', 'Gxpi2:0', ...). """
        return self.layout['indices_by_model']

    @property
    def shared_indices(self) -> dict[str, int]:
        """ Index into the parameter vector for each shared parameter, keyed by shared parameter name. """
        return self.layout['shared_indices']

    @property
    def num_parameters(self) -> int:
        return len(self.layout['names'])

    @property
    def parameter_names(self) -> list[str]:
        """ Parameter names in the order of theta, e.g. 'shared:SPAM_error_probability', 'MS:0:1.phi_error'. """
        return list(self.layout['names'])

    @property
    def parameter_bounds(self) -> list[tuple[float | None, float | None]] | None:
        """ (lower, upper) bounds for each parameter in theta order, or None if every parameter is unbounded. """
        bounds = self.layout['bounds']
        if all(b == (None, None) for b in bounds):
            return None
        return list(bounds)

    @property
    def initial_guess(self) -> Vector:
        """ Initial guess for theta, built from specify_parameter() guesses (0 by default). """
        return self.layout['initial_guess'].copy()

    def print_layout(self):
        """ Prints each entry of the parameter vector with its initial guess and bounds. """
        layout = self.layout
        print("\n --- GST parameter layout --- ")
        for i, (name, guess, bound) in enumerate(zip(layout['names'], layout['initial_guess'], layout['bounds'])):
            print(f"  [{i:3d}] {name:<45s} guess = {guess:<12.6g} bounds = {bound}")


    ### Parameter names, vectors, and values ###
    def normalize_parameter_name(self, key: str) -> str:
        """ Canonicalizes a user parameter name, e.g. '[].theta' -> 'idle.theta', 'shared:x' unchanged. """
        if key.startswith('shared:'):
            return key
        model, sep, pname = key.rpartition('.')
        if not sep:
            return key
        try:
            return f"{self.model_label(model)}.{pname}"
        except (ValueError, TypeError):
            return key

    def build_theta_from_dict(self, param_values: dict, default_value: float = 0., base: Vector | None=None) -> Vector:
        """ Builds a theta vector from a dictionary of parameter names to values.

            Accepted forms (and mixtures of them):
                - flat names as in parameter_names: {'shared:SPAM_error_probability': 1e-3, 'MS:0:1.phi_error': 0.05}
                - nested by model: {'shared': {'SPAM_error_probability': 1e-3}, 'MS:0:1': {'phi_error': 0.05}}

            Unlisted parameters take default_value, or their value in `base` if a base vector is given.
        """
        names = self.parameter_names
        if base is None:
            theta = np.full(self.num_parameters, default_value, dtype=float)
        else:
            theta = np.array(base, dtype=float, copy=True)
            if theta.shape != (self.num_parameters,):
                raise ValueError(f"Base parameter vector must have length {self.num_parameters}; received shape {theta.shape}.")

        # Flatten nested dictionaries:
        flat_values = {}
        for key, val in param_values.items():
            if isinstance(val, dict):
                for param_name, param_val in val.items():
                    if key == 'shared':
                        flat_values[f"shared:{param_name}"] = param_val
                    else:
                        flat_values[self.normalize_parameter_name(f"{key}.{param_name}")] = param_val
            else:
                flat_values[self.normalize_parameter_name(key)] = val

        # Assign values by matching names
        unmatched = set(flat_values.keys())
        for i, name in enumerate(names):
            if name in flat_values:
                theta[i] = flat_values[name]
                unmatched.discard(name)

        if unmatched:
            available = '\n '.join(names)
            raise ValueError(f"Unknown parameter names: {unmatched}.\n Available parameters:\n {available}")

        return theta

    def resolve_theta(self, parameter_values: Vector | dict | None) -> Vector:
        """ A full parameter vector from a vector, a dictionary of parameter names to values (unlisted parameters take their
            initial guesses), or None (the initial guess). """
        if parameter_values is None:
            return self.initial_guess
        if isinstance(parameter_values, dict):
            return self.build_theta_from_dict(parameter_values, base=self.initial_guess)
        theta = np.array(parameter_values, dtype=float, copy=True)
        if theta.shape != (self.num_parameters,):
            raise ValueError(f"Parameter vector must have length {self.num_parameters} (see parameter_names); received shape {theta.shape}.")
        return theta

    def model_parameters(self, theta: Vector, model: str | GstGate) -> Vector:
        """ Values of a model's arguments ('prep', 'POVM', or a gate label / GstGate) from theta, in argument order """
        label = model.label if isinstance(model, GstGate) else self.model_label(model)
        return theta[self.indices_by_model[label]]

    def get_parameter_index(self, name: str, model: str) -> int:
        """ Index in the parameter vector of parameter `name` of `model` ('shared', 'prep', 'POVM', or a gate label). """
        if model == 'shared':
            if name not in self.shared_indices:
                raise ValueError(f"Unknown shared parameter {name!r}. Shared parameters: {list(self.shared_indices.keys())}.")
            return self.shared_indices[name]
        label = self.model_label(model)
        parameter_names = self._model_parameter_names[label]
        if name not in parameter_names:
            raise ValueError(f"Model parameter {name!r} is not found in model {label!r}. The model has parameters {parameter_names}.")
        return self.indices_by_model[label][parameter_names.index(name)]

    def argument_names(self, model: str | GstGate) -> list[str]:
        """ Global parameter name (as in parameter_names) of each argument of a model, in argument order.
            Shared arguments map to their shared name, e.g. ['shared:amplitude_noise_strength', 'Gxpi2:0.phase'] """
        label = model.label if isinstance(model, GstGate) else self.model_label(model)
        names = self.layout['names']
        return [names[i] for i in self.indices_by_model[label]]

    def parameter_indices_of_models(self, models: list[str | GstGate]) -> list[int]:
        """ Sorted indices of every parameter that any of the given models depends on """
        indices = set()
        for model in models:
            label = model.label if isinstance(model, GstGate) else self.model_label(model)
            indices.update(self.indices_by_model[label])
        return sorted(indices)


    ### Model evaluation ###
    def prep_state(self, theta: Vector) -> Vector:
        """ Prep state supervector at theta; checks Tr[rho] = 1. """
        prep_state = np.asarray(self.prep_state_model(*self.model_parameters(theta, 'prep')))
        d = int(round(np.sqrt(prep_state.size)))
        trace = np.trace(prep_state.reshape(d, d))
        if np.abs(trace - 1.) > 1E-6:
            raise IonSimError(f"Prep state is not normalized, trace = {trace}")
        return prep_state

    def measurement_effects(self, theta: Vector) -> dict[str, Vector]:
        """ Measurement effects {'outcome': superbra} at theta; checks completeness (sum of effects = identity). """
        effects = self.POVM_effect_models(*self.model_parameters(theta, 'POVM'))
        total = sum(np.asarray(e) for e in effects.values())
        d = int(round(np.sqrt(total.size)))
        completeness_violation = np.linalg.norm(total.reshape(d, d) - np.eye(d))
        if np.abs(completeness_violation) > 1E-7:
            raise IonSimError(f"Measurement effect models are violating completenss constraint with residual {np.abs(completeness_violation)}")
        return effects

    def gate_process_matrix(self, gate: str | GstGate, theta: Vector):
        """ Process matrix of a gate (label or GstGate) at theta """
        gate = gate if isinstance(gate, GstGate) else gate_from_label(gate)
        return self.gate_models[gate](*self.model_parameters(theta, gate))
