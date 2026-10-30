#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

from ionsim.custom_types import Vector
from ionsim.ionsim_error import IonSimError
from ionsim.custom_math import trapz_for_matrix

import numpy as np
from dataclasses import dataclass, field
from functools import wraps
import inspect
from typing import Callable
import inspect

def _gaussian(x: float, standard_deviation: float, mean: float = 0):
    """A normalized Gaussian function."""
    sigma = standard_deviation
    norm = 1/np.sqrt(2*np.pi*sigma**2)
    return norm * np.exp(-(x-mean)**2/(2*sigma**2))

def _exponential(x: float, decay_constant: float, mean: float = 0):
    """A normalized decaying exponential function."""
    tau = decay_constant
    norm = 1/(2*tau)
    return norm * np.exp(-abs(x-mean)/tau)

def _box(x: float, half_width: float, mean: float = 0):
    """A normalized contant function with a cutoff."""
    x0 = half_width
    norm = 1/(2*x0)
    if abs(x-mean) <= x0:
        return norm
    else:
        return 0.0

_PROBABILITY_DENSITY_FUNCTIONS = {
    'gaussian': _gaussian,
    'exponential': _exponential,
    'box': _box,
}

@dataclass(frozen=True, eq=False)
class Noise:
    """A quasi-static fluctuation of a function parameter with a particular probability density function.

        - parameter_name: the fluctuating function parameter.
        - probability_density_function: pdf(x) of the displacement x of the parameter.
        - domain_arguments: quadrature grid of displacements (in units of the domain scale, if set; see below).
        - pdf_parameters: named parameters of the distribution, e.g. {'standard_deviation': 0.1}, when the pdf is built from a
            parameterized pdf (from_named_pdf or from_pdf). These can then be varied, e.g. as arguments of a circuit process
            matrix function (named '<parameter_name>_noise_<pdf parameter>'), to compute sensitivities to the noise distribution.
        - parameterized_pdf: pdf(x, **pdf_parameters).
        - domain_scale_parameter: if set (e.g. 'standard_deviation'), domain_arguments are in units of this pdf parameter, so the
            grid scales with the distribution's width as it is varied. Otherwise domain_arguments are absolute displacements.
        - domain_scale_function: alternatively, a function of pdf parameters returning the domain scale, for distributions whose
            width is not itself a pdf parameter, e.g. lambda variance: np.sqrt(variance) for a Gaussian parameterized by its
            variance. It is called with the pdf parameters named in its signature. At most one of domain_scale_parameter and
            domain_scale_function may be set.

        Realizations: the quadrature grid defines a finite set of noise realizations, the displacements x_k, with weights
        w_k proportional to pdf(x_k) times the trapezoid-rule weights of the grid, normalized to sum to 1 (so a grid that
        truncates the distribution represents the distribution conditioned on the grid's range). See realizations().
        noise.average(f) is the noise average of a function f of the displacement, sum_k w_k f(x_k), which approximates the
        integral of pdf(x) f(x) dx.
    """
    parameter_name: str
    probability_density_function: Callable
    domain_arguments: Vector
    pdf_parameters: dict = field(default_factory=dict)
    parameterized_pdf: Callable | None = None
    domain_scale_parameter: str | None = None
    domain_scale_function: Callable | None = None

    def __post_init__(self):
        object.__setattr__(self, 'pdf_parameters', dict(self.pdf_parameters))
        if self.domain_scale_parameter is not None and self.domain_scale_function is not None:
            raise ValueError("Specify at most one of domain_scale_parameter and domain_scale_function.")
        if self.domain_scale_parameter is not None and self.domain_scale_parameter not in self.pdf_parameters:
            raise ValueError(f"domain_scale_parameter {self.domain_scale_parameter!r} must be one of the pdf parameters "
                             f"{list(self.pdf_parameters)}.")
        if self.domain_scale_function is not None:
            if not callable(self.domain_scale_function):
                raise TypeError("domain_scale_function must be callable, e.g. lambda variance: np.sqrt(variance).")
            arguments = list(inspect.signature(self.domain_scale_function).parameters)
            unknown = [a for a in arguments if a not in self.pdf_parameters]
            if unknown or not arguments:
                raise ValueError(f"domain_scale_function must take pdf parameters {list(self.pdf_parameters)} as its arguments; "
                                 f"its arguments are {arguments}.")
            object.__setattr__(self, '_domain_scale_arguments', arguments)

    @classmethod
    def from_named_pdf(cls, parameter_name: str, pdf_name: str, pdf_parameters: dict[str, float],
            domain_arguments: Vector, domain_scale_parameter: str | None = None, domain_scale_function: Callable | None = None):
        """Build noise for a function parameter from the name of its probability density function ('gaussian',
            'exponential', 'box'), e.g. pdf_parameters = {'standard_deviation': 0.1}."""
        if pdf_name not in _PROBABILITY_DENSITY_FUNCTIONS:
            raise ValueError(f"Unknown pdf {pdf_name!r}; available: {list(_PROBABILITY_DENSITY_FUNCTIONS)}.")
        return cls.from_pdf(parameter_name, _PROBABILITY_DENSITY_FUNCTIONS[pdf_name], pdf_parameters, domain_arguments,
                            domain_scale_parameter, domain_scale_function)

    @classmethod
    def from_pdf(cls, parameter_name: str, pdf: Callable, pdf_parameters: dict[str, float], domain_arguments: Vector,
            domain_scale_parameter: str | None = None, domain_scale_function: Callable | None = None):
        """Build noise from a parameterized probability density function pdf(x, **pdf_parameters), e.g.

            def gaussian_by_variance(x, variance):
                return np.exp(-x**2 / (2 * variance)) / np.sqrt(2 * np.pi * variance)
            Noise.from_pdf('theta', gaussian_by_variance, {'variance': 1e-4}, np.linspace(-5, 5, 41),
                           domain_scale_function=lambda variance: np.sqrt(variance))
        """
        pdf_parameters = dict(pdf_parameters)
        parameterized_pdf = pdf
        def pdf(x):
            return parameterized_pdf(x, **pdf_parameters)
        return cls(parameter_name, pdf, domain_arguments, pdf_parameters, parameterized_pdf, domain_scale_parameter,
                   domain_scale_function)

    def argument_name(self, pdf_parameter: str) -> str:
        """Name of a pdf parameter as an argument of a function that varies it, e.g. 'theta_noise_standard_deviation'."""
        return f"{self.parameter_name}_noise_{pdf_parameter}"

    @property
    def variable_parameters(self) -> dict[str, float]:
        """{argument name: value} of the pdf parameters that can be varied (empty if the pdf is not parameterized)."""
        if self.parameterized_pdf is None:
            return {}
        return {self.argument_name(name): value for name, value in self.pdf_parameters.items()}

    def _resolved_pdf_parameters(self, pdf_parameters: dict) -> dict:
        if pdf_parameters and self.parameterized_pdf is None:
            raise ValueError("This Noise has no parameterized pdf, so its pdf parameters cannot be varied; build it with "
                             "Noise.from_named_pdf or Noise.from_pdf.")
        unknown = set(pdf_parameters) - set(self.pdf_parameters)
        if unknown:
            raise ValueError(f"Unknown pdf parameter(s) {sorted(unknown)}; the pdf parameters are {list(self.pdf_parameters)}.")
        return {**self.pdf_parameters, **pdf_parameters}

    def domain_scale(self, **pdf_parameters) -> float | None:
        """Scale of the quadrature grid (domain_scale_parameter's value or domain_scale_function's result), or None for an
            absolute grid."""
        if self.domain_scale_parameter is None and self.domain_scale_function is None:
            return None
        parameters = self._resolved_pdf_parameters(pdf_parameters)
        if self.domain_scale_parameter is not None:
            scale = parameters[self.domain_scale_parameter]
        else:
            scale = self.domain_scale_function(**{name: parameters[name] for name in self._domain_scale_arguments})
        if not np.isfinite(scale) or scale <= 0.:
            raise ValueError(f"The domain scale must be positive and finite; received {scale} at pdf parameters {parameters}. "
                             f"(For a fitted width, bound it away from zero, e.g. bounds=(1e-4, 10.).)")
        return float(scale)

    def displacements(self, **pdf_parameters) -> np.ndarray:
        """Quadrature grid of parameter displacements, x_k (scaled by the domain scale, if set)."""
        scale = self.domain_scale(**pdf_parameters)
        if scale is None:
            self._resolved_pdf_parameters(pdf_parameters)        # validates the pdf parameter names
            return np.asarray(self.domain_arguments, dtype=float)
        return scale * np.asarray(self.domain_arguments, dtype=float)

    def weights(self, **pdf_parameters) -> np.ndarray:
        """Weights w_k of the realizations (see the class docstring): pdf(x_k) times the trapezoid-rule weights of the grid,
            normalized to sum to 1."""
        return self.realizations(**pdf_parameters)[1]

    def realizations(self, **pdf_parameters) -> tuple[np.ndarray, np.ndarray]:
        """The noise realizations: (displacements x_k, weights w_k), optionally with different pdf parameter values. The noise
            average of f is sum_k w_k f(x_k); for quasistatic noise, every gate of a circuit takes the same x_k in a shot."""
        xs = self.displacements(**pdf_parameters)
        if xs.ndim != 1 or len(xs) < 2 or np.any(np.diff(xs) <= 0.):
            raise ValueError("The quadrature grid (domain_arguments) must be a 1D increasing array of at least 2 points.")
        trapezoid = np.zeros_like(xs)
        trapezoid[:-1] += np.diff(xs) / 2.
        trapezoid[1:] += np.diff(xs) / 2.
        density = np.array([self.density(x, **pdf_parameters) for x in xs], dtype=float)
        weights = density * trapezoid
        total = weights.sum()
        if not np.isfinite(total) or total <= 0.:
            raise ValueError(f"The pdf has no weight on the quadrature grid (total {total}); check domain_arguments and the pdf parameters.")
        return xs, weights / total

    def density(self, x: float, **pdf_parameters) -> float:
        """Probability density of displacement x, optionally with different pdf parameter values."""
        if not pdf_parameters:
            return self.probability_density_function(x)
        return self.parameterized_pdf(x, **self._resolved_pdf_parameters(pdf_parameters))

    def average(self, function_of_displacement: Callable, **pdf_parameters):
        """Noise average of function_of_displacement(x) (a scalar or array): sum_k w_k f(x_k) over the realizations, which
            approximates the integral of pdf(x) f(x) dx; optionally with different pdf parameter values (e.g.
            standard_deviation = 0.2)."""
        xs, weights = self.realizations(**pdf_parameters)
        ys = np.array([function_of_displacement(x) for x in xs])
        return np.tensordot(weights, ys, axes=1)

    def add_noise_to_matrix_function(self, matrix_function: Callable):
        """Replace a function with one averaged over the noisy parameter."""

        # ECM Fix 07/2026 to handle the case where a user passes in kwargs only
        @wraps(matrix_function)
        def wrapper(*args, **kwargs):
 #<<<<<<< HEAD
 #            param_in_kwargs = self.parameter_name in kwargs
 #            if param_in_kwargs:
 #                base_value = float(kwargs[self.parameter_name])
 #            else:
 #                if parameter_index >= len(args):
 #                    raise TypeError(f"Noisy parameter at position {parameter_index} was not provided positionally nor specified as a keyword argument.")
 #                base_value = float(args[parameter_index])
 #
 #            def displaced(darg):
 #                noisy_value = base_value + darg
 #                if param_in_kwargs:
 #                    return matrix_function(*args, **dict(kwargs, **{self.parameter_name: noisy_value}))
 #                call_args = list(args)
 #                call_args[parameter_index] = noisy_value
 #                return matrix_function(*call_args, **kwargs)
 #            return self.average(displaced)
 #=======
            # Locate the named parameter in the arguments, either
            # keyword or positional, so that we can change it later.
            sig = inspect.signature(matrix_function)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            if not self.parameter_name in bound.arguments:
                raise IonSimError(f"No paramter {self.parameter_name} in function to add noise to")

            base_value = bound.arguments[self.parameter_name]

            def call_noisy(delta):
                noisy_value = base_value + delta
                bound.arguments[self.parameter_name] = noisy_value
                return matrix_function(*bound.args, **bound.kwargs)

            ys = np.array([self.probability_density_function(darg) * call_noisy(darg)
                           for darg in self.domain_arguments])
            return trapz_for_matrix(ys, self.domain_arguments)
        return wrapper
