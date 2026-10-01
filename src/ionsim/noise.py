#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

from ionsim.custom_math import trapz_for_matrix
from ionsim.custom_types import Vector

import numpy as np
from dataclasses import dataclass, field
from functools import wraps
from typing import Callable

from icecream import ic

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
        - domain_arguments: quadrature grid of displacements (in units of the pdf parameter `domain_scale_parameter`, if set).
        - pdf_parameters: named parameters of the distribution, e.g. {'standard_deviation': 0.1}, when the pdf is built from a
            parameterized pdf (from_named_pdf or from_pdf). These can then be varied, e.g. as arguments of a circuit process
            matrix function (named '<parameter_name>_noise_<pdf parameter>'), to compute sensitivities to the noise distribution.
        - parameterized_pdf: pdf(x, **pdf_parameters).
        - domain_scale_parameter: if set (e.g. 'standard_deviation'), domain_arguments are in units of this pdf parameter, so the
            grid scales with the distribution's width as it is varied. Otherwise domain_arguments are absolute displacements.

        noise.average(f) is the noise average of a function f of the displacement: integral of pdf(x) f(x) dx on the grid.
    """
    parameter_name: str
    probability_density_function: Callable
    domain_arguments: Vector
    pdf_parameters: dict = field(default_factory=dict)
    parameterized_pdf: Callable | None = None
    domain_scale_parameter: str | None = None

    def __post_init__(self):
        object.__setattr__(self, 'pdf_parameters', dict(self.pdf_parameters))
        if self.domain_scale_parameter is not None and self.domain_scale_parameter not in self.pdf_parameters:
            raise ValueError(f"domain_scale_parameter {self.domain_scale_parameter!r} must be one of the pdf parameters "
                             f"{list(self.pdf_parameters)}.")

    @classmethod
    def from_named_pdf(cls, parameter_name: str, pdf_name: str, pdf_parameters: dict[str, float],
            domain_arguments: Vector, domain_scale_parameter: str | None = None):
        """Build noise for a function parameter from the name of its probability density function ('gaussian',
            'exponential', 'box'), e.g. pdf_parameters = {'standard_deviation': 0.1}."""
        if pdf_name not in _PROBABILITY_DENSITY_FUNCTIONS:
            raise ValueError(f"Unknown pdf {pdf_name!r}; available: {list(_PROBABILITY_DENSITY_FUNCTIONS)}.")
        return cls.from_pdf(parameter_name, _PROBABILITY_DENSITY_FUNCTIONS[pdf_name], pdf_parameters, domain_arguments,
                            domain_scale_parameter)

    @classmethod
    def from_pdf(cls, parameter_name: str, pdf: Callable, pdf_parameters: dict[str, float], domain_arguments: Vector,
            domain_scale_parameter: str | None = None):
        """Build noise from a parameterized probability density function pdf(x, **pdf_parameters)."""
        pdf_parameters = dict(pdf_parameters)
        parameterized_pdf = pdf
        def pdf(x):
            return parameterized_pdf(x, **pdf_parameters)
        return cls(parameter_name, pdf, domain_arguments, pdf_parameters, parameterized_pdf, domain_scale_parameter)

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

    def displacements(self, **pdf_parameters) -> np.ndarray:
        """Quadrature grid of parameter displacements (scaled by the domain scale parameter, if set)."""
        parameters = self._resolved_pdf_parameters(pdf_parameters)
        if self.domain_scale_parameter is None:
            return self.domain_arguments
        return parameters[self.domain_scale_parameter] * np.asarray(self.domain_arguments)

    def density(self, x: float, **pdf_parameters) -> float:
        """Probability density of displacement x, optionally with different pdf parameter values."""
        if not pdf_parameters:
            return self.probability_density_function(x)
        return self.parameterized_pdf(x, **self._resolved_pdf_parameters(pdf_parameters))

    def average(self, function_of_displacement: Callable, **pdf_parameters):
        """Noise average of function_of_displacement(x): the integral of pdf(x) f(x) dx by the trapezoid rule on the grid,
            optionally with different pdf parameter values (e.g. standard_deviation = 0.2)."""
        xs = self.displacements(**pdf_parameters)
        ys = np.array([self.density(x, **pdf_parameters) * function_of_displacement(x) for x in xs])
        return trapz_for_matrix(ys, xs)

    def add_noise_to_matrix_function(self, matrix_function: Callable, parameter_index: int | None):
        """Replace a function with one averaged over the noisy parameter."""
        if parameter_index is None:
            return matrix_function

        # ECM Fix 07/2026 to handle the case where a user passes in kwargs only
        @wraps(matrix_function)
        def wrapper(*args, **kwargs):
            param_in_kwargs = self.parameter_name in kwargs
            if param_in_kwargs:
                base_value = float(kwargs[self.parameter_name])
            else:
                if parameter_index >= len(args):
                    raise TypeError(f"Noisy parameter at position {parameter_index} was not provided positionally nor specified as a keyword argument.")
                base_value = float(args[parameter_index])

            def displaced(darg):
                noisy_value = base_value + darg
                if param_in_kwargs:
                    return matrix_function(*args, **dict(kwargs, **{self.parameter_name: noisy_value}))
                call_args = list(args)
                call_args[parameter_index] = noisy_value
                return matrix_function(*call_args, **kwargs)
            return self.average(displaced)
        return wrapper
