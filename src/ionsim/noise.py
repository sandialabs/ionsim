from ionsim.custom_math import trapz_for_matrix
from ionsim.custom_types import Vector

import numpy as np
from dataclasses import dataclass
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
    """A quasi-static fluctuation of a function parameter with a particular probability density function."""
    parameter_name: str
    probability_density_function: Callable
    domain_arguments: Vector

    @classmethod
    def from_named_pdf(cls, parameter_name: str, pdf_name: str, pdf_parameters: dict[str, float],
            domain_arguments: Vector):
        """Build noise for a function parameter from the name of its probability density function."""
        def pdf(x):
            return _PROBABILITY_DENSITY_FUNCTIONS[pdf_name](x, **pdf_parameters)
        return cls(parameter_name, pdf, domain_arguments)

    def add_noise_to_matrix_function(self, matrix_function: Callable, parameter_index: int | None):
        """Replace a function with one averaged over the noisy parameter."""
        if parameter_index is None:
            return matrix_function
        @wraps(matrix_function)
        def wrapper(*args, **kwargs):
            function_arguments = np.array([[float(arg) for arg in args]]*len(self.domain_arguments)) # TODO: is float right here? Then, arguments will accept ints but they must be real
            function_arguments[:, parameter_index] += self.domain_arguments
            function_values = [matrix_function(*arguments) for arguments in function_arguments]
            probs = [self.probability_density_function(darg) for darg in self.domain_arguments]
            ys = np.array([p*fv for p, fv in zip(probs, function_values)])
            return trapz_for_matrix(ys, self.domain_arguments)
        return wrapper
