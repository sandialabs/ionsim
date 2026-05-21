from ionsim.custom_math import trapz_for_matrix
from ionsim.custom_types import Vector
from ionsim.ionsim_error import IonSimError

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
    

@dataclass(frozen=True, eq=False)
class StochasticNoise:
    """ Generate stochastic noise samples for time-dependent simulations. """

    @staticmethod
    def white_noise(
        n_trajectories: int,
        target_variance: float,
        rng: np.random.Generator,
        first_time_step_all_trajectories: np.ndarray | None = None,
        time_evals: Vector | None = None,
        same_psd: bool = False,
        dt_step: float | None = None,
        remove_mean: bool = False,
        mean: float = 0.0,
    ) -> np.ndarray:
        """
        Generate white noise samples with the correct power spectral density (PSD).
        Args:
            n_trajectories: Number of trajectories (noise realizations)
            N: Number of time steps
            dt: Time step size
            target_variance: Desired variance of the noise
            rng: numpy random number generator
            first_time_step_all_trajectories: Initial values for each trajectory (optional)
            same_psd: If True, use the same PSD for all steps (for legacy compatibility)
            dt_step: Reference time step for the simulation grid (required if same_psd=True)
        Returns:
            noise_all: Array of shape (n_trajectories, N) with white noise samples
        Raises:
            IonSimError: If same_psd is True and dt_step is not provided or bigger than dt.
        """
        N = len(time_evals)
        dt = time_evals[1] - time_evals[0]

        if same_psd:
            if dt_step is None:
                raise IonSimError("dt_step must be provided if same_psd=True in white_noise().")
            if dt_step >= dt:
                raise IonSimError(f"dt_step (={dt_step}) is >= dt (={dt}); this may lead to unphysical noise bandwidth.")
            f_nye = 1 / (2 * dt_step)
        else:
            f_nye = 1 / (2 * dt)
        psd = target_variance / f_nye  # (rad/s)^2/Hz

        noise_all = np.empty((n_trajectories, N), dtype=np.float64)

        if first_time_step_all_trajectories is not None:
            first_time_step_all_trajectories = np.asarray(first_time_step_all_trajectories, dtype=np.float64)
                        
            noise_all[:, 0] = first_time_step_all_trajectories

            noise_all[:, 1:] = rng.normal(
                mean,
                np.sqrt(psd * 1/(2 * dt)),
                size=(n_trajectories, N - 1),
            )
        else:
            noise_all = rng.normal(mean, np.sqrt(psd * 1/(2 * dt)), size=(n_trajectories, N))
        if remove_mean:
            noise_all = noise_all - np.mean(noise_all, axis=1, keepdims=True)
        return noise_all

    @staticmethod
    def ou_noise(n_trajectories: int, 
                 tau_c: float, 
                 rng: np.random.Generator,
                 c: float | None = None,
                 target_variance: float | None = None,
                 first_time_step_all_trajectories: np.ndarray | None = None,
                 time_evals: Vector | None = None,
                 remove_mean: bool = False,
                 mean: float = 0.0) -> np.ndarray:
        """
        Generate Ornstein-Uhlenbeck (OU, Lorentzian) colored noise samples.
        Args:
            n_trajectories: Number of trajectories (noise realizations)
            tau_c: Correlation time (decay constant)
            target_variance: Stationary variance of the process
            rng: numpy random number generator
            first_time_step_all_trajectories: Initial values for each trajectory (optional)
            time_evals: Time evaluation points
            remove_mean: If True, remove the mean from each trajectory
            mean: Constant mean to add to the noise (default 0.0)
        Returns:
            x: Array of shape (n_trajectories, N) with OU noise samples
        """
        if target_variance is None and c is None:
            raise IonSimError("Must provide either target_variance or c for OU noise generation.")
        
        var = target_variance if target_variance is not None else c * tau_c / 2
        N = len(time_evals)
        dt = time_evals[1] - time_evals[0]
        
        phi = np.exp(-dt / tau_c)
        sd = np.sqrt(var * (1.0 - phi * phi))
        x = np.empty((n_trajectories, N), float)
        x[:, 0] = rng.normal(mean, np.sqrt(var), size=n_trajectories)
        if first_time_step_all_trajectories is not None:
            x[:, 0] = first_time_step_all_trajectories
        else:
            x[:, 0] = rng.normal(mean, np.sqrt(var), size=n_trajectories)
        # Pre-generate all innovations in one batch call to avoid N separate
        # RNG dispatches inside the loop.
        xi = rng.standard_normal((n_trajectories, N))
        for n in range(1, N):
            x[:, n] = phi * x[:, n - 1] + sd * xi[:, n]
        if remove_mean:
            x = x - np.mean(x, axis=1, keepdims=True)
            
        return x


