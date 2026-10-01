#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Any, Callable
from scipy.integrate import trapezoid as trapz
import itertools as it
from concurrent.futures import ProcessPoolExecutor
from scipy.integrate import odeint, solve_ivp, ode
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse import kron as skron 
from icecream import ic

from ionsim.custom_types import Vector, AnyMatrix 
from ionsim.ionsim_error import IonSimError

def matrix_AYB_multiply_to_superoperator(A: AnyMatrix | None, B: AnyMatrix | None=None) -> AnyMatrix:
    """Helper function to convert matrix multiplication to a superoperator form. 
        Matrix Y can be flattened column-wise, mapping to a vector "y".  

        Consider three-matrix product: A Y B ==> Oy

        A is a matrix multiplying a matrix of interest on the left.
        B is a matrix multiplying a matrix of interest on the right.

        A, Y, B are each N x N matrices, 
            O is a N^2 x N^2 matrix,
            y is a column vector with N^2 entries.  

        This function takes in A and B matrices and returns O. 
        To compute O, the general formula is: 
            A Y B --> (B^{T} kron A) y 

    """ 
    #Note: np.kron silently fails for sparse matrix inputs; instead use kron from scipy.sparse 
    if A is None and B is None:
        raise IonSimError('Input error: Specify either a left or right matrix A or B.')

    if A is not None:
        N = A.shape[0]
    
    if B is not None:
        N = B.shape[0]

    if A is not None and B is not None:
        assert N == A.shape[0]

    # Default behavior: If one matrix input is none, assume it is the identity. 
    if A is None and B is not None:
        result = skron(B.T, np.eye(N))
    elif B is None and A is not None:
        result = skron(np.eye(N), A) 
    elif A is not None and B is not None:
        result = skron(B.T, A)
    else:
        assert False, "A and B should not be None here"

    # If one or both matrices are sparse, return a sparse matrix 
    if sparse.issparse(A) or sparse.issparse(B):
        return result
    else:
        return result.toarray()

def solve_time_evolution_equation(interaction_function: Callable, initial_state_vector: Vector, duration: float,
    time_evals: Vector | None = None, ode_solver: str = 'odeintz', **kwargs):
    """Solve the time-dependent Schrodinger equation or the vectorized Lindblad master equation."""
    #print(f'Solving ODE with {ode_solver}.')
    if ode_solver == 'odeintz':
        return OdeIntz(interaction_function, initial_state_vector, duration, time_evals, **kwargs).solve()
    elif ode_solver == 'solve_ivp':
        return SolveIvp(interaction_function, initial_state_vector, duration, time_evals, **kwargs).solve()
    elif ode_solver == 'zvode':
        return ZVODE(interaction_function, initial_state_vector, duration, time_evals, **kwargs).solve()
    else:
        raise IonSimError(f'ODE solver {ode_solver} is not implemented.')

@dataclass(frozen=True, eq=False)
class OdeSolver(ABC):
    """A numerical routine to solve an ordinarty differential equation (ODE)."""
    # interaction_function: Callable
    interaction_function: Callable
    initial_vector: Vector
    duration: float
    time_evals: Vector | None

    @abstractmethod
    def solve(self):
        """Solves the ODE."""

@dataclass(frozen=True, eq=False)
class OdeIntz(OdeSolver):
    """A complex-valued version of Python's odeint routine."""
    def solve(self):
        """Solves the ODE."""
        def right_hand_side(t, y):
            return self.interaction_function(t).dot(-1j * y)
        def right_hand_side_flip_args(y, t):
            return right_hand_side(t, y)
        if self.time_evals is None:
            times = np.linspace(0, self.duration, 4)
        else:
            times = self.time_evals
        y0 = np.array(self.initial_vector, dtype='complex')
        result = odeintz(right_hand_side_flip_args, y0, times)
        return list(times), [y for y in result]

@dataclass(frozen=True, eq=False)
class SolveIvp(OdeSolver):
    """Python's solve_ivp routine."""
    def solve(self):
        """Solves the ODE."""
        def right_hand_side(t, y):
            return self.interaction_function(t).dot(-1j * y)
        y0 = np.array(self.initial_vector, dtype='complex')
        result = solve_ivp(right_hand_side, (0, self.duration), y0, t_eval=self.time_evals)
        return list(result['t']), [result['y'][:, i] for i in range(len(result['t']))]

@dataclass(frozen=True, eq=False)
class ZVODE(OdeSolver):
    """Python's zvode routine."""
    nsteps: float = 1e5
    atol: float = 1e-16
    rtol: float = 1e-14

    def solve(self):
        """Solves the ODE."""
        if self.time_evals is None:
            num_steps = 3
        else:
            num_steps = len(self.time_evals)
            assert(self.time_evals[-1] == self.duration)

        n_states = len(self.initial_vector)
        hamiltonian = self.interaction_function
        t_final = self.duration
        initial_state = self.initial_vector

        if initial_state is None:
            initial_state = _np.zeros(n_states)
            initial_state[0] = 1.

        intermediate_states = [initial_state]
        def schrodinger(t, y):
            return  -1.0j * hamiltonian(t).dot(y)
        def jacobian(t, y):
            tempham = hamiltonian(t)
            if sparse.issparse(tempham):
                return -1.0j * tempham.todense()
            else:
                return -1.0j * tempham
        r = ode(schrodinger, jacobian)
        r.set_integrator('zvode', method='adams', with_jacobian=True, atol = self.atol, rtol = self.rtol, nsteps=self.nsteps) # use method='bdf' for stiff ode
        r.set_initial_value(initial_state, 0)
        for k, t in enumerate(self.time_evals[1:], start=1):
            r.integrate(t)
            intermediate_states += [r.y]
            if not r.successful():
                raise RuntimeError(f"Integration failed at t={t}")

        return self.time_evals, intermediate_states

# working version
# @dataclass(frozen=True, eq=False)
# class ZVODE(OdeSolver):
#     """Python's zvode routine."""
#     nsteps: float = 1e6

#     def solve(self):
#         """Solves the ODE."""
#         if self.time_evals is None:
#             num_steps = 3
#         else:
#             num_steps = len(time_evals)
#             assert(time_evals[-1] == duration)

#         # TODO: remove the "propgagte" method below and just solve the ODE within the "solve" method.
#         def propagate(n_states, hamiltonian, t_final, initial_state=None, initial_time=0., display_progress=False,
#             return_intermediate=False, verbose=False, atol=1e-16, rtol=1e-14, nsteps=self.nsteps, num_steps=3):
#             """Propagate the initial wavefunction."""

#             if initial_state is None:
#                 initial_state = _np.zeros(n_states)
#                 initial_state[0] = 1.
#             if return_intermediate:
#                 # intermediate_states = []
#                 # intermediate_times = []
#                 intermediate_states = [initial_state]
#                 intermediate_times = [initial_time]

#             # Define the Schrodinger equation and the Jacobian
#             def schrodinger(t, y):
#                 return  -1.0j * hamiltonian(t).dot(y)
#             def jacobian(t, y):
#                 tempham = hamiltonian(t)
#                 if sparse.issparse(tempham):
#                     return -1.0j * tempham.todense()
#                 else:
#                     return -1.0j * tempham
#             # Instantiate the integrator
#             r = ode(schrodinger, jacobian)
#             r.set_integrator('zvode', method='adams', with_jacobian=True, atol=atol, rtol=rtol, nsteps=nsteps) # use method='bdf' for stiff ode
#             r.set_initial_value(initial_state, initial_time)
#             if display_progress or return_intermediate:
#                 # Do the integral in peices and display progress
#                 # n_steps = 1000
#                 dt = t_final/float(num_steps)
#                 if display_progress:
#                     evaluation_times = []
#                     previous_time = time()
#                 while r.successful() and r.t < t_final:
#                     r.integrate(r.t+dt)
#                     # print ''
#                     # print datetime.datetime.today()
#                     # print "%g" % (r.t)
#                     if display_progress:
#                         current_time = time()
#                         evaluation_times += [current_time-previous_time]
#                         previous_time = current_time
#                         this_step = len(evaluation_times)
#                         print('Finished step {0} of {1} in time {2}s.  Expected time remaining: {3}s'.format(
#                             this_step, num_steps, evaluation_times[-1], mean(evaluation_times) * (num_steps - this_step)))
#                     if return_intermediate:
#                         intermediate_states += [r.y]
#                         intermediate_times += [r.t]
#             else:
#                 r.integrate(t_final)
#                 if verbose:
#                     print('')
#                     print(argmax(initial_state))
#                     print(initial_time, t_final)
#                     print(datetime.datetime.today())
#                     print("%g" % (r.t))
#             if return_intermediate:
#                 return intermediate_states, intermediate_times
#             else:
#                 return r.y

#         states, times = propagate(
#             len(self.initial_vector),
#             self.interaction_function,
#             self.duration,
#             initial_state = self.initial_vector,
#             return_intermediate = True,
#             num_steps = num_steps,
#             )
#         assert(len(times) == num_steps + 1)
#         return times, states

        

# original below
# class ZVODE(OdeSolver):
#     """Python's zvode routine."""

# def propagate(n_states, hamiltonian, t_final, initial_state = None, initial_time = 0., display_progress = False,
#                   return_intermediate=False, verbose=False, atol=1e-16, rtol=1e-14, nsteps=1e6):

#     if initial_state is None:
#         initial_state = _np.zeros(n_states)
#         initial_state[0] = 1.
#     if return_intermediate:
#         intermediate_states = []
#         intermediate_times = []

#     # Define the Schrodinger equation and the Jacobian
#     def schrodinger(t, y):
#         return  -1.0j * hamiltonian(t).dot(y)
#     def jacobian(t, y):
#         tempham = hamiltonian(t)
#         if sparse.issparse(tempham):
#             return -1.0j * tempham.todense()
#         else:
#             return -1.0j * tempham
#     # Instantiate the integrator
#     r = ode(schrodinger, jacobian)
#     r.set_integrator('zvode', method='adams', with_jacobian=True, atol=atol, rtol=rtol, nsteps=nsteps) # use method='bdf' for stiff ode
#     r.set_initial_value(initial_state, initial_time)
#     if display_progress or return_intermediate:
#         # Do the integral in peices and display progress
#         n_steps = 1000
#         dt = 1.*t_final/n_steps
#         evaluation_times = []
#         previous_time = time()
#         while r.successful() and r.t < t_final:
#             r.integrate(r.t+dt)
#             # print ''
#             # print datetime.datetime.today()
#             # print "%g" % (r.t)
#             current_time = time()
#             evaluation_times += [current_time-previous_time]
#             previous_time = current_time
#             this_step = len(evaluation_times)
#             if display_progress:
#                 print('Finished step {0} of {1} in time {2}s.  Expected time remaining: {3}s'.format(
#                     this_step, n_steps, evaluation_times[-1], mean(evaluation_times) * (n_steps - this_step)))
#             if return_intermediate:
#                 intermediate_states += [r.y]
#                 intermediate_times += [r.t]
#     else:
#         r.integrate(t_final)
#         if verbose:
#             print('')
#             print(argmax(initial_state))
#             print(initial_time, t_final)
#             print(datetime.datetime.today())
#             print("%g" % (r.t))

#     # r.integrate(t_final)
#     if return_intermediate:
#         return (array(intermediate_states),array(intermediate_times))
#     else:
#         return r.y

def odeintz(func, z0, t, **kwargs):
    """An odeint-like function for complex valued differential equations."""

    # Disallow Jacobian-related arguments.
    _unsupported_odeint_args = ['Dfun', 'col_deriv', 'ml', 'mu']
    bad_args = [arg for arg in kwargs if arg in _unsupported_odeint_args]
    if len(bad_args) > 0:
        raise ValueError("The odeint argument %r is not supported by "
                         "odeintz." % (bad_args[0],))

    # Make sure z0 is a numpy array of type np.complex128.
    z0 = np.array(z0, dtype=np.complex128, ndmin=1)

    def realfunc(x, t, *args):
        z = x.view(np.complex128)
        dzdt = func(z, t, *args)
        # func might return a python list, so convert its return
        # value to an array with type np.complex128, and then return
        # a np.float64 view of that array.
        return np.asarray(dzdt, dtype=np.complex128).view(np.float64)

    result = odeint(realfunc, z0.view(np.float64), t, **kwargs)

    if kwargs.get('full_output', False):
        z = result[0].view(np.complex128)
        infodict = result[1]
        return z, infodict
    else:
        z = result.view(np.complex128)
        return z

def slow_trapz_for_matrix(ys: Vector, xs: Vector, *args, **kwargs): 
    """Apply scipy.integrate.trapz to a matrix of integrands."""
    num_rows, num_columns = ys[0].shape
    integral = np.zeros((num_rows, num_columns), dtype='complex')
    for row in range(num_rows):
        for column in range(num_columns):
            integrands = np.array([y[row, column] for y in ys])
            integral[row, column] = trapz(integrands, xs, *args, **kwargs)
    return integral

def trapz_for_matrix(ys: Vector, xs: Vector, *args, **kwargs): 
    """Apply scipy.integrate.trapz to a matrix of integrands."""
    num_rows, num_columns = ys[0].shape
    prods = list(it.product(range(num_rows), range(num_columns)))
    index_map = {k: (row, column) for k, (row, column) in enumerate(prods)}
    integrands_list = [np.array([y[row, column] for y in ys]) for row, column in prods]
    function = lambda integs: trapz(integs, xs, *args, **kwargs)
    results = [function(integs) for integs in integrands_list]
    integral = np.zeros((num_rows, num_columns), dtype='complex')
    for k, result in enumerate(results):
        row, column = index_map[k]
        integral[row, column] = result
    return integral


### Finite-difference derivatives ###
# Stencils as (offsets in units of the step h, weights). First-derivative stencils are second-order accurate; the
# one-sided versions are used next to a parameter bound or where the function is undefined (non-finite) on one side.
_FIRST_DERIVATIVE_STENCILS = {
    'central':  ((-1, 1), (-0.5, 0.5)),
    'forward':  ((0, 1, 2), (-1.5, 2., -0.5)),
    'backward': ((0, -1, -2), (1.5, -2., 0.5)),
}
_SECOND_DERIVATIVE_STENCILS = {
    'central':  ((-1, 0, 1), (1., -2., 1.)),
    'forward':  ((0, 1, 2, 3), (2., -5., 4., -1.)),
    'backward': ((0, -1, -2, -3), (2., -5., 4., -1.)),
}


def finite_difference_derivatives(function: Callable, x0: Vector, order: int = 1, evaluator_tolerance: float | None = None,
        relative_step: float | None = None, relative_step_second_order: float | None = None,
        bounds: list[tuple[float | None, float | None]] | None = None):
    """ Derivatives of a scalar- or vector-valued function of a real parameter vector by finite differences.

        - function: callable f(x) returning a scalar or an array (e.g. a vector of outcome probabilities).
        - x0: point (1D array of real parameters) where derivatives are evaluated.
        - order: 1 for the Jacobian only, 2 for the Jacobian and Hessian.
        - evaluator_tolerance: relative accuracy of f itself. Defaults to machine precision (appropriate for matrix
            exponentials); set to e.g. the ODE solver's rtol for solver-based models so that steps are sized accordingly.
        - relative_step / relative_step_second_order: override the steps used for first / second derivatives. Steps are
            h_i = relative_step * max(1, |x_i|). Defaults are tolerance^(1/3) and tolerance^(1/4), the error-optimal
            choices for central differences.
        - bounds: optional list of (lower, upper) per parameter; one-sided stencils are used where a central stencil would
            step outside the bounds. One-sided stencils are also used automatically where f is non-finite on one side.

        Returns (f0, jacobian, hessian) where jacobian[i] = df/dx_i and hessian[i, j] = d2f/dx_i dx_j, each with the
        shape of f's output; hessian is None for order = 1.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2; received {order}.")
    x0 = np.array(x0, dtype=float).reshape(-1)
    n = x0.size
    tolerance = np.finfo(float).eps if evaluator_tolerance is None else float(evaluator_tolerance)
    if tolerance <= 0.:
        raise ValueError(f"evaluator_tolerance must be positive; received {evaluator_tolerance}.")
    rel_1 = tolerance**(1./3.) if relative_step is None else float(relative_step)
    rel_2 = tolerance**(1./4.) if relative_step_second_order is None else float(relative_step_second_order)
    if bounds is None:
        bounds = [(None, None)] * n
    if len(bounds) != n:
        raise ValueError(f"bounds must have one (lower, upper) pair per parameter; expected {n}, received {len(bounds)}.")

    def _steps(rel):
        h = rel * np.maximum(1., np.abs(x0))
        return (x0 + h) - x0   # make the step exactly representable

    h_1 = _steps(rel_1)
    h_2 = _steps(rel_2) if order == 2 else None

    # Memoize function values by displacement, {(parameter index, step multiple, step size), ...} -> f
    values = {}
    def f_at(displacement: dict[int, float]):
        key = tuple(sorted((i, float(d)) for i, d in displacement.items() if d != 0.))
        if key not in values:
            x = x0.copy()
            for i, d in key:
                x[i] += d
            values[key] = np.asarray(function(x))
        return values[key]

    f0 = f_at({})
    if not np.all(np.isfinite(f0)):
        raise ValueError(f"The function is not finite at the evaluation point {x0}.")

    def _choose_stencil(i: int, h: float, reach: int) -> str:
        """ Pick central/forward/backward for parameter i, given how many steps a one-sided stencil extends. """
        lower, upper = bounds[i]
        minus_allowed = lower is None or x0[i] - h >= lower
        plus_allowed = upper is None or x0[i] + h <= upper
        if minus_allowed and plus_allowed:
            # Probe both sides; a side where f is undefined (e.g. sqrt of a negative rate) is expected and handled by falling back
            # to a one-sided stencil, so numpy's invalid-value / divide warnings are silenced for these two evaluations only
            with np.errstate(invalid='ignore', divide='ignore'):
                minus_finite = np.all(np.isfinite(f_at({i: -h})))
                plus_finite = np.all(np.isfinite(f_at({i: h})))
            if minus_finite and plus_finite:
                return 'central'
            minus_allowed, plus_allowed = minus_finite, plus_finite
        if plus_allowed and (upper is None or x0[i] + reach*h <= upper):
            return 'forward'
        if minus_allowed and (lower is None or x0[i] - reach*h >= lower):
            return 'backward'
        raise ValueError(f"Cannot take finite-difference steps of size {h:.3g} for parameter {i} at {x0[i]}: the function is "
                         f"non-finite or the bounds {bounds[i]} are too tight on both sides. Try a smaller relative step.")

    # First derivatives
    stencils_1 = [_choose_stencil(i, h_1[i], 2) for i in range(n)]
    jacobian = np.zeros((n,) + f0.shape, dtype=f0.dtype)
    for i in range(n):
        offsets, weights = _FIRST_DERIVATIVE_STENCILS[stencils_1[i]]
        jacobian[i] = sum(w * f_at({i: a*h_1[i]}) for a, w in zip(offsets, weights)) / h_1[i]

    if order == 1:
        return f0, jacobian, None

    # Second derivatives: diagonal terms from second-difference stencils, mixed terms from products of first-derivative stencils
    hessian = np.zeros((n, n) + f0.shape, dtype=f0.dtype)
    stencils_2 = [_choose_stencil(i, h_2[i], 3) for i in range(n)]
    for i in range(n):
        offsets, weights = _SECOND_DERIVATIVE_STENCILS[stencils_2[i]]
        hessian[i, i] = sum(w * f_at({i: a*h_2[i]}) for a, w in zip(offsets, weights)) / h_2[i]**2
    for i in range(n):
        offsets_i, weights_i = _FIRST_DERIVATIVE_STENCILS[stencils_2[i]]
        for j in range(i + 1, n):
            offsets_j, weights_j = _FIRST_DERIVATIVE_STENCILS[stencils_2[j]]
            mixed = sum(w_a * w_b * f_at({i: a*h_2[i], j: b*h_2[j]})
                        for a, w_a in zip(offsets_i, weights_i) for b, w_b in zip(offsets_j, weights_j))
            hessian[i, j] = hessian[j, i] = mixed / (h_2[i] * h_2[j])

    return f0, jacobian, hessian
