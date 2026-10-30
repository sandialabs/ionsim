#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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

def solve_ode(right_hand_side: Callable, initial_vector: Vector, duration: float, time_evals: Vector | None = None,
              ode_solver: str = 'odeintz', jacobian: Callable | None = None, **solver_options):
    """ Solve dy/dt = right_hand_side(t, y) for a flat complex vector y from t = 0 to t = duration. """
    solvers = {'odeintz': OdeIntz, 'solve_ivp': SolveIvp, 'zvode': ZVODE}
    if ode_solver not in solvers:
        raise IonSimError(f'ODE solver {ode_solver} is not implemented. Available solvers: {sorted(solvers)}.')
    return solvers[ode_solver](right_hand_side, initial_vector, duration, time_evals, jacobian, dict(solver_options)).solve()


def solve_time_evolution_equation(interaction_function: Callable, initial_state_vector: Vector, duration: float,
    time_evals: Vector | None = None, ode_solver: str = 'odeintz', **solver_options):
    """ Solve dy/dt = -i A(t) y, the time-dependent Schrodinger equation (A = H) or the vectorized Lindblad master equation
        (A = i L), with A(t) = interaction_function(t). See solve_ode for the solvers and solver_options. """
    def right_hand_side(t, y):
        return interaction_function(t).dot(-1j * y)

    def jacobian(t, y):
        matrix = interaction_function(t)
        return -1j * (matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix))

    return solve_ode(right_hand_side, initial_state_vector, duration, time_evals, ode_solver, jacobian=jacobian, **solver_options)


@dataclass(frozen=True, eq=False)
class OdeSolver(ABC):
    """A numerical routine to solve an ordinary differential equation dy/dt = right_hand_side(t, y)."""
    right_hand_side: Callable
    initial_vector: Vector
    duration: float
    time_evals: Vector | None
    jacobian: Callable | None = None
    solver_options: dict = field(default_factory=dict)

    @property
    def y0(self) -> np.ndarray:
        if self.initial_vector is None:
            raise IonSimError("An initial vector is required to solve the ODE.")
        return np.array(self.initial_vector, dtype='complex').ravel()

    @abstractmethod
    def solve(self):
        """Solves the ODE."""


@dataclass(frozen=True, eq=False)
class OdeIntz(OdeSolver):
    """ A complex-valued version of scipy's odeint routine. odeint has no complex Jacobian support here, so the
        jacobian is not used. Note: odeint starts the integration at time_evals[0]. """
    def solve(self):
        """Solves the ODE."""
        options = dict(self.solver_options)
        user_full_output = options.pop('full_output', False)
        times = np.linspace(0, self.duration, 4) if self.time_evals is None else np.asarray(self.time_evals)

        def right_hand_side_flip_args(y, t):
            return self.right_hand_side(t, y)

        result, info = odeintz(right_hand_side_flip_args, self.y0, times, full_output=True, **options)
        if info.get('message') != 'Integration successful.':
            raise IonSimError(f"odeint integration failed: {info.get('message')} Consider passing solver options, e.g. a "
                              f"larger mxstep (default 500 steps per output interval), more time_evals, or another ode_solver.")
        if user_full_output:
            return list(times), [y for y in result], info
        return list(times), [y for y in result]


@dataclass(frozen=True, eq=False)
class SolveIvp(OdeSolver):
    """ scipy's solve_ivp routine. Default method RK45 with rtol = 1e-3 and atol = 1e-6 (scipy defaults); pass e.g.
        method='DOP853', rtol=1e-10 for accurate results. """
    IMPLICIT_METHODS = ('Radau', 'BDF', 'LSODA')

    def solve(self):
        """Solves the ODE."""
        options = dict(self.solver_options)
        if self.jacobian is not None and options.get('method') in self.IMPLICIT_METHODS and 'jac' not in options:
            options['jac'] = self.jacobian
        result = solve_ivp(self.right_hand_side, (0, self.duration), self.y0, t_eval=self.time_evals, **options)
        if not result.success:
            raise IonSimError(f"solve_ivp integration failed: {result.message}")
        return list(result['t']), [result['y'][:, i] for i in range(len(result['t']))]


@dataclass(frozen=True, eq=False)
class ZVODE(OdeSolver):
    """ scipy's zvode routine (scipy.integrate.ode). The jacobian, when given, is used (with_jacobian=True). """
    DEFAULT_OPTIONS = {'method': 'adams', 'atol': 1e-16, 'rtol': 1e-14, 'nsteps': 1e5}  # method='bdf' for stiff ODEs

    def solve(self):
        """Solves the ODE."""
        if self.time_evals is None:
            time_evals = np.linspace(0., self.duration, 3)
        else:
            time_evals = np.asarray(self.time_evals)
            if time_evals[-1] != self.duration:
                raise IonSimError(f"The last of time_evals ({time_evals[-1]}) must equal the duration ({self.duration}) for zvode.")
        options = {**self.DEFAULT_OPTIONS, **self.solver_options}

        integrator = ode(self.right_hand_side, self.jacobian)
        integrator.set_integrator('zvode', with_jacobian=self.jacobian is not None, **options)
        integrator.set_initial_value(self.y0, time_evals[0])
        intermediate_states = [self.y0]
        for t in time_evals[1:]:
            integrator.integrate(t)
            if not integrator.successful():
                raise IonSimError(f"zvode integration failed at t = {t} (return code {integrator.get_return_code()}).")
            intermediate_states.append(integrator.y.copy())
        return list(time_evals), intermediate_states

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
