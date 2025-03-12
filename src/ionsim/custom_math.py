from ionsim.custom_types import Vector
from ionsim.ionsim_error import IonSimError

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Any, Callable
from nptyping import NDArray, Shape
from scipy.integrate import trapezoid as trapz
import itertools as it
from concurrent.futures import ProcessPoolExecutor
from scipy.integrate import odeint, solve_ivp, ode
from scipy import sparse

from icecream import ic

def solve_time_evolution_equation(interaction_function: Callable, initial_state_vector: Vector, duration: float,
    time_evals: Vector | None = None, ode_solver: str = 'odeintz', **kwargs):
    """Solve the time-dependent Schrodinger equation or the vectorized Lindblad master equation."""
    print(f'Solving ODE with {ode_solver}.')
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
    nsteps: float = 1e6

    def solve(self):
        """Solves the ODE."""
        if self.time_evals is None:
            num_steps = 3
        else:
            num_steps = len(time_evals)
            assert(time_evals[-1] == duration)

        ic(self.nsteps)

        n_states = len(self.initial_vector)
        hamiltonian = self.interaction_function
        t_final = self.duration
        initial_state = self.initial_vector

        if initial_state is None:
            initial_state = _np.zeros(n_states)
            initial_state[0] = 1.

        intermediate_states = [initial_state]
        intermediate_times = [0]
        def schrodinger(t, y):
            return  -1.0j * hamiltonian(t).dot(y)
        def jacobian(t, y):
            tempham = hamiltonian(t)
            if sparse.issparse(tempham):
                return -1.0j * tempham.todense()
            else:
                return -1.0j * tempham
        r = ode(schrodinger, jacobian)
        r.set_integrator('zvode', method='adams', with_jacobian=True, atol=1e-16, rtol=1e-14, nsteps=self.nsteps) # use method='bdf' for stiff ode
        r.set_initial_value(initial_state, 0)
        dt = t_final/float(num_steps)
        while r.successful() and r.t < t_final:
            r.integrate(r.t + dt)
            intermediate_states += [r.y]
            intermediate_times += [r.t]
        return intermediate_times, intermediate_states

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

def main():
    """Script to execute."""
    xs = np.linspace(-10, 10, 101)
    def f(x):
        return 1/np.sqrt(2*np.pi)*np.exp(-x**2/2)
    ys = [np.array([[0, f(x)], [2*f(x), (3j+1)*f(x)]]) for x in xs]

    import time
    start = time.perf_counter()
    result = slow_trapz_for_matrix(ys, xs)
    ic(result.round(14) == np.array([[0, 1], [2, (3j+1)]]))
    end = time.perf_counter()
    ic(end-start)

    start = time.perf_counter()
    result = trapz_for_matrix(ys, xs)
    ic(result.round(14) == np.array([[0, 1], [2, (3j+1)]]))
    end = time.perf_counter()
    ic(end-start)


if __name__ == '__main__':
    main()