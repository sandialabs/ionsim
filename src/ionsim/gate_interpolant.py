import numpy as np
from typing import Callable
from dataclasses import dataclass
from csaps import NdGridCubicSmoothingSpline
from itertools import product 
import inspect 
from functools import cached_property

from ionsim.custom_types import Vector, Matrix
from ionsim.noise import Noise
from ionsim.basis import Basis, StandardBasis
from ionsim.ionsim_error import IonSimError
from ionsim.hamiltonian import Hamiltonian
import ionsim.io as io 
from ionsim.process import Gate 

@dataclass(frozen=True, eq=False) 
class GateInterpolant(): 
    """ A class for building a grid of gates to do gate interpolation and compute interpolated gates. 

        - maintains a sequence of gates built at corresponding parameter grid values. 
        - assumes the gates are the same size and type 
        - uses the sequence of known gates to compute interpolated gates via cubic splines  
    """

    grid_axes: dict[str, list[Vector]] 
    gate_name: str | None
    grid: list[tuple] #| NDArray 
    basis: StandardBasis | None # gives meaning to process matrix element ordering  
    computed_gates: list[Gate] # Representing non-interpolated gates, computed on the parameter grid  

    @property
    def grids(self): 
        return list(self.grid_axes.values()) 

    @property
    def grid_lengths(self): 
        return [len(grid) for grid in self.grids]

    @property
    def parameter_list(self):
        return list(self.grid_axes.keys()) 

    @staticmethod
    def build_grid(grid_axes: dict[str, list[Vector]]):
        """ Returns the N-dimensional parameter grid as a list of coordinates in the grid. """
        return list(product(*list(grid_axes.values()))) 

    @property
    def gate_size(self):
        """ A gate is represented by an N^2 x N^2 process matrix. This function returns N^2 """
        if self.computed_gates:
            shape = self.computed_gates[0].process_matrix.shape
            assert len(shape) == 2
            assert shape[0] == shape[1]  # should they always be square? 
            return shape[0] 
        raise IonSimError("Error: There are no gates in this interpolant class instance. Size of gate is unknown.")

    @property
    def computed_gates_process_matrices(self):
        """ Return the process matrix for each gate on the grid in a list """
        return [gate.process_matrix for gate in self.computed_gates]
         

    @property
    def computed_gate_data_as_array(self):
        """ Returns the gate d^2 x d^2 process matrices on the grid as a big numpy array 
            of shape (d^2 , d^2 , len(x), len(y), ... len(z) ) where x, y, z denote interpolation grid parameters. """
        gate_data = np.empty((self.gate_size, self.gate_size, *self.grid_lengths), dtype='complex')
        for i in range(self.gate_size):
            for j in range(self.gate_size):
                gate_data[i,j] = np.array([gd[i,j] for gd in self.computed_gates_process_matrices]).reshape(*self.grid_lengths)
        return gate_data


    # TODO: add from_lindbladian_function class method? 
    @classmethod
    def from_gate_function(cls, gate_function: Callable, grid_axes: dict[str, NDArray], gate_name: str | None=None):
        """ Build gate interpolant from a gate function, which returns a gate from grid parameter values. """ 
        # NB: This is the cleanest way to handle noise. Noise is embedded in the gate function input. 
        grid = cls.build_grid(grid_axes) # a list of grid points 
        gates_on_grid = []

        # Retrieve basis of the gate 
        sample_values = grid[0]
        gate_basis = gate_function(*sample_values).basis     

        for values in grid:
            gates_on_grid.append(gate_function(*values))

        return cls(grid_axes, gate_name, grid, gate_basis, gates_on_grid)


    @classmethod
    def from_process_matrix_function(cls, process_matrix_function: Callable, grid_axes: dict[str, NDArray], gate_basis: StandardBasis, gate_name: str | None=None): 
        """ Build gate interpolant from a process matrix function. """ 
        # Build a grid and loop over every parameter value and build the gate from the process matrix function  
        grid = cls.build_grid(grid_axes)

        gates_on_grid = []
        for values in grid:
            coordinate = dict(zip(grid_axes.keys(), values))
            gates_on_grid.append(Gate.from_process_matrix_function(gate_basis, process_matrix_function, coordinate)) 

        return cls(grid_axes, gate_name, grid, gate_basis, gates_on_grid)


    @classmethod
    def from_hamiltonian_function(cls, basis: StandardBasis, hamiltonian_function: Callable, gate_duration: float, grid_axes: dict[str, NDArray], gate_name: str | None=None):
        """ Build gate interpolant from Schrodinger evolution of a Hamiltonian. 
            - requires a function that returns the hamiltonian using the interpolant grid parameters  
            - requires a fixed duration to set the hamiltonian's time evolution 
            - requires a dictionary of parameters to specify the grid
        """ 
        # TODO: Test/verify this function. 
        # Build a grid and loop over every parameter value and build the gate from the hamiltonian
        grid = cls.build_grid(grid_axes)

        # For each parameter combination in the parameter grid, compute the gate and process matrix.
        gates_on_grid = []
        for values in grid:
            coordinate = dict(zip(grid_axes.keys(), values)) 
            # TODO: accomodate args for from_hamiltonian() like DOF to trace out, etc. 
            gates_on_grid.append(Gate.from_hamiltonian_function(basis, hamiltonian_function, gate_duration, coordinate)) 

        return cls(grid_axes, gate_name, grid, basis, gates_on_grid)
    

    def compute_functional_of_gates(self, gate_property_functional: Callable) -> list:
        """ Computes a functional of the gate (e.g. gate residual) at every gate in the grid, corresponding to each element of the returned list. """ 
        functional_output = []
        for gate in self.computed_gates: 
            functional_output.append(gate_property_functional(gate))
        return functional_output

    def write_to_file(self, filename: str, attributes: dict=None):
        """ Function to write Gate Interpolant class data to an hd5f file """
        # TODO: Figure out how to read/write basis information 
        results_dict = {}
        results_dict.update(self.grid_axes) 
        if self.gate_name:
            results_dict[self.gate_name + '_gate_data'] = self.computed_gate_data_as_array
        else:
            results_dict['gate_data'] = self.computed_gate_data_as_array
        io.write_results_to_file(filename, results_dict, attributes)
        return 0

    @classmethod
    def read_from_file(cls, filename: str): 
        # TODO: In progress, needs to resolve basis read/write issue: 
        # TODO: Is it possible to read/write Basis objects? This functionality is not yet in IonSim. his class defaults to building without a basis  
        """ Function to read Gate Interpolant class data from an hd5f file and instance the class """
        results, attr_from_file = io.read_results_from_file(filename)

        _grid_axes = {}
        # Parse grid axes from reading 1D arrays; parse gate data from NDArray
        for key, value in results:
            if not isinstance(value, Vector) or not isinstance(value, Matrix):
                raise ValueError("Data from file should contain a set of 1D arrays for the grid axes and one NDArray for the gate data.") 
            if isinstance(value, Vector) and len(value.shape) == 1:
                _grid_axes[key] = results[key]

        gate_attribute = [x for x in results.keys() if x not in _grid_axes.keys()]
        if not gate_attribute:
            raise IonSimError("No gate NDArray data found in file.")
        elif len(gate_attribute) > 1:
            raise IonSimError("File should contain 1 gate data of shape (d^2, d^2, *grid_lengths).")
            
        try:
            gate_name = attr_from_file[results.keys()[0]]['gate_name']
        except:
            gate_name = None

        # Build grid and extract corresponding gates on the grid  
        grid = cls.build_grid(_grid_axes) 
        gates_on_grid = []
        gate_data = results[gate_attribute[0]]
        for values in grid:
            # Find where the parameter values in the grid 
            #parameter_coord_indices = grid.index(values)
            parameter_coord_indices = [] 
            #for axis_index, val in enumerate(values):
            for axis, val in zip(_grid_axes.keys(), values):
                parameter_coord_indices.append( np.where(_grid_axes[axis] == val )[0][0] )
            assert len(parameter_coord_indices) == len(_grid_axes.keys()) 
            gates_on_grid.append( gate_data[:, :, parameter_coord_indices] ) 

        results = self.grid_axes 
        return cls(_grid_axes, gate_name, grid, None, gates_on_grid )

    def construct_spline_for_gate_derived_matrix_property(self, gate_derived_property: AnyMatrix, complex_data: bool=True):
        """ Constructs interpolant spline for derived property that lives on the domain of the parameter grid.

            - assumes gate_derived_property is matrix input of the shape: (d, d, parameter1_grid, parameter2_grid, ... )
                where "d" is the dimension (process matrix is d^2 x d^2), 
                and parameter1_grid, parameter2_grid represent the one-dimensional parameter grids from the class's grid axes. 

            - This serves as a general (template) method for a set of gate process matrices or gate-derived properties.  

        """ 
        # Gate derived property input could be matrix-valued, e.g. Residuals[x,y] -> d^2 x d^2 Process Matrix of residuals  
        spline_reals = {}
        if complex_data:
            spline_imags = {}

        # Case where the gate-derived proeprty is matrix-valued, e.g. the property at some (x,y) grid point yields a matrix w. dimensionality of a process matrix
        size = self.gate_size 
        for i in range(size):
            for j in range(size):
                spline_reals[i,j] = NdGridCubicSmoothingSpline(self.grids, gate_derived_property[i,j].real, smooth=1) 
                if complex_data:
                    spline_imags[i,j] = NdGridCubicSmoothingSpline(self.grids, gate_derived_property[i,j].imag, smooth=1) 

        if complex_data:
            return spline_reals, spline_imags
        return spline_reals

    def construct_spline_for_gate(self, complex_data: bool=True):
        """ Constructs interpolant spline for the gate process matrices that live on the domain of the parameter grid. """ 
        return self.construct_spline_for_gate_derived_matrix_property(self.computed_gate_data_as_array, complex_data)

    def construct_spline_for_gate_derived_scalar_property(self, gate_derived_property: AnyMatrix, complex_data: bool=True):
        """ Constructs interpolant spline for derived property that lives on the domain of the parameter grid.""" 
        # Gate derived property input is a grid-dependent scalar property, e.g. Fidelity[x, y] -> Number  
        spline_reals = NdGridCubicSmoothingSpline(self.grids, gate_derived_property.real, smooth=1) 
        if complex_data:
            spline_imags = NdGridCubicSmoothingSpline(self.grids, gate_derived_property.imag, smooth=1) 
            return spline_reals, spline_imags
        return spline_reals


    #### Interpolation methods #### 
    @cached_property # TODO: decide whether this should be cached or just a property 
    def process_matrix_interpolant_function(self): 
        """ Returns a gate's process matrix interpolating function of the grid parameters, e.g. G(x,y) for x,y grid parameters""" 
        # Extract gate spline information, then build interpolant function from the splines 
        gate_spline_reals, gate_spline_imags = self.construct_spline_for_gate(complex_data=True)
        return self.interpolant_function_from_splines([gate_spline_reals, gate_spline_imags], 'process matrix')
        
    def interpolant_function_from_splines(self, property_interpolant: dict | list[dict, dict], property_name: str) -> Callable:
        """ Returns a property interpolating function of the grid parameters, e.g. F(x,y) for x,y grid parameters""" 
        args_str = ", ".join(self.parameter_list)
        N_parameters = len(self.parameter_list)

        # Build and return a general, dynamic interpolating function 
        def _interpolating_function(*args, **kwargs):
            # Interpolating function for the property specified in the interpolant.  
            if args and kwargs:
                raise ValueError("Use positional or keyword arguments, not both.")

            if args:
                if len(args) == 1 and hasattr(args[0], '__len__'):
                    # If one argument is passed (representing a list, tuple, array of arguments), convert to list
                    values = list(args[0])
                else:
                    # If several arguments are passed, this should correspond to the parameter values. 
                    values = list(args)

                if len(values) != N_parameters:
                    raise ValueError(f"Not enough parameters specified for interpolant. Expected {N_parameters}, received {len(values)}.")

                grid_coordinate = dict(zip(self.parameter_list, values))
            else:
                missing_parameters = set(self.parameter_list).difference(set(kwargs.keys()))
                extra_parameters = set(kwargs.keys()).difference(set(self.parameter_list)) 
                if missing_parameters:
                    raise ValueError(f"Function keyword arguments are missing the following parameters: {missing_parameters}")
                if extra_parameters:
                    raise ValueError(f"Additional parameters specified that are not part of the interpolation parameter list: {extra_parameters}")

                grid_coordinate = kwargs               
 
            # Convert grid coordinate dictionary to tuple 
            grid_coordinate_values = tuple(grid_coordinate[parameter_name] for parameter_name in self.parameter_list) # ensures sorted order of grid coordinate values

            # For gate interpolation, the property interpolant is of shape (d, d, *parameter_grid)
            size = self.gate_size 
            complex_data = False 
            if isinstance(property_interpolant, list) or isinstance(property_interpolant, tuple): 
                #property_interpolant = list(property_interpolant)
                complex_data = True
                function_output = np.empty((size, size), dtype=complex)
            else:
                function_output = np.empty((size, size), dtype=float)
                if not isinstance(property_interpolant, dict):
                    raise ValueError("Pass in a dictionary containing the real-valued interpolant. For complex data, pass in a tuple or list of form [dict, dict].")

            # Fill process matrix at the requested parameter coordinate values. 
            # TODO: Can we vectorize this for a list of (dx, dy) coordiantes? 
            for i in range(size):
                for j in range(size):
                    if complex_data:
                        function_output[i,j] = property_interpolant[0][i,j](grid_coordinate_values).item() + 1j*property_interpolant[1][i,j](grid_coordinate_values).item()
                    else:
                        function_output[i,j] = property_interpolant[i,j](grid_coordinate_values).item() 

            return function_output 

        # Set the signature of the function for readibility  
        sig = inspect.Signature(parameters=[inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for name in self.parameter_list] )

        # Build a function whose signature is specified by the user in the class instance (at runtime) 
        _interpolating_function.__signature__ = sig
        _interpolating_function.__name__ = property_name + "_interpolant"

        return _interpolating_function


    def interpolated_gate_from_process_matrix_interpolating_function(self, process_matrix_interpolating_function: Callable, parameter_coordinate: tuple | dict[str, float]):  
        """ Returns the gate evaluated at a grid point off of the grid domain.""" 
        if isinstance(parameter_coordinate, tuple):
            grid_values = parameter_coordinate
        elif isinstance(parameter_coordinate, dict):
            grid_values = tuple(parameter_coordiante.values())
        else:
            raise ValueError("Input either a dictionary of grid parameter values or a tuple of the values for interpolation.")

        return Gate(self.basis, process_matrix = process_matrix_interpolating_function(*grid_values)) 

    #@property
    @cached_property # TODO: decide whether this should be cached or just a property 
    def interpolated_gate_function(self) -> Callable:
        """ Returns a function that returns a Gate object evaluated at grid parameter values. """
        # Build and return a general interpolating function for a Gate object  
        # Retrieve process matrix interpolating function --> should only compute this once. 
        process_matrix_interpolating_function = self.process_matrix_interpolant_function
        args_str = ", ".join(self.parameter_list)
        N_parameters = len(self.parameter_list)

        def _gate_interpolating_function(*args, **kwargs) -> Gate:
            # Interpolating function for the gate.  
            if args and kwargs:
                raise ValueError("Use positional or keyword arguments, not both.")

            # Extract grid coordinate values from args/kwargs  
            if args:
                if len(args) == 1 and hasattr(args[0], '__len__'):
                    # If one argument is passed (representing a list, tuple, array of arguments), convert to list
                    values = list(args[0])
                else:
                    # If several arguments are passed, this should correspond to the parameter values. 
                    values = list(args)

                if len(values) != N_parameters:
                    raise ValueError(f"Not enough parameters specified for interpolant. Expected {N_parameters}, received {len(values)}.")

                grid_coordinate = dict(zip(self.parameter_list, values))
            else:
                missing_parameters = set(self.parameter_list).difference(set(kwargs.keys()))
                extra_parameters = set(kwargs.keys()).difference(set(self.parameter_list)) 
                if missing_parameters:
                    raise ValueError(f"Function keyword arguments are missing the followign parameters: {missing_parameters}")
                if extra_parameters:
                    raise ValueError(f"Additional parameters specified that are not part of the interpolation parameter list: {extra_parameters}")

                grid_coordinate = kwargs               
 
            grid_coordinate_values = tuple(grid_coordinate[parameter_name] for parameter_name in self.parameter_list) # ensures sorted order of grid coordinate values

            gate_output = self.interpolated_gate_from_process_matrix_interpolating_function(process_matrix_interpolating_function, grid_coordinate_values)
            return gate_output 

        return _gate_interpolating_function
