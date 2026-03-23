import numpy as np
#from dataclasses import dataclass, field
from typing import Callable
from dataclasses import dataclass
from csaps import NdGridCubicSmoothingSpline
from itertools import product 
import inspect 

from ionsim.custom_math import trapz_for_matrix
from ionsim.custom_types import Vector, Matrix
from ionsim.noise import Noise
from ionsim.basis import DegreeOfFreedom, Basis, StandardBasis
from ionsim.ionsim_error import IonSimError
from ionsim.hamiltonian import Hamiltonian
from ionsim.state import State
from ionsim.process import Gate 


# Class to set up a grid of gates and then interpolate using the grid. 

#class GateInterpolant(Gate): # May not need to inherit from Gate 
@dataclass(frozen=True, eq=False) 
class GateInterpolant(): 
    """ A class for building a grid of gates to do gate interpolation and build interpolated gates. 

        - maintains a sequence of gates built at corresponding parameter grid values. 
        - uses the sequence of known gates to build an interpolate gates 

    """

    #parameter_grids: list[Vector]
    grid_axes: dict[str, list[Vector]] 
    grid: list[tuple] #| NDArray 
    simulated_gates: list[Gate] # can we think of a better name? 

    noisy_parameter: str | None=None # extend to a list[str] or None if we support 2+ noisy parameters  

    # The way noise is handled assumes that possibly one of the parameters in the grid is a noise-type or noisy parameter  
    # Currently, there is a maximum of 1 noisy parameter. For more, IonSim needs an extension to multiple noisy parameter handling in Gate / Process module  
    # - this class tracks which parameter is noisy 


    # maybe this will just be a grid class? 
    #parameters # already included in Gate parent class  

    # Represent a grid point as an N-dimensional array for N parameters. 


    # TODO: build mapping between grid index and parameter human readable values,
        # e.g. grid[0, 2, 3, .. ,1] <==> what parameter values? 


 #    def __post_init__(self):
 #        # Checks that parameter grid variables correspond with gate parameters  
 #        # When should we perform this check?? We may not need to  
 #        gate_parameter_names = list(parameters.keys())
 #        grid_parameter_names = list(self.grid_axes.keys())
 #        for name in grid_parameter_names:
 #            if name not in gate_parameter_names:
 #                raise IonSimError("Invalid grid parameter name. Gate does not contain that parameter in its parameter set.")

    # Build mapping function for the grid index  
    # TODO: Option to convert grid output to numpy array. Should grid() be a list of coordiantes or a np matrix-type object?  
 #    @cachedproperty
 #    def grid(self): 
 #        """ Returns the N-dimensional parameter grid as a list of parameter value coordinates. """
 #        return list(product(*list(self.grid_axes.values()))) 

 #    def grid_size(self):
 #        return len(grid)

 #    @property
 #    def size_of_grid_for_parameter(self, parameter_name: str):
 #        return len(parameter_grids[parameter_name])

    @staticmethod
    def build_grid(grid_axes: dict[str, list[Vector]]):
        """ Returns the N-dimensional parameter grid as a list of coordinates in the grid. """
        return list(product(*list(grid_axes.values()))) 
        #return np.meshgrid(*list(1D_parameter_grids.values()), indexing = 'ij')


 #    def map_grid_coordinates_to_parameter_vector(grid_coordinates: list[int]):
 #        """ Returns the parameter values from a list of grid coordinates (e.g. [1, 0, 2, 4, ... , 1]) """
 #        
 #
 #    def return_grid_for_parameter(parameter_name: str):
 #        """ Returns the 1-dimensional grid for a parameter."""
 #        return parameter_grids[parameter_name]


    # usage maybe like GateInterpolant.build_interpolant.from_hamiltonian( , )
    # usage maybe like GateInterpolant.build_interpolant.from_lindbladian( , )
    # usage maybe like GateInterpolant.build_interpolant.from_process_matrix_function( , )
        
    #def from_hamiltonian(cls, hamiltonian: Hamiltonian, gate_duration: float, parameters: dict[str, NDArray]):



    @classmethod
    def from_gate_function(cls, gate_function: Callable, grid_axes: dict[str, NDArray]):
        """ Build gate interpolant from a gate function. """ 
        grid = cls.build_grid(grid_axes)

        gates_on_grid = []
        parameter_list = list(grid_axes.keys())
        for values in grid:
            #coordinate = dict(zip(grid_axes.keys(), values))
            gates_on_grid.append(gate_function(*values))
            #if noise_function is not None:
 #            if noise_functions:
 #                # Read the noise function for the noisy parameter 
 #                #sig = inspect.signature(noise_function)
 #                #noise_function_parameters = list(sig.parameters.keys())
 # #                noisy_parameters = []
 # #                for noisy_parameter in noise_function_parameters:
 # #                    if noisy_parameter in parameter_list:
 # #                        noisy_parameters.append(noisy_parameter) 
 #                noisy_parameters = list(noise_functions.keys())
 #                #print(noise_function_parameters)
 #                #print(parameter_list)
 #                if not noisy_parameters:
 #                    raise IonSimError("Noise info dictionary does not contain any parameters.")
 #                    #raise IonSimError("Noise function does not share any parameters with the gate interpolant.")
 #                elif len(noisy_parameters) > 1:
 #                    raise IonSimError("More than 1 noisy parameter is currently not supported in IonSim.")
 #
 #                # TODO: To extend to multiple noisy parameters, loop through noisy parameters 
 #                noisy_parameter = noisy_parameters[0]
 #                #noisy_parameter_index_in_grid = parameter_list.index(noisy_parameter)
 #
 #                noise_function = noise_functions[noisy_parameter]
 #                
 #                #noise = noise_function(coordinate[noisy_parameter_index_in_grid]) # returns an IonSim Noise object 
 #                print(coordinate)
 #                if coordinate[noisy_parameter] == 0.:
 #                    gates_on_grid.append(Gate.from_process_matrix_function(basis, process_matrix_function, coordinate)) 
 #                else:                    
 #                    noise = noise_function(coordinate[noisy_parameter]) # returns an IonSim Noise object 
 # #                    print(noise)
 # #                    print(noise.parameter_name)
 # #                    print(noise.domain_arguments)
 #                    process_matrix_function_for_noise = process_matrix_function
 #                    gates_on_grid.append(Gate.from_process_matrix_function(basis, process_matrix_function, coordinate, noise)) 
 #            else:
              #  gates_on_grid.append(Gate.from_process_matrix_function(basis, process_matrix_function, coordinate)) 

        return cls(grid_axes, grid, gates_on_grid)


    #def from_process_matrix_function(cls, process_matrix_function: Callable, grid_axes: dict[str, NDArray], basis: StandardBasis, noise_function: Callable | None): 
    @classmethod
    def from_process_matrix_function(cls, process_matrix_function: Callable, grid_axes: dict[str, NDArray], basis: StandardBasis, noise_functions: dict[str,Callable] | None): 
        """ Build gate interpolant from a process matrix function. """ 
        # Build a grid and loop over every parameter value and build the gate from the process matrix function  
        # TODO: Handle case where there is noise. A Gate is built for some fixed realization of the noise  
        grid = cls.build_grid(grid_axes)

        gates_on_grid = []
        parameter_list = list(grid_axes.keys())
        for values in grid:
            coordinate = dict(zip(grid_axes.keys(), values))
            #if noise_function is not None:
            if noise_functions:
                # Read the noise function for the noisy parameter 
                #sig = inspect.signature(noise_function)
                #noise_function_parameters = list(sig.parameters.keys())
 #                noisy_parameters = []
 #                for noisy_parameter in noise_function_parameters:
 #                    if noisy_parameter in parameter_list:
 #                        noisy_parameters.append(noisy_parameter) 
                noisy_parameters = list(noise_functions.keys())
                #print(noise_function_parameters)
                #print(parameter_list)
                if not noisy_parameters:
                    raise IonSimError("Noise info dictionary does not contain any parameters.")
                    #raise IonSimError("Noise function does not share any parameters with the gate interpolant.")
                elif len(noisy_parameters) > 1:
                    raise IonSimError("More than 1 noisy parameter is currently not supported in IonSim.")

                # TODO: To extend to multiple noisy parameters, loop through noisy parameters 
                noisy_parameter = noisy_parameters[0]
                #noisy_parameter_index_in_grid = parameter_list.index(noisy_parameter)

                noise_function = noise_functions[noisy_parameter]
                
                #noise = noise_function(coordinate[noisy_parameter_index_in_grid]) # returns an IonSim Noise object 
                print(coordinate)
                if coordinate[noisy_parameter] == 0.:
                    gates_on_grid.append(Gate.from_process_matrix_function(basis, process_matrix_function, coordinate)) 
                else:                    
                    noise = noise_function(coordinate[noisy_parameter]) # returns an IonSim Noise object 
 #                    print(noise)
 #                    print(noise.parameter_name)
 #                    print(noise.domain_arguments)
                    process_matrix_function_for_noise = process_matrix_function
                    gates_on_grid.append(Gate.from_process_matrix_function(basis, process_matrix_function, coordinate, noise)) 
            else:
                gates_on_grid.append(Gate.from_process_matrix_function(basis, process_matrix_function, coordinate)) 

        return cls(grid_axes, grid, gates_on_grid)

    #def from_gate(cls, gate: Gate, grid_axes: dict[str, NDArray], basis: StandardBasis, noises: list[Noise]): 
 #    @classmethod
 #    def from_gate(cls, gate: Gate, grid_axes: dict[str, NDArray], basis: StandardBasis, noise: Noise): 
 #        """ Build gate interpolant from a gate class. """ 
 #        # TODO: Currently assumes at most 1 noisy parameter. Extend?  
 #        # Build a grid and loop over every parameter value and build the gate from the process matrix function  
 #        grid = cls.build_grid(grid_axes)
 #        gates_on_grid = []
 #        for values in grid:
 #            coordinate = dict(zip(grid_axes.keys(), values)) 
 #            gates_on_grid.append(Gate.from_process_matrix_function(basis, process_matrix_function, coordinate, noise)) 
 #        return cls(grid_axes, grid, gates_on_grid)


 #    def from_process_matrix_function(cls, basis: Basis, process_matrix_function: Callable,
 #            parameters: dict[str, float], noise: Noise | None = None):
 #
 #        return cls(grid_axes, grid, gates_on_grid)


    @classmethod
    def from_hamiltonian_function(cls, basis: StandardBasis, hamiltonian_function: Callable, gate_duration: float, grid_axes: dict[str, NDArray]):
        """ Build gate interpolant from Schrodinger evolution of a Hamiltonian. 
            - requires a function that returns the hamiltonian using the parameters  
            - requires a fixed duration to set the hamiltonian's time evolution 
            - requires a dictionary of parameters to specify the grid
        """ 
        # Build a grid and loop over every parameter value and build the gate from the hamiltonian
        grid = cls.build_grid(grid_axes)
            # TODO: accomodate args for from_hamiltonian() like DOF to trace out, etc. 

        # For each parameter combination in the parameter grid, compute the gate and process matrix.
        gates_on_grid = []
        for values in grid:
            coordinate = dict(zip(grid_axes.keys(), values)) 
            gates_on_grid.append(Gate.from_hamiltonian_function(basis, hamiltonian_function, gate_duration, coordinate)) 

        return cls(grid_axes, grid, gates_on_grid)
    

 #    @classmethod
 #    def from_gate():
 #        """ Build gate interpolant from a gate """ 


    def return_gate_at_grid_point(self, parameter_coordinate: list[float]):
        """ Returns the gate evaluated at the grid point """ 

        # May be redundant--> interpolant should just return the known gate.
        if parameter_coordinate in self.grid:
            return self.return_gate_at_parameter_value(parameter_coordinate)
        return None # TODO replace 
        #else:
            


    

    #def build_gates_on_grid():




 #class InterpolatedGate(Gate):
 #
 #
 #    parameter_grids: list[Vector]
 # 
 #    @cachedproperty
 #    def grid(self):
 #        """ Returns the N-dimensional parameter mesh grid """
 #        return np.meshgrid(*parameter_grids, indexing = 'ij')
 #
 #    @classmethod
 #    def from_gate():
 #        """ Build gate interpolant from a gate """ 

