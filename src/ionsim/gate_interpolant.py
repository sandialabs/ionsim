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
    computed_gates: list[Gate] # Representing non-interpolated gates, computed on the parameter grid  

    noisy_parameter: str | None=None # extend to a list[str] or None if we support 2+ noisy parameters  

    # The way noise is handled assumes that possibly one of the parameters in the grid is a noise-type or noisy parameter  
    # Currently, there is a maximum of 1 noisy parameter. For more, IonSim needs an extension to multiple noisy parameter handling in Gate / Process module  
    # - this class tracks which parameter is noisy 

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
    # TODO: Option to convert grid output to numpy array. Should grid() be a list of coordinates or a np matrix-type object?  
 #    @cachedproperty
 #    def grid(self): 
 #        """ Returns the N-dimensional parameter grid as a list of parameter value coordinates. """
 #        return list(product(*list(self.grid_axes.values()))) 

 #    def grid_size(self):
 #        return len(grid)

 #    @property
 #    def size_of_grid_for_parameter(self, parameter_name: str):
 #        return len(parameter_grids[parameter_name])

    @property
    def grids(self): 
        return list(self.grid_axes.values()) 

    @property
    def grid_lengths(self): 
        return [len(grid) for grid in self.grids]

    @staticmethod
    def build_grid(grid_axes: dict[str, list[Vector]]):
        """ Returns the N-dimensional parameter grid as a list of coordinates in the grid. """
        return list(product(*list(grid_axes.values()))) 


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
        # NB: This is the cleanest way to handle noise. Noise is embedded in the gate function input. 
        # Used in the R gate example 
        grid = cls.build_grid(grid_axes)

        gates_on_grid = []
        parameter_list = list(grid_axes.keys())
        for values in grid:
            gates_on_grid.append(gate_function(*values))

        return cls(grid_axes, grid, gates_on_grid)


    #def from_process_matrix_function(cls, process_matrix_function: Callable, grid_axes: dict[str, NDArray], basis: StandardBasis, noise_function: Callable | None): 
    @classmethod
    def from_process_matrix_function(cls, process_matrix_function: Callable, grid_axes: dict[str, NDArray], basis: StandardBasis, noise_functions: dict[str,Callable] | None): 
        """ Build gate interpolant from a process matrix function. """ 
        # Build a grid and loop over every parameter value and build the gate from the process matrix function  
        # TODO: Handle case where there is noise. A Gate is built for some fixed realization of the noise.  
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


    def compute_functional_of_gates(self, gate_property_functional: Callable):
        """ Computes a functional of the gate at every gate in the grid. """ 
        # TODO: Decide whether to loop over ALL gates or just the computed / interpolated gates 
        functional_output = []
        for gate in self.computed_gates: 
            functional_output.append(gate_property_functional(gate))
        return functional_output

        #results_dictionary = {'dx' : dxs, 'dy': dys, 'relative_error': F_data}
        #R_gate_interpolant.write_to_file(data_filename, results_dictionary, attributes)

    #def write_to_file()


    #### Interpolation methods #### 
    #def construct_spline_for_gate_derived_property(gate_derived_property: dict[str, float] | dict[str, AnyMatrix] ):
    def construct_spline_for_gate_derived_property(self, gate_derived_property: AnyMatrix, dtype: 'float' | None=None):
        """ Constructs interpolant spline for derived property that lives on the domain of the parameter grid.""" 
        # Gate derived property input is a grid-dependent property, e.g. Fidelity[x, y] -> Number  
        # Gate derived property input could be matrix-valued, e.g. Residuals[x,y] -> d^2 x d^2 Process Matrix of residuals  

        if isinstance(type(gate_derived_property), AnyMatrix):
            # Case where the gate-derived proeprty is matrix-valued, e.g. the property at some grid point yields a matrix
            num_dtype = gate_derived_property.dtype
            row, cols = gate_derived_property.shape
            size = gate_derived_property.size 
            for i in range(rows):
                for j in range(cols):
                    spline_reals[i,j] = NdGridCubicSmoothingSpline(self.grids, gate_derived_property[i,j].real, smooth=1) 
        else:
            spline = NdGridCubicSmoothingSpline(self.grids, gate_derived_property, smooth=1) 



        dtype = type(gate_derived_property.values()[0])
        #property_name = gate_derived_property.key

 #        if dtype == complex:
 #
 #        else:

 #        F_spline_reals, F_spline_imags = R_gate_interpolant.construct_spline_for_gate_derived_property({'F_data':F_data})
 #
 #        #R_gate_interpolant.interpolate_gate_property
 #
 #        F_spline_reals = {}
 #        F_spline_imags = {}
 #        for i in range(size):
 #            for j in range(size):
 #                F_spline_reals[i,j] = NdGridCubicSmoothingSpline(grids, F_data[i,j].real, smooth=1)
 #                F_spline_imags[i,j] = NdGridCubicSmoothingSpline(grids, F_data[i,j].imag, smooth=1)

    def compute_interpolated_gate_at_coordinate(self, parameter_coordinate: dict[str, float]):
        """ Returns the gate evaluated at the grid point """ 

        # May be redundant--> interpolant should just return the known gate.
        if parameter_coordinate in self.grid:
            return self.return_gate_at_parameter_value(parameter_coordinate)
        return None # TODO replace 
        #else:
            


    def interpolate_gate_functional(self, gate_property_functional: Callable, parameter_coordinate: dict[str, float]):
        """ Interpolates a gate property, defined as a functional of the gate for a requested parameter coordiante value """  
        # e.g. process fidelity is a functional of the gate  
        coordinate_values = tuple(parameter_coordinate.values())



        return gate_property_functional(parameter_coordinate) 


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

