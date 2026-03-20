from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from typing import Callable
from functools import cached_property

from ionsim.basis import StandardBasis
from ionsim.degree_of_freedom import AtomicSpin 
from ionsim.operator import Operator, Coupling, EnergyShift, GeneralOperator, EnergyShiftOperator, CouplingOperator
from ionsim.custom_types import Vector, Matrix, SparseMatrix, AnyMatrix, as_dense_matrix
from ionsim.custom_math import matrix_AYB_multiply_to_superoperator, solve_time_evolution_equation
from ionsim.ionsim_error import IonSimError
from ionsim.composite_operator import CompositeOperator
from ionsim.hamiltonian import Hamiltonian
from ionsim.atomic_internal_energy_level import AtomicInternalEnergyLevel, LSFineLevel, LSHyperfineLevel, J1L2FineLevel, J1L2HyperfineLevel
from ionsim.atomic_internal_energy_level import compute_dipole_amplitude
from ionsim.config import SMALLEST_ENERGY_SCALE


@dataclass(frozen=True, eq=False)
class Dissipator(CompositeOperator):
    """Class for dissipator object which implements dissipative phenomena in the context of open quantum systems. 

        Inherits from CompositeOperator parent class. Instantiation requires a basis and 
            list of operators corresponding to Lindblad operators.  
    """
    def __post_init__(self):
        super().__post_init__()

    @staticmethod
    def lindblad_matrix_to_superoperator(L_matrix: AnyMatrix) -> AnyMatrix:
        """ Method to convert a Lindblad operator in matrix form (N x N) to a superoperator (N^2 x N^2) 
            for matrix-vector multiplication on a flattened density matrix (shape: N^2 x 1) 
    
            Lindblad master equation contains Lindblad operators {L} with 3 contributions:
            1. L rho L^†
            2. -1/2 L^† L rho 
            3. -1/2 rho L^† L 
    
            - assumes any decay rates have been lumped into definition of L matrix.
    
            For matrix-matrix operations, the mapping to the (column-stacked) superoperator (y) is: 
            A rho --> (I kron A) y  
            rho A --> (A^{T} kron I) y  
            A rho B --> (B^{T} kron A) y 
        """
        LdaggerL = np.conj(L_matrix.T) @ L_matrix # L^† L: 

        superoperator = matrix_AYB_multiply_to_superoperator(A = LdaggerL, B = None) 
        superoperator += matrix_AYB_multiply_to_superoperator(A = None, B = LdaggerL) 
        superoperator *= -0.5
    
        superoperator += matrix_AYB_multiply_to_superoperator(L_matrix, np.conjugate(L_matrix.T)) 
        return superoperator 

    @cached_property
    def lindblad_matrix_functions(self) -> list[Callable]:
        """Creates and stores a list of callables that return (N x N) matrices as a function of time. 
            Each callable corresponds to a lindblad operator in the interaction picture, which may contain time-dependence as a result of frame shifting. 

            Defines and returns a list of callables as a function of time"""

        lindblad_functions = []

        for lindblad_op in self.operators:
            lindblad_functions.append(self.create_lindblad_matrix_function(lindblad_op))

        return lindblad_functions

    def create_lindblad_matrix_function(self, lindblad_operator: Operator) -> Callable:
        """Converts a lindblad Operator object to a callable that returns a matrix at a given time point"""
        
        def _lindblad_function(t: float) -> AnyMatrix:
            if self.sparse:
                L_matrix = csr_matrix(([0], ([0], [0])), shape=(self.size, self.size), dtype='complex') 
            else:
                L_matrix = np.zeros((self.size, self.size), dtype=complex) 
            
            # Static, diagonal contribution 
            if isinstance(lindblad_operator, GeneralOperator):
                L_matrix += lindblad_operator.diagonal_contribution.static_matrix
            elif isinstance(lindblad_operator, EnergyShiftOperator):
                L_matrix += lindblad_operator.static_matrix

            if isinstance(lindblad_operator, CouplingOperator) or isinstance(lindblad_operator, GeneralOperator):
                # Must account for frame shifts in coupling operators 
                L_int, Rate = self._frame_shifted_coupling_matrix_and_rate_from_operator(lindblad_operator) 
    
                # Element-wise multiplication, compute L * exp(-1j * Rate * t)
                if self.sparse: 
                    phase_factor_minus_one = Rate.multiply(-1j*t).expm1() # equivalent to exp(-1j * Rate * t) - 1
                    Ltemp = L_int + L_int.multiply(phase_factor_minus_one) # necessary to cancel out the -1 contribution above 
                    L_matrix += Ltemp 
                else:
                    L_matrix += (L_int.toarray() * np.exp(-1j * Rate.toarray() * t))
            return L_matrix

        return _lindblad_function
            
    @cached_property
    def dissipator_matrix_function(self) -> Callable:
        """Creates N^2 x N^2 matrix representation of each Lindblad operator's action on a N^2 x 1 superoperator (y) from NxN density matrix (rho): 
                Dissipator acting on supervector from Lindblad operator i (L_i): d_i(t)y <==> L_i(t) rho L†_i(t) - 1/2 { L†_i(t) L_i(t) , rho }  

            Returns the sum of the dissipators: D(t) = sum_i d_i (t) 
        """
        # Transform to dissipator form for superoperator matrix-vector multiplication.  
        def _dissipator_matrix_function(t: float):
            if self.sparse:
                d_matrix = csr_matrix(([0], ([0], [0])), shape=(self.size**2, self.size**2), dtype='complex') 
            else:
                d_matrix = np.zeros((self.size**2, self.size**2), dtype=complex)

            for L_i in self.lindblad_matrix_functions:
                d_matrix += self.lindblad_matrix_to_superoperator(L_i(t))
            return d_matrix

        return _dissipator_matrix_function 


@dataclass(frozen=True, eq=False)
class DissipatorSpontaneousEmission(Dissipator):
    """Subclass for including spontaneous emission dynamics for a known atomic structure."""

    @classmethod
    def from_atomic_structure_data(cls, basis: StandardBasis, ground_levels: list[AtomicInternalEnergyLevel], excited_levels: list[AtomicInternalEnergyLevel], frame_energies: list[float] | None=None, sparse: bool=False, all_spins_are_same: bool = True, select_DOFs: list[AtomicSpin] | None = None):
        """ Builds dissipator for spontaneous emission from a user-specified list of excited and ground levels.  
            

            - The user specifies a list of ground levels and excited levels that they want to consider for spontaneous emission (decay) couplings.  
            - Level strings should correspond with an exciting atomic configuration file (e.g. 'S1/2,0,0'). 
        """
        if select_DOFs is None:
            DOF_list = basis.degrees_of_freedom
        else:
            DOF_list = select_DOFs

        lindblad_operators = []

        # Loop over each (spin) DOF and create a lindblad operator in that DOF's Hilbert space.
            # Then, enlarge that lindblad operator to the system basis which contains the whole Hilbert space.  
        for DOF in DOF_list: 
            if isinstance(DOF, AtomicSpin):
                # For each excited level, loop through ground levels it can decay to  
                for e_level in excited_levels:

                    # Extract lifetime and branching ratio for this excited level
                    e_lifetime = e_level.lifetime # dict with ground-manifold as key 
                    e_branching_ratios = e_level.branching_ratios

                    g_amplitudes = {} 
                    for g_level in ground_levels:
                        if e_level.energy <= g_level.energy:
                            raise IonSimError('Error: Excited level should be higher in energy than the lower level. Excited energy: {e_level.energy}, Ground energy: {g_level.energy}')
                            
                        # E1 transition considers angular momentum transfer q: -1, 0, +1
                        for q in [-1, 0, 1]:
                            # Compute E1 dipole amplitude between |e> and |g>, append if non-zero 
                            amplitude = compute_dipole_amplitude(g_level, e_level, q)
                            if np.abs(amplitude) >  SMALLEST_ENERGY_SCALE:
                                g_amplitudes[(g_level, q)] = np.abs(amplitude**2) 

                    # Get normalization by summing over amplitudes, necessary if we don't consider every decay path way  
                    amplitude_sum = sum(g_amplitudes.values())
                    assert amplitude_sum > 0., 'Error: No decay pathways for excited state {e_level.name} '
                    e_level_index = DOF.energy_levels.index(e_level)

                    for (g_level, q), weight in g_amplitudes.items():
                        if e_branching_ratios is None:
                            e_g_branching_ratio = 1.
                        else:
                            e_g_branching_ratio = e_branching_ratios[g_level.term_symbol]

                        # Spontaneous emission decay rate: 
                        decay_rate = (1./e_lifetime) * e_g_branching_ratio * g_amplitudes[(g_level, q)] / amplitude_sum    

                        # Create lowering operator  
                        g_level_index = DOF.energy_levels.index(g_level)
                        lowering_matrix = np.zeros((len(DOF.energy_levels), len(DOF.energy_levels)))
                        lowering_matrix[g_level_index, e_level_index] = 1.*np.sqrt(decay_rate) 

                        # Enlarge lowering opearator to live in entire basis 
                        if not all_spins_are_same:
                            enlarged_lowering_matrix = basis.enlarge_matrix(lowering_matrix, [DOF]) 
                            decay_operator = np.sqrt(decay_rate) * enlarged_lowering_matrix
                            lindblad_operators.append(CouplingOperator.from_matrix(basis, decay_operator, 0.))
                        else:
                            enlarged_lowering_matrices = [basis.enlarge_matrix(lowering_matrix, [spin]) for spin in basis.spin_DOFs]
                            decay_operators = [large_matrix for large_matrix in enlarged_lowering_matrices] 
                            for decay_operator in decay_operators:
                                lindblad_operators.append(CouplingOperator.from_matrix(basis, decay_operator, 0.))

            # Break from the loop if all the AtomicSpin DOFs are the same 
            if all_spins_are_same:
                break 

        if not lindblad_operators:
            raise IonSimError("Error: No branching ratios or lifetime data found. No lindblad operators were created.") 

        return cls(basis, lindblad_operators, frame_energies, sparse) 


@dataclass(frozen=True, eq=False)
class Lindbladian:
    hamiltonian: Hamiltonian | None  
    dissipator: Dissipator | None 

    def __post_init__(self):
        if self.hamiltonian and self.dissipator:
            # Check if Hamiltonian and Dissipator are the same dimensionality 
            if self.hamiltonian.size != self.dissipator.size:
                raise IonSimError('Hamiltonian and Dissipator objects must have the same size (dimensionality)')
            # Check if Hamiltonian and Dissipator have the same sparse setting.  
            if self.hamiltonian.sparse != self.dissipator.sparse: 
                raise IonSimError('Input error: Both Hamiltonian and Dissipator should have the same setting for sparse variable. e.g. Set sparse = True when constructing the Hamiltonian and Dissipator objects.')
        if (self.hamiltonian, self.dissipator) == None:
            raise IonSimError('Input error: Both Hamiltonian and Dissipator inputs are None')

    @property
    def size(self):
        # Lindbladian size is N^2, corresponding to an N^2 x N^2 superoperator representation
        if self.hamiltonian and self.dissipator:
            return self.hamiltonian.size**2
        elif self.hamiltonian and (self.dissipator is None):
            return self.hamiltonian.size**2
        elif self.dissipator and self.hamiltonian is None:
            return self.dissipator.size**2

    @cached_property
    def matrix_function(self) -> Callable:
        """ Lindbladian matrix function L(t), corresponding to an N^2 x N^2 superoperator. """
        if self.hamiltonian:
            super_ham = lambda t: -1j*(
                  matrix_AYB_multiply_to_superoperator(A = self.hamiltonian.hamiltonian_function(t), B = None) 
                - matrix_AYB_multiply_to_superoperator(A = None, B = self.hamiltonian.hamiltonian_function(t))
                )
        else:
            super_ham = lambda t: 0. 

        # For dissipation, we compensate the input by multiplying the RHS by i to get dy/dt = Ay form. 
        if self.dissipator:
            super_dissipator = lambda t: self.dissipator.dissipator_matrix_function(t) 
            #super_dissipator = lambda t: 1j*self.dissipator.dissipator_matrix_function(t) 
        else:
            super_dissipator = lambda t: 0. 

        # Lindbladian superoperator from hamiltonian and dissipation contributions: 
        lindbladian_function = lambda t: super_ham(t) + super_dissipator(t)
        return lindbladian_function

    def evolve_supervector(self, initial_supervector: Vector, duration: float, time_evals: Vector | None = None, **kwargs):
        """ Evolve a supervector by solving the time-dependent Lindblad master equation with pure dissipation (no Hamiltonian).
            e.g. evolves supervector "y" using dy/dt = Dy, where D is the N^2 x N^2 dissipator matrix. 
        """
 #        assert(self.size == len(initial_supervector))
 #        if self.hamiltonian:
 #            super_ham = lambda t: (
 #                  matrix_AYB_multiply_to_superoperator(A = self.hamiltonian.hamiltonian_function(t), B = None) 
 #                - matrix_AYB_multiply_to_superoperator(A = None, B = self.hamiltonian.hamiltonian_function(t))
 #                )
 #        else:
 #            super_ham = lambda t: 0. 
 #
 #        # solve_time_evolution_equation() assumes a Schrodinger equation form dy/dt = (-i*A)y, where i = sqrt(-1) and A <==> Hamiltonian matrix. 
 #        # For dissipation, we compensate the input by multiplying the RHS by i to get dy/dt = Ay form. 
 #        if self.dissipator:
 #            super_dissipator = lambda t: 1j*self.dissipator.dissipator_matrix_function(t) 
 #        else:
 #            super_dissipator = lambda t: 0. 
 #
 #        # Lindbladian superoperator from hamiltonian and dissipation contributions: 
 #        lindbladian_function = lambda t: super_ham(t) + super_dissipator(t)

        # solve_time_evolution_equation() assumes a Schrodinger equation form dy/dt = (-i*A)y, where i = sqrt(-1) and A <==> the function input, e.g. a Hamiltonian matrix. 
        # Therfore, we must compensate this form by multiplying by the lindbladian by i 
        assert(self.size == len(initial_supervector))
        dynamical_matrix = lambda t: self.matrix_function(t) * 1j
        return solve_time_evolution_equation(dynamical_matrix, initial_supervector, duration, time_evals, **kwargs)
