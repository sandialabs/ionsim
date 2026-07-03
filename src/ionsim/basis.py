#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

from ionsim.ionsim_error import IonSimError
from ionsim.degree_of_freedom import DegreeOfFreedom, AtomicSpin
from ionsim.atomic_internal_energy_level import AtomicInternalEnergyLevel, compute_dipole_amplitude
from ionsim.energy_level import EnergyEigenstate
from ionsim.custom_types import Vector, Matrix
from ionsim.config import NUMERICAL_EQUIVALENCE_THRESHOLD, NUMERICAL_ERROR_THRESHOLD
from ionsim.laser import Laser 

import numpy as np
from numpy.linalg import multi_dot
from typing import Callable
from functools import wraps # do I need this?
from scipy.sparse import csr_matrix
from scipy.sparse import kron as skron
from scipy import constants as const

from abc import ABC, abstractmethod
from typing import Sequence
from dataclasses import dataclass
import itertools
from functools import cached_property
import functools as ft

from icecream import ic

@dataclass(frozen=True, eq=False)
class Basis(ABC):
    """A basis of states."""
    degrees_of_freedom: Sequence[DegreeOfFreedom]

    @property
    @abstractmethod
    def vectors(self):
        """Basis-state vectors."""

    @property
    def spin_DOFs(self):
        """ Returns list of spin degrees of freedom or empty list if none. """
        spins = [DOF for DOF in self.degrees_of_freedom if isinstance(DOF, AtomicSpin)]
        return spins

    @property
    def change_of_basis_matrix(self):
        """The unitary matrix that transforms a vector in this basis to the standard basis."""
        return np.array([vector for vector in self.vectors]).T

    def transform_vector_to_standard_basis(self, vector: Vector):
        """Transform a vector in this basis to the standard basis."""
        return self.change_of_basis_matrix.dot(vector)

    def transform_vector_from_standard_basis(self, vector: Vector):
        """Transform a vector in the standard basis to this basis."""
        return (self.change_of_basis_matrix.conj().T).dot(vector)

    # TODO: we need to differentiate between "matrix" and "supermatrix" throughout code
    # since this won't work for a process matrix
    def transform_matrix_to_standard_basis(self, matrix: Matrix):
        """Transform a matrix in this basis to the standard basis."""
        return multi_dot([self.change_of_basis_matrix, matrix, self.change_of_basis_matrix.conj().T])

    def transform_matrix_from_standard_basis(self, matrix: Matrix):
        """Transform a matrix in the standard basis to this basis."""
        return multi_dot([self.change_of_basis_matrix.conj().T, matrix, self.change_of_basis_matrix])

    # def transform_supervector_to_standard_basis()
    # def transform_supermatrix_to_standard_basis()

    def compute_wavefunction_from_coefficients(self, coefficients: list[float]):
        """Compute a wavefunction from its basis-state coefficients."""
        assert(len(coefficients) == len(self.vectors)) 
        coefficients = list(np.array(coefficients)/np.linalg.norm(coefficients))
        assert(np.abs(np.abs(np.linalg.norm(coefficients)) - 1) < NUMERICAL_EQUIVALENCE_THRESHOLD)
        return sum([vector*coef for vector, coef in zip(self.vectors, coefficients)])
    
    def compute_density_matrix_from_wavefunction(self, wavefunction: Vector):
        """Compute a density matrix from a wavefunction, i.e., a pure state."""
        assert(len(wavefunction) == len(self.vectors)) 
        error = np.abs(np.abs((wavefunction.conj().T).dot(wavefunction)) - 1)
        if error > NUMERICAL_ERROR_THRESHOLD:
            raise IonSimError(f'Numerical error of {error} is greater than NUMERICAL_ERROR_THRESHOLD.')
        # assert(np.abs(np.abs((wavefunction.conj().T).dot(wavefunction)) - 1) < NUMERICAL_ERROR_THRESHOLD)
        return np.outer(wavefunction, wavefunction.conj().T)
    
    def compute_supervector_from_density_matrix(self, density_matrix: Matrix): # TODO: generalize for any basis
        """Compute a column-stacked supervector from a density matrix."""
        assert(density_matrix.shape == (len(self.vectors), len(self.vectors))) # TODO: replace with IonSimError
        return (density_matrix.T).flatten()
    
    def compute_density_matrix_from_supervector(self, supervector: Vector): # TODO: generalize for any basis
        """Compute a process matrix from a column-stacked supervector."""
        dimension = len(self.vectors)
        assert(len(supervector) == dimension**2) 
        return (supervector.reshape(dimension, dimension)).T

    def compute_projector_matrix(self, basis_vector: Vector):
        """Compute the projector matrix onto a basis vector."""
        return self.compute_density_matrix_from_wavefunction(basis_vector)

    def compute_superoperator_from_unitary_operator(self, unitary_operator: Matrix):
        """Compute a superoperator from a unitary operator in the column-stacked representation."""
        #TODO: should this function reflect the choice of basis?
        return np.kron(unitary_operator.conj(), unitary_operator)
    
    def create_superoperator_function_from_unitary_operator_function(self, unitary_operator_function: Callable):
        """Create a superoperator function from a unitary operator function in the column-stacked representation."""
        @wraps(unitary_operator_function)
        def wrapper(*args, **kwargs):
            return self.compute_superoperator_from_unitary_operator(unitary_operator_function(*args, **kwargs))
        return wrapper

    # TODO: change name to array since it works for vectors too. We use it for both row and columns vectors. 
    # TODO: implement sparse matrices
    def enlarge_matrix(self, matrix: Matrix, current_dofs: list[DegreeOfFreedom]): 
        """Enlarge the dimension of a matrix with some degrees of freedome to include all degrees of freedom in the basis."""
        if len(current_dofs) == len(self.degrees_of_freedom):
            return matrix
        if len(current_dofs) == 1:
            return self.enlarge_one_dof_matrix(matrix, current_dofs[0])
        else:
            raise IonSimError('Enlarging a matrix for more than one degree of freedom is not currently implemented.')

    def enlarge_matrix_function(self, matrix_function: Callable, current_dofs: list[DegreeOfFreedom]):
        """Enlarge the dimension of a matrix function with some degrees of freedom to include all degrees of freedom in the basis."""
        if len(current_dofs) == len(self.degrees_of_freedom):
            return matrix_function
        @wraps(matrix_function)
        def wrapper(*args, **kwargs):
            if len(current_dofs) == 1:
                return self.enlarge_one_dof_matrix(matrix_function(*args, **kwargs), current_dofs[0])
            else:
                raise IonSimError('Enlarging a matrix function for more than one degree of freedom is not currently implemented.')
        return wrapper

    def enlarge_one_dof_matrix(self, matrix: Matrix, current_dof: DegreeOfFreedom):
        """Enlarge the dimension of an matrix with one degree of freedom to include all degrees of freedom in the basis."""
        sparse = isinstance(matrix, csr_matrix) # TODO: use better design to aviod this isinstance
        if sparse: 
            kron = skron
            large_matrix = csr_matrix(([1], ([0], [0])), shape=(1, 1))
        else:
            kron = np.kron
            large_matrix = np.array([1])
        for dof in self.degrees_of_freedom:
            if dof is current_dof:
                large_matrix = kron(large_matrix, matrix)
            else:
                # large_matrix = kron(large_matrix, np.eye(matrix.shape[0]))
                large_matrix = kron(large_matrix, np.eye(len(dof.energy_levels)))
        if sparse:
            return csr_matrix(large_matrix)
        else:
            return large_matrix

    def _check_if_pauli_basis(self):
        """Check if the basis has one degree of freedom with two atomic internal energy levels."""
        if len(self.degrees_of_freedom) == 1:
            self._check_if_qubit_basis()
        else:
            raise IonSimError('The basis must have one degree of freeedom.')

    def _check_if_qubit_basis(self):
        """Check if the basis has two atomic internal energy levels in each degree of freedom."""
        if all([len(dof.energy_levels) == 2 for dof in self.degrees_of_freedom]):
            if all([
                all([isinstance(level, AtomicInternalEnergyLevel) for level in dof.energy_levels])
                for dof in self.degrees_of_freedom
            ]):
                return
        raise IonSimError('The basis must have two atomic internal energy levels in each degree of freedom.')

    # TODO: check that these are working for the new Basis class
    def change_basis_of_vector(self, vector: Vector, new_basis: 'Basis'):
        """Change the basis of a vector."""
        standard_vector = self.transform_vector_to_standard_basis(vector)
        return new_basis.transform_vector_from_standard_basis(standard_vector)

    def change_basis_of_matrix(self, matrix: Matrix, new_basis: 'Basis'):
        """Change the basis of a matrix."""
        standard_matrix = self.transform_matrix_to_standard_basis(matrix)
        return new_basis.transform_matrix_from_standard_basis(standard_matrix)


#=========================================================================================================================================================
#=========================================================================================================================================================
    # TODO: Should this method go in a different class? 
    def build_atom_laser_coupling_operators_for_ground_level(self, ground_level: AtomicInternalEnergyLevel, laser: Laser, multipole: str="E1", 
                                                            all_spins_are_same: bool = True) -> list[Operator]: 
        """ New helper method for building light-atom coupling operators from AMO physics - Atomic Structure details """ 

        if multipole == 'E1' or multipole == 'dipole':
            q = [-1, 0, 1] 
        elif multipole == 'E2':
            raise ValueError(f"Only E1 (dipole) transitions are implemented.")
            q = [-2, -2, 0, 1, 2] 
        else:
            raise ValueError(f"Multipole value must be either 'E1' or 'E2', corresponding respectively to electric dipole or quadrupole transitions.")

        for atom in self.spin_DOFs:        
            # Build coupling operator for each |g>, |e> pairing   
            atomic_levels = atom.energy_levels 

            for excited_level in atomic_levels: 
                if ground_level.name == level.name:
                    pass
                coupling_amplitudes = {}
                for _q in q: 
                    coupling_amplitudes{_q} = compute_dipole_amplitude(ground_level, excited_level, _q) 

                # Compute dot product with laser field polarization vector 
                laser.polarization.spherical_components 
                
            


            # Break from the loop if all the AtomicSpin DOFs are the same 
            if all_spins_are_same:
                break 

        return operators 


    def build_atom_laser_coupling_operator(self, atom: AtomicSpin, ground_level: AtomicInternalEnergyLevel, excited_level: AtomicInternalEnergyLevel, laser: Laser, multipole: str = "E1"): 
        """ Builds a light-atom coupling operator between two atomic levels in the full basis """  
        # Need to build the operator with access to the full basis 
        if multipole == 'E1' or multipole == 'dipole':
            q = [-1, 0, 1] 
        elif multipole == 'E2':
            raise ValueError(f"Only E1 (dipole) transitions are implemented.")
            q = [-2, -2, 0, 1, 2] 
        else:
            raise ValueError(f"Multipole value must be either 'E1' or 'E2', corresponding respectively to electric dipole or quadrupole transitions.")

        atomic_levels = atom.energy_levels 
        if (not ground_level in atomic_levels) or (not excited_level in atomic_levels):
            raise ValueError("Specified ground and excited levels must be in the atom's atomic structure.")

        coupling_amplitudes = {}
        for _q in q: 
            coupling_amplitudes{_q} = compute_dipole_amplitude(ground_level, excited_level, _q) 

        # Compute dot product with laser field polarization vector 
        # TODO: should we use vdot? 
        polarization = laser.polarization.spherical_components
        rabi_frequency = 2. * laser.peak_electric_field_magnitude * np.dot(polarization, np.array(list(coupling_amplitudes.values()))) / const.hbar 

        coupling_operator = np.zeros((2,2)) 
        ground_index = atomic_levels.index(ground_level) 
        excited_index = atomic_levels.index(excited_level) 
        # Convention: We build a raising operator only; the Hamiltonian class will supply the h.c. lowering operator 
        coupling_operator[excited_index, ground_index] = 0.5 * rabi_frequency * np.exp(1j*laser.phase) 

        # Retrieve modulation function from laser class 
        if laser.mod_functions is not None and None not in laser.mod_functions.values():
            # Unpack modulation functions and check which exist 
            amplitude_mod = laser.modulation_function['amplitude']
            phase_mod = laser.modulation_function['phase']
            frequency_mod = laser.modulation_function['phase']
            # There are several combinations of None/not None to handle: 
            if phase_mod is None and frequency_mod is None and amplitude_mod is not None:
                def mod_function(t: float):
                    return lambda t: amplitude_mod(t) 
            elif phase_mod is None and amplitude_mod is None and frequency_mod is not None:
                def mod_function(t: float):
                    return lambda t: np.exp(1j * frequency_mod(t) * t) 

            elif phase_mod is None and amplitude_mod is None and frequency_mod is not None:
                def mod_function(t: float):
                    return lambda t: np.exp(1j * frequency_mod(t) * t) 



                mod_function = laser.modulation_function['amplitude']



        return CouplingOperator.from_matrix(basis, coupling_operator, )

        #for amplitude in coupling_amplitudes.values():
            #if amplitude != 0.:
                
                 

        



    #def build_all_atom_light_coupling_operators


@dataclass(frozen=True, eq=False)
class StandardBasis(Basis):
    """A basis of energy eigenstates of a non-interacting Hamiltonian."""

    @cached_property
    def states(self): # TODO: consider renaming this "energy_eigenstates."
        """The energy eigenstates of the non-interacting Hamiltonian in this basis."""
        components_list = list(itertools.product(*[dof.energy_levels for dof in self.degrees_of_freedom]))
        return [EnergyEigenstate(list(components)) for components in components_list]

    @property
    def vectors(self):
        """Basis-state vectors corresponding to the energy eigenstates."""
        return list(np.eye(len(self.states)))

    # TODO: Check that moving the spin_DOFs property from here to Basis isn't a problem 


@dataclass(frozen=True, eq=False)
class ZPauliBasis(StandardBasis):
    """A basis in which the basis states correspond to the (plus/minus) eigenstates of the z-Pauli spin matrix."""
    degrees_of_freedom: list[AtomicSpin]

    def __post_init__(self):
        # self._check_if_pauli_basis() # TODO: should we only allow for one degree of freedom here?
        self._check_if_qubit_basis()

@dataclass(frozen=True, eq=False)
class XPauliBasis(Basis):
    """A basis in which the basis vectors correspond to the (plus/minus) eigenstates of the x-Pauli spin matrix."""
    degrees_of_freedom: list[AtomicSpin]

    def __post_init__(self):
        # self._check_if_pauli_basis() # TODO: should we only allow for one degree of freedom here?
        self._check_if_qubit_basis()

    @property
    def vectors(self):
        """Eigenstate vectors of the x-Pauli spin matrix, expressed in the z-Pauli basis."""
        plus = 1/np.sqrt(2)*np.array([1, 1])
        minus = 1/np.sqrt(2)*np.array([1, -1])
        if len(self.degrees_of_freedom) == 1:
            return [plus, minus]
        pairs = list(itertools.product(*[[plus, minus] for dof in self.degrees_of_freedom]))
        return [np.kron(*pair) for pair in pairs]

@dataclass(frozen=True, eq=False)
class YPauliBasis(Basis):
    """A basis in which the basis vectors correspond to the (plus/minus) eigenstates of the x-Pauli spin matrix."""
    degrees_of_freedom: list[AtomicSpin]

    def __post_init__(self):
        # self._check_if_pauli_basis() # TODO: should we only allow for one degree of freedom here?
        self._check_if_qubit_basis()

    @property
    def vectors(self):
        """Eigenstate vectors of the y-Pauli spin matrix, expressed in the z-Pauli basis."""
        plus = 1/np.sqrt(2)*np.array([1, 1j])
        minus = 1/np.sqrt(2)*np.array([1, -1j])
        if len(self.degrees_of_freedom) == 1:
            return [plus, minus]
        pairs = list(itertools.product(*[[plus, minus] for dof in self.degrees_of_freedom]))
        return [np.kron(*pair) for pair in pairs]

@dataclass(frozen=True, eq=False)
class XPauliAndFockBasis(Basis):
    """A basis in which the basis vectors correspond to the (plus/minus) eigenstates of the x-Pauli spin matrix and Fock states."""
    atomic_spins: list[AtomicSpin]

    @property
    def motional_modes(self):
        return [dof for dof in self.degrees_of_freedom if dof not in self.atomic_spins]

    @property
    def vectors(self):
        """Eigenstate vectors of the x-Pauli spin matrix, expressed in the z-Pauli basis."""
        plus = 1/np.sqrt(2)*np.array([1, 1])
        minus = 1/np.sqrt(2)*np.array([1, -1])
        groups = list(itertools.product(
            *[
                [plus, minus] if dof in self.atomic_spins else
                [np.eye(len(dof.energy_levels))[i] for i in range(len(dof.energy_levels))]
                for dof in self.degrees_of_freedom
            ]
        ))
        return [ft.reduce(np.kron, group) for group in groups]
