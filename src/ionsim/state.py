from ionsim.basis import Basis, StandardBasis
from ionsim.degree_of_freedom import DegreeOfFreedom
from ionsim.custom_types import Vector, Matrix
from ionsim.ionsim_error import IonSimError
from ionsim.hamiltonian import Hamiltonian
from ionsim.dissipator import Dissipator, Lindbladian

import numpy as np
# from typing import Any
from dataclasses import dataclass
from numpy.linalg import multi_dot

from icecream import ic
 
@dataclass(frozen=True, eq=False)       
class State:
    """A quantum state in a basis of states."""
    basis: Basis
    density_matrix: Matrix
    wavefunction: Vector | None = None

    @property
    def supervector(self) -> Vector:
        return self.basis.compute_supervector_from_density_matrix(self.density_matrix)

    @classmethod
    def from_coefficients(cls, basis: Basis, coefficients: list[float]): 
        """Build a state from a list of basis-state coefficients."""
        wavefunction = basis.compute_wavefunction_from_coefficients(coefficients)
        density_matrix = basis.compute_density_matrix_from_wavefunction(wavefunction)
        return cls(basis, density_matrix, wavefunction)
        
    @classmethod
    def from_wavefunction(cls, basis: Basis, wavefunction: Vector):
        """Build a state from a wavefunction in a particular basis."""
        density_matrix = basis.compute_density_matrix_from_wavefunction(wavefunction)
        return cls(basis, density_matrix, wavefunction)

    @classmethod
    def from_density_matrix(cls, basis: Basis, density_matrix: Matrix):
        """Build a state from a density matrix in a particular basis."""
        return cls(basis, density_matrix)

    @classmethod
    def from_supervector(cls, basis: Basis, supervector: Vector):
        """Build a state from a supervector in a particular basis."""
        density_matrix = basis.compute_density_matrix_from_supervector(supervector)
        return cls(basis, density_matrix)

    @classmethod
    def from_wavefunction_with_new_component(cls, basis: Basis, current_wavefunction: Vector,
            new_component: Vector, new_dofs: list[DegreeOfFreedom]):
        """Build a state from a wavefunction while adding a component with new degrees of freedom."""
        assert(len(new_dofs) == 1) # TODO generalize for adding new components with multiple DoFs.
        new_dof = new_dofs[0]
        if new_dof is basis.degrees_of_freedom[0]:
            wavefunction = np.kron(new_component, current_wavefunction)
        elif new_dof is basis.degrees_of_freedom[-1]:
            wavefunction = np.kron(current_wavefunction, new_component)
        else:
            raise IonSimError(
                'Currently, the new degree of freedom must be either the first or last '
                'degree of freedom in the full basis.'
            )
        density_matrix = basis.compute_density_matrix_from_wavefunction(wavefunction)
        return cls(basis, density_matrix, wavefunction)

    @classmethod
    def from_state(cls, basis: Basis, state: 'State'):
        """Build a state from a state, which could be in a different basis."""
        density_matrix = state.get_density_matrix_in_new_basis(basis)
        if state.wavefunction is not None:
            wavefunction = state.get_wavefunction_in_new_basis(basis)
            return cls(basis, density_matrix, wavefunction)
        return cls(basis, density_matrix)

    def propagate_using_process_matrix(self, process_matrix: Matrix):
        """
            Propagate the state by applying a process matrix to the supervector.
            This builds and returns a new state in the same basis as the initial state.
        """
        supervector = process_matrix.dot(self.supervector)
        density_matrix = self.basis.compute_density_matrix_from_supervector(supervector)
        return State(self.basis, density_matrix)

    def propagate_using_schrodinger_equation(self, hamiltonian: Hamiltonian, duration: float,
            time_evals: Vector | None = None, **kwargs):
        """
            Propagate the state by solving the time-dependent Schrodinger equation.
            This builds and returns a new state in the same basis as the initial state.
        """
        if self.wavefunction is None:
            raise IonSimError('The state must have a well-defined wavefunction.')
        times, psis = hamiltonian.evolve_wavefunction(self.wavefunction, duration, time_evals, **kwargs)
        if time_evals is None:
            density_matrix = self.basis.compute_density_matrix_from_wavefunction(psis[-1])
            return State(self.basis, density_matrix, psis[-1])
        rhos = [self.basis.compute_density_matrix_from_wavefunction(psi) for psi in psis]
        return [State(self.basis, rho, psi) for rho, psi in zip(rhos, psis)]

    def propagate_using_master_equation(self, lindbladian: Lindbladian, duration: float,
            time_evals: Vector | None = None, **kwargs):
        """
            Propagate the state by solving the time-dependent Lindblad master equation.
            This builds and returns a new state in the same basis as the initial state.
        """

        times, supervectors = lindbladian.evolve_supervector(self.supervector, duration, time_evals, **kwargs)
        if time_evals is None:
            density_matrix = self.basis.compute_density_matrix_from_supervector(supervectors[-1])
            return State(self.basis, density_matrix)
        rhos = [self.basis.compute_density_matrix_from_supervector(psi) for psi in supervectors]
        return [State(self.basis, rho) for rho in rhos]

    def get_wavefunction_in_new_basis(self, new_basis: Basis):
        """Get the wavefunction in a new basis."""
        if new_basis is self.basis or self.wavefunction is None:
            return self.wavefunction
        return self.basis.change_basis_of_vector(self.wavefunction, new_basis)

    def get_density_matrix_in_new_basis(self, new_basis: Basis):
        """Get the density matrix in a new spin basis."""
        if new_basis is self.basis:
            return self.density_matrix
        return self.basis.change_basis_of_matrix(self.density_matrix, new_basis) 
    
    def compute_state_fidelity(self, target_density_matrix: Matrix):
        """Compute the state fidelity with respect to a target density matrix."""
        return np.trace(np.dot(self.density_matrix, target_density_matrix)).real
    
    def compute_basis_state_probabilities(self):
        """Compute the probability of measuring each basis state."""
        return [
            np.trace(self.basis.compute_projector_matrix(vector).dot(self.density_matrix)).real
            for vector in self.basis.vectors
        ]

    def compute_density_matrix_traced_over_degree_of_freedom(self, degree_of_freedom: DegreeOfFreedom):
        """Compute a reduced density matrix by tracing out a degree of freedom in the basis."""
        size = len(degree_of_freedom.energy_levels)
        vecs = list(np.eye(size))
        new_size = int(len(self.basis.vectors)/size)
        small_density_matrix = np.zeros((new_size, new_size), dtype='complex')
        for n in range(size):
            bra = vecs[n].T
            ket = np.array([[c] for c in vecs[n]])
            proj_left = self.basis.enlarge_matrix(bra, [degree_of_freedom])
            proj_right = self.basis.enlarge_matrix(ket, [degree_of_freedom])
            small_density_matrix += multi_dot([proj_left, self.density_matrix, proj_right])
        return small_density_matrix

    def trace_out_degree_of_freedom(self, degree_of_freedom: DegreeOfFreedom):
        """Trace out a degree of freedom in the basis and return a new state in a new basis."""
        density_matrix = self.compute_density_matrix_traced_over_degree_of_freedom(degree_of_freedom)
        basis = StandardBasis([dof for dof in self.basis.degrees_of_freedom if dof is not degree_of_freedom])
        return State.from_density_matrix(basis, density_matrix)

    def compute_coherent_displacements(self, spin_dofs: list[DegreeOfFreedom], motional_dof: DegreeOfFreedom):
        """Compute the coherent displacement (expectation value of the lowering operator) for each spin state."""
        assert(len(self.basis.degrees_of_freedom) == len(spin_dofs) + 1) # TODO: trace out other degrees of freedom
        spin_basis = StandardBasis(spin_dofs)
        lowering = lowering_motion(len(motional_dof.energy_levels))
        diplacements = []
        for vector in spin_basis.vectors:
            spin_proj = spin_basis.compute_projector_matrix(vector)
            displacement = np.trace(np.kron(spin_proj, lowering).dot(self.density_matrix))
            diplacements.append(displacement)
        return diplacements

    # def transform_to_spin_eigenbasis(self):

def lowering_motion(fock_dimension: int):
    return np.diag([np.sqrt(n+1) for n in range(fock_dimension-1)], k=1)
