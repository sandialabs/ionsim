from ionsim.basis import Basis, StandardBasis
from ionsim.operator import Operator, CouplingOperator
from ionsim.degree_of_freedom import DegreeOfFreedom
from ionsim.custom_types import Vector, Matrix
from ionsim.ionsim_error import IonSimError
from ionsim.hamiltonian import Hamiltonian
from ionsim.dissipator import Dissipator, Lindbladian
from ionsim.named_operators import Fock 
from ionsim.collective_motional_energy_level import CollectiveMotionalEnergyLevel 

import numpy as np
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

    @property
    def motional_state(self):
        """ Motional portion of the state, if it exists. Returns None if no motional DOFs """ 
        if not self.basis.motional_modes: 
            return None 

        # Trace out spin DOFs to obtain a purely motional state  
        state = self
        for spin in self.basis.spin_DOFs:
            state = state.trace_out_degree_of_freedom(spin)            
             
        return state 

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
        lowering = Fock.lowering(len(motional_dof.energy_levels))
        diplacements = []
        for vector in spin_basis.vectors:
            spin_proj = spin_basis.compute_projector_matrix(vector)
            displacement = np.trace(np.kron(spin_proj, lowering).dot(self.density_matrix))
            diplacements.append(displacement)
        return diplacements

    ### Added methods by ECM in this branch:### 
    def compute_matrix_observable_expectation(self, observable_operator: Matrix) -> Matrix:
        """ Compute the expectation value of an operator observable, represented by <O>. 

            - Computes via Tr[O rho], where rho is the density matrix. 
            - O and rho must be in compatable bases.

        """
        # Check that observable and density matrix are compatible 
        basis_size = len(self.basis.vectors)
        if observable_operator.shape != (basis_size, basis_size):
            raise IonSimError("Observable operator must correspond with a matrix of shape: {(basis_size, basis_size)}.")

        return np.trace(observable_operator.dot(self.density_matrix))

    def compute_operator_observable_expectation(self, observable_operator: Operator) -> Matrix:
        """ Compute the expectation value of an operator observable, represented by <O>. 

            - Computes via Tr[O rho], where rho is the density matrix. 
            - O and rho must be in compatable bases.

        """
        # Check that observable and density matrix are compatible 
        if isinstance(observable_operator, CouplingOperator) and observable_operator.oscillation_rate != 0.:
            raise IonSimeError(f"Observable operator must be a static operator, but oscillation rate is {observable_operator.oscillation_rate}.")

        return self.compute_matrix_observable_expectation(observable_operator.static_matrix) 

    def compute_quadratures(self, enable_squared_quadratures: bool=False):
        """ Returns quadrature expectation values <x> and <p> for each motional mode in the state. 
            - fails if there are no motional DOFs 
            - returns lists of <x> and <p>; each list element corresponds to 1 motional mode 
            - list order matches DOF order in the state's basis 
            - option to return <x^2> and <p^2>, the expectation of the squared quadrature operators.
        """ 
        x = []        
        p = []        
        if enable_squared_quadratures:
            x2 = []        
            p2 = []        


        use_partial_trace = False 
        #import time
        #start = time.perf_counter()
        # Either compute partial trace on the density matrix or elevate observable to full space 
        # It may be cheaper to compute the expectation value in the full space 
        if not use_partial_trace :
            full_basis = self.basis

            # For each mode, build the full-space raising and lowering operators  
            for i, mode_i in enumerate(self.basis.motional_modes):
                if not isinstance(mode_i.energy_levels[0], CollectiveMotionalEnergyLevel):
                    raise IonSimError("Quadrature calculation assumes motional mode is in the Fock number state basis.")

                # Enlarge the x and p operators for the current mode 
                Fock_dim = len(mode_i.energy_levels) 
                x_enlarged = full_basis.enlarge_one_dof_matrix(Fock.position(Fock_dim), mode_i) 
                p_enlarged = full_basis.enlarge_one_dof_matrix(Fock.momentum(Fock_dim), mode_i) 

                # Compute expectation values: 
                x.append(self.compute_matrix_observable_expectation(x_enlarged))
                p.append(self.compute_matrix_observable_expectation(p_enlarged))
                if enable_squared_quadratures:
                    x2.append(self.compute_matrix_observable_expectation(x_enlarged @ x_enlarged)) 
                    p2.append(self.compute_matrix_observable_expectation(p_enlarged @ p_enlarged))

        else:
            for i, mode_i in enumerate(self.basis.motional_modes):
                if not isinstance(mode_i.energy_levels[0], CollectiveMotionalEnergyLevel):
                    raise IonSimError("Quadrature calculation assumes motional mode is in the Fock number state basis.")
            
                mode_state = self.motional_state # reset the state  
                # Trace out other modes 
                for j, mode_j in enumerate(self.basis.motional_modes):
                    if i == j:
                        continue 
                    mode_state = mode_state.trace_out_degree_of_freedom(mode_j) 
                assert mode_state is not None
                Fock_dim = len(mode_i.energy_levels) 
                # Compute expectation values: 
                x.append(mode_state.compute_matrix_observable_expectation(Fock.position(Fock_dim)))
                p.append(mode_state.compute_matrix_observable_expectation(Fock.momentum(Fock_dim)))
                if enable_squared_quadratures:
                    x2.append(mode_state.compute_matrix_observable_expectation(Fock.position(Fock_dim) @ Fock.position(Fock_dim)))
                    p2.append(mode_state.compute_matrix_observable_expectation(Fock.momentum(Fock_dim) @ Fock.momentum(Fock_dim)))

        #end = time.perf_counter()
 #        if not use_partial_trace :
 #            print(f"Promoting observable to full space")
 #        else:
 #            print(f"Using partial traces")
 #        print(f"Computing x,p took: {end-start} [s]")
        if enable_squared_quadratures:
            return x, p, x2, p2
        else:
            return x, p
                            

    def compute_wigner_distribution(self, x_grid: Vector, p_grid: Vector): 
        """ Computes W(x,p) the Wigner distribution for each motional mode in the basis; assumes a Fock basis for each mode.
            - requires a specification of the x and p grids as 1-dimensional arrays, determines resolution of Wigner distributions. 
            - returns a list of Wigner distributions -> one for each mode. 
            - returns an empty list if there are no modes in the basis.  
            - assumes the motional modes are in the Fock number state basis  
        """  
        from qutip import Qobj, wigner
        wigner_distributions = []

        # For each mode, trace out all other modes  
        for i, mode_i in enumerate(self.basis.motional_modes):
            if not isinstance(mode_i.energy_levels[0], CollectiveMotionalEnergyLevel):
                raise IonSimError("Wigner distribution calculation assumes motional mode is in the Fock number state basis.")

            mode_state = self.motional_state # reset the state  
            assert mode_state is not None

            for j, mode_j in enumerate(self.basis.motional_modes):
                if i == j:
                    continue 
                mode_state = mode_state.trace_out_degree_of_freedom(mode_j) 
            N_fock = len(mode_i.energy_levels) 
            wigner_distributions.append(wigner(Qobj(mode_state.density_matrix, dims=[[N_fock], [N_fock]]), x_grid, p_grid))

        return wigner_distributions 

    # def transform_to_spin_eigenbasis(self):
