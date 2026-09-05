#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************
import numpy as np
from dataclasses import dataclass, field
from abc import ABC
import scipy 
from scipy.integrate import quad_vec
import typing
from typing import Callable, Dict, List, Sequence, get_type_hints, get_origin, get_args
import inspect 
import functools
from functools import reduce, wraps 
from icecream import ic

from ionsim.custom_math import trapz_for_matrix
from ionsim.custom_types import Vector, Matrix
from ionsim.energy_level import EnergyEigenstate
from ionsim.noise import Noise
from ionsim.basis import DegreeOfFreedom, Basis, StandardBasis, PauliProductBasis
from ionsim.ionsim_error import IonSimError
from ionsim.hamiltonian import Hamiltonian
from ionsim.lindbladian import Lindbladian 
from ionsim.state import State
from ionsim.operator import Operator
from ionsim.ionsim_error import IonSimError
from ionsim.config import NUMERICAL_EQUIVALENCE_THRESHOLD

@dataclass(frozen=True, eq=False)
class Process(ABC): 
    """A quantum process represented in a basis of states."""
    basis: Basis
    process_matrix: Matrix
        
    def compute_process_fidelity(self, target_process_matrix: Matrix):
        """Compute the process fidelity with respect to a target process matrix."""
        total = 0
        for basis_state_vector in np.eye(len(self.process_matrix), dtype='complex'):
            # TODO: is dtype='complex' necessary here? When is it?
            final_state_vector = self.process_matrix.dot(basis_state_vector)
            target_state_vector = target_process_matrix.dot(basis_state_vector)
            total += np.dot(target_state_vector.conj().T, final_state_vector).real
        return total/len(self.process_matrix)

@dataclass(frozen=True, eq=False)
class Gate(Process):
    """A quantum gate represented in a basis of states."""
    process_matrix_function: Callable | None = None
    parameters: dict[str, float] = field(default_factory=dict)

    unitary: Matrix | None = None

    def __post_init__(self):
        # Check that process_matrix_function(*parameter_args) == process_matrix 
        parameter_names, arguments = list(self.parameters.keys()), list(self.parameters.values())
        if self.process_matrix_function:
            if not (self.process_matrix_function(*arguments) == self.process_matrix).all:
                raise IonSimError(f"Error, process matrix function and process matrix attributes do not correspond.")


    @classmethod #TODO: let default target_dofs be all degrees of freedom
    def from_unitary(cls, basis: Basis, unitary: Matrix, target_dofs: list[DegreeOfFreedom]):
        """Build a gate from a unitary-gate matrix."""
        full_unitary = basis.enlarge_matrix(unitary, target_dofs)
        process_matrix = basis.compute_superoperator_from_unitary_operator(full_unitary)
        return cls(basis, process_matrix, unitary=full_unitary)

    @classmethod
    def from_unitary_function(cls, basis: Basis, unitary_function: Callable,
            parameters: dict[str, float], target_dofs: list[DegreeOfFreedom], noise: Noise | None = None):
        """Build a gate from a unitary-gate function and its arguments."""
        parameter_names, arguments = list(parameters.keys()), list(parameters.values())
        full_function = basis.enlarge_matrix_function(unitary_function, target_dofs)
        process_matrix_function = basis.create_superoperator_function_from_unitary_operator_function(full_function)
        if noise is not None:
            noisy_parameter_index = parameter_names.index(noise.parameter_name)
            process_matrix_function = noise.add_noise_to_matrix_function(process_matrix_function, noisy_parameter_index)
        return cls(basis, process_matrix_function(*arguments), process_matrix_function, parameters)


    # TODO: Should we extend this to take more than 1 noise parameter? 
    @classmethod
    def from_process_matrix_function(cls, basis: Basis, process_matrix_function: Callable,
            parameters: dict[str, float], noise: Noise | None = None):
        """Build a gate from a process-matrix function and its arguments."""
        parameter_names, arguments = list(parameters.keys()), list(parameters.values())
        if noise is None or noise.parameter_name not in parameter_names:
            return cls(basis, process_matrix_function(*arguments), process_matrix_function, parameters)
        noisy_parameter_index = parameter_names.index(noise.parameter_name)
        process_matrix_function = noise.add_noise_to_matrix_function(process_matrix_function, noisy_parameter_index)
        return cls(basis, process_matrix_function(*arguments), process_matrix_function, parameters)


    @classmethod
    def from_hamiltonian_function(cls, basis: StandardBasis, hamiltonian_function: Callable, duration: float,  
            parameters: dict[str, float], target_dofs: list[DegreeOfFreedom], noise: Noise | None = None):
        """ Build a gate from a hamiltonian function and its arguments."""
        parameter_names, arguments = list(parameters.keys()), list(parameters.values())

        @wraps(hamiltonian_function)
        def process_matrix_function(*args, **kwargs):
            gate = cls.from_hamiltonian(basis, hamiltonian_function(*args, **kwargs))
            return gate.process_matrix

        if noise is None or noise.parameter_name not in parameter_names:
            noisy_parameter_index = parameter_names.index(noise.parameter_name)
            process_matrix_function = noise.add_noise_to_matrix_function(process_matrix_function, noisy_parameter_index)

        return cls(basis, process_matrix_function(*arguments), process_matrix_function, parameters)

    @classmethod
    def from_hamiltonian(cls, basis: StandardBasis, hamiltonian: Hamiltonian, duration: float,
            dofs_to_trace_out: list[DegreeOfFreedom] | None = None,
            initial_wavefunctions_for_dofs_to_trace_out: list[Vector] | None = None,
            projection_info: dict[StandardBasis, list[EnergyEigenstate]] | None = None, 
            ode_solver: str = 'odeintz',
            **ode_solver_kwargs): # TODO: add an option for initial density matrices for the traced out DoFs.
        """ Build a gate by solving the Schrodinger equation for a complete set of initial states.

            - optional argument to trace out DOF or project out states. 
            - for projection info, specify a dictionary with keys 'basis' : StandardBasis & 'states to project' : list[EnergyEigenstate]
            - The gate is built in the reduced or projected basis. 

        """ 
        # TODO: reconcile projection & tracing out and what the final basis is  
        if dofs_to_trace_out is not None:
            assert(initial_wavefunctions_for_dofs_to_trace_out is not None)
            assert(len(dofs_to_trace_out) == len(initial_wavefunctions_for_dofs_to_trace_out))
            assert(len(dofs_to_trace_out) == 1) # TODO: generlize for multiple traced out DoFs
            dof_to_trace_out = dofs_to_trace_out[0]
            initial_wavefunction_for_dof_to_trace_out = initial_wavefunctions_for_dofs_to_trace_out[0]
            # TODO: consider if this function should just accept a reduced basis...? ==> ECM 03/2026: Yes I think so. 


        # TODO: Consolidate tracing out and projection methods for building a process matrix from hamiltonian in an enlarged hilbert space
        if projection_info is None:
            if dofs_to_trace_out is None:
                reduced_basis = basis
            else:
                reduced_basis = StandardBasis([dof for dof in basis.degrees_of_freedom if dof not in dofs_to_trace_out])
        else:
            unwanted_state_indices = [basis.states.index(state) for state in projection_info['states to project out']] 
            computational_indices = [i for i in range(len(basis.states)) if i not in unwanted_state_indices] 
            if dofs_to_trace_out is None:
                reduced_basis = projection_info['new basis'] 
            else:
                raise IonSimError("Tracing out DOFs & projecting out states is not yet supported in this function.") 

        final_states = []
        for i, vector in enumerate(basis.vectors):
            if projection_info and (i in unwanted_state_indices):
                # Skip basis states that we will ultimately project out. 
                pass 
            else:
                if dofs_to_trace_out is None:
                    initial_state = State.from_wavefunction(basis, vector)
                else:
                    initial_state = State.from_wavefunction_with_new_component(
                        basis, vector, initial_wavefunction_for_dof_to_trace_out, [dof_to_trace_out]
                    )
                final_states.append(
                    initial_state.propagate_using_schrodinger_equation(
                        hamiltonian, duration,
                        ode_solver=ode_solver, **ode_solver_kwargs
                    )
                )
                # At the end of each evolution, project out the unwanted states.  
                # It is equivalent to do the projection at this stage (before building the process matrix) 
                #  compared to projecting the full process matrix. 
                if projection_info: 
                    final_states[-1] = final_states[-1].project_out_states(projection_info['new basis'], projection_info['states to project out']) 

        # TODO: how can we do multiprocessing outside of main?
        # from concurrent.futures import ProcessPoolExecutor
        # def propagate(vector):
        #     if traced_out_dofs is None:
        #         initial_state = State.from_wavefunction(basis, vector)
        #     else:
        #         # wavefunction = basis.build_wavefunction(vector, traced_out_initial_wavefunction, [traced_out_dof])
        #         # TODO: crate the method above and use instead of line below
        #         wavefunction = np.kron(vector, traced_out_initial_wavefunction)
        #         initial_state = State.from_wavefunction(basis, wavefunction)
        #     initial_state.propagate_using_schrodinger_equation(hamiltonian, duration)
        # with ProcessPoolExecutor() as executor:
        #     results = executor.map(propagate, reduced_basis.vectors)
        # final_states = list(results)


        if dofs_to_trace_out is None:
            final_wavefunctions = [fs.wavefunction for fs in final_states]
            unitary = np.array(final_wavefunctions).T
        else:
            unitary = None

        supervectors = []
        for final_state_p in final_states:
            for final_state in final_states: # iterate rows with inner loop for column-stacked supervectors
                density_matrix = np.outer(final_state.wavefunction, final_state_p.wavefunction.conj().T)
                if dofs_to_trace_out is None and projection_info is None:
                    spin_state = State.from_density_matrix(basis, density_matrix)
                else:
                    if dofs_to_trace_out is None:
                        # Set the reduced basis to the new basis specified by the projection 
                        reduced_basis = projection_info['new basis']        
                        spin_state = State.from_density_matrix(reduced_basis, density_matrix)
                    else:
                        spin_state = State.from_density_matrix(
                            basis, density_matrix
                        ).trace_out_degree_of_freedom(dof_to_trace_out)
                supervectors.append(spin_state.supervector)
 #=======
 #                if dofs_to_trace_out is None:
 #                    reduced_state = State.from_density_matrix(basis, density_matrix)
 #                else:
 #                    reduced_state = State.from_density_matrix(
 #                        basis, density_matrix
 #                    ).trace_out_degree_of_freedom(dof_to_trace_out)
 #                supervectors.append(reduced_state.supervector)
 #>>>>>>> main
        process_matrix = np.array(supervectors).T

        return cls(reduced_basis, process_matrix, unitary=unitary)


    @classmethod
    def from_lindbladian(cls, basis: StandardBasis, lindbladian: Lindbladian, duration: float, 
            dofs_to_trace_out: list[DegreeOfFreedom] | None = None,
            initial_density_matrices_for_dofs_to_trace_out: list[State] | None = None,
            projection_info: dict | None = None,
            lindbladian_time_independent: bool = False, 
            lindbladian_commutes_at_later_times: bool = False, 
            ode_solver: str = 'odeintz',
            **ode_solver_kwargs): # TODO: add an option for initial density matrices for the traced out DoFs.
        # TODO: reconcile projection &  tracing out and what the final basis is  
        """ Build a gate using either the matrix-exponentiated Lindbladian or by solving the Lindblad master equation for a complete set of initial states.
        
            - optional argument to trace out DOF or project out states. 
            - for projecting, specify a dictionary with keys 'basis' : StandardBasis & 'states to project' : list[EnergyEigenstate]
            - The gate is built in the reduced or projected basis. 
        """
        if dofs_to_trace_out is not None:
            raise NotImplementedError("Tracing out DOFs from a larger subspace not yet implemented.")
            assert(initial_wavefunctions_for_dofs_to_trace_out is not None)
            assert(len(dofs_to_trace_out) == len(initial_wavefunctions_for_dofs_to_trace_out))
            assert(len(dofs_to_trace_out) == 1) # TODO: generlize for multiple traced out DoFs
            dof_to_trace_out = dofs_to_trace_out[0]
            initial_wavefunction_for_dof_to_trace_out = initial_wavefunctions_for_dofs_to_trace_out[0]
 #<<<<<<< HEAD
 #            # TODO: consider if this function should just accept a reduced basis...? ==> ECM 03/2026: Yes I think so. 
 #
 #
 #        if projection_info is None:
 #            if dofs_to_trace_out is None:
 #                reduced_basis = basis
 #            else:
 #                reduced_basis = StandardBasis([dof for dof in basis.degrees_of_freedom if dof not in dofs_to_trace_out])
 #        else:
 #            unwanted_state_indices = [basis.states.index(state) for state in projection_info['states to project out']] 
 #            computational_indices = [i for i in range(len(basis.states)) if i not in unwanted_state_indices] 
 #            if dofs_to_trace_out is None:
 #                reduced_basis = projection_info['new basis'] 
 #                #reduced_basis = basis 
 #            else:
 #                raise IonSimError("Tracing out DOFs & projecting out states is not yet supported in this function.") 
 #
 #        # Use general t-dependent, non-commutating Lindbladian method unless user specifies otherwise 
 #        if lindbladian_time_independent:
 #            print(f"Lindbladian is time-independent. Simplifying computation of process matrix via direct matrix exponentiation.")
 #            # Major simplification for time-independent Lindbladians: Process matrix is simply e^{-L t}
 #            process_matrix = scipy.linalg.expm(lindbladian.matrix_function(0) * duration)
 #
 #            if projection_info:
 #                process_matrix = basis.project_superoperator(process_matrix, computational_indices) 
 #
 #        elif lindbladian_commutes_at_later_times:
 #            print(f"Lindbladian commutes at different times. Integrating Lindbladian directly in time.") 
 #            # Integrate each element of the lindbladian matrix forward in time from t = 0 to t = duration            
 #            L_integral, err = quad_vec(lindbladian.matrix_function, 0., duration)
 #
 #            process_matrix = scipy.linalg.expm(L_integral)
 #            if projection_info:
 #                process_matrix = basis.project_superoperator(process_matrix, computational_indices) 
 #
 #        else:
 #            print(f"Default method for generating process matrix from generic time-dependent, non-commutating Lindbladian.")
 #            # For general lindbladian, time-evolve each |i><j| and then reconstruct process matrix from all d^2 combinations.
 #            # 1. Create initial density matrices |i><j| for all i,j in the d-dimensional Hilbert space. 
 #            # 2. Forming |i><j| gives you 1 of the d^2 columns of the process matrix. 
 #=======

        # Check if building the gate requires projection or tracing out from a larger Hilbert space  
        if projection_info is None:
            if dofs_to_trace_out is None:
                reduced_basis = basis
            else:
                reduced_basis = StandardBasis([dof for dof in basis.degrees_of_freedom if dof not in dofs_to_trace_out])
        else:
            unwanted_state_indices = [basis.states.index(state) for state in projection_info['states to project out']] 
            computational_indices = [i for i in range(len(basis.states)) if i not in unwanted_state_indices] 
            if dofs_to_trace_out is None:
                reduced_basis = projection_info['new basis'] 
                #reduced_basis = basis 
            else:
                raise NotImplementedError("Tracing out DOFs & projecting out states is not yet supported in this function.") 
 #        if dofs_to_trace_out is None:
 #            reduced_basis = basis
 #        else:
 #            reduced_basis = StandardBasis([dof for dof in basis.degrees_of_freedom if dof not in dofs_to_trace_out])

        # Use general t-dependent, non-commutating Lindbladian method unless user specifies otherwise 
        if lindbladian_time_independent:
            # Major simplification for time-independent Lindbladians: Process matrix is simply e^{-L t}
            process_matrix = scipy.linalg.expm(lindbladian.matrix_function(0) * duration)

        elif lindbladian_commutes_at_later_times:
            # Lindbladian is time dependent but commutes with itself at later times: Integrate the lindbladian matrix forward in time from t = 0 to t = duration            
            L_integral, err = quad_vec(lindbladian.matrix_function, 0., duration)
            process_matrix = scipy.linalg.expm(L_integral)
        else:
            # For general lindbladian, time-evolve each |i><j| and then reconstruct process matrix from all d^2 combinations.
            # 1. Create initial density matrices |i><j| for all i,j in the d-dimensional Hilbert space. 
            # 2. Forming |i><j| gives you 1 of the d^2 columns of the process matrix. 
            if dofs_to_trace_out is not None:
                raise NotImplementedError(f"Building a gate on a subspace from a Lindbladian on a larger Hilbert space is not yet implemented.")

            process_matrix_columns = []
            # When projecting, loop over all vectors in the total basis and then skip the ones that will be zero, i.e. set those cols = zero and skip evolution.   
                # Projection does this redundantly by setting the appropriate parts to zero. But skipping t-evolution saves substantially on computation.
            for i, vector in enumerate(basis.vectors):
                for j, vector_p in enumerate(basis.vectors):
                    if projection_info is not None and ((i in unwanted_state_indices) or (j in unwanted_state_indices)):
                        # Skip pure non-computational basis states, e.g. Rydberg or Raman states  
                        pass 
                    else:
                        # Necessary to do |vector_p > <vector| to get correct basis ordering after projection  
                        initial_state = State.from_density_matrix(basis, np.outer(vector_p, vector))
    
                        # TODO: Include tracing out DOF functionality 
                        # Time-evolve with Lindbladian, this yields the ij'th column of the process matrix.
     #                    if dofs_to_trace_out is None:
     #                        initial_state = State.from_wavefunction(basis, vector)
     #                    else:
     #                        initial_state = State.from_wavefunction_with_new_component(
     #                            basis, vector, initial_wavefunction_for_dof_to_trace_out, [dof_to_trace_out]
     #                        )
                        final_state = initial_state.propagate_using_master_equation(lindbladian, duration, ode_solver=ode_solver, **ode_solver_kwargs)

                        if projection_info is not None: 
                            final_state = final_state.project_out_states(projection_info['new basis'], projection_info['states to project out']) 

                        # Supervector of final state gives you 1 column of the process matrix  
                        process_matrix_columns.append(final_state.supervector) 
    
            process_matrix = np.array(process_matrix_columns).T # tranpose ensures column behavior  

        return cls(reduced_basis, process_matrix, unitary=None)

    @classmethod
    def from_lindbladian_function(cls, basis: StandardBasis, lindbladian_function: Callable, duration: float,  
            parameters: dict[str, float], noise: Noise | None = None, lindbladian_time_independent: bool=False, 
            lindbladian_commutes_at_later_times: bool = False): 
        """ Build a gate from a hamiltonian function and its arguments."""
        parameter_names, arguments = list(parameters.keys()), list(parameters.values())

        @wraps(lindbladian_function)
        def process_matrix_function(*args, **kwargs):
            gate = cls.from_lindbladian(basis, lindbladian_function(*args, **kwargs), duration)
            return gate.process_matrix

        if noise is None or noise.parameter_name not in parameter_names:
            noisy_parameter_index = parameter_names.index(noise.parameter_name)
            process_matrix_function = noise.add_noise_to_matrix_function(process_matrix_function, noisy_parameter_index, 
                lindbladian_time_independent = lindbladian_time_independent, lindbladian_commutes_at_later_times = lindbladian_commutes_at_later_times)

        return cls(basis, process_matrix_function(*arguments), process_matrix_function, parameters)

    def compute_pauli_error_rates(self) -> dict[str, float]:
        """ Computes Pauli channel error rates, returned in a dictionary with entries (channel name, error rate) """ 
        # Basis safety checks:  
        basis = self.basis
        if not isinstance(basis, PauliProductBasis):
            # Create Pauli product basis 
            pauli_group_basis = PauliProductBasis(self.basis.degrees_of_freedom)

            # TODO: Consider automatically converting the gate to standard basis, then to Pauli basis 
            if isinstance(basis, StandardBasis):
                pauli_transfer_matrix = pauli_group_basis.superoperator_to_pauli_transfer_matrix(self.process_matrix, basis)
            else: 
                raise IonSimError(f"Gate must be in the standard basis or pauli group basis to compute Pauli error rates.")
        else:
            pauli_group_basis = self.basis 
            pauli_transfer_matrix = self.process_matrix # pointer assigment  

        # Extract error channel rate from Pauli transfer matrix for each pauli group operator. 
        # Walsh-Hadamard transform relates eigenvalues of PTM to to error rates in Pauli channel representation. 
        # TODO: Add option to go beyond Pauli twirled approximation (get eigs of PTM and ensure consistent basis state ordering)
        error_rates = pauli_group_basis.walsh_hadamard_transformation_matrix @ np.diag(pauli_transfer_matrix)
        return dict(zip(pauli_group_basis.vector_labels, error_rates)) 
 
    # Putting this method here (in process.py) instead of basis.py avoids circular import issue  
    def convert_to_pauli_basis(self):
        """ Converts a Gate object to the Pauli Product basis. Returns a Gate object """ 
        if isinstance(self.basis, PauliProductBasis):
            return self

        if not isinstance(self.basis, StandardBasis):
            raise IonSimError(f"Gate input should be in the Standard Basis. Other transformations are not yet implemented in IonSim.") 

        # Create a PauliProductbasis 
        spins = self.basis.degrees_of_freedom 
        pauli_basis = PauliProductBasis(spins) # will automatically check that DOFs are qubits only 

        if self.process_matrix_function:
            @wraps(self.process_matrix_function)
            def ptm_function(*args, **kwargs):
                return pauli_basis.superoperator_to_pauli_transfer_matrix(self.process_matrix_function(*args, **kwargs), self.basis)
            return Gate.from_process_matrix_function(basis = pauli_basis, process_matrix_function = ptm_function, parameters = self.parameters) 
        else:
            pauli_transfer_matrix = pauli_basis.superoperator_to_pauli_transfer_matrix(self.process_matrix, self.basis)
            return Gate(basis = pauli_basis, process_matrix = pauli_transfer_matrix) 

# @dataclass(frozen=True, eq=False)
# class PauliGate(Gate):
#     """A quantum gate in the z-Pauli spin basis.""" # TODO: Should we say "qubit basis" instead?

#     # TODO: check if basis is a z-Pauli spin basis.
#     # TODO: does this class require a StandardBasis input?

#     @staticmethod
#     def get_unitary(name: str):
#         """Get a unitary-gate matrix from its name."""
#         return _UNITARY_GATES[name]

#     @classmethod
#     def from_named_unitary(cls, basis: Basis, name: str, *args, **kwargs):
#         """Build a gate from the name of a unitary gate."""
#         unitary = cls.get_unitary(name)
#         return cls.from_unitary(basis, unitary, *args, **kwargs)

#     @staticmethod
#     def get_unitary_function(name: str):
#         """Get a unitary-gate function from its name."""
#         return _UNITARY_GATE_FUNCTIONS[name]

#     @classmethod
#     def from_named_unitary_function(cls, basis: Basis, name: str, *args, **kwargs):
#         """Build a gate from the name of a unitary-gate function."""
#         unitary_function = cls.get_unitary_function(name)
#         return cls.from_unitary_function(basis, unitary_function, *args, **kwargs)


@dataclass(frozen=True, eq=False)
class Circuit(Process):
    """A quantum circuit (i.e., a series of gates) in a basis of states."""
    gates: list[Gate]

    @classmethod
    def from_gates(cls, gates: list[Gate], noise: Noise | None = None):
        """Build a circuit from a series of gates in the same basis."""
        if any(gate.basis is not gates[0].basis for gate in gates):
            raise IonSimError('All gates in a circuit must be in the same basis.')

        if len(gates) == 0:
            raise IonSimError(f"List of gates must not be empty, received: {gates}.")

        deterministic = (noise is None) or all([noise.parameter_name not in gate.parameters for gate in gates])
        if deterministic: 
            process_matrix = _combine_process_matrices([gate.process_matrix for gate in gates])
            return cls(gates[0].basis, process_matrix, gates, circuit_process_matrix_function)
        pmats_list = []

        for gate in gates:
            if gate.process_matrix_function is not None and noise.parameter_name in gate.parameters:
                arguments = np.array(list(gate.parameters.values()))
                vec = np.array([1 if noise.parameter_name == name else 0 for name in gate.parameters])
                pmats = [gate.process_matrix_function(*list(arguments + darg * vec)) for darg in noise.domain_arguments]
            else:
                pmats = [gate.process_matrix for darg in noise.domain_arguments]
            pmats_list.append(pmats)
        new_pmats_list = [[pmats[i] for pmats in pmats_list] for i in range(len(pmats_list[0]))]
        process_mats = [_combine_process_matrices(ps) for ps in new_pmats_list]
        probs = [noise.probability_density_function(darg) for darg in noise.domain_arguments]
        ys = np.array([p * chi for p, chi in zip(probs, process_mats)])
        process_matrix = trapz_for_matrix(ys, noise.domain_arguments) 
        return cls(gates[0].basis, process_matrix, gates)

    def predict_outcome_probability(self, initial_state: State, outcome_operator: Operator) -> float:
        """ Computes a probability of observing an outcome when applying the circuit to a state. 
            
            Outcome is specified as a POVM projector operator, e.g. |0><0| 
        
        """ 
        return predict_outcome_probability_from_process_matrix(initial_state, self.process_matrix, outcome_operator)

    def predict_outcome_probabilities(self, initial_state: State, outcome_operators: list[Operator]) -> list[float]:
        """ Computes a list of probabilities of observing outcomes when applying the circuit to a state. 
            
            Outcomes are specified as a list of projector operators, e.g. [|0><0|, |1><1|]. 
        
        """ 
        outcome_probabilities = []
        # It is more efficient to evaluate the circuit's action on the state ONCE and then loop over outcome operators. 

        # Propagate the init state using the circuit 
        propagated_state = initial_state.propagate_using_process_matrix(self.process_matrix) 

        # Represent state and outcome operators in superket/superbra form 
        for outcome_op in outcome_operators:
            outcome_probabilities.append((outcome_op.superbra @ propagated_state.supervector).real) 

        return np.array(outcome_probabilities)


    def build_outcome_probability_function(self, initial_state: State, outcome_operator: Operator) -> Callable:
        """ Returns a function that returns an outcome probability as a function of circuit model parameters """  

        def scalar_function(circuit_process_matrix):
            return predict_outcome_probability_from_process_matrix(initial_state, circuit_process_matrix, outcome_operator)

        @wraps(self.process_matrix_function)
        def outcome_probability(**kwargs):
            """Outcome probability given a circuit acted on an initial state."""
            return scalar_function(self.process_matrix_function(**kwargs))
        outcome_probability_function.scalar_function = scalar_function # Needed by jax for gradient / derivative work 
        outcome_probability_function.process_matrix_function = self.process_matrix_function 
        return outcome_probability_function

    def build_outcome_probabilities_function(self, initial_state: State, outcome_operators: list[Operator]) -> Callable:
        """ Returns a function that returns a vector of outcome probabilities as a function of circuit model parameters """  
        if self.process_matrix_function is None:
            return [None] * len(outcome_operators)

        outcome_matrix = np.vstack([outcome_op.superbra for outcome_op in outcome_operators])
        def vector_function(circuit_process_matrix):
            return predict_outcome_probabilities_from_process_matrix(initial_state, circuit_process_matrix, outcome_matrix)

        def outcome_probabilities_function(**kwargs):
            return vector_function(self.process_matrix_function(**kwargs)) 

        outcome_probabilities_function.__signature__ = self.process_matrix_function.__signature__
        outcome_probabilities_function.__name__ = "outcome_probabilities" 
        outcome_probabilities_function.__doc__ = "Outcome probabilities given a circuit acted on an initial state.\n"
        outcome_probabilities_function.vector_function = vector_function # Needed by jax for jacobian 
        outcome_probabilities_function.process_matrix_function = self.process_matrix_function 
        return outcome_probabilities_function

        
def _combine_process_matrices(process_matrices: list[Matrix]):
    """Combine a series of process matrices (in chronological order) into a single process matrix for the whole circuit."""
    if len(process_matrices) == 1:
        return process_matrices[0]
    else:
        return np.linalg.multi_dot(process_matrices[::-1])


def predict_outcome_probability_from_process_matrix(initial_state: State, process_matrix: Matrix, outcome_operator: Operator) -> float:
    """ Predicts the outcome of a process matrix on a state after measurement/projection <==> outcome operator """   
    propagated_state = initial_state.propagate_using_process_matrix(process_matrix)
    # Using @ operator facilitates jax compatibility; np.dot does not 
    return (outcome_operator.superbra @ propagated_state.supervector).real  

def predict_outcome_probabilities_from_process_matrix(initial_state: State, process_matrix: Matrix, outcome_matrix: Matrix) -> Vector:
    """ Predicts the probabilities of outcomes of a process matrix on a state after measurement/projection <==> outcome operator """   
    propagated_state = initial_state.propagate_using_process_matrix(process_matrix)
    # Using @ operator facilitates jax compatibility; np.dot does not 
    return (outcome_matrix @ propagated_state.supervector).real  

