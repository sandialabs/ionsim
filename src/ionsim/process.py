from ionsim.custom_math import trapz_for_matrix
from ionsim.custom_types import Vector, Matrix
from ionsim.noise import Noise
from ionsim.basis import DegreeOfFreedom, Basis, StandardBasis
from ionsim.ionsim_error import IonSimError
from ionsim.hamiltonian import Hamiltonian
from ionsim.dissipator import Lindbladian 
from ionsim.operator import Operator
from ionsim.state import State


import numpy as np
from dataclasses import dataclass, field
from abc import ABC
import scipy 
import typing
from typing import Callable, Dict, List, Sequence, get_type_hints, get_origin, get_args
import inspect 
from functools import reduce 
from icecream import ic

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
            ode_solver: str = 'odeintz',
            **ode_solver_kwargs): # TODO: add an option for initial density matrices for the traced out DoFs.
        """Build a gate by solving the Schrodinger equation for a complete set of initial states.""" 
        if dofs_to_trace_out is not None:
            assert(initial_wavefunctions_for_dofs_to_trace_out is not None)
            assert(len(dofs_to_trace_out) == len(initial_wavefunctions_for_dofs_to_trace_out))
            assert(len(dofs_to_trace_out) == 1) # TODO: generlize for multiple traced out DoFs
            dof_to_trace_out = dofs_to_trace_out[0]
            initial_wavefunction_for_dof_to_trace_out = initial_wavefunctions_for_dofs_to_trace_out[0]
            # TODO: consider if this function should just accept a reduced basis...?

        if dofs_to_trace_out is None:
            reduced_basis = basis
        else:
            reduced_basis = StandardBasis([dof for dof in basis.degrees_of_freedom if dof not in dofs_to_trace_out])

        import time
        final_states = []
        ic(len(reduced_basis.vectors))
        ic(reduced_basis.vectors)
        for vector in reduced_basis.vectors:
            if dofs_to_trace_out is None:
                initial_state = State.from_wavefunction(basis, vector)
            else:
                initial_state = State.from_wavefunction_with_new_component(
                    basis, vector, initial_wavefunction_for_dof_to_trace_out, [dof_to_trace_out]
                )
            ic(len(initial_state.wavefunction))
            # start = time.perf_counter()
            final_states.append(
                initial_state.propagate_using_schrodinger_equation(
                    hamiltonian, duration,
                    ode_solver=ode_solver, **ode_solver_kwargs
                )
            )
            # end = time.perf_counter()
            # ic(f'State propagation took {end-start} seconds.')

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
                if dofs_to_trace_out is None:
                    spin_state = State.from_density_matrix(basis, density_matrix)
                else:
                    spin_state = State.from_density_matrix(
                        basis, density_matrix
                    ).trace_out_degree_of_freedom(dof_to_trace_out)
                supervectors.append(spin_state.supervector)
        process_matrix = np.array(supervectors).T

        return cls(reduced_basis, process_matrix, unitary=unitary)


    @classmethod
    def from_lindbladian(cls, basis: StandardBasis, lindbladian: Lindbladian, duration: float, 
            dofs_to_trace_out: list[DegreeOfFreedom] | None = None,
            initial_density_matrices_for_dofs_to_trace_out: list[State] | None = None,
            lindbladian_time_independent: bool = False, 
            lindbladian_commutes_at_later_times: bool = False, 
            ode_solver: str = 'odeintz',
            **ode_solver_kwargs): # TODO: add an option for initial density matrices for the traced out DoFs.
        """ Build a gate using either the matrix-exponentiated Lindbladian or by solving the Lindblad master equation for a complete set of initial states.
            - optional argument to trace out DOF or project out states. 
            - The gate is built in the reduced or projected basis. 
        """
        # TODO: Include tracing out DOF functionality 
        if dofs_to_trace_out is not None:
            assert(initial_wavefunctions_for_dofs_to_trace_out is not None)
            assert(len(dofs_to_trace_out) == len(initial_wavefunctions_for_dofs_to_trace_out))
            assert(len(dofs_to_trace_out) == 1) # TODO: generlize for multiple traced out DoFs
            dof_to_trace_out = dofs_to_trace_out[0]
            initial_wavefunction_for_dof_to_trace_out = initial_wavefunctions_for_dofs_to_trace_out[0]
            # TODO: consider if this function should just accept a reduced basis...? ==> ECM 03/2026: Yes I think so. 

        if dofs_to_trace_out is None:
            reduced_basis = basis
        else:
            reduced_basis = StandardBasis([dof for dof in basis.degrees_of_freedom if dof not in dofs_to_trace_out])

        # Use general t-dependent, non-commutating Lindbladian method unless user specifies otherwise 
        if lindbladian_time_independent:
            #print(f"Lindbladian is time-independent. Simplifying computation of process matrix via direct matrix exponentiation.")
            # Major simplification for time-independent Lindbladians: Process matrix is simply e^{-L t}
            process_matrix = scipy.linalg.expm(lindbladian.matrix_function(0) * duration)

        elif lindbladian_commutes_at_later_times:
            #print(f"Lindbladian commutes at different times. Integrating Lindbladian directly in time.") 
            # Lindbladian is time dependent but commute with itself at later times: Integrate each element of the lindbladian matrix forward in time from t = 0 to t = duration            
            L_integral, err = quad_vec(lindbladian.matrix_function, 0., duration)

            process_matrix = scipy.linalg.expm(L_integral)

        else:
            #print(f"Default method for generating process matrix from generic time-dependent, non-commutating Lindbladian.")
            # For general lindbladian, time-evolve each |i><j| and then reconstruct process matrix from all d^2 combinations.
            # 1. Create initial density matrices |i><j| for all i,j in the d-dimensional Hilbert space. 
            # 2. Forming |i><j| gives you 1 of the d^2 columns of the process matrix. 

            process_matrix_columns = []
            # When projecting, loop over all vectors in the total basis and then skip the ones that will be zero, i.e. set those cols = zero and skip evolution.   
                # Projection does this redundantly by setting the appropriate parts to zero. But skipping t-evolution saves substantially on computation.
            for i, vector in enumerate(basis.vectors):
                for j, vector_p in enumerate(basis.vectors):
                    # Necessary to do |vector_p > <vector| to get correct basis ordering after projection  
                    initial_state = State.from_density_matrix(basis,  np.outer(vector_p, vector))
        
                    # TODO: Include tracing out DOF functionality 
                    # Time-evolve with Lindbladian, this yields the ij'th column of the process matrix.
         #            if dofs_to_trace_out is None:
         #                initial_state = State.from_wavefunction(basis, vector)
         #            else:
         #                initial_state = State.from_wavefunction_with_new_component(
         #                    basis, vector, initial_wavefunction_for_dof_to_trace_out, [dof_to_trace_out]
         #                )
                    final_state = initial_state.propagate_using_master_equation(lindbladian, duration, ode_solver=ode_solver, **ode_solver_kwargs)
    
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
    process_matrix_function: Callable | None = None
    #parameters: None | dict[str, float] = field(default_factory=dict) 

 #    def __post_init__(self):
 #        # Check that process_matrix_function(*parameter_args) == process_matrix 
 #        parameter_names, arguments = list(self.parameters.keys()), list(self.parameters.values())
 #        if self.process_matrix_function:
 #            if not (self.process_matrix_function(*arguments) == self.process_matrix).all:
 #                raise IonSimError(f"Error, process matrix function and process matrix attributes do not correspond.")


    @classmethod
    def from_gates(cls, gates: list[Gate], noise: Noise | None = None):
        """Build a circuit from a series of gates in the same basis."""
        if any(gate.basis is not gates[0].basis for gate in gates):
            raise IonSimError('All gates in a circuit must be in the same basis.')
        if noise is None or all([noise.parameter_name not in gate.parameters for gate in gates]):
            process_matrix = _combine_process_matrices([gate.process_matrix for gate in gates])
            return cls(gates[0].basis, process_matrix, gates)
        pmats_list = []

        circuit_process_matrix_function = None # default 
        if all(gate.process_matrix_function is None for gate in gates):
            # Compile gate function list (in circuit order) and then reverse by circuit convention  
            gate_functions = []
            for gate in gate:
                gate_functions.append(gate.process_matrix_function)

            # Reverse gate function order by convention (last gate in original list is first gate to apply)  
            gate_functions = gate_functions[::-1]
            circuit_process_matrix_function = Circuit_Process_Matrix_Function_Helper(gate_functions)

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
        return cls(gates[0].basis, process_matrix, gates, circuit_process_matrix_function)

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
            outcome_probabilities.append(np.dot(outcome_op.superbra, propagated_state.supervector).real) 

        return outcome_probabilities


    def build_outcome_probability_function(self, initial_state: State, outcome_operator: Operator) -> Callable:
        """ Returns a function that returns an outcome probability as a function of circuit model parameters """  
        if self.process_matrix_function is None:
            return None 

        @wraps(self.process_matrix_function)
        def outcome_probability_function(*args, **kwargs) -> float:
            circuit_process_matrix = self.process_matrix_function(*args, **kwargs)
            return predict_outcome_probability_from_process_matrix(initial_state, circuit_process_matrix, outcome_operator)
            
        return outcome_probability_function


    def build_outcome_probabilities_function(self, initial_state: State, outcome_operators: list[Operator]) -> Callable:
        """ Returns a function that returns a list of outcome probabilities as a function of circuit model parameters """  
        if self.process_matrix_function is None:
            return [None for _ in range(len(outcome_operators))] 

        # Although a list of functions (each fxn corresponding to an outcome) is more intuitive,  
        #   it is more efficient to evaluate the circuit process matrix once and then loop over outcome operators. 
        @wraps(self.process_matrix_function)
        def outcome_probabilities_function(*args, **kwargs) -> list[float]:
            circuit_process_matrix = self.process_matrix_function(*args, **kwargs)
            propagated_state = initial_state.propagate_using_process_matrix(circuit_process_matrix)
            outcome_probabilities = []
            for operator in outcome_operators:
                outcome_probabilities.append(np.dot(outcome_op.superbra, propagated_state.supervector).real)
            return outcome_probabilities
            
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
    return np.dot(outcome_operator.superbra, propagated_state.supervector).real



#import jax 
#import jax.numpy as jnp 
    
#jax.config.update("jax_enable_x64", True)

#class Circuit_Process_Matrix_Helper():
#@dataclass(frozen=True, eq=False)
class Circuit_Process_Matrix_Function_Helper():
    """ Builds a single process matrix function for a circuit, represented as a composition of gates 
            where each gate is represented by its own gate process matrix function. 

        - This class builds a single callable that returns the process matrix for the circuit. 

        - The class organizes and tracks each gate model input arguments in order to avoid namespace 
            conflicts.  

        - Functions that are repeated are computed once and reused to avoid excess compution.  

        - Includes JAX functionality for derivative computation of the proess matrix function w.r.t. gate parameters
            - requires jax, jaxlib  
    """ 

    def __init__(self, gate_models: Sequence[Callable], separator: str = "__"):
        """ 
            gate_models: a sequence to represent the order of gates applied in the circuit 

            separator: a string used for namespacing parameters within a gate model, e.g. "X_pi2__thetaX"
                refers to the parameter arg thetaX within the gate model function X_pi2.  

        """
        self.gate_sequence = list(gate_models)
        self.separator = seperator 

        self.unique_functions = list({id(f): f for f in self.gate_sequence}.values())

        self._param_map: dict[str, tuple] = {} # namespaced name -> (function name, original name)
        self._name_to_function: dict[str, Callable] = {}
        self._type_hints: dict[str, type] = {}

        self._build_signature()


    def _build_signature(self):
        """ Builds the circuit process matrix function signature """ 
        params = []
        seen_names = set()
        for f in self.unique_functions:
            fname = f.__name__
            if fname in seen_names:
                raise ValueError(f"Duplicate function name '{fname}' -- two distinct functions can't share a name. Rename one or pass explicit names.")
            seen_names.add(fname)
            self._name_to_func[fname] = f

            sig = inspect.signature(f)
            try:
                hints = get_type_hints(f)
            except Exception:
                hints = {} # skip if this fails 
        
            for name, param in sig.parameters.items():
                if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    raise NotImplementedError(f"{fname} uses *args/**kwargs, not supported.")
                namespace = f"{fname}{self.separator}{name}"
                if namespace in self._param_map:
                    raise ValueError(f"Parameter collision in '{namespace}'")
                self._param_map[namespace] = (fname, name)
                if name in hints:
                    self._type_hints[namespace] = hints[name]
                params.append(param.replace(name=namespace, kind=inspect.Parameter.KEYWORD_ONLY))

        self.__signature__ = inspect.Signature(params)


    def check_types(self, **kwargs) -> list[str]:
        """ Returns human readable messages for type-hint vs. actual type mismatch """

        errors = []
        for namespace, value, in kwargs.items():
            expected = self._type_hints.get(namespace)
            if expected is None:
                continue
            if not self._type_matches(value, expected):
                fname, orig_name = self._param_map[namespace]
                expected_repr = getattr(expected, "__name__", str(expected))
                errors.append(f"{namespace} (={value!r}) expected {expected_repr} for {fname}'s '{orig_name}', got {type(value).__name__}")
        return errors 

    def missing_required(self, **kwargs) -> list[str]:
        """ Returns namespaced names of required parameters not supplied"""
        provided = set(kwargs)
        return [name for name, p in self.__signature__.parameters.items() 
                if p.default is inspect.Parameter.empty and name not in provided]


    def __call__(self):
        """ method for Callable behavior """
        missing = self.missing_required(**kwargs)
        if missing:
            by_func: dict[str,list] = {}
            for namespace in missing:
                fname, orig_name = self._param_map[namespace]
                by_func.setdefault(fname, []).append(orig_name)
            details = "; ".join(f"{fname} missing {names}" for fname, names in by_func.items())
            raise TypeError(f"Missing required argument(s): {details}")

        valid_names = set(self.__signature__.parameters)
        unexpected = set(kwargs) - valid_names

        if unexpected:
            raise TypeError(f"Unexpected argument(s): {sorted(unexpected)}. Valid arguments are: {sorted(valid_names)}")
        
        errors_type = self.check_types(**kwargs)
        if errors_type:
            raise TypeError("Type mismatch: " + "; ".join(errors_type)) 

        bound = self.__signature__.bind(**kwargs)
        bound.apply_defaults()

        per_function_kwargs = {fname: {} for fname in self._name_to_func}
        for namespace, value in bound.arguments.items():
            fname, orig_name = self.p_param_map[namespace]
            per_function_kwargs[fname][orig_name] = value

        # call each unique function once and store the result to reuse at every instance of that function 
        results = {fname: f(**per_function_kwargs[fname]) for fname, f in self._name_to_func.items()}

        # Build list of matrices 
        matrices = [results[f.__name__] for f in self.sequence]

        return reduce(lambda g1,g2: g1 @ g2, matrices)





                




# Possible idea: Build a IonSim circuit object from a ParsedCircuit object, requires gate models + basis  
#@dataclass(frozen=True,eq=False)
#class GST_Circuit():
#    """ Class containing IonSim GST Circuit Objects, containing lists of gates"""
#
#    name: str
#    prep_circuit: Circuit  
#    germ_circuit: Circuit 
#    measure_circuit: Circuit 
#    germ_power: int 
#    counts: dict[str, int] | None=None 
#    
#    @property
#    def expanded_circuit(self) -> list[Gate]:
#        """ List of gates, expanded (no germ power included) """
#        return self.prep_circuit.gates + self.germ_circuit.gates * self.germ_power.gates + self.measure_circuit.gates
#
#    def __repr__(self):
#        readable_name = " ".join(repr(gate.name) for gate in self.expanded_gates) or "(empty)"
#        return f"GST_Circuit({gates_readable}, counts={self.counts})"
