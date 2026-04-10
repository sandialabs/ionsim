import numpy as np
from numpy.linalg import multi_dot
from typing import Callable
from functools import wraps # do I need this?
from scipy.sparse import csr_matrix
from scipy.sparse import kron as skron
from itertools import product

from abc import ABC, abstractmethod
from typing import Sequence
from dataclasses import dataclass
import itertools
from functools import cached_property
import functools as ft
from icecream import ic

from ionsim.ionsim_error import IonSimError
from ionsim.degree_of_freedom import DegreeOfFreedom, AtomicSpin
from ionsim.atomic_internal_energy_level import AtomicInternalEnergyLevel
from ionsim.energy_level import EnergyEigenstate
from ionsim.custom_types import Vector, Matrix
from ionsim.named_operators import Pauli
from ionsim.config import NUMERICAL_EQUIVALENCE_THRESHOLD, NUMERICAL_ERROR_THRESHOLD


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

    def project_superoperator(self, superoperator: Matrix, indices_to_project_into: list[int]) -> Vector: 
        """Project a superoperator to a lower-dimensional subspace"""
        # TODO: Verify that this is correct and we don't need to transpose anything 
        # We know the computational indices for d x d matrix, but what about d^2 x d^2 ?
        superoperator_indices = [i + j*len(self.states)  # column-wise superoperator convention
            for j in indices_to_project_into 
            for i in indices_to_project_into] 
        return superoperator[np.ix_(superoperator_indices, superoperator_indices)]


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

    # TODO: This is from old IonSim and needs to be adapted for new IonSim 
    def compute_choi_jamiolkowski_process_matrix(self, process: Matrix, representation = 'superoperator'):
        """ Return the Choi-Jamiolkowski representation of a quantum process """
        # TODO: Add methods as necessary to accept different representations
        allowed_representations = ['superoperator', 'unitary']
        process = np.array(process)
        if representation == 'unitary':
            process = np.kron(process.conj(), process)
            representation = 'superoperator'
        if representation == 'superoperator':
            # Superoperator is the linear operator acting on vec(rho)
            dimension = int(np.sqrt(process.shape[0]))
            jamiolkowski_matrix = np.zeros([dimension**2, dimension**2], dtype='complex')
            for i in range(dimension**2):
                Ei_vec= np.zeros(dimension**2)
                Ei_vec[i] = 1
                output = unvec(np.dot(process,Ei_vec))
                jamiolkowski_matrix += np.kron(output, unvec(Ei_vec))
            return jamiolkowski_matrix
        else:
            print('Input representation must be one of: ', allowed_representations)





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


@dataclass(frozen=True, eq=False)
class PauliProductBasis(Basis):
    """ A basis in the N-qubit Pauli group, which forms an orthonormal basis for the Hilbert-Schmidt space of d^2 x d^2 operators.  
        - Basis vectors are basis operators in the Pauli group, which span d^2 unique d x d operators. 
        - This basis is over-specified with respect to d x d states.  
    """
    degrees_of_freedom: list[AtomicSpin]  

    def __post_init__(self):
        self._check_if_qubit_basis()


    # TODO: Consider cacheing these vectors to avoid building a list of d^2, dxd matrices every time you call this object.  
    # TODO: The memory cost of storing this will need to be weighed against the time computing these vectors.  
    @property
    def vectors(self) -> list[Vector]:
        """ Normalized basis vectors corresponding to vectorized (column-wise flattened) Pauli operator products: vec(P_i)/sqrt(2^{N}) """
        N = len(self.degrees_of_freedom)
        return [(op.T).flatten()/(2**(0.5*N)) for op in Pauli.product_operators(N)] 

    @property
    def vector_labels(self):
        """ Returns list of labels corresponding to each Pauli product operator basis vectors, e.g. "XIY" for 3 qubits. """  
        # Convention in IonSim is to use the single-qubit pauli vector in the following order: (I, X, Y, Z) 
        single_qubit_pauli_vector = ['I', 'X', 'Y', 'Z'] 
        N = len(self.degrees_of_freedom)
        pauli_op_labels  = ["".join(label) for label in product(single_qubit_pauli_vector, repeat = N)]
 #        for label in product(single_qubit_pauli_vector, repeat=N): 
 #            # operators are tuples containing the single-qubit Pauli matrices 
 #            pauli_op_labels.append("".join(label)) 
        return pauli_op_labels


    @staticmethod
    def label_of_pauli_transfer_matrix_element(self, i: int, j: int):
        """ Returns Pauli operator label corresponding to the R[i,j] for a Pauli transfer matrix R """ 
        return self.vector_labels[i], self.vector_labels[j]


    @staticmethod
    def pauli_to_symplectic(pauli_label: str):
        """ Converts Pauli operator bit string label """ 
        encoding = {'I': (0,0), 'X': (1,0), 'Y': (1,1), 'Z': (0,1)}
        a = [encoding[p][0] for p in pauli_label]
        b = [encoding[p][1] for p in pauli_label]
        return np.array(a + b, dtype=int)

    @property
    def walsh_hadamard_transformation_matrix(self, include_normalization: bool=True) -> Matrix:
        """ Walsh-Hadamard transformation matrix for transforming between Pauli transfer matrix (PTM) 
             eigenvalues and Pauli channel error rates. 

                    W_{m,n} = (-1)^phi(m, n)   , parity phi(m,n) = 0 if P_m, P_n commute and 1 if they anticommute.  
            
            such that lambda = W @ q. 

            - lambda is a vector of PTM eigenvalues (fidelities), describing how well a Pauli observable is preserved in the process.
            - q is a vector of Pauli channel error rates. 

        """

        size = len(self.vector_labels) # d^2        
        W = np.zeros((size, size))
 
 #        for m, Pm in enumerate(self.vectors):
 #            for n, Pn in enumerate(self.vectors):
 #                # Expensive approach: phi(m,n) computed by trace operation 
 #                W[m,n] = (-1)**self.symplectic_inner_product(Pm, Pn) 
 #                #W[m,n] = np.real( np.trace(Pm @ Pn @ Pm @ Pn) ) # always either +1 or -1  

        # Convert pauli labels to binary representation 
        symplectic_encodings = np.array([self.pauli_to_symplectic(label) for label in self.vector_labels])
        N = len(self.degrees_of_freedom)

        A, B = symplectic_encodings[:, :N], symplectic_encodings[:, N:]

        # Compute phi(m,n) via matrix multiplication mod 2         
        Phi = (A @ B.T + B @ A.T) % 2  # matrix of integer -1, +1 values 

        W = ((-1)**Phi).astype(float)
        if include_normalization:
            # W is normalized by d^2
            W *= 1./float(size)

        return W 


 #    def compute_basis_coefficients(self, superoperator: Matrix) -> list[float]:
 #        """ Computes Pauli-product basis coefficient for an input superoperator via c_i = Tr[P_{i} A ]. 
 #            - P_{i} represents the normalized i'th basis vector of the Pauli product basis.  
 #            - A is the input superoperator  
 #            - Basis coefficients are defined via A = sum_{i} c_i P_i over all d^2 Pauli operators.
 #        """  
 #        # TODO: add this computation 
 #        assert superoperator.shape = (len(vectors), len(vectors))
 #        return [np.trace(Pauli_operator @ superoperator for Pauli_operator in self.vectors) ]
 #
 #    def build_superoperator_from_coefficients(self, coefficients: list[float] | Vector[float] ):
 #        """ Computes a superoperator from its pauli-product operator expansion coefficients: A = sum_{i} c_i P_i  
 #            - P_{i} represents the i'th basis vector of the Pauli product basis.  
 #            - A is the input superoperator with basis coefficients {c_i}  
 #            - Basis coefficients are defined via A = sum_{i} c_i P_i over all d^2 Pauli operators.
 #        """  
 #        # TODO: add this computation 
 #        assert len(coefficients) == len(vectors) # d^2 necessary coefficients 
 #        superoperator = np.zeros((len(coefficients), len(coefficients), dtype=complex) # should it be real floats only? 
 #        #for 
 #        return [np.trace(Pauli_operator @ superoperator for Pauli_operator in self.vectors) ]

 #    @staticmethod
 #    def superoperator_to_pauli_transfer_matrix(self, superoperator: Matrix) -> Matrix:
    def superoperator_to_pauli_transfer_matrix(self, superoperator: Matrix, superoperator_basis: StandardBasis) -> Matrix:
        """ Converts a superoperator to a Pauli Transfer Matrix via

            R = U S U^{dagger}

            R is the Pauli Transfer Matrix  
            U is a unitary change-of-basis matrix ,defined as U_(mu, :) = vec(P_{mu})*  
            S is the input superoperator 
        """
        if not isinstance(superoperator_basis, StandardBasis):
            raise IonSimError(f"Gate input should be in the Standard Basis. Other transformations are not yet implemented in IonSim.") 
        assert superoperator.shape == (len(self.vectors), len(self.vectors))
        # Get change of basis matrix 
        U = np.array(self.vectors).conj() 

        # If S represents a completely positive, trace-preserving (CPTP) map, R will be purely real. 
        pauli_transfer_matrix = np.real(U @ superoperator @ (U.T).conj() )

        # TODO: test and verify these formulae 
        return pauli_transfer_matrix 

    def convert_gate_to_pauli_basis(self, gate: Gate) -> Gate:
        """ Converts a Gate object to the Pauli product basis """ 
        if not isinstance(gate.basis, StandardBasis):
            raise IonSimError(f"Gate input should be in the Standard Basis. Other transformations are not yet implemented in IonSim.") 

        if gate.process_matrix_function:
            @wraps(gate.process_matrix_function)
            def ptm_function(*args, **kwargs):
                return self.superoperator_to_pauli_transfer_matrix(gate.process_matrix_function(*args, **kwargs), gate.basis)
            return Gate.from_process_matrix_function(basis = self, process_matrix_function = ptm_function, parameters = gate.parameters) 
        else:
            pauli_transfer_matrix = self.superoperator_to_pauli_transfer_matrix(gate.process_matrix, gate.basis)
            return Gate(basis = self, process_matrix = pauli_transfer_matrix) 


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
