import numpy as np
from itertools import product

class Pauli:

    X = np.array(
        [[0, 1],
         [1, 0]],
    )
    Y = np.array(
        [[0, -1j],
         [1j, 0]],
    )
    Z = np.array(
        [[1, 0],
         [0, -1]],
    )
    I = np.array(
        [[1, 0],
         [0, 1]],
    )

    # Attribute for the single-qubit Pauli vector: \sigma = (I, X, Y, Z)
    vector: list[Matrix] = [I, X, Y, Z]


    ''' Raising/lowering operators assume |g> corresponds to row/column 1 and |e> corresponds to row/column 2 '''
    plus = np.array(
        [[0, 0],
         [1, 0]],
    )
    minus = np.array(
        [[0, 1],
         [0, 0]],
    )

    @classmethod
    def product_operators(cls, N_qubits: int) -> list[Matrix]:
        """ Helper function to compute a N-qubit Pauli operators. d = 2^N for N qubits. 
            
            - returns a list of pauli operators. 
            - there are d^2 Pauli operators, each a d x d matrix. 

        """ 
        # Safety checks: 
        if N_qubits <= 0 :
            raise ValueError(f"Number of qubits cannot be negative or zero. Received N_qubits = {N_qubits}.")
            
        if N_qubits == 1:
            return self.vector

        pauli_operators = []
        for operators in product(cls.vector, repeat=N_qubits): 
            # operators are tuples containing the single-qubit Pauli matrices 
            P = operators[0]
            for P_prime in operators[1:]:
                P = np.kron(P, P_prime) 
            pauli_operators.append(P)

        return pauli_operators


class Fock:

    from scipy.special import genlaguerre

    @staticmethod
    def lowering(fock_dimension: int):
        """The lowering operator for a harmonic oscillator."""
        return np.diag([np.sqrt(n+1) for n in range(fock_dimension-1)], k=1)

    @classmethod
    def raising(cls, fock_dimension: int):
        """The raising operator for a harmonic oscillator."""
        return cls.lowering(fock_dimension).conj().T

    @classmethod
    def number(cls, fock_dimension: int):
        return np.diag(np.arange(fock_dimension))

    @staticmethod
    def debye_waller_lowering(fock_dimension: int, lamb_dicke_parameter: float):
        """The lowering operator for a harmonic oscillator."""
        xsq = lamb_dicke_parameter**2
        # dw_fac = lambda n: np.exp(-xsq/2) * (1 + n - n*(n+1)/2 * xsq) / np.sqrt(n+1)
        # dw_fac = lambda n: np.exp(-xsq/2) * np.sqrt(1+n)*(1 - n/2 * xsq)
        from scipy.special import genlaguerre
        dw_fac = lambda n: np.exp(-xsq/2) * genlaguerre(n, 1)(xsq) / np.sqrt(n+1)
        return np.diag([dw_fac(n) for n in range(fock_dimension-1)], k=1)

    @classmethod
    def debye_waller_raising(cls, fock_dimension: int, lamb_dicke_parameter: float):
        """The raising operator for a harmonic oscillator."""
        return cls.debye_waller_lowering(fock_dimension, lamb_dicke_parameter).conj().T

class Unitary:

    @staticmethod
    def R(phi: float, theta: float):
        """A single-qubit rotation on the Bloch sphere."""
        sigma_phi = np.cos(phi) * Pauli.X + np.sin(phi) * Pauli.Y
        return np.exp(1j*theta/2) * (np.cos(theta/2) * Pauli.I - 1j*np.sin(theta/2) * sigma_phi)

    X = R(0, np.pi)
    sqrtX = R(0, np.pi/2)
    Y = R(np.pi/2, np.pi)
    sqrtY = R(np.pi/2, np.pi/2)
    I = R(0, 0)

    @staticmethod
    def MS(phi: float, theta: float):
        """The Molmer-Sorensen entangling gate."""
        sigma_phi = np.cos(phi) * Pauli.X + np.sin(phi) * Pauli.Y
        return np.cos(theta/2) * np.kron(Pauli.I, Pauli.I) - 1j*np.sin(theta/2) * np.kron(sigma_phi, sigma_phi)
