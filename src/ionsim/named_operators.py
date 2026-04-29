import numpy as np

from icecream import ic

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

    ''' Raising/lowering operators assume |g> corresponds to row/column 1 and |e> corresponds to row/column 2 '''
    plus = np.array(
        [[0, 0],
         [1, 0]],
    )
    minus = np.array(
        [[0, 1],
         [0, 0]],
    )

    ''' Projectors for single-qubit basis states: |0><0| and |1><1| ''' 
    projector_0 = np.array(
        [[1, 0],
         [0, 0]],
    )

    projector_1 = np.array(
        [[0, 0],
         [0, 1]],
    )


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
        """A single-qubit rotation on the XY plane of the Bloch sphere."""
        sigma_phi = np.cos(phi) * Pauli.X + np.sin(phi) * Pauli.Y
        # TODO: Resolve --> why is there overall phase of exp(i theta / 2)? Usually R(phi, theta) is exp[-i (theta/2) sigma_phi ]  
        return np.exp(1j*theta/2) * (np.cos(theta/2) * Pauli.I - 1j*np.sin(theta/2) * sigma_phi)

    @staticmethod
    def R_bloch(bloch_vector: list[float]):
        """ A single-qubit rotation on the Bloch sphere via the Bloch vector v = (v1, v2, v3): 

            U = exp( -i (v1 * sigma_x + v2*sigma_y + v3*sigma_z) ) 
              = cos(alpha) I - i sin(alpha) (v_hat dot sigma) 

            Unit vector: v_hat = v / |v|, sigma = (X, Y, Z) 

            ex] Recover X_pi2 (sqrtX) gate via v1 = (π/2)/2
        """
        # Magnitude of the Bloch vector: 
        alpha = 0.
        for v in bloch_vector:
            alpha += v**2 

        alpha = np.sqrt(alpha)
        TOL = 1E-10

        if alpha < TOL:
            return np.eye(2, dtype=complex) 
        
        # Pauli spin vector:  
        n_vector = [v / alpha for v in bloch_vector] 
        sigma_n = (n_vector[0] * Pauli.X + n_vector[1] * Pauli.Y + n_vector[2] * Pauli.Z ) # to absorb normalization  
        return np.cos(alpha) * Pauli.I - 1j*np.sin(alpha) * sigma_n

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
