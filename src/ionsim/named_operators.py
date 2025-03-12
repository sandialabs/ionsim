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
    plus = np.array(
        [[0, 0],
         [1, 0]],
    )
    minus = np.array(
        [[0, 1],
         [0, 0]],
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

def main():
    """Script to execute if module is ran directly."""
    from scipy.linalg import expm

    ic(Unitary.Y.round(14) == Pauli.Y)

    phi, theta = np.pi/8, -2*np.pi/3
    sig_phi = np.cos(phi) * Pauli.X + np.sin(phi) * Pauli.Y
    ic(
        expm(-1j * theta/2 * np.kron(sig_phi, sig_phi)).round(14) == (
            + np.cos(theta/2) * np.kron(Pauli.I, Pauli.I) - 1j*np.sin(theta/2) * np.kron(sig_phi, sig_phi)
        ).round(14)
    )

if __name__ == '__main__':
    main()

