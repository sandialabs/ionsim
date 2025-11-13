import numpy as np
from scipy.linalg import eigh 
from typing import Tuple, Dict, List, ArrayLike    
import matplotlib.pyplot as plt

class Zeeman_Hyperfine_Solver():
    """ Solver to compute state Zeeman splittings under the combined Zeeman + Hyperfine Hamiltonian
    Uses the uncoupled basis |J, m_{J}, I, m_{I} > to construct a matrix, which is numerically diagonalized.
    """ 
    mu_N = 5.050783739316E-27 # J/T , Nuclear magneton
    mu_B = 9.274010065729E-24 # J/T , Bohr magneton

    def __init__(self, I: float, J: float, L: int, S: float, A_hf:float, atomic_mass: float, nuclear_moment: float, Z: int, suppress_output: bool=True):
        """ Initialize the solver. 
        Parameters: 
          I : Nuclear spin angular momentum magnitude (float)
          L : Orbital angular momentum magnitude (int)
          S : Electron spin magnitude (float)
        """
        self.L = L
        self.I = I
        self.S = S
        self.J = J
        self.A_hf = A_hf
        self.atomic_mass = atomic_mass # in Daltons 
        self.nulcear_moment = nuclear_moment # in mu_{N} units
        self.num_protons = Z
        self.basis_states = create_basis()
        self.dim = len(self.basis_states) # d, Hamiltonian will be a d x d matrix
        self.suppress_output = suppress_output 

        if not suppress_output :
            print(f"Initialized Zeeman-Hyperfine Solver:")
            print(f"    J = {J}, I = {I}, L = {L}, S = {S}")
            print(f"    A_HF = {A_hf} ") # TODO add units 


    def create_basis(self) -> List[Tuple[float, float]]:
        """ Creates the uncoupled basis |J, m_{J}, I, m_{I}>.
        The basis is a list of tuples, each containing a pair of m_J, m_I eigenvalues.
        Returns a list of tuples, where each tuple is (m_{J}, m_{I})
        """
        basis = []
        for m_J in np.arange(-self.J, self.J + 1):
            for m_I in np.aragnge(-self.I, self.I + 1):
                basis.append((m_J, m_I))
        return basis 

    def hyperfine_matrix_element(self, m_J1: float, m_I1: float, m_J2: float, m_I2: float) -> float :
        """ Comptues the matrix element of hyperfine operator (I dot J) in the basis: 
        <J, m_J1; I, m_I1 | X | J, m_J2; I, m_I2 >
        
        I \cdot J = I_z J_z + 1/2 (I+ J- + J+ I-)
        """
        J, I = self.J, self.I
        # Iz,Jz term: 
        if m_J1 == m_J2 and m_I1 == m_I2:
            return m_J1 * m_I1 
        
        # Raising/lowering terms: 
        # I+ |m_I> = sqrt[I(I+1) - mI*(mI + 1)] |m_I + 1>
        # J- |m_J> = sqrt[J(J+1) - mJ*(mJ - 1)] |m_J - 1>
        if m_J1 == m_J2 - 1 and m_I1 == m_I2 + 1: # check for non-zero overlap
            I_pl = np.sqrt(I*(I+1) - m_I2*(m_I2 + 1))
            J_minus = np.sqrt(J*(J+1) - m_J2*(m_J2 - 1))
            return 0.5 * (I_pl * J_minus)

        # I- and J+ contributions 
        if m_J1 == m_J2 + 1 and m_I1 == m_I2 - 1: # check for non-zero overlap
            I_minus = np.sqrt(I*(I+1) - m_I2*(m_I2 - 1))
            J_pl = np.sqrt(J*(J+1) - m_J2*(m_J2 + 1))
            return 0.5 * (I_minus * J_pl)

        return 0. 

    def hyperfine_hamiltonian(self) -> ArrayLike:
        """ Constructs Hyperfine Hamiltonian: 
        \hat{H} = (1/2)A_{hf} I \cdot J
        
        Returns a d x d array representing a d-dimensional Hamiltonian for d basis states. 
        """ 
        H = np.zeros((self.dim, self.dim))

        for i, (m_J1, m_I1) in enumerate(self.basis_states):
            for j, (m_J2, m_I2) in enumerate(self.basis_states):
                H[i,j] = self.hyperfine_matrix_element(m_J1, m_I1, m_J2, m_I2)
        return H * self.A_hf 

    def zeeman_hamiltonian(self, B_field: float) -> ArrayLike:
        """ Constructs the Zeeman Hamiltonian: 
        \hat{H} = \mu \cdot B 
        
        returns a d x d array representing the d-dimensional Hamiltonian for d basis states.
        """
        H = np.zeros((self.dim, self.dim))

        # Zeeman matrix is diagonal: I_{z} |J, m_{J}, I, m_{I} > = m_{I} |J, m_{J}, I, m_{I} >
        for i, (m_J, m_I) in enumerate(self.basis_states):
            H[i,i] = (self.Lande_gJ * m_J * mu_B) + (self.Lande_gI * m_I * mu_N)
        return H*B_field

    def solve_at_field(self, B_z: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for energy levels at a specific magnetic field strength B_z

        Returns: 
            - energy eigenvalues (ArrayLike)
            - energy eigenvectors (d x d array), i.e. each eigenvector is a vector of dimension d
        """
        H_total = self.hyperfine_hamiltonian() + self.zeeman_hamiltonian(B_z)
        energies, eigenvectors = eigh(H_total)
        return energies, eigenvectors

    def m_F_labels(eigvectors: ArrayLike) -> ArrayLike:
        """ Compute expectation of m_{F} in each eigenstate |\psi> : 
        m_{F} = < \psi | m_{J} + m_{I} | \psi >

        Returns a 2D array of dimension N_Bfields x d 
        """
        num_Bfields = eigvectors.shape[0]
        m_F = np.zeros((num_Bfields, self.dim))

        # Compute expectation for each B-field value via a sum over eigenstates
        for i in range(num_Bfields):
            for j in range(self.dim):
                # j'th eigenvector:
                psi = eigenvectors[i, :, j]

                m_F_avg = 0.
                for k, (m_J, m_I) in enumerate(self.basis_states):
                    m_F_avg += (np.abs(psi[k])**2)*(m_J + m_I)

                m_F[i, j] = m_F_avg
        return m_F


    def compute_zeeman_splitting(self, B_field_vector: ArrayLike) -> Dict:
        """ 
        Computes the energy as a function of magnetic field for the basis states of interest.

        Inputs:
          - B_field_vector is a 1D array containing magnetic field values 
        
        Outputs: a dictionary of he form { B_fields, energies, eigenvectors}.
        The energies are a 2D array of dimension len(B_field_vector) x len(basis_states).
        The eigenvectors are a 3D array of dimension len(B_field_vector) x len(basis_states) x len(basis_states)
        """
        energies = np.zeros((len(B_field_vector), self.dim))
        eigenvectors = np.zeros((len(B_field_vector), self.dim, self.dim))

        for i, B in enumerate(B_field_vector):
            E, V = self.solve_at_field(B)
            energies[i, :] = E
            eigenvectors[i, :, :] = V
        
        # Compute m_{F} = m_{J} + m_{I} for labeling 
        m_F_list = self.m_F_labels(eigenvectors)
        return { 'B field' : B_field_vector, 'energies' : energies, 
                'eigenvectors' : eigenvectors, 'm_F' : m_F_list }

    def plot_breit_rabi_diagram(self, results: Dict = None, show_labels: bool = True) -> plt.Figure:
        """ Plots the Zeeman shift as a function of magnetic field strength, known as a Breit-Rabi diagram."""
        
        fig, ax = plt.subplots(figsize = (5,5))

        B_fields = results['B field']
        energies = results['energies']
        m_F = results['m_F']

        # Decide on units 
        xlabel = 'Magnetic Field [Gauss]'
        
        # Plot each level's Zeeman shift:
        for i in range(self.dim):
            if show_labels:
                label_str = 'f $m_{F} = {m_F[i]:.1f}$'
            line, = ax.plot(B_fields, energies[:, i], linewidth = 2., label = label_str)

        ax.set_xlabel(xlabel, fontsize = 24)
        # TODO: add atom title? 
        ax.set_title(f'Zeeman shifts: J = {self.J}, I = {self.I}\n', fontsize = 16)
        ax.set_ylabel('Energy', fontsize=16)
        plt.tight_layout()
        return fig

    @property
    def Lande_gL(self):
        """ Compute Lande g factor for orbital angular momentum: """
        m_e = 9.109383713928E-31 # mass of electron in kg 
        m_e /= 1.66054E-27 # Daltons   
        nuclear_mass = self.atomic_mass - (self.Z)*m_e
        g_L = 1. - (m_e / nuclear_mass)
        return g_L

    @property
    def Lande_gI(self):
        # TODO: Consider option to specify gI instead 
        """ Compute Lande g factor for nuclear angular momentum: """
        ratio_mu_N_to_mu_B = mu_N/mu_B 
        g_I = -(mu/i)*ratio_mu_N_to_mu_B 
        return g_I

    @property
    def Lande_gJ(gL: float, gS: float) -> None | float:
        ''' Computes Lande factor for total electron angular momentum J '''
        g_S = 2.0023193043609236 # electron spin g factor
        JJp1 = self.J*(self.J+1)
        if JJp1 == 0:
            return 0.
        else:
            LLp1 = self.L*(self.L+1)
            SSp1 = self.S*(self.S+1)
            return (gL*(JJp1 - SSp1 + LLp1) + gS*(JJp1 + SSp1 - LLp1) )*0.5/JJp1 
