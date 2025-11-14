import numpy as np
from scipy.linalg import eigh 
from typing import Tuple, Dict, List  
from numpy.typing import NDArray
import matplotlib.pyplot as plt

class Zeeman_Hyperfine_Solver():
    """ Solver to compute state Zeeman splittings under the combined Zeeman + Hyperfine Hamiltonian
    Uses the uncoupled basis |J, m_{J}, I, m_{I} > to construct a matrix, which is numerically diagonalized.
    """ 

    def __init__(self, I: float, J: float, L: int, S: float, A_hf:float, atomic_mass: float | None=None, 
                nuclear_moment: float | None = None, Z: int | None = None, gI: float | None = None, 
                suppress_output: bool=True):
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
        self.A_hf = A_hf  # rad/s
        self.atomic_mass = atomic_mass # in Daltons 
        if nuclear_moment is None and gI is None:
            raise ValueError('Input error: Either nuclear moment or gJ must be non-zero.')
        self.nuclear_moment = nuclear_moment # in mu_{N} units
        self.gI = gI
        self.Z = Z
        self.basis_states = self.create_basis()
        self.dim = len(self.basis_states) # d, Hamiltonian will be a d x d matrix
        self.suppress_output = suppress_output 

        self.mu_N = 5.050783739316E-27 # J/T , Nuclear magneton
        self.mu_B = 9.274010065729E-24 # J/T , Bohr magneton 

        if not suppress_output :
            print(f"Initialized Zeeman-Hyperfine Solver:")
            print(f"    J = {J}, I = {I}, L = {L}, S = {S}")
            print(f"    A_HF = {A_hf} ") # TODO add units 
            print(f"    basis states (m_J, m_I): ")
            print(*self.basis_states)


    def create_basis(self) -> List[Tuple[float, float]]:
        """ Creates the uncoupled basis |J, m_{J}, I, m_{I}>.
        The basis is a list of tuples, each containing a pair of m_J, m_I eigenvalues.
        Returns a list of tuples, where each tuple is (m_{J}, m_{I})
        """
        basis = []
        for m_J in np.arange(-self.J, self.J + 1):
            for m_I in np.arange(-self.I, self.I + 1):
                basis.append((m_J, m_I))
        return basis 

    def hyperfine_matrix_element(self, m_J1: float, m_I1: float, m_J2: float, m_I2: float) -> float :
        """ Comptues the matrix element of hyperfine operator (I dot J) in the basis: 
        <J, m_J1; I, m_I1 | X | J, m_J2; I, m_I2 >
        
        I dot J = I_z J_z + 1/2 (I+ J- + J+ I-)
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

    def hyperfine_hamiltonian(self) -> NDArray:
        """ Constructs Hyperfine Hamiltonian: 
        H = (1/2) A_{hf} I dot J
        
        Returns a d x d array representing a d-dimensional Hamiltonian for d basis states. 
        """ 
        H = np.zeros((self.dim, self.dim))

        for i, (m_J1, m_I1) in enumerate(self.basis_states):
            for j, (m_J2, m_I2) in enumerate(self.basis_states):
                H[i,j] = self.hyperfine_matrix_element(m_J1, m_I1, m_J2, m_I2)
        return H * self.A_hf 


    def zeeman_hamiltonian(self, B_field: float) -> NDArray:
        """ Constructs the Zeeman Hamiltonian: 
        H = mu dot B 
        
        returns a d x d array representing the d-dimensional Hamiltonian for d basis states.
        """
        H = np.zeros((self.dim, self.dim))

        h = 6.62607015E-34 # Planck's constant, J s 
        # mu_N in J/T, divide by h gives you 1/(sT), multiply by 2\pi to get rad/s / T
        units_scaling = 2*np.pi*1E-4/h # Hz / Gauss

        # Zeeman matrix is diagonal: I_{z} |J, m_{J}, I, m_{I} > = m_{I} |J, m_{J}, I, m_{I} >
        for i, (m_J, m_I) in enumerate(self.basis_states):
            H[i,i] = (self.Lande_gJ * m_J * self.mu_B * units_scaling) + (self.Lande_gI * m_I * self.mu_B * units_scaling)
        return (H * B_field) # rad/s
        # There's a convention where the nuclear Zeeman term has \mu_{B} as its prefactor, with 
        # the Lande-gI factor taking into account the nuclear magneton, e.g.: mu_B x gI x I x Bz / hbar 


    def solve_at_field(self, B_z: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for energy levels at a specific magnetic field strength B_z

        Returns: 
            - energy eigenvalues (NDArray)
            - energy eigenvectors (d x d array), i.e. each eigenvector is a vector of dimension d
        """
        H_total = self.hyperfine_hamiltonian() + self.zeeman_hamiltonian(B_z)
        energies, eigenvectors = eigh(H_total)
        return energies, eigenvectors


    def m_F_labels(self, eigvectors: NDArray) -> NDArray:
        """ Compute expectation of m_{F} in each eigenstate |psi> : 
        m_{F} = < psi | m_{J} + m_{I} | psi >

        Returns a 2D array of dimension N_Bfields x d 
        """
        #print(eigvectors.shape)
        if len(eigvectors.shape) == 2:
            num_Bfields = 1
        else:
            num_Bfields = eigvectors.shape[0]
        m_F = np.zeros((num_Bfields, self.dim))

        # Compute expectation for each B-field value via a sum over eigenstates
        for i in range(num_Bfields):
            for j in range(self.dim):
                # j'th eigenvector:
                if num_Bfields == 1:
                    psi = eigvectors[:, j]
                else:
                    psi = eigvectors[i, :, j]

                m_F_avg = 0.
                for k, (m_J, m_I) in enumerate(self.basis_states):
                    m_F_avg += (np.abs(psi[k])**2)*(m_J + m_I)

                m_F[i, j] = m_F_avg
        return m_F


    def compute_zeeman_splitting(self, B_field_vector: NDArray) -> Dict:
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

    
    def get_state_energy(self, energies: NDArray, eigenvectors: NDArray, F: Optional[float] = None, 
                            m_F: Optional[float] = None, tolerance: float = 0.1) -> float:
        """ Returns the energy for an angular momentum state of interest |F, mF> 
        At zero or low field, F is a good quantum number. At finite field, m_F is conserved but F is not. 
        Assumes energies and eigenvectors at one magnetic field condition.
        Returns a dictionary with state energy among other properties. """
        m_F_values = self.m_F_labels(eigenvectors)[0]
        matching_states = []
        
        # From list of F values and mF values, map input to index for corresponding state (F,mF)
        F_range = np.arange(np.abs(self.I - self.J), self.I + self.J + 1)
        num_states_per_F = (2*F_range + 1).astype(int)
        
        try:
            F_indx = list(F_range).index(F) 
        except:
            raise ValueError(f"Invalid F value. Please choose in the range {F_range}")

        left_indx = np.sum(num_states_per_F[0:F_indx]) # number of states to ignore 

        mF_subarray = m_F_values[left_indx:left_indx + num_states_per_F[F_indx]] # to search for mF state 
        try:
            match_index = np.where(np.abs(mF_subarray - m_F) < 0.1)[0][0]
            #match_index = list(mF_subarray).index(m_F)
        except:
            raise ValueError(f"Invalid mF value. Please choose in the range {mF_subarray}")

        return energies[left_indx + match_index]

        # m_F_values array has structure based on num_states_per_F
        # start_indx = int(num_states_per_F[F_indx]) - 1
        # end_indx = 
        # mF_list_F = np.arange(-F, F+1)
        # num_mF_states = len(mF_list_F)
        # mF_indx = np.argmin(np.abs(mF_list_F - m_F))
        # # Search sub-array of m_F_values to find index where desired mF occurs 
        # match_index = 

        # # Return energy of the F, mF state:
        # solver_indx = num_mF_states*(F-1) + mF_indx
        # return 

        # TODO need to return state within correct F manifold 
        # e.g. for Rb87, F = 1 and mF=-1 currently returns mF=-1 from both F = 1 and 2 manifolds for s orbital
        # 2F+1 states per level 
        # F_solver = (self.dim - 1)*0.5


        # # Find matching states:
        # for i in range(self.dim):
        #     if m_F is not None:
        #         if abs(m_F_values[0, i] - m_F) > tolerance :
        #             continue
        
        #     if F is not None:
        #         F_char = self.F_character(eigenvectors[:, i], F)
        #         if F_char < 0.1:
        #             continue
        #     else:
        #         F_char = None
        #     # Dominant basis state: 
        #     psi = eigenvectors[:, i]
        #     dominant_indx = np.argmax(np.abs(psi)**2)
        #     dominant_coeff = psi[dominant_indx]
        #     dominant_basis = self.basis_states[dominant_indx]

        #     matching_states.append({
        #         'state_index' : i,
        #         'energy' : energies[i],
        #         'F_character' : F_char
        #         'm_F' : m_F_values[0, i],
        #         'dominant basis' : dominant_basis
        #     })

        # # Safety Checks:
        # if len(matching_states) == 0.:
        #     raise ValueError(f"No state found matching F = {F}, m_F = {m_F}")

        # if len(matching_states) > 1:
        #     if matching_states[0]['energy'] != matching_states[1]['energy']:
        #         print(f"Energy of state {matching_states[0]['dominant basis']} = {matching_states[0]['energy']}")
        #         print(f"Energy of state {matching_states[1]['dominant basis']} = {matching_states[1]['energy']}")
        #         raise ValueError(f"Zeeman shifts are degenerate at this field condition.")

        # return matching_states[0]


    def plot_breit_rabi_diagram(self, results: Dict = None, show_labels: bool = True, energy_units: str | None=None) -> plt.Figure:
        """ Plots the Zeeman shift as a function of magnetic field strength, known as a Breit-Rabi diagram."""
        
        fig, ax = plt.subplots(figsize = (6,6))

        B_fields = results['B field']
        energies = results['energies'] / (2.*np.pi)
        m_F = results['m_F']

        # Decide on units 
        xlabel = 'Magnetic Field [Gauss]'
        
        # Plot each level's Zeeman shift:
        for i in range(self.dim):
            if show_labels:
                label_str = f' $m_F = ${m_F[-1, i]:.1f}' # use final field value for label 
                line, = ax.plot(B_fields, energies[:, i], linewidth = 2., label = label_str)
            else:
                line, = ax.plot(B_fields, energies[:, i], linewidth = 2.)

        ax.set_xlabel(xlabel, fontsize = 24)
        # TODO: add atom title? 
        ax.set_title(f'Zeeman shifts: J = {self.J}, I = {self.I}\n', fontsize = 16)
        ax.set_ylabel('Energy [Hz]', fontsize=16)
        plt.tight_layout()
        if show_labels:
            plt.legend()
        return fig

    @property
    def Lande_gL(self):
        """ Compute Lande g factor for orbital angular momentum: """
        m_e = 9.109383713928E-31 # mass of electron in kg 
        m_e /= 1.66054E-27 # Daltons   
        if self.Z is None:
            return 1.
        else:
            nuclear_mass = self.atomic_mass - (self.Z)*m_e # Daltons
            #g_L = 1. - (m_e / nuclear_mass)
            return 1. - (m_e / nuclear_mass)

    @property
    def Lande_gI(self):
        """ Compute Lande g factor for nuclear angular momentum: """
        if self.gI is None:
            assert(self.nuclear_moment != None)
            ratio_mu_N_to_mu_B = self.mu_N/self.mu_B 
            self.gI = -(self.nuclear_moment/self.I)*ratio_mu_N_to_mu_B 
        #print(f"gI = {self.gI}")
        return self.gI

    @property
    def Lande_gJ(self) -> None | float:
        ''' Computes Lande factor for total electron angular momentum J '''
        gS = 2.0023193043609236 # electron spin g factor
        JJp1 = self.J*(self.J+1)
        if JJp1 == 0:
            return 0.
            # return (self.Lande_gL + gS)*0.5 via limit process? 
        else:
            LLp1 = self.L*(self.L+1)
            SSp1 = self.S*(self.S+1)
            return (self.Lande_gL*(JJp1 - SSp1 + LLp1) + gS*(JJp1 + SSp1 - LLp1) )*0.5/JJp1 
