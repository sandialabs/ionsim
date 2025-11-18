import numpy as np
from scipy.linalg import eigh 
from typing import Tuple, Dict, List  
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import pint 

class Zeeman_Hyperfine_Solver():
    """ Solver to compute state Zeeman splittings under the combined Zeeman + Hyperfine Hamiltonian
    Uses the uncoupled basis |J, m_{J}, I, m_{I} > to construct a matrix, which is numerically diagonalized.
    """ 

    def __init__(self, I: float, J: float, L: int, S: float, A_hf:float, atomic_mass: float | None=None, 
                nuclear_moment: float | None = None, Z: int | None = None, gI: float | None = None, 
                suppress_output: bool=True, freq_units: str = 'Hz', magnetic_field_units = 'gauss'):
        """ Initialize the solver. 
        Parameters: 
          I : Nuclear spin angular momentum magnitude (float)
          L : Orbital angular momentum magnitude (int)
          S : Electron spin magnitude (float)

        Optional Parameters:
          Z : Atomic number (# of protons of an element)
          Nuclear moment
          gI: Lande "g factor" for nuclear angular momentum. 
          freq_units: A string denoting the frequency units for plotting and returning energies 
          magnetic_field_units: A string denoting the magnetic field units to use when plotting 
        """
        # Set up member variables from inputs
        self.L = L
        self.I = I
        self.S = S
        self.J = J
        self.A_hf = A_hf # input units should match freq_units specification; default expected in Hz 
        self.atomic_mass = atomic_mass # in Daltons 
        if nuclear_moment is None and gI is None:
            raise ValueError('Input error: Either nuclear moment or gJ must be non-zero.')
        self.nuclear_moment = nuclear_moment # in mu_{N} units
        self.gI = gI
        self.Z = Z

        # Create set of basis states in terms of Tuples (m_{J}, m_{I})
        self.basis_states = self.create_basis()
        self.dim = len(self.basis_states) # d, Hamiltonian will be a d x d matrix
        self.suppress_output = suppress_output 

        self.mu_N = 5.050783739316E-27 # J/T , Nuclear magneton
        self.mu_B = 9.274010065729E-24 # J/T , Bohr magneton 

        # Parse the desired unit handling
        # The solver works in rad/s internally. Unit conversions may be specified by the user 
        self.freq_units_str = freq_units 
        self.magnetic_units_str = magnetic_field_units 

        # Create unit registry from pint for internal unit management 
        self.unit_reg = pint.UnitRegistry()
        self.internal_freq_units = self.unit_reg(freq_units)
        self.internal_magnetic_units = self.unit_reg(magnetic_field_units)
        
        # --- Unit Conversions ---
        self.atomic_mass *= self.unit_reg("dalton")

        # Hyperfine constant 
        self.A_hf *= self.internal_freq_units 

        # mu_N and mu_B. Convert to desired magnetic field units 
        self.mu_N *= self.unit_reg.joule / self.unit_reg.tesla
        self.mu_B *= self.unit_reg.joule / self.unit_reg.tesla
        try: 
            self.mu_N = self.mu_N.to("joule/" + self.magnetic_units_str)
            self.mu_B = self.mu_B.to("joule/" + self.magnetic_units_str)
        except:
            # For Tesla to Gauss conversion, pint fails without specifying Gaussian context 
            if self.magnetic_units_str.lower() == 'gauss' or self.magnetic_units_str.low() == "g" :
                self.mu_N = self.mu_N.to("joule/gauss", "Gau")
                self.mu_B = self.mu_B.to("joule/gauss", "Gau")

        if not suppress_output :
            print(f"Initialized Zeeman-Hyperfine Solver:")
            print(f"    J = {self.J}, I = {self.I}, L = {self.L}, S = {self.S}")
            print(f"    A_HF = {self.A_hf} ") 
            print(f"    basis states (m_J, m_I): ")
            self.print_basis_states()



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

    def print_basis_states(self):
        for state in self.basis_states:
            print(f"    (mJ, mI) = {state[0]},{state[1]}")

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

    def hyperfine_hamiltonian(self): #-> NDArray:
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
        H = µ . B 
        
        returns a d x d array representing the d-dimensional Hamiltonian for d basis states.
        """
        H = np.zeros((self.dim, self.dim))
        B_field *= self.internal_magnetic_units

        h = self.unit_reg.planck_constant # 6.62607015E-34 # Planck's constant, J s 
        planck_inverse = (1. / h) # Convert from Joule to Hz via Plancks constant  

        # Convention where the nuclear Zeeman term has \mu_{B} as its prefactor, with gI factor taking into account the nuclear magneton, e.g.: mu_B x gI x I x Bz / hbar 
        # Zeeman matrix is diagonal: I_{z} |J, m_{J}, I, m_{I} > = m_{I} |J, m_{J}, I, m_{I} >
        for i, (m_J, m_I) in enumerate(self.basis_states):
            H[i,i] = (self.Lande_gJ * m_J) + (self.Lande_gI * m_I)

        # Apply the units to the entire array simultaneously
        return (H * B_field * self.mu_B * planck_inverse).to(self.internal_freq_units) 


    def solve_at_field(self, B_z: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize the Hamiltonian for eigen-energies at a specific magnetic field strength B_z.

        Returns: 
            - energy eigenvalues (NDArray)
            - energy eigenvectors (d x d array), i.e. each eigenvector is a vector of dimension d
        """
        H_total = self.hyperfine_hamiltonian() + self.zeeman_hamiltonian(B_z)
        energies, eigenvectors = eigh(H_total.magnitude)
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
        Returns the state energy. """
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

    def get_state_energy_from_mJmI_pair(self, energies: NDArray, mJ: float, mI: float) -> float:
        """ Returns the energy for a basis angular momentum state of interest: |mJ, mI> 
        At zero or low field, F is a good quantum number. At high field, (mJ, mI) are good quantum numbers/
        Assumes energies and eigenvectors at one magnetic field condition.
        Returns state energy. """
        # Get index at mJ, mI pair 
        mJ_states = [state[0] for state in self.basis_states]
        mI_states = [state[1] for state in self.basis_states]
        if mI not in mI_states or mJ not in mJ_states:
            raise ValueError("Requested (mJ, mI) state not found. Please request a valid state.")

        requested_state = (mJ, mI)

        state_indx = -1
        try:
            state_indx = next(i for i, state in enumerate(self.basis_states) if state == requested_state)
        except StopIteration:
            raise("Basis state not found in the list of basis states.")

        return energies[state_indx]


    def plot_breit_rabi_diagram(self, results: Dict = None, show_labels: bool = True, energy_units: str | None=None) -> plt.Figure:
        """ Plots the Zeeman shift as a function of magnetic field strength, known as a Breit-Rabi diagram."""
        
        fig, ax = plt.subplots(figsize = (6,6))

        B_fields = results['B field']
        energies = results['energies']
        m_F = results['m_F']

        xlabel = 'Magnetic Field [' + self.magnetic_units_str + ']'

        # Get number of total |F| states 
        F_range = np.arange(np.abs(self.I - self.J), self.I + self.J + 1)
        F_states = len(F_range)
        num_states_per_F = (2*F_range + 1).astype(int)

        if show_labels: 
            pcnt_offset = 1.0 # vertical offset for F hyperfine label 
            for i, f in enumerate(F_range):
                # Place label above the highest state curve within F manifold at low-field  
                state_indx = num_states_per_F[i] + np.sum(num_states_per_F[0:i])
                ax.text(-0.06*np.abs(B_fields[-1] - B_fields[0]), energies[0, state_indx-1]*(pcnt_offset), f"$F = {int(f)}$", ha='center', va='center')

        # Plot each level's Zeeman shift:
        for i in range(self.dim):
            line, = ax.plot(B_fields, energies[:, i], linewidth = 2.)
            if show_labels:
                # TODO: Check labels for accuracy  
                label_str = f"({self.basis_states[i][0]}, {self.basis_states[i][1]})"
                ax.text(B_fields[-1]*1.02, energies[-1, i], label_str, va='center', ha='left', color = 'darkblue', fontsize=8)

        ax.set_xlabel(xlabel, fontsize = 24)
        if show_labels:
            ax.text(B_fields[-1]*0.8, 0.,'High field ($m_{J}$, $m_{I}$)', va = 'center', ha='left')
            ax.set_xlim(-0.15*np.abs(B_fields[-1] - B_fields[0]), B_fields[-1]*1.2)
        ax.set_title(f'Zeeman shifts: L = {self.L}, J = {self.J}, I = {self.I}\n', fontsize = 16)
        ax.set_ylabel('$E/h$ \n [' + self.freq_units_str + ']', rotation = 0, fontsize=16, labelpad = 15)
        plt.tight_layout()
        return fig, ax

    @property
    def Lande_gL(self):
        """ Compute Lande g factor for orbital angular momentum: """
        # g_L = 1. - (m_e / nuclear_mass)
        m_e = 9.109383713928E-31 * self.unit_reg("kg") # mass of electron in kg 
        m_e = m_e.to("dalton") # Daltons   
        # TODO: Change this to an exception to raise rather than having this hidden logic 
        # - implement a unit test to see if yaml file has the Z argument. This would keep future developers consistent. 
        if self.Z is None:
            return 1.
        else:
            nuclear_mass = self.atomic_mass - (self.Z)*m_e # Daltons
            return 1. - (m_e / nuclear_mass)

    @property
    def Lande_gI(self):
        """ Compute Lande g factor for nuclear angular momentum: """
        if self.gI is None:
            assert self.nuclear_moment != None, "Please specify nuclear magnetic moment or Lande gI factor."
            ratio_mu_N_to_mu_B = self.mu_N/self.mu_B 
            self.gI = -(self.nuclear_moment/self.I)*ratio_mu_N_to_mu_B 
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
