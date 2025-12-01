import numpy as np
from scipy.linalg import eigh 
from typing import Tuple, Dict, List  
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import pint 
import math

class Zeeman_Hyperfine_Solver():
    """ Solver to compute state Zeeman splittings under the combined Zeeman + Hyperfine Hamiltonian
    Uses the uncoupled basis |J, m_{J}, I, m_{I} > to construct a matrix, which is numerically diagonalized.
    """ 

    def __init__(self, I: float, J: float, L: int, S: float, A_hf:float, atomic_mass: float | None=None, 
                nuclear_moment: float | None = None, Z: int | None = None, gI: float | None = None, 
                suppress_output: bool=True, freq_units: str = 'Hz', magnetic_field_units = 'gauss', mode: str = 'exact'):
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
        if self.Z is None:
            raise ValueError('Input error: Specify atomic number of the atom.')
        
        # Set up the mode for solving:
        # Options: 
        #  "exact": considers both hyperfine and zeeman hamiltonian
        #  "weak field": weak field limit, considers an effective linear Zeeman Hamiltonian 
        self.mode = mode.lower()
        if self.mode != 'exact' and self.mode != 'weak field':
            raise ValueError("Invalid mode specified. Mode variable must be either 'exact' or 'weak field'.")

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
            self.print_basis_states()



    def create_basis(self) -> List[Tuple[float, float]]:
        """ Creates a basis depending on the solver mode. 
        For exact mode, the uncoupled basis |J, m_{J}, I, m_{I}> is used.
        The basis is a list of tuples, each containing a pair of m_J, m_I eigenvalues.
        Returns a list of tuples, where each tuple is (m_{J}, m_{I})

        For weak field mode, the coupled basis |F, mF> is used.
        The basis is a list of tuples containing F, mF pairs.
        """
        basis = []
        
        if self.mode == 'exact':
            for m_J in np.arange(-self.J, self.J + 1):
                for m_I in np.arange(-self.I, self.I + 1):
                    basis.append((m_J, m_I))
        elif self.mode == 'weak field':
            # |F, mF> are good quatnum numbers
            # The |mJ, mI> states are coupled. 
            F_range = np.arange(np.abs(self.I - self.J), self.I + self.J + 1) 
            for f in F_range:
                for mF in np.arange(-f, f+1):
                    basis.append((f, mF))
        else:
            raise ValueError("Invalid mode specified. Mode variable must be either 'exact' or 'weak field'.")
        return basis 

    def print_basis_states(self):
        """ Function for printing out basis state information """
        if self.mode == 'exact':
            state_str = '(mJ, mI)'
        elif self.mode == 'weak field':
            state_str = '(F, mF)'
        print(f"    basis states " + state_str + ": ")
        for state in self.basis_states:
            print(f"    " + state_str + f" = {state[0]},{state[1]}")

    def hyperfine_matrix_element(self, m_J1: float, m_I1: float, m_J2: float, m_I2: float) -> float :
        """ Comptues the matrix element of hyperfine operator (I dot J) in the basis: 
        <J, m_J1; I, m_I1 | X | J, m_J2; I, m_I2 >
        
        I dot J = I_z J_z + 1/2 (I+ J- + J+ I-)
        """
        assert self.mode == 'exact', "Error: Incorrect mode specified by user."

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

    def hyperfine_hamiltonian(self): 
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
        if self.mode == 'exact':       
            for i, (m_J, m_I) in enumerate(self.basis_states):
                H[i,i] = (self.Lande_gJ * m_J) + (self.Lande_gI * m_I)
        elif self.mode.lower() == 'weak field' :
            for i, (f, mF) in enumerate(self.basis_states): 
                H[i,i] = (self.Lande_gF(f) * mF)
        else:
            raise ValueError("Invalid mode specified. Mode variable must be either 'exact' or 'weak field'.")

        # Apply the units to the entire array simultaneously
        return (H * B_field * self.mu_B * planck_inverse).to(self.internal_freq_units) 

    def solve_at_field(self, B_z: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize the Hamiltonian for eigen-energies at a specific magnetic field strength B_z.

        Returns: 
            - energy eigenvalues (NDArray)
            - energy eigenvectors (d x d array), i.e. each eigenvector is a vector of dimension d
        """
        H_total = self.zeeman_hamiltonian(B_z)
        if self.mode == 'exact':
            H_total += self.hyperfine_hamiltonian() 
        energies, eigenvectors = eigh(H_total.magnitude)
        # Note: These energies are not in the sorted order that matches the basis states
        return energies, eigenvectors


    def F_character(self, psi: NDArray, F: float) -> float:
        """ Compute how much of |F, mF> character an eigenstate has. 
        Requires Clebsch-Gordan decomposition to project onto |F,mF>

        Returns the probability of finding state in F manifold (p in [0,1])"""

        from scipy.special import comb 
        J, I = self.J, self.I
        if F < abs(J - I) or F > J + I :
            return 0.

        tol = 1e-8
        probability = 0.
        for mF in np.arange(-F, F + 1, 1):
            amplitude = 0.
            for k, (mJ, mI) in enumerate(self.basis_states):
                if abs(mJ + mI - mF) < tol:
                    # C-G coeff:
                    cg = self.Clebsch_Gordan(J, I, F, mJ, mI, mF)
                    amplitude += cg * psi[k]

            probability += np.abs(amplitude)**2 

        return probability


    def Clebsch_Gordan(self, j1: float, j2: float, j: float, m1: float, m2: float, m: float) -> float : 
        """ Compute Clebsch-Gordan coefficient using standard formulae from quantum mechanics texts """
        tol = 1e-9
        if (abs(m1 + m2 - m) > tol):
            return 0.
        if j < abs(j1 - j2) or j > j1 + j2:
            return 0.
        if abs(m1) > j1 or abs(m2) > j2 or abs(m) > j:
            return 0.

        prefactor = np.sqrt(
             (2*j + 1) * math.factorial(int(j + j1 - j2)) * 
             math.factorial(int(j - j1 + j2)) * math.factorial(int(j1 + j2 - j)) / math.factorial(int(j1 + j2 + j + 1))
        )

        prefactor *= np.sqrt(
            math.factorial(int(j + m)) * math.factorial(int(j-m)) * math.factorial(int(j1 + m1)) * 
            math.factorial(int(j1 - m1)) * math.factorial(int(j2 + m2)) * math.factorial(int(j2 - m2))
        )

        k_min = int(max(0, j2 - j - m1, j1 - j + m2))
        k_max = int(min(j1 + j2 - j, j1 - m1, j2 + m2))

        S = 0.
        for k in range(k_min, k_max + 1):
            S += ((-1)**k / (math.factorial(k) * math.factorial(int(j1 + j2 - j - k)) * 
            math.factorial(int(j1 - m1 - k)) * math.factorial(int(j2 + m2 - k)) * 
            math.factorial(int(j - j2 + m1 + k)) * math.factorial(int(j - j1 - m2 + k))))
        
        return prefactor*S

    def mJ_mI_labels(self, eigvectors: NDArray):
        """ Function to compute the estimated mJ, mI labels for the eigenvectors.
            - assuming the eigenvectors are for a single magnetic field value """

        if len(eigvectors.shape) == 2:
            num_Bfields = 1
        else:
            num_Bfields = eigvectors.shape[0]

        mJ_labels = np.zeros((num_Bfields, self.dim))
        mI_labels = np.zeros_like(mJ_labels)

        for i in range(num_Bfields):
            for j in range(self.dim):
                psi = eigvectors[i, :, j]

                # Find basis with maximum probability 
                probs = np.abs(psi)**2
                dominant_indx = np.argmax(probs)

                # (mJ, mI) for this basis state 
                mJ, mI = self.basis_states[dominant_indx]

                mJ_labels[i, j] = mJ
                mI_labels[i, j] = mI

        return mJ_labels, mI_labels


    def F_mF_labels(self, eigvectors: NDArray):
        """ Computes state labels |F, mF> for an energy eigenstates.
        
        Compute expectation of m_{F} in each eigenstate |psi> : 
        m_{F} = < psi | m_{J} + m_{I} | psi >

        Returns a 2D array of dimension N_Bfields x d 
        """
        if len(eigvectors.shape) == 2:
            num_Bfields = 1
        else:
            num_Bfields = eigvectors.shape[0]

        m_F = np.zeros((num_Bfields, self.dim))
        dominant_F = np.zeros_like(m_F)
        F_range = np.arange(np.abs(self.I - self.J), self.I + self.J + 1)

        # Compute expectation for each B-field value via a sum over eigenstates
        for i in range(num_Bfields):
            for j in range(self.dim):
                # j'th eigenvector:
                if num_Bfields == 1:
                    psi = eigvectors[:, j]
                else:
                    psi = eigvectors[i, :, j]

                m_F_avg = 0.
                if self.mode == 'exact' :
                    for k, (m_J, m_I) in enumerate(self.basis_states):
                        m_F_avg += (np.abs(psi[k])**2)*(m_J + m_I)
                elif self.mode == 'weak field' :
                    for  k, (f, mF) in enumerate(self.basis_states):
                        m_F_avg += (np.abs(psi[k])**2)*(mF)

                m_F[i, j] = m_F_avg
                # Compute F character 
                F_chars = {}
                for F in F_range:
                    F_char = self.F_character(psi, F)
                    F_chars[F] = F_char
                # Dominant F value: 
                dominant_F[i,j] = max(F_chars.items(), key = lambda x: x[1])[0]

        return dominant_F, m_F


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
        F_list, m_F_list = self.F_mF_labels(eigenvectors)
        mJ_list, mI_list = self.mJ_mI_labels(eigenvectors)
        if self.mode == 'exact' :
            return { 'B field' : B_field_vector, 'energies' : energies, 
                'eigenvectors' : eigenvectors, 'F' : F_list, 'm_F' : m_F_list,
                'mJ' : mJ_list, 'mI' : mI_list}
        elif self.mode == 'weak field':
            return { 'B field' : B_field_vector, 'energies' : energies, 
                'eigenvectors' : eigenvectors, 'F' : F_list, 'm_F' : m_F_list}

    
    def get_state_energy(self, energies: NDArray, eigenvectors: NDArray, F: float, 
                            m_F: float, tolerance: float = 1e-8) -> float:
        """ Returns the energy for an angular momentum state of interest |F, mF> 
        At zero or low field, F is a good quantum number. At finite field, m_F is conserved but F is not. 
        Assumes energies and eigenvectors at one magnetic field condition.
        Returns the state energy. """
        F_vals, m_F_values = self.F_mF_labels(eigenvectors)

        F_vals = F_vals[0]
        m_F_values = m_F_values[0]

        matching_states = []
        requested_state = (F, m_F)
        
        # From list of F values and mF values, map input to index for corresponding state (F,mF)
        F_range = np.arange(np.abs(self.I - self.J), self.I + self.J + 1)
        num_states_per_F = (2*F_range + 1).astype(int)
        try:
            F_indx = list(F_range).index(F) 
        except:
            raise ValueError(f"Invalid F value. Please choose in the range {F_range}")

        approximate_states = list(zip(F_vals, m_F_values)) 
        state_indx = -1
        for i, value in enumerate(m_F_values):
            if np.abs(value - m_F) < tolerance :
                state_indx = i
        
        if state_indx == -1:
            raise ValueError(f"Basis state mF = {m_F} not found in list of basis states: {m_F_values}.")
        return energies[state_indx]


    def get_state_energy_from_mJmI_pair(self, energies: NDArray, eigenvectors: NDArray, mJ: float, mI: float) -> float:
        """ Returns the energy for a basis angular momentum state of interest: |mJ, mI> 
        At zero or low field, F is a good quantum number. At high field, (mJ, mI) are good quantum numbers/
        Assumes energies and eigenvectors at one magnetic field condition.
        Returns state energy. """
        assert self.mode == 'exact', "Error. Request energies from (mJ, mI) pairs only in exact basis."
        # Get index at mJ, mI pair 
        mJ_states = [state[0] for state in self.basis_states]
        mI_states = [state[1] for state in self.basis_states]
        if mI not in mI_states or mJ not in mJ_states:
            raise ValueError("Requested (mJ, mI) state not found. Please request a valid state.")

        requested_state = (mJ, mI)

        # Find eigenstate with domaintn mJ, mI character
        best_match = -1
        best_overlap = 0
        # Index of mJ,mI pair in the basis 
        basis_indx = self.basis_states.index((mJ, mI))

        # Compute probability of being in (mJ, mI) state 
        for i in range(self.dim):
            overlap = np.abs(eigenvectors[basis_indx, i])**2
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = i
        return energies[best_match]


    def plot_breit_rabi_diagram(self, results: Dict = None, show_labels: bool = True, energy_units: str | None=None) -> plt.Figure:
        """ Plots the Zeeman shift as a function of magnetic field strength, known as a Breit-Rabi diagram."""
        
        fig, ax = plt.subplots(figsize = (6,6))

        B_fields = results['B field']
        energies = results['energies']
        m_F = results['m_F']
        F_values = results['F']

        xlabel = 'Magnetic Field [' + self.magnetic_units_str + ']'

        if show_labels: 
            # Add label for |F| manifold on the left of the curves near B = 0 
            if self.mode == 'exact':
                F_range = np.arange(np.abs(self.I - self.J), self.I + self.J + 1)
                num_states_per_F = (2*F_range + 1).astype(int)
                for i, f in enumerate(F_range):
                    f_label = f"$F = {f:.1f}$" if f % 1 != 0 else f"$F = {int(f)}$"
                    # Place label above the highest state curve within F manifold at low-field  
                    state_indx = num_states_per_F[i] + np.sum(num_states_per_F[0:i])
                    ax.text(-0.08*np.abs(B_fields[-1] - B_fields[0]), energies[0, state_indx-1], f_label, ha='center', va='center')
            elif self.mode == 'weak field':
                for i, (f, mf) in enumerate(self.basis_states):
                    f_label = f"$F = {f:.1f}$" if f % 1 != 0 else f"$F = {int(f)}$"
                    ax.text(-0.08*np.abs(B_fields[-1] - B_fields[0]), energies[0, i], f_label, ha='center', va='center')

        # Plot each level's Zeeman shift:
        if self.mode == 'exact' :
            mJ_labels = results['mJ']
            mI_labels = results['mI']

        for i in range(self.dim):
            # Plot each curve with same color to prevent confusion with level mixing in intermediate region 
            line, = ax.plot(B_fields, energies[:, i], linewidth = 1.25, color='k') 

            if show_labels:
                if self.mode == "exact" :
                    label_str = f"({mJ_labels[-1, i]}, {mI_labels[-1, i]})" 
                elif self.mode == "weak field" :
                    label_str = f"({F_values[-1,i]},{m_F[-1,i]})" 
                ax.text(B_fields[-1]*1.02, energies[-1, i], label_str, va='center', ha='left', color = 'darkblue', fontsize=8)

        ax.set_xlabel(xlabel, fontsize = 24)
        if show_labels:
            # Option 1, right section 
            x_annotate = B_fields[1]*0.8
            y_annotate = 0.
            # Option 2 (better), top left corner 
            x_annotate = -0.025*np.abs(B_fields[-1] - B_fields[0])
            y_annotate = np.max(energies)
            if self.mode == 'exact':
                ax.text(x_annotate, y_annotate,'High field ($m_{J}$, $m_{I}$)', va = 'center', ha='left')
            elif self.mode == 'weak field':
                ax.text(x_annotate, y_annotate,'($F$, $m_{F}$)', va = 'center', ha='left')
            ax.set_xlim(-0.15*np.abs(B_fields[-1] - B_fields[0]), B_fields[-1]*1.2)
        ax.set_title(f'Zeeman shifts: L = {self.L}, J = {self.J}, I = {self.I}\n', fontsize = 16)
        ax.set_ylabel('$E/h$ \n [' + self.freq_units_str + ']', rotation = 0, fontsize=16, labelpad = 15)
        plt.tight_layout()
        return fig, ax

    @property
    def Lande_gL(self):
        """ Compute Lande g factor for orbital angular momentum: """
        """ gL = 1. - (m_e / nuclear_mass) """
        m_e = 9.109383713928E-31 * self.unit_reg("kg") # mass of electron in kg 
        m_e = m_e.to("dalton") # Daltons   
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
            return (self.Lande_gL*(JJp1 - SSp1 + LLp1) + gS*(JJp1 + SSp1 - LLp1) )*0.5/JJp1 #

    def Lande_gF(self, f: float) -> float:
        """ Lande g factor for total hyperfine angular momentum. 
        - takes in "F" the total angular momentum magnitude as an input """
        ffp1 = f*(f+1)
        if ffp1 == 0:
            raise ValueError('Division by zero error. Do not use this function if F = 0.')
        if self.Lande_gJ is None:
            return self.Lande_gI
        IIp1 = self.I*(self.I+1)
        JJp1 = self.J*(self.J+1)
        return ((self.Lande_gJ*0.5*(ffp1 - IIp1 + JJp1)) + self.Lande_gI*0.5*(ffp1 + IIp1 - JJp1))/ffp1
