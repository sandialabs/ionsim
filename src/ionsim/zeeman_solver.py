import numpy as np
from scipy import constants
from scipy.linalg import eigh 
from scipy.special import comb 
from numpy.typing import NDArray
import pint 
import math


class ZeemanHyperfineSolver():
    """ Solver to compute state Zeeman splittings under the combined Zeeman + Hyperfine Hamiltonian
    Uses the uncoupled basis |J, m_{J}, I, m_{I} > to construct a matrix, which is numerically diagonalized.
    """ 

    def __init__(self, i: float, j: float, l: int, s: float, hyperfine_a:float, atomic_mass: float | None=None, 
                nuclear_moment: float | None = None, z: int | None = None, gi: float | None = None, 
                freq_units: str = 'Hz', magnetic_field_units = 'gauss', approximation: str | None=None):
        """ Initialize the solver. 
        Parameters: 
          i : Nuclear spin angular momentum magnitude (float)
          l : Orbital angular momentum magnitude (int)
          s : Electron spin magnitude (float)

        Optional Parameters:
          Z : Atomic number (# of protons of an element)
          Nuclear moment
          gi: Lande "g factor" for nuclear angular momentum. 
          freq_units: A string denoting the frequency units for returning energies 
          magnetic_field_units: A string denoting the magnetic field units to use 
        """
        self.l = l
        self.i = i
        self.s = s
        self.j = j
        self.hyperfine_a = hyperfine_a # input units should match freq_units specification; default expected in Hz 
        self.atomic_mass = atomic_mass # in Daltons 
        self.nuclear_moment = nuclear_moment # in mu_{N} units
        self.gi = gi
        if nuclear_moment is None and gi is None:
            raise ValueError('Input error: Either nuclear moment or gj must be non-zero.')
        self.z = z
        if self.z is None:
            raise ValueError('Input error: Specify atomic number of the atom.')
        
        # Set up the approximation for solving:
        # Options: 
        #  None: An exact treatment, considers both hyperfine and zeeman hamiltonian
        #  "weak field": weak field limit, considers an effective linear Zeeman Hamiltonian 
        if type(approximation) == str:
          self.approximation = approximation.lower()
        else:
          self.approximation = approximation

        if self.approximation is not None and self.approximation != 'weak field':
            raise ValueError("Invalid approximation specified. Approximation variable must be either None or 'weak field'.")

        # Create set of basis states in terms of Tuples (m_{J}, m_{I})
        self.basis_states = self.create_basis()
        self.dim = len(self.basis_states) # d, Hamiltonian will be a d x d matrix

        # Retrieve constants and manipulate units internally with pint 
        self.mu_n = constants.physical_constants['nuclear magneton'][0] # J/T , Nuclear magneton
        self.mu_b = constants.physical_constants['Bohr magneton'][0] # J/T , Bohr magneton 

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
        self.hyperfine_a *= self.internal_freq_units 

        # mu_n and mu_b. Convert to desired magnetic field units 
        self.mu_n *= self.unit_reg.joule / self.unit_reg.tesla
        self.mu_b *= self.unit_reg.joule / self.unit_reg.tesla
        try: 
            self.mu_n = self.mu_n.to("joule/" + self.magnetic_units_str)
            self.mu_b = self.mu_b.to("joule/" + self.magnetic_units_str)
        except:
            # For Tesla to Gauss conversion, pint fails without specifying Gaussian context 
            if self.magnetic_units_str.lower() == 'gauss' or self.magnetic_units_str.lower() == "g" :
                self.mu_n = self.mu_n.to("joule/gauss", "Gau")
                self.mu_b = self.mu_b.to("joule/gauss", "Gau")


    def create_basis(self) -> list[tuple[float, float]]:
        """ Creates a basis depending on the solver approximation. 
        For exact, the uncoupled basis |J, m_{J}, I, m_{I}> is used.
        The basis is a list of tuples, each containing a pair of m_J, m_I eigenvalues.
        Returns a list of tuples, where each tuple is (m_{J}, m_{I})

        For weak field approximation, the coupled basis |F, mF> is used.
        The basis is a list of tuples containing F, mF pairs.
        """
        basis = []
        
        if self.approximation is None:
            for mj in np.arange(-self.j, self.j + 1):
                for mi in np.arange(-self.i, self.i + 1):
                    basis.append((mj, mi))
        elif self.approximation == 'weak field':
            # |F, mF> are good quantum numbers
            # The |mj, mi> states are coupled. 
            f_range = np.arange(np.abs(self.i - self.j), self.i + self.j + 1) 
            for f in f_range:
                for mf in np.arange(-f, f+1):
                    basis.append((f, mf))
        else:
            raise ValueError("Invalid approximation specified. `approximation` constructor argument must be either None or 'weak field'.")
        return basis 


    def hyperfine_matrix_element(self, mj1: float, mi1: float, mj2: float, mi2: float) -> float :
        """ Comptues the matrix element of hyperfine operator (I dot J) in the basis: 
        <J, mj1; I, mi1 | X | J, mj2; I, mi2 >
        
        I dot J = I_z J_z + 1/2 (I+ J- + J+ I-)
        """
        j, i = self.j, self.i
        # Iz,Jz term: 
        if mj1 == mj2 and mi1 == mi2:
            return mj1 * mi1 
        
        # Raising/lowering terms: 
        # I+ |m_I> = sqrt[I(I+1) - mi*(mi + 1)] |m_I + 1>
        # J- |m_J> = sqrt[J(J+1) - mj*(mj - 1)] |m_J - 1>
        if mj1 == mj2 - 1 and mi1 == mi2 + 1: # check for non-zero overlap
            i_plus = np.sqrt(i*(i+1) - mi2*(mi2 + 1))
            j_minus = np.sqrt(j*(j+1) - mj2*(mj2 - 1))
            return 0.5 * (i_plus * j_minus)

        # I- and J+ contributions 
        if mj1 == mj2 + 1 and mi1 == mi2 - 1: # check for non-zero overlap
            i_minus = np.sqrt(i*(i+1) - mi2*(mi2 - 1))
            j_plus = np.sqrt(j*(j+1) - mj2*(mj2 + 1))
            return 0.5 * (i_minus * j_plus)

        return 0. 

    def hyperfine_hamiltonian(self) -> NDArray: 
        """ Constructs Hyperfine Hamiltonian: 
        H = (1/2) hyperfineA I dot J
        
        Returns a d x d array representing a d-dimensional Hamiltonian for d basis states. 
        """ 
        H = np.zeros((self.dim, self.dim))

        for i, (mj1, mi1) in enumerate(self.basis_states):
            for j, (mj2, mi2) in enumerate(self.basis_states):
                H[i,j] = self.hyperfine_matrix_element(mj1, mi1, mj2, mi2)
        return H * self.hyperfine_a 

    def zeeman_hamiltonian(self, magnetic_field: float) -> NDArray:
        """ Constructs the Zeeman Hamiltonian: 
        H = µ . B 
        
        returns a d x d array representing the d-dimensional Hamiltonian for d basis states.
        """
        H = np.zeros((self.dim, self.dim))
        magnetic_field *= self.internal_magnetic_units

        h = self.unit_reg.planck_constant # 6.62607015E-34 # Planck's constant, J s 
        planck_inverse = (1. / h) # Convert from Joule to Hz via Plancks constant  

        # Convention where the nuclear Zeeman term has \mu_{B} as its prefactor, with gi factor taking into account the nuclear magneton, e.g.: mu_b x gi x I x Bz / hbar 
        # Zeeman matrix is diagonal: I_{z} |J, m_{J}, I, m_{I} > = m_{I} |J, m_{J}, I, m_{I} >
        if self.approximation is None:       
            for i, (mj, mi) in enumerate(self.basis_states):
                H[i,i] = (self.lande_gj * mj) + (self.lande_gi * mi)
        elif self.approximation == 'weak field' :
            for i, (f, mf) in enumerate(self.basis_states): 
                H[i,i] = (self.lande_gf(f) * mf)
        else:
            raise ValueError("Invalid approximation specified. `approximation` constructor argument must be either None or 'weak field'.")

        # Apply the units to the entire array simultaneously
        return (H * magnetic_field * self.mu_b * planck_inverse).to(self.internal_freq_units) 

    def solve_at_field(self, magnetic_field: float) -> tuple[NDArray, NDArray]:
        """
        Diagonalize the Hamiltonian for eigen-energies at a specific magnetic field strength magnetic_field.

        Returns: 
            - energy eigenvalues (NDArray)
            - energy eigenvectors (d x d array), i.e. each eigenvector is a vector of dimension d

        Note: These energies are not in the sorted order that matches the basis states. 
        eigh() returns energies, eigenvectors in ascending order (lowest energy first) 
        """
        H_total = self.zeeman_hamiltonian(magnetic_field)
        if self.approximation is None:
            H_total += self.hyperfine_hamiltonian() 
        energies, eigenvectors = eigh(H_total.magnitude)
        return energies, eigenvectors

    def f_character(self, eigenvector: NDArray, f: float) -> float:
        """ Compute how much of |F, mF> character an eigenstate has. 
        Requires Clebsch-Gordan decomposition to project onto |F,mF>

        Returns the probability of finding state in F manifold (p in [0,1])"""

        j, i = self.j, self.i
        if f < abs(j - i) or f > j + i :
            return 0.

        tol = 1e-8
        probability = 0.
        for mf in np.arange(-f, f + 1, 1):
            amplitude = 0.
            for k, (mj, mi) in enumerate(self.basis_states):
                if abs(mj + mi - mf) < tol:
                    # C-G coeff:
                    cg = self.Clebsch_Gordan(j, i, f, mj, mi, mf)
                    amplitude += cg * eigenvector[k]

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

    def mj_mi_labels(self, eigvectors: NDArray) -> tuple[NDArray, NDArray]:
        """ Function to compute the estimated mj, mi labels for the eigenvectors.
            - assuming the eigenvectors are for a single magnetic field value """

        if len(eigvectors.shape) == 2:
            num_magnetic_fields = 1
        else:
            num_magnetic_fields = eigvectors.shape[0]

        mj_labels = np.zeros((num_magnetic_fields, self.dim))
        mi_labels = np.zeros_like(mj_labels)

        for i in range(num_magnetic_fields):
            for j in range(self.dim):
                psi = eigvectors[i, :, j]

                # Find basis with maximum probability 
                probs = np.abs(psi)**2
                dominant_indx = np.argmax(probs)

                # (mj, mi) for this basis state 
                mj, mi = self.basis_states[dominant_indx]

                mj_labels[i, j] = mj
                mi_labels[i, j] = mi

        return mj_labels, mi_labels

    def f_mf_labels(self, eigvectors: NDArray) -> tuple[NDArray, NDArray]:
        """ Computes state labels |F, mF> for an energy eigenstates.
        
        Compute expectation of m_{F} in each eigenstate |psi> : 
        m_{F} = < psi | m_{J} + m_{I} | psi >

        Returns a 2D array of dimension N_Bfields x d 
        """
        if len(eigvectors.shape) == 2:
            num_magnetic_fields = 1
        else:
            num_magnetic_fields = eigvectors.shape[0]

        mf_array = np.zeros((num_magnetic_fields, self.dim))
        dominant_f = np.zeros_like(mf_array)
        f_range = np.arange(np.abs(self.i - self.j), self.i + self.j + 1)

        # Compute expectation for each B-field value via a sum over eigenstates
        for i in range(num_magnetic_fields):
            for j in range(self.dim):
                # j'th eigenvector:
                if num_magnetic_fields == 1:
                    psi = eigvectors[:, j]
                else:
                    psi = eigvectors[i, :, j]

                mf_avg = 0.
                if self.approximation is None :
                    for k, (mj, mi) in enumerate(self.basis_states):
                        mf_avg += (np.abs(psi[k])**2)*(mj + mi)
                elif self.approximation == 'weak field' :
                    for  k, (f, mf) in enumerate(self.basis_states):
                        mf_avg += (np.abs(psi[k])**2)*(mf)

                mf_array[i, j] = mf_avg
                # Compute F character 
                f_chars = {}
                for f in f_range:
                    f_char = self.f_character(psi, f)
                    f_chars[f] = f_char
                # Dominant F value: 
                dominant_f[i,j] = max(f_chars.items(), key = lambda x: x[1])[0]

        return dominant_f, mf_array

    def compute_zeeman_splitting(self, magnetic_fields: NDArray) -> dict:
        """ 
        Computes the energy as a function of magnetic field for the basis states of interest.

        Inputs:
          - magnetic_fields is a 1D array containing magnetic field values 
        
        Outputs: a dictionary of the form { 'magnetic field', energies, eigenvectors}.
        The energies are a 2D array of dimension len(magnetic_fields) x len(basis_states).
        The eigenvectors are a 3D array of dimension len(magnetic_fields) x len(basis_states) x len(basis_states)
        """
        energies = np.zeros((len(magnetic_fields), self.dim))
        eigenvectors = np.zeros((len(magnetic_fields), self.dim, self.dim))

        for i, magnetic_field in enumerate(magnetic_fields):
            energy, eigenvector = self.solve_at_field(magnetic_field)
            energies[i, :] = energy
            eigenvectors[i, :, :] = eigenvector 
        
        # Compute m_{F} = m_{J} + m_{I} for labeling 
        f_list, mf_list = self.f_mf_labels(eigenvectors)
        mj_list, mi_list = self.mj_mi_labels(eigenvectors)
        if self.approximation is None :
            return { 'magnetic field' : magnetic_fields, 'energies' : energies, 
                'eigenvectors' : eigenvectors, 'f' : f_list, 'mf' : mf_list,
                'mj' : mj_list, 'mi' : mi_list}
        elif self.approximation == 'weak field':
            return { 'magnetic field' : magnetic_fields, 'energies' : energies, 
                'eigenvectors' : eigenvectors, 'f' : f_list, 'mf' : mf_list}

    
    def get_state_energy(self, energies: NDArray, eigenvectors: NDArray, f: float, 
                            mf: float, tolerance: float = 1e-6) -> float:
        """ Returns the energy for an angular momentum state of interest |F, mF> 
        At zero or low field, F is a good quantum number. At finite field, mf is conserved but F is not. 
        Assumes energies and eigenvectors at one magnetic field condition.
        Returns the state energy. """
        f_values, mf_values = self.f_mf_labels(eigenvectors)

        f_values = f_values[0]
        mf_values = mf_values[0]

        matching_states = []
        requested_state = (f, mf)
        
        # From list of F values and mF values, map input to index for corresponding state (F,mF)
        f_range = np.arange(np.abs(self.i - self.j), self.i + self.j + 1)
        num_states_per_f = (2*f_range + 1).astype(int)
        try:
            f_indx = list(f_range).index(f)
        except Exception as exc:
            raise ValueError(f"Invalid F value {F}. Please choose in the range {f_range}") from exc
         
        approximate_states = list(zip(f_values, mf_values)) 
        state_indx = -1
        # mf_values lists all mF indices (for all F states considered); 
        # When looping over mF values, make sure the mF value corresponds to the requested F state 
        for i, value in enumerate(mf_values):
            if (np.abs(value - mf) < tolerance) and (np.abs(f_values[i] - f) < tolerance ):
                assert state_indx == -1 # Enforces that there is only 1 correct match 
                state_indx = i
        
        if state_indx == -1:
            raise ValueError(f"Basis state mF = {mf} not found in list of basis states: {mf_values}.")
        return energies[state_indx]


    def get_state_energy_from_mjmi_pair(self, energies: NDArray, eigenvectors: NDArray, mj: float, mi: float) -> float:
        """ Returns the energy for a basis angular momentum state of interest: |mj, mi> 
        At zero or low field, F is a good quantum number. At high field, (mj, mi) are good quantum numbers/
        Assumes energies and eigenvectors at one magnetic field condition.
        Returns state energy. """
        assert self.approximation is None, "Error. Request energies from (mj, mi) pairs only in exact basis."
        # Get index at mj, mi pair 
        mj_states = [state[0] for state in self.basis_states]
        mi_states = [state[1] for state in self.basis_states]
        if mi not in mi_states or mj not in mj_states:
            raise ValueError(f"Requested (mj, mi) ({mj}, {mi}) state not found. Please request a valid state.")

        requested_state = (mj, mi)

        # Find eigenstate with dominant mj, mi character
        best_match = -1
        best_overlap = 0
        # Index of mj,mi pair in the basis 
        basis_indx = self.basis_states.index((mj, mi))

        # Compute probability of being in (mj, mi) state 
        for i in range(self.dim):
            overlap = np.abs(eigenvectors[basis_indx, i])**2
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = i
        assert best_match != -1, "overlap with all eigenvectors is zero"
        return energies[best_match]


    @property
    def lande_gl(self):
        """ Compute Lande g factor for orbital angular momentum: """
        """ gL = 1. - (m_e / nuclear_mass) """
        m_e = constants.electron_mass * self.unit_reg("kg") # mass of electron in kg 
        m_e = m_e.to("dalton") # Daltons   
        # TODO: implement a unit test to see if yaml file has the Z argument. This would keep future developers consistent. 
        if self.z is None:
            return 1.
        else:
            nuclear_mass = self.atomic_mass - (self.z)*m_e # Daltons
            return 1. - (m_e / nuclear_mass)

    @property
    def lande_gi(self):
        """ Compute Lande g factor for nuclear angular momentum: """
        if self.gi is None:
            assert self.nuclear_moment != None, "Please specify nuclear magnetic moment or Lande gi factor."
            ratio_mu_n_to_mu_b = self.mu_n/self.mu_b 
            self.gi = -(self.nuclear_moment/self.i)*ratio_mu_n_to_mu_b 
        return self.gi

    @property
    def lande_gj(self) -> None | float:
        ''' Computes Lande factor for total electron angular momentum J '''
        gs = np.abs(constants.physical_constants['electron g factor'][0])  # electron spin g factor. 
        jjp1 = self.j*(self.j+1)
        if jjp1 == 0:
            return 0.
        else:
            llp1 = self.l*(self.l+1)
            ssp1 = self.s*(self.s+1)
            return (self.lande_gl*(jjp1 - ssp1 + llp1) + gs*(jjp1 + ssp1 - llp1) )*0.5/jjp1

    def lande_gf(self, f: float) -> float:
        """ Lande g factor for total hyperfine angular momentum. 
        - takes in "F" the total angular momentum magnitude as an input """
        ffp1 = f*(f+1)
        if ffp1 == 0:
            raise ValueError('Division by zero error. Do not use this function if F = 0.')
        iip1 = self.i*(self.i+1)
        jjp1 = self.j*(self.j+1)
        return ((self.lande_gj*0.5*(ffp1 - iip1 + jjp1)) + self.lande_gi*0.5*(ffp1 + iip1 - jjp1))/ffp1
