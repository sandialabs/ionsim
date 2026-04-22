import numpy as np
from numpy.typing import NDArray
import scipy.constants as const
import warnings
import scipy.optimize as opt    
from ionsim.custom_types import Matrix, Vector
from ionsim.degree_of_freedom import AtomicSpin
from ionsim.basis import Basis 

########## List of substantive changes: 
## 1. Made dimensionless variable function take in inputs so the user can choose the charge, mass, and trap freq scales to use.   
## 2. Added a helper function to pack/unpack the ion coordinates to reduce code. 

########## List of non-substantive changes: 
## 1. Variable changes for readability (e.g. omega_x instead of wx) 
## 2. Type-hinting in functions  
## 3. Consolidated the functions for solving for ion positions equilibria 
## 4. Moved eigenvector helper functions into the class. 
## 5. Reduced number of matrices that we store as class attributes: Need to check whether we want to be storing those. 

########## Questions: 
# -1. Should the branch sorting be the default option? This seems like what we would want to do for our problems  
# 0. LD parameter matrix shape? 
# 1. Is there a need for the "has run" boolean? e.g. is there a time where it stays False? Ah I think it refers to whether the eqb solver has been successfully executed. 
# 2. Should the dimensionless parameters have a naming convention, e.g. omega_x --> omega_x_ND 
# 3. What sets the phases of the Lamb-Dicke parameters?  
# 4. Does the LD calculation properly do k dot r?
# 5. Can we vectorize this to convert loops into vector operations?  
# 6. See TODO's 


def characteristic_length(q: float, mass: float, omega: float) -> float:
    """ Computes characteristic length in trapped ion system """
    k_e = 1 / (4 * np.pi * const.epsilon_0)  # Coulomb constant
    l0 = ((k_e * q ** 2) / (.5 * mass * omega ** 2)) ** (1 / 3)
    return l0


class TrappedIonModeAnalysis:
    def __init__(self, num_ions: int, omega_x: float, omega_y: float, omega_z: float, atomic_masses: Vector | float, atomic_numbers: Vector | int): 
        """ Class for determining properties of trapped-ion phonon modes, requiring the following system parameters:

            - num_ions: the number of ions
            - omega_x: harmonic trap frequency in the x direction in units of rad/s. 
            - omega_y: harmonic trap frequency in the y direction in units of rad/s. 
            - omega_z: harmonic trap frequency in the z direction. This is the "axial" direction used for 1D chains.  
            - atomic masses: array of atomic masses or a single number => same mass for each ion. Units are amu (atomic mass units) 
            - atomic numbers: array of (Z) atomic numbers (# of protons of an element) or a single number => all ions are the same.  

        """ 
        # TODO: Decide how to handle units and how much of pint we should use.  
        self.num_ions = num_ions

        self.atomic_masses = self.convert_to_array(atomic_masses) * const.u # kg from amu 
        self.atomic_numbers = self.convert_to_array(atomic_numbers)

        # Safety checks: 
        assert len(self.atomic_masses) == num_ions
        assert len(self.atomic_numbers) == num_ions

        # Charge of all the protons -> total nuclear charge:
        self.nuclear_charges = self.atomic_numbers * const.e # Z * e, const.e ==> elementary charge in Coulombs 

        # Trapping frequencies for each ion 
        # TODO: loop and vectorize trap frequencies?  
        self.omega_x = self.calculate_species_trap_frequencies(omega_x)   
        self.omega_y = self.calculate_species_trap_frequencies(omega_y)
        self.omega_z = self.calculate_species_trap_frequencies(omega_z)    

        # Check that the trap is stable: 
        if not self.trap_is_stable():
            raise IonSimError(f"Trap is unstable from negative trap frequencies.")

    def calculate_species_trap_frequencies(self, omega: float) -> Vector: 
        """ Computes relative trap frequencies for each species:

            w_i = sqrt(q_{i} m_0 / m_{i} q_0) omega

            from omega_(secular, i) = sqrt(q_i V_0 / (2 m_i d^2) ) for the i'th ion. 

        """ 
        # assume that the trapping frequency given corresponds to the first ion species 
        q0 = self.nuclear_charges[0] # charge of first ion 
        m0 = self.atomic_masses[0] # charge of first ion 
        species_trap_frequencies = np.sqrt(self.nuclear_charges * m0 /(q0 * self.atomic_masses)) * omega 
        return species_trap_frequencies 
   
    def trap_is_stable(self):
        # Checks that all trap frequencies are positive  
        return np.all(self.omega_z > 0) and np.all(self.omega_y > 0) and np.all(self.omega_x > 0)

    def convert_to_array(self, x: float | int | list | NDArray) -> NDArray:
        """ Converts a scalar to a vector or returns the vector """ 
        if not hasattr(x, "__len__"):
            x = np.ones(self.num_ions) * x 
            return x
        return np.array(x) 

    #def dimensionless_parameters(self):
    #def nondimensionalize_parameters(self):
    def convert_parameters_to_dimensionless(self, charge_scale: float=1., mass_scale: float=1., trap_freq_scale: float=1.):
        """ Compute dimensionless parameters, using first ion's properties and axial (z) trapping frequency. 

            - characteristic length
            - characteristic time 
            - characteristic velocity 
            - characteristic energy 

        """
        # Check positivity:
        if trap_freq_scale <= 0:
            raise IonSimError(f"Trap frequency scale should be positive. Received {trap_freq_scale}")
        if mass_scale <= 0:
            raise IonSimError(f"Mass scale should be positive. Received {mass_scale}")
 #        if charge_scale <= 0:
 #            raise IonSimError(f"Charge scale should be positive. Received {charge_scale}")

        # Store the scales so they are retrievable by the user 
        self.charge_scale = charge_scale
        self.mass_scale = mass_scale
        self.trap_freq_scale = trap_freq_scale

        # Normalize to the first ion species and axial (z) trap frequency 
        # ion properties    
        self.m = self.atomic_masses / self.mass_scale 
        self.q = self.nuclear_charges / self.charge_scale 

        # trap frequencies
        self.wz = self.omega_z / self.trap_freq_scale 
        self.wy = self.omega_y / self.trap_freq_scale 
        self.wx = self.omega_x / self.trap_freq_scale  
        self.trap_frequencies_ND = [self.wx, self.wy, self.wz]

        # TODO: Is hbar set to 1? Reconcile hbar somewhere  
        # Compute characteristic length, time, velocity, and energy scales and store in a dictionary  
        self.characteristic_parameters = {}
        self.characteristic_parameters['length'] = characteristic_length(self.charge_scale, self.mass_scale, self.trap_freq_scale)  
        self.characteristic_parameters['time'] = 1 / trap_freq_scale # characteristic time
        self.characteristic_parameters['velocity'] = self.characteristic_parameters['length'] * trap_freq_scale  # characteristic velocity
        self.characteristic_parameters['energy'] = 0.5 * mass_scale * self.characteristic_parameters['velocity'] ** 2  # characteristic energy  


    def get_eigenvector_norm(self, eigenvector: Vector, H: Matrix) -> float:
        """ Computes the norm of the eigenvector en w.r.t. the Hamiltonian H. """
        norm = np.sqrt(eigenvector.T.conj() @ H @ eigenvector)
        return norm
    
    
    def get_canonical_transformation(self, H, eigenvectors, eigenvalues: Vector | None=None):
        """ Given the eigensolve of the dynamical matrix, get the transform matrix to the canonical coordinates.
            X = S X', where X' = (Q,P)^T and X = (q,p)^T
        """
        ## TODO: understand the sign in the transformation matrix
        sign = -1
        num_coords, num_eigenvalues = np.shape(eigenvectors)
        assert num_coords //2 == num_eigenvalues 
        T = np.zeros((num_coords,num_coords),dtype=complex) 
        eigenvectors = self.normalize_eigenvectors(eigenvectors, H, eigenvalues)
        T = np.sqrt(2)*np.concatenate((np.real(eigenvectors), sign*np.imag(eigenvectors)), axis=1)
        # the sign is likely due to the convention of writing the time evolution as exp(-iwt)
        return T

    def check_for_zero_modes(self):
        assert np.all(self.eigvals > 0), "All eigenvalues must be positive"   

    def check_outer_relation(self, H: Matrix): 
        D = self.build_symplectic_matrix() @ H
        Eval, Evec = np.linalg.eig(D)
        _, en = self.sort_modes(Eval,Evec)
        en = self.normalize_eigenvectors(en,H)
        Outers = np.zeros((6*self.num_ions,6*self.num_ions),dtype=complex)
        for i in range(6*self.num_ions):
            norm = self.get_eigenvector_norm(en[:,i],H)
            Outers = Outers + np.outer(en[:,i],en[:,i].conj())/norm 
        I_right = H @ Outers
        I_left  = Outers @ H
        eye = np.eye(6*self.num_ions,dtype=complex)
        np.set_printoptions(precision=2, suppress=True) 
        try:
            assert np.allclose(I_left,eye) 
            assert np.allclose(I_right,eye)
        except AssertionError:
            warnings.warn("Outer relation check failed")

    def check_diagonalization(self, T: Matrix, S: Matrix, E: Matrix) -> bool:
        M = np.linalg.inv(T) @ S
        H_diag = M.T @ E @ M    
        H_diag_check = np.diag(np.tile(self.eigvals,2)) 
        np.set_printoptions(precision=2, suppress=True) 
        try:
            assert np.allclose(H_diag,H_diag_check)
        except AssertionError:
            has_duplicate_eigenvalues = np.any(np.triu(np.isclose(eigenvalues[:, None], eigenvalues[None, :], atol = 1E-6), k=1))
            warnings.warn("Diagnolization check failed")    
            print("has duplicate eigenvalues: ", has_duplicate_eigvals)

    #def run(self):

    def solve_ion_trap_equilibrium(self): 
        """ Diagonalizes the Coulomb + harmonic trap Hamiltonian for a system of ions. """
        # Convert to dimensionless units using axial trap frequency and first ion's mass and charge  
        # TODO: take in input here or in class constructor for mass, charge, trap scales. 
        self.convert_parameters_to_dimensionless(self.nuclear_charges[0], self.atomic_masses[0], self.omega_z[0])
        
        self.equilibrium_positions = self.solve_for_equilibrium_positions()
        self.reindex_ions()

        # Compute helper matrices for computing normal mode properties 
        mass_matrix = self.build_mass_matrix(self.m)  
        E_matrix = self.build_E_matrix(self.equilibrium_positions, mass_matrix)  
        T_matrix = self.build_momentum_transform_matrix(mass_matrix)
        H_matrix = self.compute_H_matrix(T_matrix, E_matrix)   

        self.eigvals, self.eigvecs = self.calculate_normal_modes(H_matrix)
        self.eigvecs_vel = self.get_eigenvectors_xv_coords(T_matrix, self.eigvecs)    
        self.check_for_zero_modes() 
        S_matrix = self.get_canonical_transformation(H_matrix, self.eigvecs, self.eigvals) 

        # Perform checks 
        self.check_outer_relation(H_matrix)
        self.check_diagonalization(T_matrix, S_matrix, E_matrix)
        #self.hasrun = True  


    def normalize_eigenvectors(self, eigvecs, H: Matrix, eigenvalues: Vector | None=None): 
        """ Rescale the eigenvectors ens w.r.t. the Hamiltonian H. """
        eigvecs_rescaled = np.zeros_like(eigvecs,dtype=complex)
        num_coords, num_eigenvalues = np.shape(eigvecs)
        if eigenvalues is None:
            eigenvalues = np.ones(num_eigenvalues)
        for i in range(num_eigenvalues):
            en = eigvecs[:,i].reshape(num_coords,1)
            norm = self.get_eigenvector_norm(en,H)
            eigvecs_rescaled[:,i] = en[:,0]/norm
            eigvecs_rescaled[:,i] *= np.sqrt(eigenvalues[i])
        return eigvecs_rescaled 


    def get_eigenvectors_xv_coords(self,T,ens): 
        ens_vel = np.zeros_like(ens,dtype=complex)
        ens_vel[:,:] = np.linalg.inv(T) @ ens
        return ens_vel 


    def calculate_normal_modes(self, H_matrix: Matrix) -> (Vector, list[Vector]):
        """ Compute normal mode frequencies from eigenvalues of the matrix D. Returns eigenvalues and eigenvectors.

            Solves det( D + i*omega*Identity) == 0 for omega.
            
            The matrix "D" is defind as D = JH, where J = (0 , I ; I, 0) (eq. 1.67 of thesis)

        """
        J = self.build_symplectic_matrix()
        D_matrix = J @ H_matrix  
        # u_n(t) = exp(-i w_n t) u_n(0), minus by convention
        eigvals, eigvecs = np.linalg.eig(-D_matrix)   
        eigvals, eigvecs = self.organize_modes(eigvals, eigvecs)
        eigvecs = self.normalize_eigenvectors(eigvecs, H_matrix) 

        return eigvals, eigvecs 


    def ion_coordinates_from_flattened(self, flattened_coordinate_vector: Vector) -> tuple[Vector, Vector, Vector]:
        """ Returns arrays corresponding to an ion's x, y, and z coordinates, respectively, from a flattened vector u:
                x = u[0:N], y = u[N:2N], z[2N:]

            e.g. x[1], y[1], z[1] is the x, y, z coordinate values for ion 2. 
        """
        x = flattened_coordinate_vector[0:self.num_ions]
        y = flattened_coordinate_vector[self.num_ions:2*self.num_ions]
        z = flattened_coordinate_vector[2*self.num_ions:]
        return x, y, z 
        

    def reindex_ions(self):  
        """ Re-indexes the ions to order based on the distance from the center of the trap, where smallest index is closest to the center """
        # TODO: How are even/odd cases is handled? 
        x,y,z = self.ion_coordinates_from_flattened(self.equilibrium_positions)
        r = np.sqrt(x**2 + y**2 + z**2)
        idx = np.argsort(r)

        reindexed_positions = np.hstack((x[idx], y[idx], z[idx]))
        self.equilibrium_positions = reindexed_positions 
        self.m = self.m[idx]
        self.q = self.q[idx]
        self.atomic_masses = self.atomic_masses[idx]
        self.nuclear_charges = self.nuclear_charges[idx]
        self.atomic_numbers = self.atomic_numbers[idx]


    def solve_for_equilibrium_positions(self, positions_guess: Vector | None=None):
        """ Solves for equilibrium position vector: u, which represents a flattened spatial grid. """
        # TODO: option for an initial guess choice? 
        # Set the initial guess for the solver  
        if positions_guess == None:
            # Initialize a random guess  
            u0 = np.zeros(3*self.num_ions)
            u0[:] = (np.random.rand(3*self.num_ions) * 2 - 1) * self.num_ions 
        else:
            u0 = positions_guess

        # Solve for the equilibrium positions by minimizing the potential energy (trap + Coulomb) 
        bfgs_tolerance = 1e-34
        solver_output = opt.minimize(self.potential_energy, u0, method='BFGS', jac=self.force,
                                    options={'gtol': bfgs_tolerance, 'disp': False})
        equilibrium_positions = solver_output.x 

        # Get potential energy at equilibrium positions  
        self.p0 = self.potential_energy(equilibrium_positions)
        return equilibrium_positions 


    def potential_trap(self, positions: Vector) -> Vector:
        """ Computes trapping potential, takes in a flattened (1D) array of all position coordinates """
        r = self.ion_coordinates_from_flattened(positions)
        V_trap = 0.
        for coord, omega in zip(r, self.trap_frequencies_ND): 
            V_trap += 0.5 * np.sum((self.m * omega**2) * coord**2)
        return V_trap
    
    def potential_coulomb(self, positions: Vector) -> Vector:
        """ Computes Coulomb potential, takes in a flattened (1D) array of all position coordinates """
        r = self.ion_coordinates_from_flattened(positions)
        dr = []
        for coord in r:
            dr.append(coord[:, np.newaxis] - coord)

        rsep = np.sqrt(dr[0]**2 + dr[1]**2 + dr[2]**2).astype(np.float64)
        qq = (self.q * self.q[:, np.newaxis]).astype(np.float64)    
        with np.errstate(divide='ignore'):
            V_Coulomb = np.sum( np.where(rsep != 0., qq / rsep, 0) ) / 2 # divide by 2 to avoid double counting
        V_Coulomb *= .5 
        return V_Coulomb

    def potential_energy(self, positions: Vector) -> Vector:
        """ Computes the potential energy of each ion in its coordinate basis """ 
        return self.potential_trap(positions) + self.potential_coulomb(positions)   

    def force_trap(self, positions: Vector):
        r = self.ion_coordinates_from_flattened(positions)

        F_r = []
        for coord, omega in zip(r, self.trap_frequencies_ND): 
            F_r.append(self.m * omega**2 * coord)

        force_trap = np.hstack(F_r)
        return force_trap

    def force_coulomb(self, positions: Vector): 
        """ Computes the Coulomb force by derivative w.r.t. ion coordinates """
        r = self.ion_coordinates_from_flattened(positions)
        dr = []

        for coord in r:
            dr.append(coord[:, np.newaxis] - coord)
        rsep = np.sqrt(dr[0]**2 + dr[1]**2 + dr[2]**2).astype(np.float64)
        qq = (self.q * self.q[:, np.newaxis]).astype(np.float64)    

        with np.errstate(divide='ignore', invalid='ignore'):
            rsep3 = np.where(rsep != 0., rsep ** (-3), 0)

        F_r = [] # Force in each direction 
        for dx in dr:
            f_r = dx * rsep3 * qq
            F_r.append(-np.sum(np.array(f_r), axis=1))

        force_coulomb = np.hstack(F_r)
        force_coulomb *= 0.5    
        return force_coulomb

    def force(self, positions: Vector) -> Vector:
        """ Computes the force vector, e.g. derivative of the potential energy w.r.t. ion position: dU/dx, dU/dy, dU/dz. 

            - returns a one-dimensional length-3N array characterizing forces w.r.t. ion coordinates for N ions. 
        
        """
        Force = self.force_coulomb(positions) + self.force_trap(positions)  
        return Force

    def hessian_trap(self, positions: Vector) -> Matrix:
        """ Computes the Hessian of the trap  """ 
        H_rr = []
        for i, omega in enumerate(self.trap_frequencies_ND): 
            H_rr.append(np.diag(self.m * omega**2 * np.ones(self.num_ions)))
        zeros = np.zeros((self.num_ions, self.num_ions))  
        H = np.block([[H_rr[0], zeros, zeros], [zeros, H_rr[1], zeros], [zeros, zeros, H_rr[2]]])
        return H

    def hessian_coulomb(self, positions: Vector) -> Matrix:
        """ Computes the Hessian of the Coulomb interaction """ 
        r = self.ion_coordinates_from_flattened(positions)
        dr = []

        for coord in r:
            dr.append(coord[:, np.newaxis] - coord)
        rsep = np.sqrt(dr[0]**2 + dr[1]**2 + dr[2]**2).astype(np.float64)
        qq = (self.q * self.q[:, np.newaxis]).astype(np.float64)    

        with np.errstate(divide='ignore'):
            rsep5 = np.where(rsep != 0., rsep ** (-5), 0)

        dr_sq = np.array(dr)**2
        rsep2 = rsep ** 2

        # X derivatives, Y derivatives for alpha != beta
        H_rr = (rsep2 - 3 * dr_sq) * rsep5 # form: [Hxx, Hyy, Hzz] 

        # Above, for alpha == beta
        # Compute diagonals: 
        for i in range(3):
            H_rr[i][np.diag_indices(self.num_ions)] = -np.sum(H_rr[i], axis=0)
        dx = dr[0]
        dy = dr[1]
        dz = dr[2]

        # Off-diagonal elements:
        Hxy = -3 * dx * dy * rsep5
        Hxy[np.diag_indices(self.num_ions)] = 3 * np.sum(dx * dy * rsep5, axis=0)
        Hxz = -3 * dx * dz * rsep5
        Hxz[np.diag_indices(self.num_ions)] = 3 * np.sum(dx * dz * rsep5, axis=0)
        Hyz = -3 * dy * dz * rsep5
        Hyz[np.diag_indices(self.num_ions)] = 3 * np.sum(dy * dz * rsep5, axis=0)
        
        H_rr *= qq
        Hxy *= qq
        Hxz *= qq
        Hyz *= qq

        H_coulomb = np.block([[H_rr[0], Hxy, Hxz], [Hxy, H_rr[1], Hyz], [Hxz, Hyz, H_rr[2]]])
        H_coulomb /= 2
        return H_coulomb


    def hessian(self, positions: Vector) -> Matrix:
        """ Computes total Hessian (trap + Coulomb interaction) """
        H = self.hessian_coulomb(positions) + self.hessian_trap(positions)  
        return H

    def build_mass_matrix(self, masses: Vector) -> Matrix: 
        """ Builds a diagonal mass matrix representing the ions """ 
        return np.diag(np.tile(masses, 3)) 

    def build_E_matrix(self, positions: Vector, mass_matrix: Matrix) -> Matrix: 
        """ 6N x 6N energy matrix for N ions """ 
        PE_matrix = np.zeros((3*self.num_ions, 3*self.num_ions), dtype=np.complex128)
        KE_matrix = np.zeros((3*self.num_ions, 3*self.num_ions), dtype=np.complex128)
        E_matrix = np.zeros((6*self.num_ions, 6*self.num_ions), dtype=np.complex128)

        PE_matrix = self.hessian(positions)
        KE_matrix = mass_matrix 
        zeros = np.zeros((3*self.num_ions, 3*self.num_ions)) 
        E_matrix = np.block([[PE_matrix, zeros], [zeros, KE_matrix]])
        return E_matrix

    def compute_H_matrix(self, T_matrix, E_matrix):  
        """ Computes 6N x 6N Hermitian Hamiltonian matrix in (q, p) canonical coordinates """ 
        T_matrix_inv = np.linalg.inv(T_matrix)  
        H_matrix = T_matrix_inv.T @ E_matrix @ T_matrix_inv
        return H_matrix 

    def build_momentum_transform_matrix(self, mass_matrix: Matrix) -> Matrix:
        # assuming no magnetic field
        eye = np.eye(3*self.num_ions)  
        zeros = np.zeros((3*self.num_ions, 3*self.num_ions))
        T = np.block([[eye, zeros], [zeros, mass_matrix]])  
        return T    

    def build_symplectic_matrix(self) -> Matrix:
        """ Computes J a 6N x 6N symplectic matrix defined as J = (0 , I ; I, 0) (eq. 1.67 of thesis) """
        zeros = np.zeros((3*self.num_ions, 3*self.num_ions), dtype=np.complex128)
        I = np.eye(3*self.num_ions, dtype=np.complex128)
        J = np.block([[zeros, I], [-I, zeros]])
        return J    

    ### Mode organizing helper methods  
    def sort_modes(self, eigvals, eigvecs):
       eigvals = np.imag(eigvals)
       sort_dex = np.argsort(eigvals)
       eigvals = eigvals[sort_dex]
       eigvecs = eigvecs[:,sort_dex]
       return eigvals, eigvecs  
    
    def split_modes(self, eigvals, eigvecs): 
       half = len(eigvals) // 2
       eigvals = eigvals[half:]
       eigvecs = eigvecs[:,half:]
       return eigvals, eigvecs
    

    def organize_modes(self, eigvals, eigvecs):  
        eigvals, eigvecs = self.sort_modes(eigvals, eigvecs)    
        eigvals, eigvecs = self.split_modes(eigvals, eigvecs)
        return eigvals, eigvecs 


    def calculate_mode_participation_factors(self) -> Matrix:
        """ Computes ion-mode participation factors, related to the Lamb-Dicke parameters.

            Mode participation factors (eta) take the following matrix form:
             
            shape: (dimension, ion, mode), for N ions this is (3, N, N) 
            e.g. eta[1, 2, 3] is eta in the "y" direction, ion 1, and the 2nd mode. 

            For N ions, this form organizes the 3N modes into N modes per direction "d", where d = x, y, z. 

        """ 

        # TODO: Question: shouldn't the shape be (d, N, N) for d dimensions (3), N ions, and N modes per direction. 
        eigvecs = self.eigvecs
        num_coords, num_modes = np.shape(eigvecs) 
        num_ions = num_modes // 3
        mode_participation_factors = np.zeros((3, num_ions, num_modes), dtype = np.complex128)
        for mode_index in range(num_modes):
            for pos_coord in range(num_coords//2):
                direction_index = pos_coord // num_ions
                ion_index = pos_coord % num_ions
                prefactor = np.sqrt(2 * self.m[ion_index] * self.eigvals[mode_index] )
                zpm_dimensionful = np.sqrt(const.hbar / (2 * self.mass_scale * self.trap_freq_scale))
                mode_participation_factors[direction_index, ion_index, mode_index] = zpm_dimensionful * prefactor * eigvecs[pos_coord, mode_index]
        return mode_participation_factors

    # Derived properties 
    def compute_reference_single_ion_lamb_dicke_factors(self, wavenumber: float) -> (float, float, float):
        """ Computes analytical Lamb-Dicke parameter by eta = k * sqrt(hbar / m omega) for an ion in a light-field with wavevector |k| 

            - wavenumber: wavevector magnitude |k| in units of 1 / m  

        """
        # TODO: better way to handle units? 
        # Convention to use first value of trap frequency arrays (representing one of the ions)  
        trap_frequencies = np.array([self.omega_x[0], self.omega_y[0], self.omega_z[0]])        
        mass = self.atomic_masses[0] 
        eta_x, eta_y, eta_z = wavenumber * np.sqrt(const.hbar / (2 * mass * trap_frequencies))
        return eta_x, eta_y, eta_z    # ignore the phase
 
    
#classes
class GeneralizedModeAnalysisWithBranchSortedModes(TrappedIonModeAnalysis):
 
    def _xyz_classify_modes(self, evecs):
        N_coords, N_modes = np.shape(evecs)
        N_ions = N_coords // 6
        classifier = np.zeros(N_modes, dtype=int)
        for mode_index in range(N_modes):
            x_score = np.sum(np.abs(evecs[0:N_ions, mode_index])) + np.sum(np.abs(evecs[3*N_ions:4*N_ions, mode_index]))
            y_score = np.sum(np.abs(evecs[N_ions:2*N_ions, mode_index])) + np.sum(np.abs(evecs[4*N_ions:5*N_ions, mode_index]))
            z_score = np.sum(np.abs(evecs[2*N_ions:3*N_ions, mode_index])) + np.sum(np.abs(evecs[5*N_ions:6*N_ions, mode_index]))
            scores = [x_score, y_score, z_score]
            max_index = np.argmax(scores)
            classifier[mode_index] = max_index
        for direction in range(3):
            count = np.sum(classifier == direction)
            assert count == N_modes // 3, f"Classification error: direction {direction} has {count} modes, expected {N_modes // 3}"
        return classifier
 
    def sort_by_branch(self, evals, evecs):
        """ Sorts the eigenvalues, eigenvectors by mode branch (radial x, radial y, axial (z)) """
        classifier = self._xyz_classify_modes(evecs)
        # within each branch, sort by frequency
        N_ions = len(evals) // 3
        sorted_by_branch_evals = np.zeros_like(evals)
        sorted_by_branch_evecs = np.zeros_like(evecs)
        for direction in range(3):
            direction_indices = np.where(classifier == direction)[0]
            # they are already sorted by frequency from the original mode analysis code, so we can just take them in order
            sorted_by_branch_evals[direction*N_ions:(direction+1)*N_ions] = evals[direction_indices]
            sorted_by_branch_evecs[:, direction*N_ions:(direction+1)*N_ions] = evecs[:, direction_indices]
        return sorted_by_branch_evals, sorted_by_branch_evecs

    # Override from parent  
    def organize_modes(self, eigvals, eigvecs):        
        eigvals, eigvecs = self.sort_modes(eigvals, eigvecs)
        eigvals, eigvecs = self.split_modes(eigvals, eigvecs)
        eigvals, eigvecs = self.sort_by_branch(eigvals, eigvecs)
        return eigvals, eigvecs
 
 
    def reindex_ions_by_z(self): 
        # based on the position along z, lowest i, is 0, up to N - 1
        x,y,z = self.ion_coordinates_from_flattened(self.equilibrium_positions)
 
        idx = np.argsort(z)
        self.equilibrium_positions = np.hstack((x[idx], y[idx], z[idx]))
 
        # Reindex all other arrays accordingly
        self.m = self.m[idx]
        self.q = self.q[idx]
        self.atomic_masses = self.atomic_masses[idx]
        self.nuclear_charges = self.nuclear_charges[idx]
        self.atomic_numbers = self.atomic_numbers[idx]

        # trapping frequencies
        self.omega_x = self.omega_x[idx]
        self.omega_y = self.omega_y[idx]
        self.omega_z = self.omega_z[idx]
        # all ions are the same so this is safe. TODO: When does this change? 
        self.convert_parameters_to_dimensionless(self.nuclear_charges[0], self.atomic_masses[0], self.omega_z[0])
 
    #def run(self):
    def solve_ion_trap_equilibrium(self): 
        """ Diagonalizes the Coulomb + harmonic trap Hamiltonian for a system of ions. """
        self.convert_parameters_to_dimensionless(self.nuclear_charges[0], self.atomic_masses[0], self.omega_z[0])
        
        self.equilibrium_positions = self.solve_for_equilibrium_positions()
        self.reindex_ions_by_z()
        mass_matrix = self.build_mass_matrix(self.m)  
        E_matrix = self.build_E_matrix(self.equilibrium_positions, mass_matrix)  
        T_matrix = self.build_momentum_transform_matrix(mass_matrix)
        H_matrix = self.compute_H_matrix(T_matrix, E_matrix)   

        self.eigvals, self.eigvecs = self.calculate_normal_modes(H_matrix)
        self.eigvecs_vel = self.get_eigenvectors_xv_coords(T_matrix, self.eigvecs)    
        self.check_for_zero_modes()
        self.check_outer_relation(H_matrix)

        S_matrix = self.get_canonical_transformation(H_matrix, self.eigvecs, self.eigvals) 
        self.check_diagonalization(T_matrix, S_matrix, E_matrix)
        #self.hasrun = True  


# Extra methods?
    def return_equilibrium_positions(self, dimensionless: bool) -> Vector:
        """ Return equilibrium positions in either dimensionless or SI units (inverse meters) """
        if dimensionless:
            return self.equilibrium_positions
        else:
            return self.equilibrium_positions * self.characteristic_parameters['length']


    @classmethod
    def from_species(cls, species_name: str, num_ions: int, omega_x: float, omega_y: float, omega_z: float): 
        """ Build the mode analysis class from a species name, corresponding to a system of N ions under harmonic trapping. """ 
        # Import necessary data for the species 
        species_data = AtomicSpin.get_config_data(species_name)
        atomic_mass = species_data['mass']
        atomic_number = species_data['Z']
        # Construct the class 
        return cls(num_ions, omega_x, omega_y, omega_z, atomic_mass, atomic_number)

    @classmethod
    def from_atomic_spin_basis(cls, atomic_structure_basis: StandardBasis, omega_x: float, omega_y: float, omega_z: float): 
        """ Build the mode analysis class from a basis of AtomicSpin degrees of freedom under harmonic trapping. """ 
        # Extract number of ions and the mass and atomic number from the DOF in the basis 
        DOFs = atomic_structure_basis.degrees_of_freedom
        num_ions = len(DOFs)
        atomic_masses = []
        atomic_numbers = []

        for DOF in DOFs:
            if not isinstance(DOF, AtomicSpin):
                raise IonSimError("Atomic structure basis should only contain AtomicSpin or AtomicStructure objects. No motional modes should be included.")
                atomic_masses.append(DOF.atomic_mass) 
                atomic_numbers.append(DOF.atomic_number) 
        # Construct the class 
        return cls(num_ions, omega_x, omega_y, omega_z, atomic_mass, atomic_number)


 
#You could redefine the equilibrium finding function to assert that the equilibrium is linear. For example, make a wrapper inside the function for the potential, Jacobian, and Hessian that forces x_i and y_i = 0. 
 
#This is the code for the Lamb-Dicke parameters: (not that overall phases don't matter here, but the relative phase does. We could pin down the phase with some convention like, "each mode's first non-negative LD value is defined positive." I also include some extra helper functions. 
def two_central_ion_separation(wz_Hz, l_2_cent):
    mass_yb_amu = 170.936 # TODO: hardcoded for now
    four_ion_analysis = GeneralizedModeAnalysisWithBranchSortedModes(num_ions=4, omega_x=2*np.pi*11e6, omega_y=2*np.pi*10e6, omega_z=2*np.pi*wz_Hz, atomic_masses=mass_yb_amu, atomic_numbers = 70)
    four_ion_analysis.solve_ion_trap_equilibrium()
    positions_z = four_ion_analysis.equilibrium_positions[2*4:] * four_ion_analysis.characteristic_parameters['length']
    central_ions = np.argsort(np.abs(positions_z))[:2]
    dl = np.abs(positions_z[central_ions[0]] - positions_z[central_ions[1]])
    return dl - l_2_cent
    
def find_wz_for_desired_central_ion_separation(l_2_cent, bounds=(0.1e6, 0.5e6)):
    result = root_scalar(two_central_ion_separation, args=(l_2_cent,), bracket=bounds, method='bisect')
    if not result.converged:
        raise ValueError("Root finding did not converge. Try adjusting the bounds or check the function behavior.")
    return result.root
    
 
def calc_mode_energies(res, Fock_cutoffs):
    energies_m1 = np.empty(len(res.states))
    energies_m2 = np.empty(len(res.states))
    N_op = qt.num(Fock_cutoffs[0])
    eye = qt.qeye(Fock_cutoffs[0])
    spin_eye = qt.qeye(2)
    E1_op = qt.tensor([spin_eye, N_op, eye])
    E2_op = qt.tensor([spin_eye, eye, N_op])
    for k, state in enumerate(res.states):
        energies_m1[k] = qt.expect(E1_op, state)
        energies_m2[k] = qt.expect(E2_op, state)
    return energies_m1, energies_m2


if __name__ == '__main__':
    print(two_central_ion_separation(10. * 1E6 * 2*np.pi, 0.05)) 
