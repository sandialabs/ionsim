import numpy as np
#from generalized_mode_analysis import GeneralizedModeAnalysis as mode_analyzer
#plt.style.use('ionsim')
from time import time as timer
from numpy.typing import NDArray
import scipy.constants as const
import warnings
import scipy.optimize as opt    




########## List of substantive changes by Ethan: 
## 1. Made dimensionless variable function take in inputs so the user can choose the charge, mass, and trap freq scales to use.   
## 2. Added a helper function to pack/unpack the ion coordinates to reduce code. 

########## List of non-substantive changes by Ethan: 
## 1. Variable changes for readable (e.g. omega_x instead of wx) 
## 2. Type-hinting in functions  
## 3. Consolidated the functions for solving for ion positions equilibria 



def characteristic_length(q: float, mass: float, omega: float):
    """ Computes characteristic length in trapped ion system """
    k_e = 1 / (4 * np.pi * const.epsilon_0)  # Coulomb constant
    l0 = ((k_e * q ** 2) / (.5 * mass * omega ** 2)) ** (1 / 3)
    return l0

def get_norm(en,H):
    """
    Get the norm of the eigen vector en w.r.t. the Hamiltonian H.
    """
    norm = np.sqrt(en.T.conj() @ H @ en)
    return norm


def normalize_eigen_vectors(ens,H,evs=None): 
    """
    Rescale the eigen vectors ens w.r.t. the Hamiltonian H.
    """
    ens_rescaled = np.zeros_like(ens,dtype=complex)
    num_coords,num_evs = np.shape(ens)
    if evs is None:
        evs = np.ones(num_evs)
    for i in range(num_evs):
        en = ens[:,i].reshape(num_coords,1)
        norm = get_norm(en,H)
        ens_rescaled[:,i] = en[:,0]/norm
        ens_rescaled[:,i] *= np.sqrt(evs[i])
    return ens_rescaled 



def get_canonical_transformation(H,ens,evs=None):
    """
    Given the eigen-solve of the dynamical matrix, get the transform matrix 
    to the canonical coordinates.
    X = S X', where X' = (Q,P)^T and X = (q,p)^T
    """
    ## TODO: understand the sign in the transformation matrix
    sign = -1
    num_coords, num_evs = np.shape(ens)
    assert num_coords //2 == num_evs    
    T = np.zeros((num_coords,num_coords),dtype=complex) 
    ens = normalize_eigen_vectors(ens,H,evs=evs)
    T = np.sqrt(2)*np.concatenate((np.real(ens), sign*np.imag(ens)), axis=1)
    # the sign is likely due to the convention of writing the time evolution as exp(-iwt)
    return T


def convert_to_array(x: float | int | list | NDArray):
    """ Converts a scalar to a vector or returns the vector """ 
    if not hasattr(x, "__len__"):
        x = np.ones(N) * x 
        return x
    return np.array(x) 


class TrappedIonModeAnalysis:
    def __init__(self, num_ions: int, omega_x: float, omega_y: float, omega_z: float, atomic_masses: Vector[float] | float, atomic_numbers: Vector[int] | int): 
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

        self.atomic_masses = convert_to_array(atomic_masses) * const.u # kg from amu 
        self.atomic_numbers = convert_to_array(atomic_numbers)

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
            raise IonSimError(f"Error: Trap is unstable from negative trap frequencies.")

        #self.initial_equilibrium_guess = None
        self.hasrun = False 
    

    def calculate_species_trap_frequencies(self, omega: float) -> Vector[float]: 
        """ Computes relative trap frequencies for each species:

            w_i = sqrt(q_{i} m_0 / m_{i} q_0) omega 

        """ 
        # TODO: Does Wes have a ref. for this? 
        # assume that the trapping frequency given corresponds to the first ion species 
        q0 = self.nuclear_charges[0] # charge of first ion 
        m0 = self.atomic_masses[0] # charge of first ion 
        species_trap_frequencies = np.sqrt(self.nuclear_charges * m0 /(q0 * self.atomic_masses)) * omega 
        return species_trap_frequencies 
   

    def trap_is_stable(self):
        # Checks that all trap frequencies are positive  
        return np.all(self.omega_z > 0) and np.all(self.omega_y > 0) and np.all(self.omega_x > 0)


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

        # TODO: Is hbar set to 1? Reconcile hbar somewhere  
        # Compute characteristic length, time, velocity, and energy scales and store in a dictionary  
        self.characteristic_parameters = {}
        self.characteristic_parameters['length'] = characteristic_length(self.charge_scale, self.mass_scale, self.trap_freq_scale)  
        self.characteristic_parameters['time'] = 1 / trap_freq_scale # characteristic time
        self.characteristic_parameters['velocity'] = self.characteristic_parameters['length'] * trap_freq_scale  # characteristic velocity
        self.characteristic_parameters['energy'] = 0.5 * mass_scale * self.characteristic_parameters['velocity'] ** 2  # characteristic energy  

    def check_for_zero_modes(self):
        assert np.all(self.eigvals > 0), "All eigenvalues must be positive"   

    def check_outer_relation(self): 
        H = self.H_matrix.copy()       
        D = self.get_symplectic_matrix() @ H
        Eval, Evec = np.linalg.eig(D)
        _, en = self.sort_modes(Eval,Evec)
        en = self.normalize_eigen_vectors(en,H)
        Outers = np.zeros((6*self.num_ions,6*self.num_ions),dtype=complex)
        for i in range(6*self.num_ions):
            norm = get_norm(en[:,i],H)
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



    def has_duplicate_eigvals(eigvals):
        evs = eigvals.copy()    
        return np.any(np.triu(np.isclose(evs[:, None], evs[None, :], atol=1e-6), k=1))  
    


    def check_diagnolization(self):
        M = np.linalg.inv(self.T_matrix) @ self.S_matrix    
        H_diag = M.T @ self.E_matrix @ M    
        H_diag_check = np.diag(np.tile(self.eigvals,2)) 
        np.set_printoptions(precision=2, suppress=True) 
        try:
            assert np.allclose(H_diag,H_diag_check)
        except AssertionError:
            warnings.warn("Diagnolization check failed")    
            print("has duplicate eigenvalues: ", self.has_duplicate_eigvals(self.eigvals))



    def checks(self):   
        self.check_outer_relation()
        self.check_diagnolization()

    def run(self):
        #self.dimensionless_parameters()
        # Convert to dimensionless units using axial trap frequency and first ion's mass and charge  
        self.convert_parameters_to_dimensionless(self.nuclear_charges[0], self.atomic_masses[0], self.omega_z[0])
        #assert self.trap_is_stable()   # checked in the constructor  
        
        #self.u = self.calculate_equilibrium_positions()
        self.u = self.solve_for_equilibrium_positions()
        self.reindex_ions()
        self.E_matrix = self.get_E_matrix(self.u)  
        self.T_matrix = self.get_momentum_transform() 
        self.H_matrix = self.get_H_matrix(self.T_matrix, self.E_matrix)   

        self.eigvals, self.eigvecs = self.calculate_normal_modes(self.H_matrix)
        self.eigvecs_vel = self.get_eigen_vectors_xv_coords(self.T_matrix,self.eigvecs)    
        self.check_for_zero_modes() 
        self.S_matrix = self.get_canonical_transformation() 
        self.checks() 
        self.hasrun = True  
    

    def get_canonical_transformation(self):
        return get_canonical_transformation(self.H_matrix,self.eigvecs,evs=self.eigvals)    



    def normalize_eigen_vectors(self, eigvecs, H_matrix,evs=None):
        return normalize_eigen_vectors(eigvecs,H_matrix,evs=evs) 



    def get_eigen_vectors_xv_coords(self,T,ens): 
        ens_vel = np.zeros_like(ens,dtype=complex)
        ens_vel[:,:] = np.linalg.inv(T) @ ens
        return ens_vel 


    def calculate_normal_modes(self, H_matrix: Matrix) -> (Vector, list[Vector]):
        """ Compute normal mode frequencies from eigenvalues of the matrix D. Returns eigenvalues and eigenvectors.

            Solves det( D + i*omega*Identity) == 0 for omega.
            
            The matrix "D" is defind as D = JH, where J = (0 , I ; I, 0) (eq. 1.67 of thesis)

        """
        J = self.get_symplectic_matrix()
        D_matrix = J @ H_matrix  
        # u_n(t) = exp(-i w_n t) u_n(0), minus by convention
        eigvals, eigvecs = np.linalg.eig(-D_matrix)   
        eigvals, eigvecs = self.organize_modes(eigvals, eigvecs)
        eigvecs = self.normalize_eigen_vectors(eigvecs, H_matrix) 

        return eigvals, eigvecs 


    def ion_coordinates_from_flattened(self, flattened_coordinate_vector: Vector) -> tuple(Vector, Vector, Vector):
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
        x,y,z = self.ion_coordinates_from_flattened(self.u)
        r = np.sqrt(x**2 + y**2 + z**2)
        idx = np.argsort(r)

        reindexed_positions = np.hstack((x[idx], y[idx], z[idx]))
        self.u = reindexed_positions 
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


    def potential_trap(self, positions: Vector):
        """ Computes trapping potential, takes in a flattened (1D) array of all position coordinates """
        x,y,z = self.ion_coordinates_from_flattened(positions)
        V_trap = 0.5 * np.sum((self.m * self.wx ** 2) * x ** 2) + \
            0.5 * np.sum((self.m * self.wy ** 2) * y ** 2) + \
                0.5 * np.sum((self.m * self.wz ** 2) * z ** 2)
        return V_trap
    
    def potential_coulomb(self, positions: Vector):
        """ Computes Coulomb potential, takes in a flattened (1D) array of all position coordinates """
        x,y,z = self.ion_coordinates_from_flattened(positions)

        dx = x[:, np.newaxis] - x
        dy = y[:, np.newaxis] - y
        dz = z[:, np.newaxis] - z
        rsep = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2).astype(np.float64)
        qq = (self.q * self.q[:, np.newaxis]).astype(np.float64)    

        with np.errstate(divide='ignore'):
            V_Coulomb = np.sum( np.where(rsep != 0., qq / rsep, 0) ) / 2 # divide by 2 to avoid double counting
        V_Coulomb *= .5 
        return V_Coulomb

    def potential_energy(self, positions):
        return self.potential_trap(positions) + self.potential_coulomb(positions)   


    def force_trap(self, positions):
        x,y,z = self.ion_coordinates_from_flattened(positions)

        Ftrapx = self.m * self.wx**2 * x
        Ftrapy = self.m * self.wy**2 * y
        Ftrapz = self.m * self.wz**2 * z

        force_trap = np.hstack((Ftrapx, Ftrapy, Ftrapz))
        return force_trap



    def force_coulomb(self, positions): 
        x,y,z = self.ion_coordinates_from_flattened(positions)

        dx = x[:, np.newaxis] - x
        dy = y[:, np.newaxis] - y
        dz = z[:, np.newaxis] - z
        rsep = np.sqrt(dx**2 + dy**2 + dz**2).astype(np.float64)
        qq = (self.q * self.q[:, np.newaxis]).astype(np.float64)    

        with np.errstate(divide='ignore', invalid='ignore'):
            rsep3 = np.where(rsep != 0., rsep ** (-3), 0)

        fx = dx * rsep3 * qq
        fy = dy * rsep3 * qq
        fz = dz * rsep3 * qq    

        Fx = -np.sum(fx, axis=1)
        Fy = -np.sum(fy, axis=1)
        Fz = -np.sum(fz, axis=1)

        force_coulomb = np.hstack((Fx, Fy, Fz))
        force_coulomb *= 0.5    
        return force_coulomb



    def force(self, positions):
        Force = self.force_coulomb(positions) + self.force_trap(positions)  
        return Force

    def force(self, positions):
        Force = self.force_coulomb(positions) + self.force_trap(positions)  
        return Force


    #### Matrix methods 
    def hessian_trap(self, positions: Vector):
        """ Computes the Hessian of the trap  """ 
        Hxx = np.diag(self.m * (self.wx**2) * np.ones(self.num_ions))
        Hyy = np.diag(self.m * (self.wy**2) * np.ones(self.num_ions))  
        Hzz = np.diag(self.m * (self.wz**2) * np.ones(self.num_ions))  
        zeros = np.zeros((self.num_ions, self.num_ions))  
        H = np.block([[Hxx, zeros, zeros], [zeros, Hyy, zeros], [zeros, zeros, Hzz]])
        return H

    def hessian_coulomb(self, positions: Vector):
        """ Computes the Hessian of the Coulomb interaction """ 
        x,y,z = self.ion_coordinates_from_flattened(positions)
        
        dx = x[:, np.newaxis] - x
        dy = y[:, np.newaxis] - y
        dz = z[:, np.newaxis] - z
        rsep = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2).astype(np.float64)  
        qq = (self.q * self.q[:, np.newaxis]).astype(np.float64)    

        with np.errstate(divide='ignore'):
            rsep5 = np.where(rsep != 0., rsep ** (-5), 0)

        dxsq = dx ** 2
        dysq = dy ** 2
        dzsq = dz ** 2
        rsep2 = rsep ** 2

        # X derivatives, Y derivatives for alpha != beta
        Hxx = (rsep2 - 3 * dxsq) * rsep5
        Hyy = (rsep2 - 3 * dysq) * rsep5
        Hzz = (rsep2 - 3 * dzsq) * rsep5

        # Above, for alpha == beta
        Hxx[np.diag_indices(self.num_ions)] = -np.sum(Hxx, axis=0)
        Hyy[np.diag_indices(self.num_ions)] = -np.sum(Hyy, axis=0)
        Hzz[np.diag_indices(self.num_ions)] = -np.sum(Hzz, axis=0)

        Hxy = -3 * dx * dy * rsep5
        Hxy[np.diag_indices(self.num_ions)] = 3 * np.sum(dx * dy * rsep5, axis=0)
        Hxz = -3 * dx * dz * rsep5
        Hxz[np.diag_indices(self.num_ions)] = 3 * np.sum(dx * dz * rsep5, axis=0)
        Hyz = -3 * dy * dz * rsep5
        Hyz[np.diag_indices(self.num_ions)] = 3 * np.sum(dy * dz * rsep5, axis=0)
        
        Hxx *= qq
        Hyy *= qq
        Hzz *= qq
        Hxy *= qq
        Hxz *= qq
        Hyz *= qq

        H_coulomb = np.block([[Hxx, Hxy, Hxz], [Hxy, Hyy, Hyz], [Hxz, Hyz, Hzz]])
        H_coulomb /= 2
        return H_coulomb


    def hessian(self, positions: Vector):
        """ Computes total Hessian (trap + Coulomb interaction) """
        H = self.hessian_coulomb(positions) + self.hessian_trap(positions)  
        return H

    def get_mass_matrix(self, m): 
        return np.diag(np.tile(m, 3)) 

    def get_E_matrix(self, positions: Vector): 
        """ 6N x 6N energy matrix for N ions """ 
        PE_matrix = np.zeros((3*self.num_ions, 3*self.num_ions), dtype=np.complex128)
        KE_matrix = np.zeros((3*self.num_ions, 3*self.num_ions), dtype=np.complex128)
        E_matrix = np.zeros((6*self.num_ions, 6*self.num_ions), dtype=np.complex128)

        PE_matrix = self.hessian(positions)
        KE_matrix = self.get_mass_matrix(self.m) 
        zeros = np.zeros((3*self.num_ions, 3*self.num_ions)) 
        E_matrix = np.block([[PE_matrix, zeros], [zeros, KE_matrix]])
        return E_matrix

    def get_H_matrix(self, T_matrix, E_matrix):  
        """ Computes 6N x 6N Hermitian Hamiltonian matrix in (q, p) canonical coordinates """ 
        T_matrix_inv = np.linalg.inv(T_matrix)  
        H_matrix = T_matrix_inv.T @ E_matrix @ T_matrix_inv
        return H_matrix 

    def get_momentum_transform(self):
        # assuming no magnetic field
        mass_matrix = self.get_mass_matrix(self.m)  
        eye = np.eye(3*self.num_ions)  
        zeros = np.zeros((3*self.num_ions, 3*self.num_ions))
        T = np.block([[eye, zeros], [zeros, mass_matrix]])  
        return T    

    def get_symplectic_matrix(self):
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
 
    def organize_modes(self, eigvals, eigvecs):        
        eigvals, eigvecs = self.sort_modes(eigvals, eigvecs)
        eigvals, eigvecs = self.split_modes(eigvals, eigvecs)
        eigvals, eigvecs = self.sort_by_branch(eigvals, eigvecs)
        return eigvals, eigvecs
 
 
    def reindex_ions_by_z(self): 
        # based on the position along z, lowest i, is 0, up to N - 1
        x,y,z = self.ion_coordinates_from_flattened(self.u)
 
        idx = np.argsort(z)
        self.u = np.hstack((x[idx], y[idx], z[idx]))
 
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
        #self.dimensionless_parameters()
        self.convert_parameters_to_dimensionless(self.nuclear_charges[0], self.atomic_masses[0], self.omega_z[0])
 
 
    def run(self):
        #self.dimensionless_parameters()
        self.convert_parameters_to_dimensionless(self.nuclear_charges[0], self.atomic_masses[0], self.omega_z[0])
        #assert self.trap_is_stable()    
        
        self.u = self.solve_for_equilibrium_positions()
        self.reindex_ions_by_z()
        self.E_matrix = self.get_E_matrix(self.u)  
        self.T_matrix = self.get_momentum_transform()
        self.H_matrix = self.get_H_matrix(self.T_matrix, self.E_matrix)   
        self.eigvals, self.eigvecs = self.calculate_normal_modes(self.H_matrix)
        self.eigvecs_vel = self.get_eigen_vectors_xv_coords(self.T_matrix,self.eigvecs)    
        self.check_for_zero_modes()
        self.S_matrix = self.get_canonical_transformation()
        self.checks()
        self.hasrun = True  
    
 
#You could redefine the equilibrium finding function to assert that the equilibrium is linear. For example, make a wrapper inside the function for the potential, Jacobian, and Hessian that forces x_i and y_i = 0. 
 
#This is the code for the Lamb-Dicke parameters: (not that overall phases don't matter here, but the relative phase does. We could pin down the phase with some convention like, "each mode's first non-negative LD value is defined positive." I also include some extra helper functions. 
def two_central_ion_separation(wz_Hz, l_2_cent):
    mass_yb_amu = 170.936 # TODO: hardcoded for now
    four_ion_analysis = GeneralizedModeAnalysisWithBranchSortedModes(N=4, wz=2*np.pi*wz_Hz, wy=2*np.pi*10e6, wx=2*np.pi*11e6, ionmass_amu=mass_yb_amu)
    four_ion_analysis.run()
    positions_z = four_ion_analysis.u[2*4:] * four_ion_analysis.characteristic_parameters['length']
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
 
def calculate_mode_participation_factors(mode_analysis):
    eigvecs = mode_analysis.eigvecs
    num_coords, num_modes = np.shape(eigvecs)
    num_ions = num_modes // 3
    mode_participation_factors = np.zeros((3, num_ions, num_modes), dtype = np.complex128)
    for mode_index in range(num_modes):
        for pos_coord in range(num_coords//2):
            direction_index = pos_coord // num_ions
            ion_index = pos_coord % num_ions
            factor = np.sqrt(2* mode_analysis.m[ion_index] * mode_analysis.eigvals[mode_index] )
            zpm_dimensionful = np.sqrt(const.hbar / (2* mode_analysis.mass_scale * mode_analysis.trap_freq_scale))
            mode_participation_factors[direction_index, ion_index, mode_index] = zpm_dimensionful * factor * eigvecs[pos_coord, mode_index]
    return mode_participation_factors
 
 
 
def calc_single_ion_ld_factors(omega, k, mass_amu):
    z0 = np.sqrt(const.hbar / (2 * mass_amu * const.u * omega))
    return k * z0 # ignore the phase
 
def check_single_ion_case():
    # for a single ion, the mode participation factors should just be the LD factors for each mode and direction.
    # this is a good sanity check to make sure the mode participation factor calculation is correct.
    wz = 2 * np.pi * .5e6  # axial trap frequency
    wy = 2 * np.pi * 1.9e6  # radial trap frequency
    wx = 2 * np.pi * 10e6  # radial trap frequency, something high to avoid any issues with mode ordering
    mass_yb_amu = 170.936
    k = 2 * np.pi / 355e-9
    atomic_nums = np.array([70]) 
    num_ions = 1

    mode_analysis_one = TrappedIonModeAnalysis(num_ions, wx, wy, wz, np.ones(num_ions)*mass_yb_amu, atomic_nums) 
    mode_analysis_one.run()
    mode_participation_factors_one = calculate_mode_participation_factors(mode_analysis_one)
    ld_factors_one = k * mode_participation_factors_one
    eta_x = calc_single_ion_ld_factors(wx, k, mass_yb_amu)
    eta_y = calc_single_ion_ld_factors(wy, k, mass_yb_amu)
    eta_z = calc_single_ion_ld_factors(wz, k, mass_yb_amu)
    ld_factors_one_analytical = np.zeros((3, 1, 3), dtype = np.complex128)
    ld_factors_one_analytical[0, 0, 2] = eta_x
    ld_factors_one_analytical[1, 0, 1] = eta_y
    ld_factors_one_analytical[2, 0, 0] = eta_z
    print(f"\nLD factors: {ld_factors_one_analytical}")
    print("\nRatio of single ion LD factors from mode participation calculation to analytical calculation: \n", ld_factors_one / ld_factors_one_analytical)
 
def check_two_ion_case():
    wz = 2 * np.pi * .5e6  # axial trap frequency
    wy = 2 * np.pi * 1.5e6  # radial trap frequency
    wx = 2 * np.pi * 2e6  # radial trap frequency
    wy_tilt = np.sqrt(wy**2 - wz**2)
    wx_tilt = np.sqrt(wx**2 - wz**2)
    mass_yb_amu = 170.936
    k = 2 * np.pi / 355e-9
    mode_analysis_two= mode_analyzer(N =2, wz = wz, wy = wy, wx = wx, ionmass_amu= mass_yb_amu)
    mode_analysis_two.run()
    mode_participation_factors_two= calculate_mode_participation_factors(mode_analysis_two)
    ld_factors_two = k * mode_participation_factors_two
    eta_x = calc_single_ion_ld_factors(wx, k, mass_yb_amu)
    eta_y = calc_single_ion_ld_factors(wy, k, mass_yb_amu)
    eta_z = calc_single_ion_ld_factors(wz, k, mass_yb_amu)
    eta_COM_x = eta_x / np.sqrt(2)
    eta_COM_y = eta_y / np.sqrt(2)
    eta_COM_z = eta_z / np.sqrt(2)
    eta_stretch_z = eta_z /np.sqrt(2) /3**(1/4)
    # I don't know the analytical expressions for the tilt modes off the top of my head... let's assume its the square root of the normalized mode frequency
    eta_tilt_x = eta_x / np.sqrt(2) / np.sqrt(wx_tilt / wx)
    eta_tilt_y = eta_y / np.sqrt(2) / np.sqrt(wy_tilt / wy)
    ld_factors_two_analytical = np.zeros((3, 2, 6), dtype = np.complex128)
    ld_factors_two_analytical[0, 0, 5] = eta_COM_x
    ld_factors_two_analytical[0, 1, 5] = eta_COM_x
    ld_factors_two_analytical[0, 0, 4] = eta_tilt_x
    ld_factors_two_analytical[0, 1, 4] = -eta_tilt_x
    ld_factors_two_analytical[1, 0, 3] = eta_COM_y
    ld_factors_two_analytical[1, 1, 3] = eta_COM_y
    ld_factors_two_analytical[1, 0, 2] = eta_tilt_y
    ld_factors_two_analytical[1, 1, 2] = -eta_tilt_y
    ld_factors_two_analytical[2, 0, 0] = eta_COM_z
    ld_factors_two_analytical[2, 1, 0] = eta_COM_z
    ld_factors_two_analytical[2, 0, 1] = eta_stretch_z
    ld_factors_two_analytical[2, 1, 1] = -eta_stretch_z
    # these seem to work out!
    print(f"\nLD factors: {ld_factors_two_analytical}")
    print("Ratio of two ion LD factors from mode participation calculation to analytical calculation: \n", ld_factors_two / ld_factors_two_analytical)
 
 

 
### Example usage  ###  
if __name__ == '__main__':
    # Test analysis: 
    #run()
    #check_two_ion_case() 
    check_single_ion_case()

