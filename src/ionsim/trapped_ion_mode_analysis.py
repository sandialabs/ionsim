import numpy as np
from numpy.typing import NDArray
import scipy.constants as const
import warnings
import scipy.optimize as opt
from ionsim.custom_types import Matrix, Vector
from ionsim.degree_of_freedom import AtomicStructure, MotionalMode
from ionsim.basis import StandardBasis
from ionsim.ionsim_error import IonSimError

## References:
# https://arxiv.org/abs/2007.12725
# https://arxiv.org/abs/quant-ph/9702053

# Trap types whose per-species secular frequencies are derived from the reference ion's frequencies
RF_PAUL_TRAP_TYPES = ('linear', 'surface')
TRAP_TYPES = RF_PAUL_TRAP_TYPES + ('custom',)

# Relative tolerance used to decide which eigenvector components are "tied" for largest magnitude
_GAUGE_TIE_RTOL = 1e-6
# Relative size of imaginary parts below which gauge-fixed participation factors are returned as real
_REAL_RTOL = 1e-9


def characteristic_length(q: float, mass: float, omega: float) -> float:
    """ Computes characteristic length in trapped ion system: Coulomb + harmonic oscillator physics.

        l0 = (2 k_e q^2 / (m omega^2))^(1/3), which is the equilibrium separation of two identical ions
        in a harmonic well of frequency omega.
    """
    k_e = 1 / (4 * np.pi * const.epsilon_0)  # Coulomb constant
    l0 = ((k_e * q ** 2) / (.5 * mass * omega ** 2)) ** (1 / 3)
    return l0


class TrappedIonCrystal:
    def __init__(self, num_ions: int, omega_x: float | Vector, omega_y: float | Vector, omega_z: float | Vector,
                 atomic_masses: Vector | float, atomic_charges: Vector | float,
                 mode_organization: str = 'frequency_only', reindexing_strategy: str = 'distance',
                 trap_type: str = 'linear', dc_radial_split: tuple[float, float] = (0.5, 0.5),
                 principal_axes: Matrix | None = None):
        """ Class for determining properties of trapped-ion phonon modes, requiring the following system parameters:

            - num_ions: the number of ions
            - omega_x, omega_y, omega_z: secular trap frequencies [rad/s] along the trap's principal axes x, y, z.
                z is the "axial" direction used for 1D chains. For trap_type 'linear' or 'surface' these are the
                frequencies of the *reference ion* (the first ion in the user-supplied lists); frequencies of other
                species are derived from them (see calculate_species_trap_frequencies). For trap_type 'custom'
                they may be scalars or length-num_ions arrays and are used for each ion as given.
            - atomic_masses: array of atomic masses or a single number => same mass for each ion. Units are amu.
            - atomic_charges: array of ion charge states in units of the elementary charge e (e.g. 1 for Yb+,
                2 for a doubly-ionized ion), or a single number => all ions have the same charge.
            - mode_organization: strategy for organizing modes ('frequency_only' or 'branch_sorted')
            - reindexing_strategy: strategy for reindexing ions after the equilibrium solve
                ('distance', 'z_axis', or 'none'). See reindex_ions.
            - trap_type: 'linear' or 'surface' (RF Paul traps: RF pseudopotential confines radially (x, y),
                DC potential confines axially (z)), or 'custom' (per-ion frequencies given directly).
            - dc_radial_split: (s_x, s_y) with s_x + s_y = 1. The DC axial curvature produces an anti-confining
                radial curvature -s_x * k_z along x and -s_y * k_z along y (Laplace's equation). (0.5, 0.5) is a
                symmetric DC quadrupole; other values describe an additional DC radial quadrupole. Only used for
                'linear'/'surface'.
            - principal_axes: optional 3x3 array whose rows are the trap principal axes (x, y, z), expressed in
                the lab frame. Rows are normalized; they must be mutually orthogonal and right-handed. All
                internal quantities are computed in the trap (principal-axis) frame; wavevectors passed to the
                Lamb-Dicke methods are given in the lab frame. Defaults to the identity (lab frame == trap frame).

            Ion bookkeeping: after solve_ion_trap_equilibrium the ions may be reindexed. All per-ion arrays are
            permuted consistently, and `ion_indices[k]` gives the user-supplied (original) index of the ion now
            stored at index k.
        """
        if num_ions < 1:
            raise IonSimError(f"num_ions must be positive. Received {num_ions}")
        self.num_ions = num_ions

        # Store configuration strategies
        self.mode_organization = mode_organization
        self.reindexing_strategy = reindexing_strategy
        self.trap_type = trap_type

        # Validate configuration
        if mode_organization not in ['frequency_only', 'branch_sorted']:
            raise ValueError(f"Invalid mode_organization: {mode_organization}. Must be 'frequency_only' or 'branch_sorted'")
        if reindexing_strategy not in ['distance', 'z_axis', 'none']:
            raise ValueError(f"Invalid reindexing_strategy: {reindexing_strategy}. Must be 'distance', 'z_axis', or 'none'")
        if trap_type not in TRAP_TYPES:
            raise ValueError(f"Invalid trap_type: {trap_type}. Must be one of {TRAP_TYPES}")

        # Keep a copy of the user inputs (original ion order) so the configuration can be rebuilt
        self._input_atomic_masses_amu = self.convert_to_array(atomic_masses).astype(float)
        self._input_atomic_charges = self.convert_to_array(atomic_charges).astype(float)
        self.dc_radial_split = self._check_dc_radial_split(dc_radial_split)
        self.principal_axes = self._check_principal_axes(principal_axes)

        self.atomic_masses = self._input_atomic_masses_amu * const.u  # kg from amu
        self.atomic_charges = self._input_atomic_charges.copy()      # charge state, units of e
        self._check_ion_property_consistency()

        # Total ionic charge in Coulombs
        self.charges = self.atomic_charges * const.e

        # Reference ion: the first ion in the user-supplied order. Its properties set the dimensionless scales
        # and (for RF Paul traps) the frequencies omega_x, omega_y, omega_z refer to it.
        self.reference_mass = self.atomic_masses[0]
        self.reference_charge = self.charges[0]

        # Trapping frequencies for each ion (rad/s, trap frame)
        self.omega_x, self.omega_y, self.omega_z = self.calculate_species_trap_frequencies(omega_x, omega_y, omega_z)
        self.reference_trap_frequencies = np.array([self.omega_x[0], self.omega_y[0], self.omega_z[0]])

        # Check that the trap is stable:
        if not self.trap_is_stable():
            raise IonSimError("Trap is unstable: all trap frequencies must be positive.")

        # Original index of each stored ion (updated by reindexing)
        self.ion_indices = np.arange(self.num_ions)

        # Dimensionless parameters are fixed by the reference ion and never rescaled by reindexing
        self.set_up_dimensionless_parameters(self.reference_charge, self.reference_mass, self.omega_z[0])

    # ------------------------------------------------------------------------------------------------
    # Input validation and species handling
    # ------------------------------------------------------------------------------------------------
    def _check_ion_property_consistency(self):
        if len(self.atomic_masses) != self.num_ions:
            raise ValueError(f"Expected {self.num_ions} atomic masses, received {len(self.atomic_masses)}")
        if len(self.atomic_charges) != self.num_ions:
            raise ValueError(f"Expected {self.num_ions} atomic charges, received {len(self.atomic_charges)}")
        if np.any(self.atomic_masses <= 0):
            raise IonSimError(f"Atomic masses must be positive. Received {self._input_atomic_masses_amu} amu")
        if np.any(self.atomic_charges == 0):
            raise IonSimError(f"Ion charge states must be nonzero (neutral atoms cannot be trapped here). "
                              f"Received {self.atomic_charges}")
        if not (np.all(self.atomic_charges > 0) or np.all(self.atomic_charges < 0)):
            raise IonSimError(f"All ions must have charges of the same sign. Received {self.atomic_charges}")

    @staticmethod
    def _check_dc_radial_split(dc_radial_split) -> NDArray:
        split = np.asarray(dc_radial_split, dtype=float)
        if split.shape != (2,):
            raise ValueError(f"dc_radial_split must be a pair (s_x, s_y). Received {dc_radial_split}")
        if not np.isclose(np.sum(split), 1.0, rtol=0, atol=1e-12):
            raise ValueError(f"dc_radial_split must satisfy s_x + s_y = 1 (Laplace's equation). Received {dc_radial_split}")
        return split

    @staticmethod
    def _check_principal_axes(principal_axes) -> Matrix:
        """ Returns the 3x3 rotation matrix R whose rows are the unit trap axes in lab coordinates.
            v_trap = R @ v_lab and v_lab = R.T @ v_trap.
        """
        if principal_axes is None:
            return np.eye(3)
        R = np.array(principal_axes, dtype=float)
        if R.shape != (3, 3):
            raise ValueError(f"principal_axes must be a 3x3 array (rows are the x, y, z trap axes). Received shape {R.shape}")
        norms = np.linalg.norm(R, axis=1)
        if np.any(norms == 0):
            raise ValueError("principal_axes rows must be nonzero vectors")
        R = R / norms[:, np.newaxis]
        if not np.allclose(R @ R.T, np.eye(3), rtol=0, atol=1e-9):
            raise ValueError(f"principal_axes rows must be mutually orthogonal. Received {principal_axes}")
        if np.linalg.det(R) < 0:
            raise ValueError("principal_axes must form a right-handed set (x cross y = z)")
        return R

    def calculate_species_trap_frequencies(self, omega_x, omega_y, omega_z) -> tuple[Vector, Vector, Vector]:
        """ Computes the secular trap frequencies of each ion along the trap principal axes.

            'custom': frequencies are used as given (scalars are broadcast to every ion).

            'linear' / 'surface' (RF Paul traps, secular approximation, micromotion neglected): with the reference
            ion (charge q0, mass m0) having secular frequencies (wx, wy, wz) and r_i = (q_i m0) / (q0 m_i),

                axial (DC only):          w_{z,i}^2 = r_i wz^2
                radial (RF pseudo + DC):  w_{x,i}^2 = r_i^2 (wx^2 + s_x wz^2) - s_x r_i wz^2   (same for y with s_y)

            This follows from the DC spring constant scaling as q and the RF pseudopotential spring constant scaling
            as q^2 / m, with the DC axial curvature defocusing the radial directions by s_x, s_y (s_x + s_y = 1).
            RF and DC curvatures are assumed to share the same principal axes.
        """
        if self.trap_type == 'custom':
            frequencies = [self.convert_to_array(w).astype(float) for w in (omega_x, omega_y, omega_z)]
            for w, label in zip(frequencies, 'xyz'):
                if len(w) != self.num_ions:
                    raise ValueError(f"omega_{label} must be a scalar or have length {self.num_ions}")
            return tuple(frequencies)

        for w, label in zip((omega_x, omega_y, omega_z), 'xyz'):
            if np.ndim(w) != 0:
                raise ValueError(f"For trap_type '{self.trap_type}', omega_{label} must be a scalar (the reference ion's "
                                 f"frequency). Use trap_type='custom' to specify per-ion frequencies.")
            if w <= 0:
                raise IonSimError(f"Trap is unstable: omega_{label} must be positive. Received {w}")

        s_x, s_y = self.dc_radial_split
        r = (self.charges * self.reference_mass) / (self.reference_charge * self.atomic_masses)

        rf_x0 = omega_x ** 2 + s_x * omega_z ** 2   # RF pseudopotential curvature of the reference ion (as w^2)
        rf_y0 = omega_y ** 2 + s_y * omega_z ** 2
        if omega_z ** 2 <= 0 or rf_x0 <= 0 or rf_y0 <= 0:
            raise IonSimError("Reference trap frequencies are inconsistent with an RF Paul trap: the implied RF "
                              "pseudopotential or DC axial curvature is not confining.")

        wz2 = r * omega_z ** 2
        wx2 = r ** 2 * rf_x0 - s_x * r * omega_z ** 2
        wy2 = r ** 2 * rf_y0 - s_y * r * omega_z ** 2
        for w2, label in zip((wx2, wy2, wz2), 'xyz'):
            if np.any(w2 <= 0):
                unstable = np.flatnonzero(w2 <= 0)
                raise IonSimError(f"Ions {unstable.tolist()} are not confined along {label} for this trap "
                                  f"(secular frequency^2 = {w2[unstable]}).")
        return np.sqrt(wx2), np.sqrt(wy2), np.sqrt(wz2)

    def trap_is_stable(self):
        # Checks that all trap frequencies are positive
        return np.all(self.omega_z > 0) and np.all(self.omega_y > 0) and np.all(self.omega_x > 0)

    def convert_to_array(self, x: float | int | list | NDArray) -> NDArray:
        """ Converts a scalar to a length-num_ions vector or returns the vector """
        if np.ndim(x) == 0:
            return np.ones(self.num_ions) * x
        return np.array(x)

    # ------------------------------------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------------------------------------
    def lab_to_trap_frame(self, vector: Vector) -> Vector:
        """ Rotates a lab-frame vector (or array of shape (3, ...)) into the trap principal-axis frame. """
        return np.tensordot(self.principal_axes, np.asarray(vector), axes=(1, 0))

    def trap_to_lab_frame(self, vector: Vector) -> Vector:
        """ Rotates a trap-frame vector (or array of shape (3, ...)) into the lab frame. """
        return np.tensordot(self.principal_axes.T, np.asarray(vector), axes=(1, 0))

    def _wavevector_in_trap_frame(self, wavevector: Vector) -> Vector:
        k = np.asarray(wavevector, dtype=float)
        if k.shape != (3,):
            raise ValueError(f"Wavevector must be a 3-element array (kx, ky, kz) in the lab frame. Received {wavevector}")
        return self.lab_to_trap_frame(k)

    # ------------------------------------------------------------------------------------------------
    # Dimensionless setup
    # ------------------------------------------------------------------------------------------------
    def set_up_dimensionless_parameters(self, charge_scale: float=1., mass_scale: float=1., trap_freq_scale: float=1.):
        """ Compute dimensionless parameters, using the reference ion's properties and axial (z) trapping frequency.

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

        # Store the scales so they are retrievable by the user
        self.charge_scale = charge_scale
        self.mass_scale = mass_scale
        self.trap_freq_scale = trap_freq_scale

        # ion properties
        self.m = self.atomic_masses / self.mass_scale
        self.q = self.charges / self.charge_scale

        # trap frequencies
        self.wz = self.omega_z / self.trap_freq_scale
        self.wy = self.omega_y / self.trap_freq_scale
        self.wx = self.omega_x / self.trap_freq_scale
        self.trap_frequencies_ND = [self.wx, self.wy, self.wz]

        # Compute characteristic length, time, velocity, and energy scales and store in a dictionary
        self.characteristic_parameters = {}
        self.characteristic_parameters['length'] = characteristic_length(self.charge_scale, self.mass_scale, self.trap_freq_scale)
        self.characteristic_parameters['time'] = 1 / self.trap_freq_scale # characteristic time
        self.characteristic_parameters['velocity'] = self.characteristic_parameters['length'] * self.trap_freq_scale  # characteristic velocity
        self.characteristic_parameters['energy'] = 0.5 * self.mass_scale * self.characteristic_parameters['velocity'] ** 2  # characteristic energy

    # ------------------------------------------------------------------------------------------------
    # Eigenvector utilities and checks
    # ------------------------------------------------------------------------------------------------
    def get_eigenvector_norm(self, eigenvector: Vector, H: Matrix) -> float:
        """ Computes the norm of the eigenvector en w.r.t. the Hamiltonian H. """
        norm = np.sqrt(eigenvector.T.conj() @ H @ eigenvector)
        return norm

    def get_canonical_transformation(self, H, eigenvectors, eigenvalues: Vector | None=None):
        """ Given the eigensolve of the dynamical matrix, get the transform matrix to the canonical coordinates.
            X = S X', where X' = (Q,P)^T and X = (q,p)^T
        """
        # the sign due to the convention of writing the time evolution as exp(-iwt)
        sign = -1
        num_coords, num_eigenvalues = np.shape(eigenvectors)
        assert num_coords //2 == num_eigenvalues
        eigenvectors = self.normalize_eigenvectors(eigenvectors, H, eigenvalues)
        T = np.sqrt(2)*np.concatenate((np.real(eigenvectors), sign*np.imag(eigenvectors)), axis=1)
        return T

    def check_for_zero_modes(self):
        if not np.all(self.eigvals > 0):
            raise IonSimError("Found non-positive normal mode frequencies: the equilibrium found is not a stable "
                              "minimum (e.g. a linear chain beyond the zigzag transition).")

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
        if not (np.allclose(I_left,eye) and np.allclose(I_right,eye)):
            warnings.warn("Outer relation check failed")

    def _has_degenerate_modes(self, atol: float = 1E-6) -> bool:
        return bool(np.any(np.triu(np.isclose(self.eigvals[:, None], self.eigvals[None, :], atol=atol), k=1)))

    def check_diagonalization(self, T: Matrix, S: Matrix, E: Matrix) -> bool:
        M = np.linalg.inv(T) @ S
        H_diag = M.T @ E @ M
        H_diag_check = np.diag(np.tile(self.eigvals,2))
        if np.allclose(H_diag,H_diag_check):
            return True
        warnings.warn(f"Diagonalization check failed (degenerate modes present: {self._has_degenerate_modes()})")
        return False

    # ------------------------------------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------------------------------------
    def solve_ion_trap_equilibrium(self, positions_guess: Vector | None = None):
        """ Solve for equilibrium positions and analyze normal modes by minimizing the Coulomb +
                harmonic trap energy functional for a system of ions.

            This method:
            1. Solves for equilibrium positions (dimensionless units are fixed by the reference ion at construction)
            2. Reindexes ions according to the reindexing strategy (all per-ion arrays are permuted together)
            3. Computes normal modes (with branch sorting if requested) and fixes the phase of each mode
            4. Performs validation checks

            - positions_guess: optional dimensionless flattened initial guess [x_1..x_N, y_1..y_N, z_1..z_N].
                By default a deterministic linear chain along the weakest-confinement axis, in the user-supplied
                ion order, is used. For mixed species this order determines which species sits where.
        """
        self.equilibrium_positions = self.solve_for_equilibrium_positions(positions_guess)
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

    @property
    def normal_mode_frequencies(self):
        """ Normal mode frequencies in rad/s, in the order of the stored modes. """
        self._ensure_mode_analysis_run()
        return self.eigvals * self.trap_freq_scale

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

    def fix_eigenvector_phases(self, eigvecs: Matrix) -> Matrix:
        """ Removes the arbitrary per-mode phase returned by the eigensolver.

            Each mode u_n (a column of eigvecs) is multiplied by exp(-i theta_n) such that its largest-magnitude
            position component is real and positive. Ties (components within a relative 1e-6 of the largest
            magnitude, e.g. COM or stretch modes of identical ions) are broken by the lowest coordinate index in
            the flattened order [x_1..x_N, y_1..y_N, z_1..z_N].

            Without magnetic fields the position part of a non-degenerate mode is real up to one phase, so after
            this the position part (and hence the Lamb-Dicke parameters) is real, and the displacement operator is
            r_{i,d} = sum_n r^{zpf}_{i,d,n} (a_n + a_n^dagger). Relative signs between ions within a mode are
            physical; the overall sign of each mode is this convention. Degenerate subspaces are not treated.
        """
        num_position_coords = 3 * self.num_ions
        fixed = np.array(eigvecs, dtype=complex)
        for n in range(fixed.shape[1]):
            position_part = fixed[:num_position_coords, n]
            magnitudes = np.abs(position_part)
            reference_index = np.flatnonzero(magnitudes >= (1 - _GAUGE_TIE_RTOL) * magnitudes.max())[0]
            fixed[:, n] *= np.exp(-1j * np.angle(position_part[reference_index]))
        return fixed

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
        eigvecs = self.fix_eigenvector_phases(eigvecs)

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

    # ------------------------------------------------------------------------------------------------
    # Reindexing
    # ------------------------------------------------------------------------------------------------
    def reindex_ions(self, strategy: str | None = None):
        """ Re-indexes the ions according to `strategy` (defaults to self.reindexing_strategy):

            - 'distance': ascending distance from the trap center (index 0 is closest to the center)
            - 'z_axis':   ascending z (axial) coordinate (index 0 is the most negative z)
            - 'none':     keep the current order

            Every per-ion array is permuted with the same permutation, and ion_indices tracks the original
            (user-supplied) index of each ion. Dimensionless scales are not changed.
        """
        strategy = self.reindexing_strategy if strategy is None else strategy
        x, y, z = self.ion_coordinates_from_flattened(self.equilibrium_positions)
        if strategy == 'distance':
            idx = np.argsort(np.sqrt(x**2 + y**2 + z**2), kind='stable')
        elif strategy == 'z_axis':
            idx = np.argsort(z, kind='stable')
        elif strategy == 'none':
            idx = np.arange(self.num_ions)
        else:
            raise ValueError(f"Invalid reindexing strategy: {strategy}")
        self._apply_ion_permutation(idx)

    def reindex_ions_by_z(self):
        """ Re-indexes ions by ascending position along the z-axis. """
        self.reindex_ions('z_axis')

    def _apply_ion_permutation(self, idx: NDArray):
        """ Permutes all per-ion quantities by idx (new index k <- old index idx[k]). """
        idx = np.asarray(idx)
        x, y, z = self.ion_coordinates_from_flattened(self.equilibrium_positions)
        self.equilibrium_positions = np.hstack((x[idx], y[idx], z[idx]))
        for name in ('atomic_masses', 'atomic_charges', 'charges', 'm', 'q',
                     'omega_x', 'omega_y', 'omega_z', 'wx', 'wy', 'wz', 'ion_indices'):
            setattr(self, name, getattr(self, name)[idx])
        self.trap_frequencies_ND = [self.wx, self.wy, self.wz]

    # ------------------------------------------------------------------------------------------------
    # Equilibrium
    # ------------------------------------------------------------------------------------------------
    def _initial_chain_axis(self) -> int:
        """ Axis (0, 1, 2 = x, y, z) along which the default initial guess places the ions: the weakest
            confinement axis of the reference ion. """
        return int(np.argmin(self.reference_trap_frequencies))

    def initial_positions_guess(self) -> Vector:
        """ Deterministic dimensionless initial guess: a linear chain in the current ion order along the
            weakest-confinement axis, plus a small transverse perturbation so that a zigzag/2D configuration
            can form if it is energetically favored. """
        N = self.num_ions
        axis = self._initial_chain_axis()
        w_axis = self.reference_trap_frequencies[axis] / self.trap_freq_scale
        spacing = w_axis ** (-2 / 3)  # two-ion spacing in units of the characteristic length
        u0 = np.zeros(3 * N)
        u0[axis * N:(axis + 1) * N] = (np.arange(N) - (N - 1) / 2) * spacing
        alternating = (-1.0) ** np.arange(N)
        transverse_axes = [d for d in range(3) if d != axis]
        u0[transverse_axes[0] * N:(transverse_axes[0] + 1) * N] = 1e-3 * spacing * alternating
        u0[transverse_axes[1] * N:(transverse_axes[1] + 1) * N] = 0.5e-3 * spacing * alternating
        return u0

    def solve_for_equilibrium_positions(self, positions_guess: Vector | None=None):
        """ Solves for equilibrium position vector: u, which represents a flattened spatial grid (dimensionless). """
        if positions_guess is None:
            u0 = self.initial_positions_guess()
        else:
            u0 = np.asarray(positions_guess, dtype=float)
            if u0.shape != (3 * self.num_ions,):
                raise ValueError(f"positions_guess must have shape ({3 * self.num_ions},)")

        # Minimize the potential energy (trap + Coulomb), then polish with Newton steps on the analytic Hessian
        solver_output = opt.minimize(self.potential_energy, u0, method='BFGS', jac=self.force,
                                     options={'gtol': 1e-10, 'disp': False})
        u = solver_output.x
        for _ in range(10):
            gradient = self.force(u)
            if np.max(np.abs(gradient)) < 1e-14:
                break
            hessian = self.hessian(u)
            if np.min(np.linalg.eigvalsh(hessian)) <= 0:
                break  # not at a minimum; leave it to check_for_zero_modes to report
            u = u - np.linalg.solve(hessian, gradient)

        residual = np.max(np.abs(self.force(u)))
        if residual > 1e-9:
            warnings.warn(f"Equilibrium solve may not have converged: max |dU/du| = {residual:.2e} (dimensionless)")

        # Get potential energy at equilibrium positions
        self.p0 = self.potential_energy(u)
        return u

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
        """ Computes the gradient of the potential energy w.r.t. ion position: dU/dx, dU/dy, dU/dz.

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
        """ Computes the Hessian of the Coulomb interaction.

            For i != j:  d^2 V / dr_{a,i} dr_{b,j} = q_i q_j (delta_ab r_ij^2 - 3 dr_a dr_b) / r_ij^5
            For i == j:  d^2 V / dr_{a,i} dr_{b,i} = - sum_{j != i} (the i != j expression above)
        """
        r = self.ion_coordinates_from_flattened(positions)
        dr = [coord[:, np.newaxis] - coord for coord in r]
        rsep = np.sqrt(dr[0]**2 + dr[1]**2 + dr[2]**2).astype(np.float64)
        qq = (self.q * self.q[:, np.newaxis]).astype(np.float64)

        with np.errstate(divide='ignore'):
            rsep5 = np.where(rsep != 0., rsep ** (-5), 0)

        blocks = [[None] * 3 for _ in range(3)]
        for a in range(3):
            for b in range(3):
                block = qq * ((a == b) * rsep**2 - 3 * dr[a] * dr[b]) * rsep5  # zero on the diagonal (rsep5 = 0)
                np.fill_diagonal(block, -np.sum(block, axis=1))
                blocks[a][b] = block

        H_coulomb = np.block(blocks)
        H_coulomb /= 2  # consistent with the factor of 1/2 in potential_coulomb
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

    # ------------------------------------------------------------------------------------------------
    # Mode organization
    # ------------------------------------------------------------------------------------------------
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

        if self.mode_organization == 'branch_sorted':
            eigvals, eigvecs = self.sort_by_branch(eigvals, eigvecs)

        return eigvals, eigvecs

    def _xyz_classify_modes(self, evecs):
        """Classify modes by their dominant spatial direction (principal axes x, y, z)."""
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
            if count != N_modes // 3:
                raise IonSimError(f"Branch classification error: direction {direction} has {count} modes, expected "
                                  f"{N_modes // 3}. Branch sorting requires a linear chain along z.")
        return classifier

    def sort_by_branch(self, evals, evecs):
        """ Sorts the eigenvalues, eigenvectors by mode branch (radial x, radial y, axial (z)).
            Only makes sense for linear chain configurations where Hessian is block diagonal. """
        classifier = self._xyz_classify_modes(evecs)
        # within each branch, sort by frequency
        N_ions = len(evals) // 3
        sorted_by_branch_evals = np.zeros_like(evals)
        sorted_by_branch_evecs = np.zeros_like(evecs)
        for direction in range(3):
            direction_indices = np.where(classifier == direction)[0]
            # they are already sorted by frequency, so take them in order
            sorted_by_branch_evals[direction*N_ions:(direction+1)*N_ions] = evals[direction_indices]
            sorted_by_branch_evecs[:, direction*N_ions:(direction+1)*N_ions] = evecs[:, direction_indices]
        return sorted_by_branch_evals, sorted_by_branch_evecs

    # ------------------------------------------------------------------------------------------------
    # Participation factors and Lamb-Dicke parameters
    # ------------------------------------------------------------------------------------------------
    def calculate_mode_participation_factors(self, frame: str = 'trap') -> Matrix:
        """ Computes ion-mode participation factors (zero-point displacement amplitudes, in meters), related to the
            Lamb-Dicke parameters by eta[ion, mode] = k . r_zpf[:, ion, mode].

            Shape (3, N, 3N): result[d, i, n] is the zero-point displacement of ion i along direction d in mode n,
            such that the displacement operator is r_{d,i} = sum_n result[d, i, n] (a_n + a_n^dagger).
            For a single ion this reduces to sqrt(hbar / (2 m omega)).

            - frame: 'trap' (directions are the principal axes x, y, z) or 'lab' (rotated with principal_axes).

            The phase of each mode is fixed by fix_eigenvector_phases, so the result is real. If the result has
            non-negligible imaginary parts (degenerate modes), a complex array is returned with a warning.
        """
        self._ensure_mode_analysis_run()
        if frame not in ('trap', 'lab'):
            raise ValueError(f"frame must be 'trap' or 'lab'. Received {frame}")
        N = self.num_ions
        position_part = self.eigvecs[:3 * N, :]  # dimensionless, normalized so that u^dagger H u = 1
        # Dimensionful zero-point amplitude: l0 * sqrt(hbar_ND * w_n) * u = sqrt(hbar / (2 m0 w0)) * sqrt(2 w_n) * u
        zpm_dimensionful = np.sqrt(const.hbar / (2 * self.mass_scale * self.trap_freq_scale))
        participation = zpm_dimensionful * np.sqrt(2 * self.eigvals)[np.newaxis, :] * position_part
        participation = participation.reshape(3, N, 3 * N)
        if frame == 'lab':
            participation = self.trap_to_lab_frame(participation)
        return self._real_if_possible(participation)

    def _real_if_possible(self, array: NDArray) -> NDArray:
        scale = np.max(np.abs(array))
        if scale == 0 or np.max(np.abs(np.imag(array))) <= _REAL_RTOL * scale:
            return np.real(array)
        warnings.warn("Mode participation factors are complex after phase fixing (degenerate modes present: "
                      f"{self._has_degenerate_modes()}). Phases within degenerate subspaces are not fixed.")
        return array

    def calculate_lamb_dicke_parameters(self, wavevector: Vector) -> Matrix:
        """ Lamb-Dicke parameter matrix eta[ion, mode] = k . r_zpf for a lab-frame wavevector k = (kx, ky, kz) [1/m].
            Shape (num_ions, 3 * num_ions). Real, with the per-mode sign convention of fix_eigenvector_phases.
            For a Raman transition use the difference wavevector k1 - k2.
        """
        k_trap = self._wavevector_in_trap_frame(wavevector)
        return np.einsum('d,dim->im', k_trap, self.calculate_mode_participation_factors(frame='trap'))

    # Derived properties
    def compute_reference_single_ion_lamb_dicke_factors(self, wavenumber: float) -> (float, float, float):
        """ Computes analytical Lamb-Dicke parameter by eta = k * sqrt(hbar / 2 m omega) for ion 0 in a light-field
            with wavevector |k| along each principal axis.
            - wavenumber: wavevector magnitude |k| in units of 1 / m
        """
        trap_frequencies = np.array([self.omega_x[0], self.omega_y[0], self.omega_z[0]])
        mass = self.atomic_masses[0]
        eta_x, eta_y, eta_z = wavenumber * np.sqrt(const.hbar / (2 * mass * trap_frequencies))
        return eta_x, eta_y, eta_z

    def return_equilibrium_positions(self, dimensionless: bool, frame: str = 'trap') -> Vector:
        """ Return flattened equilibrium positions [x_1..x_N, y_1..y_N, z_1..z_N], dimensionless or in meters,
            in the trap or lab frame. """
        self._ensure_mode_analysis_run()
        positions = self.equilibrium_positions.copy()
        if frame == 'lab':
            positions = self.trap_to_lab_frame(positions.reshape(3, self.num_ions)).reshape(-1)
        elif frame != 'trap':
            raise ValueError(f"frame must be 'trap' or 'lab'. Received {frame}")
        if dimensionless:
            return positions
        return positions * self.characteristic_parameters['length']

    # ------------------------------------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------------------------------------
    @classmethod
    def from_species(cls, species_name: str, num_ions: int, omega_x: float, omega_y: float, omega_z: float, **kwargs):
        """Build the mode analysis class from a species name, corresponding to a system of N identical ions.
            Additional keyword arguments are passed to the constructor."""
        species_data = AtomicStructure.get_config_data(species_name)
        return cls(num_ions, omega_x, omega_y, omega_z, species_data['mass'], species_data['charge'], **kwargs)

    @classmethod
    def from_atomic_structure_basis(cls, atomic_structure_basis: StandardBasis, omega_x: float, omega_y: float,
                                    omega_z: float, **kwargs):
        """ Build the mode analysis class from a basis of AtomicStructure degrees of freedom. The order of the DOFs
            sets the ion order (the first DOF is the reference ion). Additional keyword arguments are passed to the
            constructor. After solving, ion_indices maps stored ions back to the DOF order. """
        dofs = atomic_structure_basis.degrees_of_freedom
        for dof in dofs:
            if not isinstance(dof, AtomicStructure):
                raise IonSimError("Atomic structure basis should only contain AtomicStructure objects. No motional modes should be included.")
        atomic_masses = [dof.atomic_mass for dof in dofs]
        atomic_charges = [dof.atomic_charge for dof in dofs]
        return cls(len(dofs), omega_x, omega_y, omega_z, atomic_masses, atomic_charges, **kwargs)

    # ------------------------------------------------------------------------------------------------
    # IonSim DOFs and per-mode access
    # ------------------------------------------------------------------------------------------------
    @staticmethod
    def _parse_fock_dimensions_input_dimensions(fock_dimensions: Vector | int, num_modes: int) -> NDArray:
        dims = np.atleast_1d(np.asarray(fock_dimensions))
        if dims.size == 1:
            dims = np.full(num_modes, dims.item())
        elif dims.size != num_modes:
            raise ValueError(f"Received {dims.size} Fock dimensions for {num_modes} modes")
        if not np.all(np.equal(np.mod(dims, 1), 0)) or np.any(dims < 1):
            raise ValueError(f"Fock dimensions must be positive integers. Received {fock_dimensions}")
        return dims.astype(int)

    def build_mode_DOFs(self, mode_indices: list[int], fock_dimensions: Vector | int) -> list[MotionalMode]:
        """ Builds and returns a list of IonSim Motional Degrees of Freedom, with frequencies in rad/s.

            - mode_indices: global mode indices (0 to 3N - 1, in the stored mode order)
            - fock_dimensions: one Fock dimension per mode, or a single value applied to all modes
        """
        self._ensure_mode_analysis_run()
        fock_dimensions = self._parse_fock_dimensions_input_dimensions(fock_dimensions, len(mode_indices))
        frequencies = self.normal_mode_frequencies
        return [MotionalMode.from_frequency(frequencies[mode_index], int(fock_dim))
                for mode_index, fock_dim in zip(mode_indices, fock_dimensions)]

    def _ensure_mode_analysis_run(self):
        """Ensure that mode analysis has ran. If not, run it automatically."""
        if not hasattr(self, 'eigvals') or not hasattr(self, 'eigvecs'):
            self.solve_ion_trap_equilibrium()

    def get_mode_participation_factors_for_mode(self, mode_index: int, frame: str = 'trap') -> Matrix:
        """ Returns mode participation factors for a specific mode by its index.

            Output: Array of shape (3, num_ions) where result[dimension, ion] gives the participation factor
                for that dimension and ion in the specified mode.
        """
        return self.calculate_mode_participation_factors(frame=frame)[:, :, mode_index]

    def get_lamb_dicke_parameters_for_mode(self, mode_index: int, wavevector: Vector) -> Vector:
        """ Return Lamb-Dicke parameters for a specific mode by its index.

                - mode_index: Index of the mode of interest
                - wavevector: lab-frame wavevector k as a 3-element array (kx, ky, kz) in 1/m

                Returns an array of shape (num_ions,) where result[ion] gives the LD parameter for that ion.
                Sign convention: see fix_eigenvector_phases.
        """
        return self.calculate_lamb_dicke_parameters(wavevector)[:, mode_index]

    def get_mode_properties(self, mode_index: int) -> dict:
        """Get comprehensive properties for a specific mode. Returns a dictionary containing:
                - 'frequency': Mode frequency in rad/s
                - 'frequency_MHz': Mode frequency in MHz (convenience)
                - 'participation_factors': Array of shape (3, num_ions) (trap frame)
                - 'mode_index': the global mode index
        """
        self._ensure_mode_analysis_run()
        frequency = self.eigvals[mode_index] * self.trap_freq_scale
        return {
            'frequency': frequency,  # rad/s
            'frequency_MHz': frequency / (2 * np.pi * 1e6),  # MHz
            'participation_factors': self.get_mode_participation_factors_for_mode(mode_index),
            'mode_index': mode_index
        }


class IonChain(TrappedIonCrystal):
    """ Subclass for setting up and analyzing linear (1D) ion chains.

        In a linear ion chain, ions are aligned along the z principal axis (axial direction), with no
        displacement in the x and y directions (radial directions).

        This class accomplishes:
            1. Enforce linear chain configuration
            2. Provide specialized methods for linear chain analysis
            3. Offer convenience methods for common linear chain scenarios
    """
    BRANCHES = ('x', 'y', 'z')

    def __init__(self, num_ions: int, omega_x: float | Vector, omega_y: float | Vector, omega_z: float | Vector,
                 atomic_masses: np.ndarray | float, atomic_charges: np.ndarray | float, trap_type: str = 'linear',
                 dc_radial_split: tuple[float, float] = (0.5, 0.5), principal_axes: Matrix | None = None):
        """ Initialize a linear ion chain analysis with branch-sorted modes and ions indexed by ascending z.
            See TrappedIonCrystal for the arguments. For mixed species, the user-supplied order is the order of
            the ions along +z. """
        super().__init__(num_ions, omega_x, omega_y, omega_z, atomic_masses, atomic_charges,
                         mode_organization='branch_sorted', reindexing_strategy='z_axis', trap_type=trap_type,
                         dc_radial_split=dc_radial_split, principal_axes=principal_axes)

    def _initial_chain_axis(self) -> int:
        return 2  # always seed a chain along the axial (z) direction

    def _global_mode_index(self, branch: str, mode_index_in_branch: int) -> int:
        if branch not in self.BRANCHES:
            raise ValueError("Branch must be 'x', 'y', or 'z'")
        if mode_index_in_branch < 0 or mode_index_in_branch >= self.num_ions:
            raise ValueError(f"Mode index in branch must be between 0 and {self.num_ions-1}")
        return self.BRANCHES.index(branch) * self.num_ions + mode_index_in_branch

    def _branch_slice(self, branch: str) -> slice:
        start = self._global_mode_index(branch, 0)
        return slice(start, start + self.num_ions)

    def get_axial_modes(self) -> tuple:
        """ Returns eigenvalues, eigenvectors for the axial normal modes """
        self._ensure_mode_analysis_run()
        s = self._branch_slice('z')
        return self.eigvals[s], self.eigvecs[:, s]

    def get_radial_modes(self, direction='x') -> tuple:
        """ Returns eigenvalues, eigenvectors for the radial normal modes in the specified direction.
                - direction : str; 'x' or 'y' for radial direction """
        if direction not in ['x', 'y']:
            raise ValueError("Direction must be 'x' or 'y'")
        self._ensure_mode_analysis_run()
        s = self._branch_slice(direction)
        return self.eigvals[s], self.eigvecs[:, s]

    def get_axial_ion_positions(self):
        """ Returns an array of ion axial positions along the z-axis (meters, ascending). """
        self._ensure_mode_analysis_run()
        x, y, z = self.ion_coordinates_from_flattened(self.equilibrium_positions)
        return np.sort(z * self.characteristic_parameters['length'])

    def get_ion_spacing(self):
        """ Returns an array of ion spacings between consecutive ions along the z-axis. """
        return np.diff(self.get_axial_ion_positions())

    def get_center_of_mass_position(self) -> tuple:
        """ Get the center of mass position of the ion chain (trap frame, meters). """
        self._ensure_mode_analysis_run()
        x, y, z = self.ion_coordinates_from_flattened(self.equilibrium_positions * self.characteristic_parameters['length'])
        total_mass = np.sum(self.atomic_masses)
        com_x = np.sum(x * self.atomic_masses) / total_mass
        com_y = np.sum(y * self.atomic_masses) / total_mass
        com_z = np.sum(z * self.atomic_masses) / total_mass
        return com_x, com_y, com_z

    def get_axial_mode_frequencies(self):
        """ Returns the frequencies [rad/s] of the axial modes. """
        axial_eigvals, _ = self.get_axial_modes()
        return axial_eigvals * self.trap_freq_scale

    def get_radial_mode_frequencies(self, direction='x'):
        """ Returns the frequencies of the radial modes in rad/s for the specified direction ('x' or 'y'). """
        radial_eigvals, _ = self.get_radial_modes(direction)
        return radial_eigvals * self.trap_freq_scale

    def get_mode_participation_factors_by_branch(self):
        """ Returns a dictionary of mode participation factors organized by branch (x, y, z).

            Output dictionary with keys 'x', 'y', 'z' containing the participation factors along that branch's
            direction. Each value is a 2D array of shape (num_ions, num_modes_per_branch)
        """
        mode_pf = self.calculate_mode_participation_factors()
        return {branch: mode_pf[d, :, self._branch_slice(branch)] for d, branch in enumerate(self.BRANCHES)}

    def get_mode_properties_by_branch(self, branch: str, mode_index_in_branch: int) -> dict:
        """ Returns a dictionary containing properties (freq, participations) for a specific mode within a branch (x, y, or z).

                - branch: 'x', 'y', or 'z' for the mode branch
                - mode_index_in_branch: Index of the mode within the specified branch (0 to num_ions-1),
                    in ascending frequency
        """
        properties = super().get_mode_properties(self._global_mode_index(branch, mode_index_in_branch))
        properties['branch'] = branch
        properties['mode_index_in_branch'] = mode_index_in_branch
        return properties

    def get_lamb_dicke_parameters_by_branch(self, branch: str, mode_index_in_branch: int,
                                          wavevector: Vector) -> Vector:
        """Get Lamb-Dicke parameters for a specific mode within a branch.

            Parameters:
                branch: 'x', 'y', or 'z' for the mode branch
                mode_index_in_branch: Index of the mode within the specified branch
                wavevector: lab-frame wavevector k as a 3-element array (1/m)

            Returns:
                Array of Lamb-Dicke parameters for each ion in the specified mode
        """
        return super().get_lamb_dicke_parameters_for_mode(self._global_mode_index(branch, mode_index_in_branch), wavevector)

    def calculate_lamb_dicke_parameters_by_branch(self, wavevector: Vector) -> dict:
        """ Calculate Lamb-Dicke parameters for all ions and modes, organized by branch.

            The Lamb-Dicke parameter eta = k . r_zpf is the ratio of the spatial extent of the ion's zero-point
            motion to the reduced wavelength of the laser, projected onto the mode.

            Parameters:
                wavevector : lab-frame k = (kx, ky, kz) as a 3-element array in units of 1/m

            Returns:
                Dictionary with keys 'x', 'y', 'z'. Each value is a 2D array of shape (num_ions, num_modes_per_branch)
                representing eta[ion_index, mode_index] for the modes of that branch.
        """
        total_ld_params = self.calculate_lamb_dicke_parameters(wavevector)
        return {branch: total_ld_params[:, self._branch_slice(branch)] for branch in self.BRANCHES}

    def build_mode_DOFs_from_branch(self, branch: str, mode_indices: list[int], fock_dimensions: Vector | int) -> list[MotionalMode]:
        """ Builds and returns a list of IonSim Motional Degrees of Freedom.

            - branch specifies the direction: "x", "y", "z"
            - mode_indices corresponds to a mode's index within a branch (instead of the entire set of 3N modes for N ions)
            - Applies each fock dimension to each mode, or applies the same fock dimension to all the modes
            - ex] branch = 'x', mode_indices = [0,1] (tilt and COM), fock_dimensions = 6
        """
        global_indices = [self._global_mode_index(branch, idx) for idx in mode_indices]
        return self.build_mode_DOFs(global_indices, fock_dimensions)

    def calculate_lamb_dicke_parameters_full(self, wavevector: Vector) -> Matrix:
        """ Calculate full Lamb-Dicke parameter matrix from a lab-frame laser wavevector k = (kx, ky, kz)

            Returns:
                - Total LD parameter matrix of shape (num_ions, num_modes) where eta[ion, mode]
                    gives the total LD parameter k . r_zpf. Sign convention: see fix_eigenvector_phases.
        """
        return self.calculate_lamb_dicke_parameters(wavevector)

    def get_axial_lamb_dicke_parameters(self, wavevector: Vector) -> Matrix:
        """ Get Lamb-Dicke parameters for axial (z) modes only.

            Returns: Lamb-Dicke parameters for axial modes, NDArray of shape (num_ions, num_axial_modes)
        """
        return self.calculate_lamb_dicke_parameters_by_branch(wavevector)['z']

    def get_radial_lamb_dicke_parameters(self, wavevector: Vector, direction: str) -> Matrix:
        """ Get Lamb-Dicke parameters for radial modes in a specified direction.

            Returns: Lamb-Dicke parameters for radial modes, array of shape (num_ions, num_radial_modes)
        """
        if direction not in ['x', 'y']:
            raise ValueError("Direction must be 'x' or 'y'")
        return self.calculate_lamb_dicke_parameters_by_branch(wavevector)[direction]

    def get_two_central_ion_separation(self) -> float:
        """ Calculate the separation between the two central ions in the chain. Returns: Distance between the two central ions in meters """
        self._ensure_mode_analysis_run()
        x, y, z = self.ion_coordinates_from_flattened(self.equilibrium_positions)
        z_dimensionful = z * self.characteristic_parameters['length']
        central_ions = np.argsort(np.abs(z_dimensionful))[:2]
        return np.abs(z_dimensionful[central_ions[0]] - z_dimensionful[central_ions[1]])

    def find_axial_frequency_for_desired_central_ion_separation(self, target_separation: float,
                                                               bounds: tuple = (0.1e6, 0.5e6)) -> float:
        """ Computes the reference-ion axial frequency wz [rad/s] required to achieve a desired separation between
            the central ions, holding the reference radial frequencies and all other settings fixed.

            Parameters:
                - target_separation : float; Desired separation between central ions in meters
                - bounds : tuple; (min_freq, max_freq) in Hz for the root finding algorithm
        """
        from scipy.optimize import root_scalar
        if self.num_ions < 2:
            raise IonSimError("At least two ions are required to define a separation")

        def separation_error(wz_Hz):
            temp_analysis = IonChain(
                num_ions=self.num_ions,
                omega_x=self.reference_trap_frequencies[0],
                omega_y=self.reference_trap_frequencies[1],
                omega_z=2*np.pi*wz_Hz,
                atomic_masses=self._input_atomic_masses_amu,
                atomic_charges=self._input_atomic_charges,
                trap_type=self.trap_type,
                dc_radial_split=self.dc_radial_split,
                principal_axes=self.principal_axes,
            )
            temp_analysis.solve_ion_trap_equilibrium()
            return temp_analysis.get_two_central_ion_separation() - target_separation

        if self.trap_type == 'custom':
            raise IonSimError("find_axial_frequency_for_desired_central_ion_separation is not supported for trap_type='custom'")

        result = root_scalar(separation_error, bracket=bounds, method='bisect')
        if not result.converged:
            raise ValueError("Root finding did not converge. Try adjusting the bounds or check the function behavior.")

        return 2 * np.pi * result.root

    def print_chain_summary(self):
        """ Print a summary of the linear ion chain configuration. """
        self._ensure_mode_analysis_run()
        wx0, wy0, wz0 = self.reference_trap_frequencies / (2 * np.pi * 1e6)
        print()
        print(f" ----- Linear Ion Chain Analysis Summary -----")
        print(f"Number of ions: {self.num_ions}")
        print(f"Trap type: {self.trap_type}")
        print(f"Reference-ion trap frequencies: x={wx0:.2f} MHz, y={wy0:.2f} MHz, z={wz0:.2f} MHz")
        print(f"Ion masses along +z: {np.round(self.atomic_masses / const.u, 3)} amu (original indices {self.ion_indices})")

        spacing = self.get_ion_spacing()
        z_coordinates = self.get_axial_ion_positions()
        print(f"Ion axial coordinates (z-axis): {z_coordinates*1e6} µm")
        print(f"Ion spacing (z-axis): {spacing*1e6} µm")

        com_x, com_y, com_z = self.get_center_of_mass_position()
        print(f"Center of mass position: ({com_x*1e6:.2f} µm, {com_y*1e6:.2f} µm, {com_z*1e6:.2f} µm)")

        axial_freqs = self.get_axial_mode_frequencies()
        radial_x_freqs = self.get_radial_mode_frequencies('x')
        radial_y_freqs = self.get_radial_mode_frequencies('y')

        print(f"\nAxial mode frequencies: {axial_freqs/(2*np.pi)/1e6} MHz")
        print(f"Radial X mode frequencies: {radial_x_freqs/(2*np.pi)/1e6} MHz")
        print(f"Radial Y mode frequencies: {radial_y_freqs/(2*np.pi)/1e6} MHz")
