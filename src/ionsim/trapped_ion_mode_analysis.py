from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import scipy.constants as const
import warnings
import scipy.optimize as opt
from dataclasses import dataclass
from enum import Enum, StrEnum

from ionsim.custom_types import Matrix, Vector
from ionsim.degree_of_freedom import AtomicStructure, MotionalMode
from ionsim.basis import StandardBasis
from ionsim.ionsim_error import IonSimError
from ionsim.config import NUMERICAL_EQUIVALENCE_THRESHOLD, SMALLEST_ENERGY_SCALE

## References:
# https://arxiv.org/abs/2007.12725
# https://arxiv.org/abs/quant-ph/9702053

class ModeOrganization(StrEnum):
    """ Order of the stored normal modes: ascending frequency, or grouped by branch (radial0, radial1, axial),
        each in ascending frequency. """
    FREQUENCY_ONLY = 'frequency_only'
    BRANCH_SORTED = 'branch_sorted'


class ReindexingStrategy(StrEnum):
    """ How ions are reindexed after the equilibrium solve (see _TrappedIonSystem.reindexing_order). """
    DISTANCE = 'distance'
    Z_AXIS = 'z_axis'
    NONE = 'none'


class FrameType(StrEnum):
    """ Coordinate frame for returned vectors: the trap principal-axis frame or the lab frame. """
    TRAP = 'trap'
    LAB = 'lab'


class BranchType(StrEnum):
    """ Mode branch of a linear chain, labeled by the trap principal axis along which the ions move. """
    RADIAL0 = 'radial0' # first radial principal axis
    RADIAL1 = 'radial1' # second radial principal axis
    AXIAL = 'axial' # along the chain's axis

    @property
    def axis(self) -> int:
        """ Index of the principal axis (0, 1, 2 for radial0, radial1, axial). """
        return list(BranchType).index(self)

    @property
    def is_radial(self) -> bool:
        return self is not BranchType.AXIAL


class TrapType(StrEnum):
    """ How the per-ion secular frequencies are obtained (see _TrappedIonSystem.calculate_species_trap_frequencies).

        - LINEAR, SURFACE: RF Paul traps. Frequencies of other species are derived from the reference ion's.
        - CUSTOM: per-ion frequencies are given directly.
    """
    LINEAR = 'linear'
    SURFACE = 'surface'
    CUSTOM = 'custom'

    @property
    def is_rf_paul_trap(self) -> bool:
        return self in (TrapType.LINEAR, TrapType.SURFACE)



# Relative tolerance used to decide which eigenvector components are "tied" for largest magnitude
_GAUGE_TIE_RTOL = 1e-6
# Relative size of imaginary parts below which gauge-fixed participation factors are returned as real
_REAL_RTOL = 1e-9


class TrappedIonCrystal:
    """ Equilibrium configuration and normal modes of a crystal of trapped ions.

        The equilibrium (trap + Coulomb) and normal-mode problem is solved when the object is constructed; the
        object then exposes read-only results: normal mode frequencies, zero-point displacements (mode participation factors), 
        Lamb-Dicke parameters, equilibrium positions, and IonSim motional-mode degrees of freedom. 
    """
    # Axis (0, 1, 2 = x, y, z) along which the default initial guess seeds a chain; None => weakest confinement axis
    _INITIAL_CHAIN_AXIS: int | None = None

    def __init__(self, num_ions: int, omega_x: float | Vector, omega_y: float | Vector, omega_z: float | Vector,
                 atomic_masses: Vector | float, atomic_charges: Vector | float,
                 mode_organization: ModeOrganization | str = ModeOrganization.FREQUENCY_ONLY,
                 reindexing_strategy: ReindexingStrategy | str = ReindexingStrategy.DISTANCE,
                 trap_type: TrapType | str = TrapType.LINEAR, dc_radial_split: tuple[float, float] = (0.5, 0.5),
                 principal_axes: Matrix | None = None, positions_guess: Vector | None = None):
        """ Sets up and solves the trapped-ion equilibrium and normal-mode problem, requiring the following parameters:

            - num_ions: the number of ions
            - omega_x, omega_y, omega_z: secular trap frequencies [rad/s] along the trap's principal axes x, y, z.
                z is the "axial" direction used for 1D chains. For trap_type 'linear' or 'surface' these are the
                frequencies of the *reference ion* (the first ion in the user-supplied lists); frequencies of other
                species are derived from them.  For trap_type 'custom' they may be scalars or length-num_ions arrays
                and are used for each ion as given.
            - atomic_masses: array of atomic masses or a single number => same mass for each ion. Units are amu.
            - atomic_charges: array of ion charge states in units of the elementary charge e (e.g. 1 for Yb+,
                2 for a doubly-ionized ion), or a single number => all ions have the same charge.
            - mode_organization: ModeOrganization (or its string value) for organizing modes
                ('frequency_only' or 'branch_sorted')
            - reindexing_strategy: ReindexingStrategy (or its string value) for reindexing ions after the
                equilibrium solve ('distance', 'z_axis', or 'none').
            - trap_type: TrapType (or its string value): 'linear' or 'surface' (RF Paul traps: RF pseudopotential
                confines radially (x, y), DC potential confines axially (z)), or 'custom' (per-ion frequencies
                given directly).
            - dc_radial_split: (s_x, s_y) with s_x + s_y = 1. The DC axial curvature produces an anti-confining
                radial curvature -s_x * k_z along x and -s_y * k_z along y (Laplace's equation). (0.5, 0.5) is a
                symmetric DC quadrupole; other values describe an additional DC radial quadrupole. Only used for
                'linear'/'surface'.
            - principal_axes: optional 3x3 array whose rows are the trap principal axes (x, y, z), expressed in
                the lab frame. Rows are normalized; they must be mutually orthogonal and right-handed. All
                internal quantities are computed in the trap (principal-axis) frame; wavevectors passed to the
                Lamb-Dicke methods are given in the lab frame. Defaults to the identity (lab frame == trap frame).
            - positions_guess: optional initial guess for the equilibrium solve, as a dimensionless (units of
                characteristic_length) trap-frame flattened array [x_1..x_N, y_1..y_N, z_1..z_N] in the
                user-supplied ion order. By default a deterministic linear chain in the user-supplied ion order is
                used; for mixed species this order determines which species sits where.

            Ion bookkeeping: the ions may be reindexed after the equilibrium solve. All per-ion arrays are
            permuted consistently, and `ion_indices[k]` gives the user-supplied (original) index of the ion now
            stored at index k.
        """
        self._system = _TrappedIonSystem(num_ions, omega_x, omega_y, omega_z, atomic_masses, atomic_charges,
                                         mode_organization=mode_organization, reindexing_strategy=reindexing_strategy,
                                         trap_type=trap_type, dc_radial_split=dc_radial_split, principal_axes=principal_axes, 
                                         initial_chain_axis=self._INITIAL_CHAIN_AXIS, positions_guess=positions_guess)

    @classmethod
    def from_species(cls, species_name: str, num_ions: int, omega_x: float, omega_y: float, omega_z: float, **kwargs):
        """Build the mode analysis class from a species name, corresponding to a system of N identical ions.
            Additional keyword arguments are passed to the constructor."""
        species_data = AtomicStructure.get_config_data(species_name)
        return cls(num_ions, omega_x, omega_y, omega_z, species_data['mass'], species_data['charge'], **kwargs)

    @classmethod
    def from_atomic_structure_basis(cls, atomic_structure_basis: StandardBasis, omega_x: float, omega_y: float, omega_z: float, **kwargs):
        """ Build the mode analysis class from a basis of AtomicStructure degrees of freedom. The order of the DOFs
            sets the ion order (the first DOF is the reference ion). Additional keyword arguments are passed to the
            constructor. ion_indices maps stored ions back to the DOF order. """
        dofs = atomic_structure_basis.degrees_of_freedom
        for dof in dofs:
            if not isinstance(dof, AtomicStructure):
                raise IonSimError("Atomic structure basis should only contain AtomicStructure objects. No motional modes should be included.")
        atomic_masses = [dof.atomic_mass for dof in dofs]
        atomic_charges = [dof.atomic_charge for dof in dofs]
        return cls(len(dofs), omega_x, omega_y, omega_z, atomic_masses, atomic_charges, **kwargs)

    @property
    def num_ions(self) -> int:
        return self._system.num_ions

    @property
    def num_modes(self) -> int:
        return 3 * self._system.num_ions

    @property
    def trap_type(self) -> TrapType:
        return self._system.trap_type

    @property
    def mode_organization(self) -> ModeOrganization:
        return self._system.mode_organization

    @property
    def reindexing_strategy(self) -> ReindexingStrategy:
        return self._system.reindexing_strategy

    @property
    def dc_radial_split(self) -> Vector:
        return self._system.dc_radial_split.copy()

    @property
    def principal_axes(self) -> Matrix:
        """ 3x3 rotation matrix whose rows are the trap principal axes in lab coordinates. """
        return self._system.principal_axes.copy()

    @property
    def ion_indices(self) -> Vector:
        """ ion_indices[k] is the user-supplied (original) index of the ion stored at index k. """
        return self._system.ions.original_indices.copy()

    @property
    def atomic_masses(self) -> Vector:
        """ Ion masses in kg. """
        return self._system.ions.masses.copy()

    @property
    def atomic_charges(self) -> Vector:
        """ Ion charge states in units of e. """
        return self._system.ions.charge_states.copy()

    @property
    def charges(self) -> Vector:
        """ Ion charges in Coulombs. """
        return self._system.ions.charges

    @property
    def omega_x(self) -> Vector:
        """ Secular trap frequency of each ion along the x principal axis [rad/s]. """
        return self._system.ions.trap_frequencies[0].copy()

    @property
    def omega_y(self) -> Vector:
        """ Secular trap frequency of each ion along the y principal axis [rad/s]. """
        return self._system.ions.trap_frequencies[1].copy()

    @property
    def omega_z(self) -> Vector:
        """ Secular trap frequency of each ion along the z (axial) principal axis [rad/s]. """
        return self._system.ions.trap_frequencies[2].copy()

    @property
    def reference_trap_frequencies(self) -> NDArray:
        """ Secular trap frequencies (x, y, z) of the reference ion (first user-supplied ion) [rad/s]. """
        return self._system.reference_trap_frequencies.copy()

    @property
    def characteristic_length(self) -> float:
        """ Length scale of the dimensionless problem [m] (see characteristic_length). """
        return self._system.scales.length

    @property
    def characteristic_time(self) -> float:
        return self._system.scales.time

    @property
    def characteristic_velocity(self) -> float:
        return self._system.scales.velocity

    @property
    def characteristic_energy(self) -> float:
        return self._system.scales.energy

    def lab_to_trap_frame(self, vector: Vector) -> Vector:
        """ Rotates a lab-frame vector (or array of shape (3, ...)) into the trap principal-axis frame. """
        return np.tensordot(self._system.principal_axes, np.asarray(vector), axes=(1, 0))

    def trap_to_lab_frame(self, vector: Vector) -> Vector:
        """ Rotates a trap-frame vector (or array of shape (3, ...)) into the lab frame. """
        return np.tensordot(self._system.principal_axes.T, np.asarray(vector), axes=(1, 0))

    def _wavevector_in_trap_frame(self, wavevector: Vector) -> Vector:
        k = np.asarray(wavevector, dtype=float)
        if k.shape != (3,):
            raise ValueError(f"Wavevector must be a 3-element array (kx, ky, kz) in the lab frame. Received {wavevector}")
        return self.lab_to_trap_frame(k)

    @property
    def normal_mode_frequencies(self) -> Vector:
        """ Normal mode frequencies in rad/s, in the order of the stored modes. """
        return self._system.normal_mode_frequencies.copy()

    def return_equilibrium_positions(self, dimensionless: bool, frame: FrameType | str = FrameType.TRAP) -> Vector:
        """ Return flattened equilibrium positions [x_1..x_N, y_1..y_N, z_1..z_N], dimensionless or in meters,
            in the trap or lab frame. """
        frame = FrameType(frame)
        positions = self._system.equilibrium_positions.copy()
        if frame is FrameType.LAB:
            positions = self.trap_to_lab_frame(positions.reshape(3, self.num_ions)).reshape(-1)
        if dimensionless:
            return positions
        return positions * self.characteristic_length

    def calculate_mode_participation_factors(self, frame: FrameType | str = FrameType.TRAP) -> Matrix:
        """ Returns ion-mode participation factors (zero-point displacement amplitudes, in meters), related to the
            Lamb-Dicke parameters by eta[ion, mode] = k . r_zpf[:, ion, mode].

            Shape (3, N, 3N): result[d, i, n] is the zero-point displacement of ion i along direction d in mode n,
            such that the displacement operator is r_{d,i} = sum_n result[d, i, n] (a_n + a_n^dagger).
            For a single ion this reduces to sqrt(hbar / (2 m omega)).

            - frame: 'trap' (directions are the principal axes x, y, z) or 'lab' (rotated with principal_axes).

            Sign convention: in each mode, the largest-magnitude component (over all ions and directions, trap
            frame) is positive; ties are broken by the lowest index in the order [x_1..x_N, y_1..y_N, z_1..z_N].
            Relative signs within a mode are physical. The result is real; if degenerate modes leave
            non-negligible imaginary parts, it is complex and a warning is issued when the object is constructed.
        """
        frame = FrameType(frame)
        participation = self._system.participation_factors.copy()
        if frame is FrameType.LAB:
            participation = self.trap_to_lab_frame(participation)
        return participation

    def calculate_lamb_dicke_parameters(self, wavevector: Vector) -> Matrix:
        """ Lamb-Dicke parameter matrix eta[ion, mode] = k . r_zpf for a lab-frame wavevector k = (kx, ky, kz) [1/m].
            Shape (num_ions, 3 * num_ions). Real, with the sign convention of calculate_mode_participation_factors.
            For a Raman transition use the difference wavevector k1 - k2.
        """
        k_trap = self._wavevector_in_trap_frame(wavevector)
        return np.einsum('d,dim->im', k_trap, self._system.participation_factors)

    def compute_reference_single_ion_lamb_dicke_factors(self, wavenumber: float) -> (float, float, float):
        """ Computes analytical Lamb-Dicke parameter by eta = k * sqrt(hbar / 2 m omega) for ion 0 in a light-field
            with wavevector |k| along each principal axis.
            - wavenumber: wavevector magnitude |k| in units of 1 / m
        """
        trap_frequencies = np.array([self.omega_x[0], self.omega_y[0], self.omega_z[0]])
        mass = self.atomic_masses[0]
        eta_x, eta_y, eta_z = wavenumber * np.sqrt(const.hbar / (2 * mass * trap_frequencies))
        return eta_x, eta_y, eta_z

    def build_mode_DOFs(self, mode_indices: list[int], fock_dimensions: Vector | int) -> list[MotionalMode]:
        """ Builds and returns a list of IonSim Motional Degrees of Freedom, with frequencies in rad/s.

            - mode_indices: global mode indices (0 to 3N - 1, in the stored mode order)
            - fock_dimensions: one Fock dimension per mode, or a single value applied to all modes
            - Each DOF is named 'mode <global index>'
        """
        mode_indices = self._standardize_mode_indices(mode_indices)
        fock_dimensions = _standardize_fock_dimensions(fock_dimensions, len(mode_indices))
        frequencies = self._system.normal_mode_frequencies
        return [MotionalMode.from_frequency(frequencies[mode_index], int(fock_dim), name=f'mode {mode_index}')
                for mode_index, fock_dim in zip(mode_indices, fock_dimensions)]

    def get_mode_participation_factors_for_mode(self, mode_index: int, frame: FrameType | str = FrameType.TRAP) -> Matrix:
        """ Returns mode participation factors for a specific mode by its index.

            Output: Array of shape (3, num_ions) where result[dimension, ion] gives the participation factor
                for that dimension and ion in the specified mode.
        """
        return self.calculate_mode_participation_factors(frame=frame)[:, :, mode_index]

    def get_lamb_dicke_parameters_for_mode(self, mode_index: int, wavevector: Vector) -> Vector:
        """ Return Lamb-Dicke parameters for a specific mode by its index.

                - wavevector: lab-frame wavevector k as a 3-element array (kx, ky, kz) in 1/m

                Returns an array of shape (num_ions,) where result[ion] gives the LD parameter for that ion.
                Sign convention: see calculate_mode_participation_factors.
        """
        return self.calculate_lamb_dicke_parameters(wavevector)[:, mode_index]

    def get_lamb_dicke_parameters_for_ions_and_modes(self, ion_indices: list[int], mode_indices: list[int], wavevector: Vector) -> Matrix:
        """ Lamb-Dicke parameters eta[i, j] for ion ion_indices[i] and mode mode_indices[j], shape
            (len(ion_indices), len(mode_indices)): the sub-matrix of calculate_lamb_dicke_parameters(wavevector), in the
            requested order (the layout of the lamb_dicke_parameters argument of the Laser sideband builders).

            - ion_indices: stored ion indices (rows of calculate_lamb_dicke_parameters; ion_indices[k] of this class
                gives the user-supplied index of stored ion k)
            - mode_indices: global mode indices (0 to 3N - 1, in the stored mode order)
            - wavevector: lab-frame wavevector k as a 3-element array (kx, ky, kz) in 1/m
        """
        rows = self._standardize_ion_indices(ion_indices)
        columns = self._standardize_mode_indices(mode_indices)
        return self.calculate_lamb_dicke_parameters(wavevector)[np.ix_(rows, columns)]

    def _standardize_ion_indices(self, ion_indices: list[int]) -> NDArray:
        return _standardize_indices(ion_indices, self.num_ions)

    def _standardize_mode_indices(self, mode_indices: list[int]) -> NDArray:
        return _standardize_indices(mode_indices, self.num_modes)

    def get_mode_properties(self, mode_index: int) -> dict:
        """Get comprehensive properties for a specific mode. Returns a dictionary containing frequency_angular, frequency_MHZ, participation_factors, and mode_index. """
        frequency_angular = self._system.normal_mode_frequencies[mode_index]
        return {
            'frequency_angular': frequency_angular,  # rad/s
            'frequency_MHz': frequency_angular / (2 * np.pi * 1e6),  # MHz
            'participation_factors': self.get_mode_participation_factors_for_mode(mode_index), # shape (3, num ions) in trap frame
            'mode_index': mode_index
        }


class IonChain(TrappedIonCrystal):
    """ Subclass for setting up and analyzing linear (1D) ion chains.

        Ions are aligned along the z principal axis (axial direction), with no
        displacement in the radial directions). Modes are grouped into the branches
        BranchType.RADIAL0, BranchType.RADIAL1, and BranchType.AXIAL (strings also accepted). 

        This class provides specialized methods and helper methods for linear chains.  
    """
    _INITIAL_CHAIN_AXIS = BranchType.AXIAL.axis  # always seed a chain along the axial direction

    def __init__(self, num_ions: int, omega_x: float | Vector, omega_y: float | Vector, omega_z: float | Vector,
                 atomic_masses: np.ndarray | float, atomic_charges: np.ndarray | float,
                 trap_type: TrapType | str = TrapType.LINEAR, dc_radial_split: tuple[float, float] = (0.5, 0.5),
                 principal_axes: Matrix | None = None, positions_guess: Vector | None = None):
        """ Sets up and solves a linear ion chain with branch-sorted modes and ions indexed by ascending axial position.
            See TrappedIonCrystal for the arguments. For mixed species, the user-supplied order is the order of
            the ions along +axial direction (e.g. +z). """
        super().__init__(num_ions, omega_x, omega_y, omega_z, atomic_masses, atomic_charges,
                         mode_organization=ModeOrganization.BRANCH_SORTED,
                         reindexing_strategy=ReindexingStrategy.Z_AXIS, trap_type=trap_type,
                         dc_radial_split=dc_radial_split, principal_axes=principal_axes,
                         positions_guess=positions_guess)

    def get_global_mode_index(self, branch: BranchType | str, mode_index_in_branch: int) -> int:
        """ Global mode index (0 to 3N - 1) of the mode mode_index_in_branch (ascending frequency) of a branch
            (BranchType or 'radial0', 'radial1', 'axial'). Global indices select modes in build_mode_DOFs,
            get_lamb_dicke_parameters_for_ions_and_modes, and the Laser sideband builders *_from_trap. """
        branch = BranchType(branch)
        if mode_index_in_branch < 0 or mode_index_in_branch >= self.num_ions:
            raise IonSimError(f"Mode index in branch must be between 0 and {self.num_ions-1}")
        return branch.axis * self.num_ions + mode_index_in_branch

    def _branch_slice(self, branch: BranchType | str) -> slice:
        start = self.get_global_mode_index(branch, 0)
        return slice(start, start + self.num_ions)

    @staticmethod
    def _radial_branch(direction: BranchType | str) -> BranchType:
        direction = BranchType(direction)
        if not direction.is_radial:
            raise IonSimError(f"Radial direction must be '{BranchType.RADIAL0}' or '{BranchType.RADIAL1}'. "f"Received '{direction}'")
        return direction

    def _axial_positions(self) -> NDArray:
        """ Axial (z) coordinate of each stored ion, in meters. """
        positions = self.return_equilibrium_positions(dimensionless=False).reshape(3, self.num_ions)
        return positions[BranchType.AXIAL.axis]

    def get_axial_ion_positions(self):
        """ Returns an array of ion axial positions along the axial direction (meters, ascending). """
        return np.sort(self._axial_positions())

    def get_ion_spacing(self):
        """ Returns an array of ion spacings between consecutive ions along axial direction. """
        return np.diff(self.get_axial_ion_positions())

    def get_center_of_mass_position(self) -> tuple:
        """ Get the center of mass position of the ion chain (trap frame, meters). """
        positions = self.return_equilibrium_positions(dimensionless=False).reshape(3, self.num_ions)
        masses = self._system.ions.masses
        com_x, com_y, com_z = positions @ masses / np.sum(masses)
        return com_x, com_y, com_z

    def get_axial_mode_frequencies(self):
        """ Returns the frequencies [rad/s] of the axial modes. """
        return self._system.normal_mode_frequencies[self._branch_slice(BranchType.AXIAL)].copy()

    def get_radial_mode_frequencies(self, direction: BranchType | str = BranchType.RADIAL0):
        """ Returns the frequencies of the radial modes in rad/s for the specified direction
            (BranchType.RADIAL0 / BranchType.RADIAL1, or 'radial0' / 'radial1'). """
        return self._system.normal_mode_frequencies[self._branch_slice(self._radial_branch(direction))].copy()

    def get_mode_participation_factors_by_branch(self):
        """ Returns a dictionary of mode participation factors organized by branch (radial0, radial1, axial).

            Output dictionary keyed by BranchType (the keys also match 'radial0', 'radial1', 'axial') containing the
            participation factors along that branch's principal axis. Each value is a 2D array of shape
            (num_ions, num_modes_per_branch)
        """
        mode_pf = self.calculate_mode_participation_factors()
        return {branch: mode_pf[branch.axis, :, self._branch_slice(branch)] for branch in BranchType}

    def get_mode_properties_by_branch(self, branch: BranchType | str, mode_index_in_branch: int) -> dict:
        """ Returns a dictionary containing properties (freq, participations) for a specific mode within a branch.

            -   branch: BranchType (or 'radial0', 'radial1', 'axial') for the mode branch
            -   mode_index_in_branch: Index of the mode within the specified branch (0 to num_ions-1),
                    in ascending frequency
        """
        branch = BranchType(branch)
        properties = self.get_mode_properties(self.get_global_mode_index(branch, mode_index_in_branch))
        properties['branch'] = branch
        properties['mode_index_in_branch'] = mode_index_in_branch
        return properties

    def get_lamb_dicke_parameters_by_branch(self, branch: BranchType | str, mode_index_in_branch: int, wavevector: Vector) -> Vector:
        """ Get Lamb-Dicke parameters for a specific mode within a branch.

            -   branch: BranchType (or 'radial0', 'radial1', 'axial') for the mode branch
            -   mode_index_in_branch: Index of the mode within the specified branch
            -   wavevector: lab-frame wavevector k as a 3-element array (1/m)

            Returns an array of Lamb-Dicke parameters for each ion in the specified mode.
        """
        return self.get_lamb_dicke_parameters_for_mode(self.get_global_mode_index(branch, mode_index_in_branch), wavevector)

    def calculate_lamb_dicke_parameters_by_branch(self, wavevector: Vector) -> dict:
        """ Calculate Lamb-Dicke parameters (k dot dr) for all ions and modes, organized by branch. Ratio of the spatial extent of 
            the ion's zero-point motion to the reduced wavelength of the laser, projected onto the mode.

            -   wavevector : lab-frame k = (kx, ky, kz) as a 3-element array in units of 1/m

            Returns a dictionary keyed by BranchType or string ('radial0', 'radial1', 'axial'). Each value is a 2D array of shape
                (num_ions, num_modes_per_branch) representing eta[ion_index, mode_index] for the modes of that branch.
        """
        total_ld_params = self.calculate_lamb_dicke_parameters(wavevector)
        return {branch: total_ld_params[:, self._branch_slice(branch)] for branch in BranchType}

    def build_mode_DOFs_from_branch(self, branch: BranchType | str, mode_indices: list[int], fock_dimensions: Vector | int) -> list[MotionalMode]:
        """ Builds and returns a list of IonSim Motional Degrees of Freedom for a branch type (BranchType or string representation)
            - mode_indices corresponds to a mode's index within a branch 
            - Applies each fock dimension to each mode, or applies the same fock dimension to all the modes
            - The DOFs are named by their global mode index (see build_mode_DOFs and get_global_mode_index)
        """
        global_indices = [self.get_global_mode_index(branch, idx) for idx in mode_indices]
        return self.build_mode_DOFs(global_indices, fock_dimensions)

    def calculate_lamb_dicke_parameters_full(self, wavevector: Vector) -> Matrix:
        """ Calculate full Lamb-Dicke parameter matrix from a lab-frame laser wavevector k = (kx, ky, kz)

            Returns LD parameter matrix of shape (num_ions, num_modes) where eta[ion, mode] gives the total LD parameter k . r_zpf. 
                Sign convention: see calculate_mode_participation_factors.
        """
        return self.calculate_lamb_dicke_parameters(wavevector)

    def get_axial_lamb_dicke_parameters(self, wavevector: Vector) -> Matrix:
        """ Get Lamb-Dicke parameters for axial (z) modes only.

            Returns Lamb-Dicke parameters for axial modes, NDArray of shape (num_ions, num_axial_modes)
        """
        return self.calculate_lamb_dicke_parameters_by_branch(wavevector)[BranchType.AXIAL]

    def get_radial_lamb_dicke_parameters(self, wavevector: Vector, direction: BranchType | str) -> Matrix:
        """ Get Lamb-Dicke parameters for radial modes in a specified direction (BranchType.RADIAL0 / BranchType.RADIAL1, or 'radial0' / 'radial1').
            Returns: Lamb-Dicke parameters for radial modes, array of shape (num_ions, num_radial_modes)
        """
        direction = self._radial_branch(direction)
        return self.calculate_lamb_dicke_parameters_by_branch(wavevector)[direction]

    def get_two_central_ion_separation(self) -> float:
        """ Calculate the separation between the two central ions in the chain. Returns: Distance between the two central ions in meters """
        z_dimensionful = self._axial_positions()
        central_ions = np.argsort(np.abs(z_dimensionful))[:2]
        return np.abs(z_dimensionful[central_ions[0]] - z_dimensionful[central_ions[1]])

    def find_axial_frequency_for_desired_central_ion_separation(self, target_separation: float,
                                                               bounds: tuple = (0.1e6, 0.5e6)) -> float:
        """ Computes the reference-ion axial frequency wz [rad/s] required to achieve a desired separation between
            the central ions, holding the reference radial frequencies and all other settings fixed.

                - target_separation : float; Desired separation between central ions in meters
                - bounds : tuple; (min_freq, max_freq) in Hz for the root finding algorithm
        """
        if self.num_ions < 2:
            raise IonSimError("At least two ions are required to define a separation")
        if not self.trap_type.is_rf_paul_trap:
            raise IonSimError(f"find_axial_frequency_for_desired_central_ion_separation is not supported for trap_type='{self.trap_type}'")

        input_ions = self._system.input_ions
        def separation_error(wz_Hz):
            temp_analysis = IonChain(
                num_ions=self.num_ions,
                omega_x=self.reference_trap_frequencies[0],
                omega_y=self.reference_trap_frequencies[1],
                omega_z=2*np.pi*wz_Hz,
                atomic_masses=input_ions.masses / const.u,
                atomic_charges=input_ions.charge_states,
                trap_type=self.trap_type,
                dc_radial_split=self.dc_radial_split,
                principal_axes=self.principal_axes,
            )
            return temp_analysis.get_two_central_ion_separation() - target_separation

        result = opt.root_scalar(separation_error, bracket=bounds, method='bisect')
        if not result.converged:
            raise IonSimError("Root finding did not converge. Try adjusting the bounds or check the function behavior.")

        return 2 * np.pi * result.root

    def chain_summary(self) -> dict:
        """ Return a summary of the linear ion chain configuration in physical units. """
        chain_summary = {}
        chain_summary["Number of ions"] = self.num_ions 
        chain_summary["Trap type"] = self.trap_type # or string rep?  
        chain_summary["Ion masses amu"] = np.round(self.atomic_masses / const.u, 3)

        spacing = self.get_ion_spacing()
        z_coordinates = self.get_axial_ion_positions()
        chain_summary["Ion axial spacing microns"] = spacing * 1E6 
        chain_summary["Ion coordinates microns"] = z_coordinates * 1E6 

        com_x, com_y, com_z = self.get_center_of_mass_position()
        chain_summary["Center of mass positions microns"] = (com_x*1E6, com_y*1E6, com_z*1E6) 

        axial_freqs = self.get_axial_mode_frequencies()
        radial0_freqs = self.get_radial_mode_frequencies(BranchType.RADIAL0)
        radial1_freqs = self.get_radial_mode_frequencies(BranchType.RADIAL1)

        chain_summary["Axial mode frequencies MHz"] = axial_freqs/(2. * np.pi)/1E6
        chain_summary["Radial 0 mode frequencies MHz"] = radial0_freqs/(2. * np.pi)/1E6
        chain_summary["Radial 1 mode frequencies MHz"] = radial1_freqs/(2. * np.pi)/1E6

        return chain_summary



def characteristic_length(q: float, mass: float, omega: float) -> float:
    """ Computes characteristic length in trapped ion system: Coulomb + harmonic oscillator physics.

        l0 = (2 k_e q^2 / (m omega^2))^(1/3), which is the equilibrium separation of two identical ions
        in a harmonic well of frequency omega.
    """
    k_e = 1 / (4 * np.pi * const.epsilon_0)  # Coulomb constant
    l0 = ((k_e * q ** 2) / (.5 * mass * omega ** 2)) ** (1 / 3)
    return l0

def _standardize_fock_dimensions(fock_dimensions: Vector | int, num_modes: int) -> NDArray:
    """ Broadcasts a single Fock dimension to num_modes modes, or checks that one is given per mode. """
    dims = np.atleast_1d(np.asarray(fock_dimensions))
    if dims.size == 1:
        dims = np.full(num_modes, dims.item())
    elif dims.size != num_modes:
        raise IonSimError(f"Received {dims.size} Fock dimensions for {num_modes} modes")
    if not np.all(np.equal(np.mod(dims, 1), 0)) or np.any(dims < 1):
        raise IonSimError(f"Fock dimensions must be positive integers. Received {fock_dimensions}")
    return dims.astype(int)


def _standardize_indices(indices: list[int], size: int) -> NDArray:
    """ Validates a list of integer indices into range(size) (repeats allowed) and returns them as an int array. """
    array = np.asarray(indices)
    if array.ndim != 1:
        raise IonSimError(f"Indices must be a list of integers. Received {indices}")
    if array.size and (array.dtype.kind not in 'iu' or np.any(array < 0) or np.any(array >= size)):
        raise IonSimError(f"Indices must be integers between 0 and {size - 1}. Received {indices}")
    return array.astype(int)


def _enlarge_scalar_to_vector(x: float | Vector, num_ions: int, name: str) -> NDArray:
    """ Broadcasts a scalar to a length-num_ions float array, or checks that an array has length num_ions. """
    array = np.full(num_ions, x, dtype=float) if np.ndim(x) == 0 else np.asarray(x, dtype=float)
    if array.shape != (num_ions,):
        raise IonSimError(f"{name} must be a scalar or have length {num_ions}. Received {x}")
    return array


@dataclass(frozen=True)
class _Ions:
    """ Per-ion properties, all in the same ion order. """
    masses: NDArray             # kg, shape (N,)
    charge_states: NDArray      # units of e, shape (N,)
    trap_frequencies: NDArray   # secular frequencies along the x, y, z principal axes [rad/s], shape (3, N)
    original_indices: NDArray   # user-supplied index of each ion, shape (N,)

    @property
    def charges(self) -> NDArray:
        """ Ion charges in Coulombs. """
        return self.charge_states * const.e

    def permuted(self, order: NDArray) -> _Ions:
        """ The same ions in a new order: new index k <- current index order[k]. """
        return _Ions(self.masses[order], self.charge_states[order], self.trap_frequencies[:, order],
                     self.original_indices[order])


@dataclass(frozen=True)
class _DimensionlessScales:
    """ Units of the dimensionless problem, set by the reference ion's mass, charge, and axial frequency. """
    mass: float       # kg
    charge: float     # C
    frequency: float  # rad/s

    @property
    def length(self) -> float:
        return characteristic_length(self.charge, self.mass, self.frequency)

    @property
    def time(self) -> float:
        return 1 / self.frequency

    @property
    def velocity(self) -> float:
        return self.length * self.frequency

    @property
    def energy(self) -> float:
        return 0.5 * self.mass * self.velocity ** 2


@dataclass(frozen=True)
class _CoulombTrapPotential:
    """ Dimensionless potential energy of N ions in a harmonic trap with Coulomb repulsion, in units of
        mass * frequency^2 * length^2 of the _DimensionlessScales:

            U(u) = sum_{d,i} 1/2 m_i w_{d,i}^2 r_{d,i}^2 + 1/2 sum_{i<j} q_i q_j / |r_i - r_j|

        Positions u are flattened as [x_1..x_N, y_1..y_N, z_1..z_N] (trap frame, units of length).
    """
    m: NDArray  # masses, shape (N,)
    q: NDArray  # charges, shape (N,)
    w: NDArray  # trap frequencies felt by the ion, along x, y, z, shape (3, N)

    @classmethod
    def from_ions(cls, ions: _Ions, scales: _DimensionlessScales) -> _CoulombTrapPotential:
        return cls(ions.masses / scales.mass, ions.charges / scales.charge, ions.trap_frequencies / scales.frequency)

    @property
    def num_ions(self) -> int:
        return len(self.m)

    def _separations(self, u: Vector) -> tuple[NDArray, NDArray, NDArray]:
        """ Pairwise displacements dr[d, i, j] = r_{d,i} - r_{d,j}, distances |r_i - r_j|, and q_i q_j. """
        r = u.reshape(3, self.num_ions)
        dr = r[:, :, np.newaxis] - r[:, np.newaxis, :]
        distances = np.sqrt(np.sum(dr ** 2, axis=0))
        qq = self.q[:, np.newaxis] * self.q[np.newaxis, :]
        return dr, distances, qq

    def energy(self, u: Vector) -> float:
        r = u.reshape(3, self.num_ions)
        _, distances, qq = self._separations(u)
        with np.errstate(divide='ignore'):
            coulomb = 0.25 * np.sum(np.where(distances != 0., qq / distances, 0.))  # 1/2 * (sum over i != j) / 2
        return 0.5 * np.sum(self.m * self.w ** 2 * r ** 2) + coulomb

    def gradient(self, u: Vector) -> Vector:
        """ dU/du, shape (3N,) """
        r = u.reshape(3, self.num_ions)
        dr, distances, qq = self._separations(u)
        with np.errstate(divide='ignore'):
            inverse_cubed = np.where(distances != 0., distances ** (-3), 0.)
        coulomb = -0.5 * np.sum(dr * inverse_cubed * qq, axis=2)
        return (self.m * self.w ** 2 * r + coulomb).reshape(-1)

    def hessian(self, u: Vector) -> Matrix:
        """ d^2U / du du, shape (3N, 3N). For the Coulomb part:

            For i != j:  d^2 V / dr_{a,i} dr_{b,j} = q_i q_j (delta_ab r_ij^2 - 3 dr_a dr_b) / r_ij^5
            For i == j:  d^2 V / dr_{a,i} dr_{b,i} = - sum_{j != i} (the i != j expression above)
        """
        N = self.num_ions
        dr, distances, qq = self._separations(u)
        with np.errstate(divide='ignore'):
            inverse_fifth = np.where(distances != 0., distances ** (-5), 0.)
        blocks = [[None] * 3 for _ in range(3)]
        for a in range(3):
            for b in range(3):
                block = qq * ((a == b) * distances ** 2 - 3 * dr[a] * dr[b]) * inverse_fifth  # zero on the diagonal
                np.fill_diagonal(block, -np.sum(block, axis=1))
                blocks[a][b] = 0.5 * block  # consistent with the factor of 1/2 on the Coulomb energy
        trap = np.diag((self.m * self.w ** 2).reshape(-1))
        return np.block(blocks) + trap


class _TrappedIonSystem:
    """ Internal implementation of the trapped-ion equilibrium and normal-mode physics problem.

        The constructor runs each step of the problem in order, passing results explicitly from one step to the
        next (the static methods below have no side effects):

            1. Standardize and validate inputs
            2. Solve for the equilibrium positions with the ions in the user-supplied order.
            3. Reindex the ions according to the reindexing strategy.
            4. Solve for the normal modes at the equilibrium.
            5. Compute the zero-point displacements and mode participation factors.

        Everything here is in the trap (principal-axis) frame and, unless stated otherwise, in dimensionless units
        set by the reference ion (the first ion in the user-supplied order). Only the results needed by the public
        classes TrappedIonCrystal and IonChain are stored.
    """
    def __init__(self, num_ions: int, omega_x: float | Vector, omega_y: float | Vector, omega_z: float | Vector,
                 atomic_masses: Vector | float, atomic_charges: Vector | float, mode_organization: ModeOrganization, 
                 reindexing_strategy: ReindexingStrategy, trap_type: TrapType, dc_radial_split: tuple[float, float], 
                 principal_axes: Matrix | None, initial_chain_axis: int | None = None, positions_guess: Vector | None = None, 
                 solver_options: dict | None=None):
        # 1. Inputs
        if num_ions < 1:
            raise IonSimError(f"num_ions must be positive. Received {num_ions}")
        self.num_ions = num_ions
        self.mode_organization = ModeOrganization(mode_organization)
        self.reindexing_strategy = ReindexingStrategy(reindexing_strategy)
        self.trap_type = TrapType(trap_type)
        self.dc_radial_split = self.standardize_dc_radial_split(dc_radial_split)
        self.principal_axes = self.standardize_principal_axes(principal_axes)

        masses, charge_states = self.standardize_ion_properties(num_ions, atomic_masses, atomic_charges)
        trap_frequencies = self.calculate_species_trap_frequencies(omega_x, omega_y, omega_z, masses, charge_states, self.trap_type, self.dc_radial_split)

        # Ions in the user-supplied order; the first one is the reference ion
        placeholder_ion_indices = np.arange(num_ions)
        self.input_ions = _Ions(masses, charge_states, trap_frequencies, placeholder_ion_indices)
        self.reference_trap_frequencies = trap_frequencies[:, 0].copy()
        self.scales = _DimensionlessScales(mass=masses[0], charge=self.input_ions.charges[0], frequency=trap_frequencies[2, 0])

        # 2. Equilibrium positions, ions in the user-supplied order
        potential = _CoulombTrapPotential.from_ions(self.input_ions, self.scales)
        if positions_guess is None:
            axis = int(np.argmin(self.reference_trap_frequencies)) if initial_chain_axis is None else initial_chain_axis
            positions_guess = self.initial_positions_guess(num_ions, axis, self.reference_trap_frequencies[axis] / self.scales.frequency)

        # Solve for equilibrium positions 
        default_solver_options = {'gtol': 1e-10, 'disp': False}
        if solver_options is None:
            solver_options = default_solver_options 
        positions = self.solve_equilibrium_positions(potential, positions_guess, solver_options)

        # 3. Reindexing
        order = self.reindexing_order(positions, self.reindexing_strategy)
        self.ions = self.input_ions.permuted(order)
        self.potential = _CoulombTrapPotential.from_ions(self.ions, self.scales)
        self.equilibrium_positions = positions.reshape(3, num_ions)[:, order].reshape(-1)

        # 4. Compute normal mode frequencies -- eigenvalues and eigenvectors 
        eigenvalues, eigenvectors = self.compute_normal_modes(self.potential.hessian(self.equilibrium_positions),
                                                            self.potential.m, self.mode_organization)
        self.normal_mode_frequencies = eigenvalues * self.scales.frequency

        # 5. Compute participation factors via zero point displacements  
        self.participation_factors = self.calculate_zero_point_displacements(eigenvalues, eigenvectors, self.scales)

    @staticmethod
    def standardize_ion_properties(num_ions: int, atomic_masses: Vector | float, atomic_charges: Vector | float) -> tuple[NDArray, NDArray]:
        """ Returns (masses in kg, charge states in units of e), each of length num_ions, after validation. """
        masses_amu = _enlarge_scalar_to_vector(atomic_masses, num_ions, 'atomic_masses')
        charge_states = _enlarge_scalar_to_vector(atomic_charges, num_ions, 'atomic_charges')
        if np.any(masses_amu <= 0):
            raise IonSimError(f"Atomic masses must be positive. Received {masses_amu} amu")
        if np.any(charge_states == 0):
            raise IonSimError(f"Ion charge states must be nonzero (neutral atoms cannot be trapped here). "
                              f"Received {charge_states}")
        if not (np.all(charge_states > 0) or np.all(charge_states < 0)):
            raise IonSimError(f"All ions must have charges of the same sign. Received {charge_states}")
        return masses_amu * const.u, charge_states

    @staticmethod
    def standardize_dc_radial_split(dc_radial_split) -> NDArray:
        split = np.asarray(dc_radial_split, dtype=float)
        if split.shape != (2,):
            raise IonSimError(f"dc_radial_split must be a pair (s_x, s_y). Received {dc_radial_split}")
        if not np.isclose(np.sum(split), 1.0, rtol=0, atol=1e-12):
            raise IonSimError(f"dc_radial_split must satisfy s_x + s_y = 1 (Laplace's equation). Received {dc_radial_split}")
        return split

    @staticmethod
    def standardize_principal_axes(principal_axes) -> Matrix:
        """ Returns the 3x3 rotation matrix R whose rows are the unit trap axes in lab coordinates.
            v_trap = R @ v_lab and v_lab = R.T @ v_trap.
        """
        if principal_axes is None: 
            return np.eye(3)
        R = np.array(principal_axes, dtype=float)
        if R.shape != (3, 3):
            raise IonSimError(f"principal_axes must be a 3x3 array (rows are the x, y, z trap axes). Received shape {R.shape}")
        norms = np.linalg.norm(R, axis=1)
        if np.any(norms == 0):
            raise IonSimError("principal_axes rows must be nonzero vectors")
        R = R / norms[:, np.newaxis]
        if not np.allclose(R @ R.T, np.eye(3), rtol=0, atol=1e-9):
            raise IonSimError(f"principal_axes rows must be mutually orthogonal. Received {principal_axes}")
        if np.linalg.det(R) < 0:
            raise IonSimError("principal_axes must form a right-handed set (x cross y = z)")
        return R

    @staticmethod
    def calculate_species_trap_frequencies(omega_x, omega_y, omega_z, masses: Vector, charge_states: Vector,
                                           trap_type: TrapType, dc_radial_split: Vector) -> NDArray:
        """ Returns the secular trap frequencies of each ion along the trap principal axes, shape (3, N) [rad/s].

            'custom': frequencies are used as given (scalars are broadcast to every ion).

            'linear' / 'surface' (RF Paul traps, secular approximation, micromotion neglected): with the reference
            ion (index 0; charge q0, mass m0) having secular frequencies (wx, wy, wz) and r_i = (q_i m0) / (q0 m_i),

                axial (DC only):          w_{z,i}^2 = r_i wz^2
                radial (RF pseudo + DC):  w_{x,i}^2 = r_i^2 (wx^2 + s_x wz^2) - s_x r_i wz^2   (same for y with s_y)

            This follows from the DC spring constant scaling as q and the RF pseudopotential spring constant scaling
            as q^2 / m, with the DC axial curvature defocusing the radial directions by s_x, s_y (s_x + s_y = 1).
            RF and DC curvatures are assumed to share the same principal axes.
        """
        num_ions = len(masses)
        if not trap_type.is_rf_paul_trap:
            frequencies = np.array([_enlarge_scalar_to_vector(w, num_ions, f'omega_{label}')
                                    for w, label in zip((omega_x, omega_y, omega_z), 'xyz')])
            if not np.all(frequencies > 0):
                raise IonSimError(f"Trap is unstable: all trap frequencies must be positive. Received {frequencies}")
            return frequencies

        for w, label in zip((omega_x, omega_y, omega_z), 'xyz'):
            if np.ndim(w) != 0:
                raise IonSimError(f"For trap_type '{trap_type}', omega_{label} must be a scalar (the reference ion's "
                                 f"frequency). Use trap_type='custom' to specify per-ion frequencies.")
            if isinstance(w, complex):
                raise IonSimError(f"Secular frequencies should be real-valued and positive. Received {w}.")
            if w <= 0:
                raise IonSimError(f"Trap is unstable: omega_{label} must be positive. Received {w}")

        s_x, s_y = dc_radial_split
        r = (charge_states * masses[0]) / (charge_states[0] * masses)

        rf_x0 = omega_x ** 2 + s_x * omega_z ** 2   # RF pseudopotential curvature of the reference ion (as w^2)
        rf_y0 = omega_y ** 2 + s_y * omega_z ** 2
        if rf_x0 <= 0 or rf_y0 <= 0:
            raise IonSimError(f"Reference trap frequencies {rf_x0=}, {rf_y0=} are inconsistent with an RF Paul trap: the implied RF "
                              "pseudopotential is not confining.")

        frequencies_squared = np.array([r ** 2 * rf_x0 - s_x * r * omega_z ** 2,
                                        r ** 2 * rf_y0 - s_y * r * omega_z ** 2,
                                        r * omega_z ** 2])
        for w2, label in zip(frequencies_squared, 'xyz'):
            if np.any(w2 <= 0):
                unstable = np.flatnonzero(w2 <= 0)
                raise IonSimError(f"Ions {unstable.tolist()} are not confined along {label} for this trap "
                                  f"(secular frequency^2 = {w2[unstable]}).")
        return np.sqrt(frequencies_squared)

    @staticmethod
    def initial_positions_guess(num_ions: int, axis: int, axis_frequency: float) -> Vector:
        """ Deterministic dimensionless initial guess: a linear chain in the given ion order along `axis` (0, 1, 2
            for x, y, z), plus a small transverse perturbation so that a zigzag/2D configuration can form if it is
            energetically favored. axis_frequency is the reference ion's dimensionless frequency along `axis`. """
        spacing = axis_frequency ** (-2 / 3)  # two-ion spacing in units of the characteristic length
        alternating = (-1.0) ** np.arange(num_ions)
        transverse_axes = [d for d in range(3) if d != axis]
        u0 = np.zeros((3, num_ions))
        # The following was found to work well via heuristics
        u0[axis] = (np.arange(num_ions) - (num_ions - 1) / 2) * spacing
        u0[transverse_axes[0]] = 1e-3 * spacing * alternating
        u0[transverse_axes[1]] = 0.5e-3 * spacing * alternating
        return u0.reshape(-1)

    @staticmethod
    def solve_equilibrium_positions(potential: _CoulombTrapPotential, positions_guess: Vector, solver_options: dict | None=None) -> Vector:
        """ Minimizes the potential energy starting from positions_guess (dimensionless, flattened). Returns the
            equilibrium positions in the same ion order. """
        u0 = np.asarray(positions_guess, dtype=float)
        if u0.shape != (3 * potential.num_ions,):
            raise IonSimError(f"positions_guess must have shape ({3 * potential.num_ions},)")

        # Minimize with BFGS, then polish with Newton steps on the analytic Hessian
        u = opt.minimize(potential.energy, u0, method='BFGS', jac=potential.gradient, options=solver_options).x

        max_attempts = 10
        for _ in range(max_attempts):
            gradient = potential.gradient(u)
            if np.max(np.abs(gradient)) < NUMERICAL_EQUIVALENCE_THRESHOLD:
                break
            hessian = potential.hessian(u)
            if np.min(np.linalg.eigvalsh(hessian)) <= 0:
                break  # not at a minimum; compute_normal_modes reports it
            u = u - np.linalg.solve(hessian, gradient)

        residual = np.max(np.abs(potential.gradient(u)))
        if residual > 1e-9:
            warnings.warn(f"Equilibrium solve may not have converged: max |dU/du| = {residual:.2e} (dimensionless)")
        return u

    @staticmethod
    def reindexing_order(positions: Vector, strategy: ReindexingStrategy) -> NDArray:
        """ Returns the permutation (new index k <- current index order[k]) that reindexes the ions:

            - 'distance': ascending distance from the trap center (index 0 is closest to the center)
            - 'z_axis':   ascending z (axial) coordinate (index 0 is the most negative z)
            - 'none':     keep the current order

            Ties (keys equal to 1e-9 in dimensionless units, e.g. the two end ions of a symmetric chain under
            'distance') keep the current order, so the result does not depend on rounding.
        """
        r = positions.reshape(3, -1)
        if strategy is ReindexingStrategy.NONE:
            return np.arange(r.shape[1])
        key = np.linalg.norm(r, axis=0) if strategy is ReindexingStrategy.DISTANCE else r[2]
        return np.argsort(np.round(key, decimals=9), kind='stable')

    @staticmethod
    def compute_normal_modes(hessian: Matrix, masses: NDArray, mode_organization: ModeOrganization) -> tuple[NDArray, NDArray]:
        """ Normal modes of small oscillations about the equilibrium (dimensionless; no magnetic field).

            With canonical coordinates X = (q, p), p = M dq/dt, the Hamiltonian is 1/2 X^T H X with
            H = (K, 0; 0, M^-1), and Hamilton's equations are dX/dt = D X with D = J H, J = (0, I; -I, 0)
            (eq. 1.67 of thesis). Modes evolve as u_n exp(-i w_n t), so u_n are eigenvectors of -D with
            eigenvalue i w_n.

            Returns (w, u): the 3N positive frequencies w_n and the 6N x 3N eigenvectors u_n, ordered by
            mode_organization, normalized so that u_n^dagger H u_n = 1, and with the phase convention of
            fix_eigenvector_phases.
        """
        num_coordinates = len(masses) * 3
        mass_diagonal = np.tile(masses, 3)
        zeros = np.zeros((num_coordinates, num_coordinates))
        identity = np.eye(num_coordinates)
        H = np.block([[hessian, zeros], [zeros, np.diag(1 / mass_diagonal)]])
        J = np.block([[zeros, identity], [-identity, zeros]])

        # u_n(t) = exp(-i w_n t) u_n(0), minus by convention; keep the positive-frequency half
        eigenvalues, eigenvectors = np.linalg.eig(-J @ H)
        order = np.argsort(eigenvalues.imag)[num_coordinates:]
        frequencies, eigenvectors = eigenvalues.imag[order], eigenvectors[:, order]
        if not np.all(frequencies > 0):
            raise IonSimError("Found non-positive normal mode frequencies: the equilibrium found is not a stable "
                              "minimum (e.g. a linear chain beyond the zigzag transition).")

        if mode_organization is ModeOrganization.BRANCH_SORTED:
            order = _TrappedIonSystem.branch_sorted_order(eigenvectors)
            frequencies, eigenvectors = frequencies[order], eigenvectors[:, order]

        norms = np.sqrt(np.real(np.einsum('in,ij,jn->n', eigenvectors.conj(), H, eigenvectors)))
        eigenvectors = _TrappedIonSystem.fix_eigenvector_phases(eigenvectors / norms)

        _TrappedIonSystem.check_normal_modes(H, frequencies, eigenvectors)
        return frequencies, eigenvectors

    @staticmethod
    def branch_sorted_order(eigenvectors: Matrix) -> NDArray:
        """ Permutation that groups modes (columns, assumed in ascending frequency) by the principal axis they
            mostly move along: radial0 (x), radial1 (y), axial (z), keeping ascending frequency within each branch.
            Only makes sense for a linear chain along the axial principal axis. """
        num_coordinates, num_modes = eigenvectors.shape
        num_ions = num_coordinates // 6
        # amplitude of each mode along each axis, summing positions and momenta: shape (3, num_modes)
        amplitudes = np.abs(eigenvectors).reshape(2, 3, num_ions, num_modes).sum(axis=(0, 2))
        dominant_axis = np.argmax(amplitudes, axis=0)
        for branch in BranchType:
            count = np.sum(dominant_axis == branch.axis)
            if count != num_ions:
                raise IonSimError(f"Branch classification error: branch '{branch}' has {count} modes, expected "
                                  f"{num_ions}. Branch sorting requires a linear chain along the axial principal axis.")
        return np.concatenate([np.flatnonzero(dominant_axis == branch.axis) for branch in BranchType])

    @staticmethod
    def fix_eigenvector_phases(eigenvectors: Matrix) -> Matrix:
        """ Removes the arbitrary per-mode phase returned by the eigensolver.

            Each mode u_n (a column) is multiplied by exp(-i theta_n) such that its largest-magnitude position
            component is real and positive. Ties (components within a relative 1e-6 of the largest magnitude, e.g.
            COM or stretch modes of identical ions) are broken by the lowest coordinate index in the flattened
            order [x_1..x_N, y_1..y_N, z_1..z_N].

            Without magnetic fields the position part of a non-degenerate mode is real up to one phase, so after
            this the position part (and hence the Lamb-Dicke parameters) is real, and the displacement operator is
            r_{i,d} = sum_n r^{zpf}_{i,d,n} (a_n + a_n^dagger). Relative signs between ions within a mode are
            physical; the overall sign of each mode is this convention. Degenerate subspaces are not treated.
        """
        position_part = eigenvectors[:eigenvectors.shape[0] // 2]
        magnitudes = np.abs(position_part)
        reference_rows = np.argmax(magnitudes >= (1 - _GAUGE_TIE_RTOL) * magnitudes.max(axis=0), axis=0)
        reference_components = position_part[reference_rows, np.arange(position_part.shape[1])]
        return eigenvectors * np.exp(-1j * np.angle(reference_components))

    @staticmethod
    def check_normal_modes(H: Matrix, frequencies: NDArray, eigenvectors: Matrix):
        """ Warns if the H-normalized modes u_n (with their conjugates) are not complete, or if the canonical
            transformation S = sqrt(2) [Re(u) sqrt(w), -Im(u) sqrt(w)] does not diagonalize H to diag(w, w). """
        degenerate = _TrappedIonSystem.has_degenerate_modes(frequencies)
        projector = eigenvectors @ eigenvectors.conj().T
        completeness = projector + projector.conj()  # sum over u_n u_n^dagger and its conjugate (-w_n) partner
        identity = np.eye(H.shape[0])
        if not (np.allclose(H @ completeness, identity) and np.allclose(completeness @ H, identity)):
            warnings.warn(f"Outer relation check failed (degenerate modes present: {degenerate})")

        scaled = eigenvectors * np.sqrt(frequencies)
        S = np.sqrt(2) * np.concatenate((np.real(scaled), -np.imag(scaled)), axis=1)
        if not np.allclose(S.T @ H @ S, np.diag(np.tile(frequencies, 2))):
            warnings.warn(f"Diagonalization check failed (degenerate modes present: {degenerate})")

    @staticmethod
    def has_degenerate_modes(frequencies: NDArray, atol: float = SMALLEST_ENERGY_SCALE) -> bool:
        return bool(np.any(np.triu(np.isclose(frequencies[:, None], frequencies[None, :], atol=atol), k=1)))

    @staticmethod
    def calculate_zero_point_displacements(frequencies: NDArray, eigenvectors: Matrix,
                                           scales: _DimensionlessScales) -> Matrix:
        """ Zero-point displacement amplitudes (mode participation factors) in meters, trap frame, shape (3, N, 3N):
            result[d, i, n] is the zero-point displacement of ion i along principal axis d in mode n.

            With u_n normalized so that u_n^dagger H u_n = 1, the dimensionful amplitude is
            l0 * sqrt(hbar_ND * w_n) * u_n = sqrt(hbar / (2 m0 w0)) * sqrt(2 w_n) * u_n.

            Real after phase fixing; complex (with a warning) if degenerate modes leave non-negligible imaginary parts.
        """
        num_ions = eigenvectors.shape[0] // 6
        position_part = eigenvectors[:3 * num_ions]
        zero_point_scale = np.sqrt(const.hbar / (2 * scales.mass * scales.frequency))
        displacements = (zero_point_scale * np.sqrt(2 * frequencies) * position_part).reshape(3, num_ions, 3 * num_ions)

        largest = np.max(np.abs(displacements))
        if largest == 0 or np.max(np.abs(displacements.imag)) <= _REAL_RTOL * largest:
            return displacements.real
        warnings.warn("Mode participation factors are complex after phase fixing (degenerate modes present: "
                      f"{_TrappedIonSystem.has_degenerate_modes(frequencies)}). Phases within degenerate subspaces "
                      "are not fixed.")
        return displacements
