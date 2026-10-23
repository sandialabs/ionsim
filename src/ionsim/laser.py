#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import constants as const
from scipy import integrate
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from numpy.typing import NDArray
import sympy 
from sympy.physics.wigner import clebsch_gordan

from ionsim.basis import Basis 
from ionsim.atomic_internal_energy_level import AtomicInternalEnergyLevel, compute_multipole_amplitude, compute_rabi_frequency_between_atomic_levels
from ionsim.degree_of_freedom import DegreeOfFreedom, AtomicStructure
from ionsim.custom_types import Matrix, Vector
from ionsim.config import NUMERICAL_EQUIVALENCE_THRESHOLD, NUMERICAL_ERROR_THRESHOLD, SMALLEST_ENERGY_SCALE 
from ionsim.ionsim_error import IonSimError
from ionsim.operator import Operator, CouplingOperator

## TODO: Add first-order sideband transitions? 

## Vector helper operations
def _unit_vector(vec: Vector) -> Vector:
    """ Returns a real unit vector of an input vector """ 
    v = np.asarray(vec)
    norm = np.linalg.norm(v)
    if norm < 1E-9:
        raise ValueError("Cannot normalize a vector with norm near zero. Computed norm {norm}")
    return v/norm

def _perpendicular_basis(n_hat: Vector, ref_axis: Vector | None=None) -> tuple[Vector, Vector]:
    """ Orthonormal transverse unit vectors (e1, e2) for a direction n_hat, with (e1, e2, n_hat) right-handed (e2 = n_hat x e1).

        - ref_axis is None: the polar-coordinate unit vectors (theta_hat, phi_hat) of n_hat. For n_hat = z this is (x, y),
          so the spherical basis of a z quantization axis is the standard e_{+1} = -(x + iy)/sqrt(2), e_{-1} = (x - iy)/sqrt(2).
          At the poles (n_hat = +/-z), phi = 0 is used.
        - ref_axis given: e1 is the component of ref_axis perpendicular to n_hat, normalized, so e1 points along ref_axis
          as seen in the transverse plane.
    """
    n_hat = _unit_vector(n_hat)
    if ref_axis is None:
        transverse = np.hypot(n_hat[0], n_hat[1])  # sin(theta)
        # Snap to the pole within a tolerance: roundoff in n_hat would otherwise rotate (e1, e2) by an arbitrary angle phi.
        phi = 0. if transverse < NUMERICAL_EQUIVALENCE_THRESHOLD else np.arctan2(n_hat[1], n_hat[0])
        theta = np.arctan2(transverse, n_hat[2])
        e1 = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    else:
        ref_axis = np.asarray(ref_axis, dtype=float)
        e1 = ref_axis - np.dot(ref_axis, n_hat) * n_hat
        if np.linalg.norm(e1) < NUMERICAL_ERROR_THRESHOLD * np.linalg.norm(ref_axis):
            raise IonSimError(f"Reference axis {ref_axis} is (nearly) parallel to the direction {n_hat}; it does not define a transverse axis.")
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n_hat, e1)
    return e1, e2


class BeamProfile(ABC):
    """ Beam profile represented by a spatial envelope as well as power, peak-field relationship"""
    # TODO: Re-think / check naming conventions  
    ## TODO: Should we make power an attribute and make those functions properties? 

    @abstractmethod
    def peak_electric_field_magnitude(self, power: float) -> float:
        """ Peak electirc field atmplitude E0 [V/m] given power in Watts. """ 

    @abstractmethod
    def relative_envelope(self, r: Vector, n_hat: Vector, k: float) -> complex:
        """ Complex relative spatial envelope at position vector r. The exp(i k r) is handled explicitly in the laser class so should not be handled here. """ 


@dataclass(frozen=True, eq=False)
class PlaneWave(BeamProfile):
    """ Plane wave exp(i k r) """
    intensity: float 

    def peak_electric_field_magnitude(self, power: float) -> float:
        """ Peak electric field magnitude |E0| """ 
        return np.sqrt(2. * self.intensity / (const.c * const.epsilon_0))

    def relative_envelope(self, r, n_hat, k) -> complex:
        return 1. + 0j


# TODO: add a beam profile constructor by name, e.g. 'gaussian' 
# Change name to just "Gaussian" (rm "beam")
@dataclass(frozen=True, eq=False)
class GaussianBeam(BeamProfile):
    """ Gaussian beam """  
    waist: float # meters  
    wavelength: float # meters  
    focus: Vector = field(default_factory=lambda: np.zeros(3)) 

    def peak_electric_field_magnitude(self, power: float) -> float:
        """ Peak electric field magnitude |E0| in free space """ 
        I0 = 2. * power / (np.pi * (self.waist ** 2)) # intensity, Watts/m^2 
        return np.sqrt(2. * I0 / (const.c * const.epsilon_0))

    def relative_envelope(self, r: Vector, n_hat: Vector, k: float) -> complex:
        dr = r - self.focus
        z = np.dot(dr, n_hat)

        # What is rho? 
        rho_vector = dr - z*n_hat
        rho = np.linalg.norm(rho_vector)

        n = 1. # refractive index of free space  
        zR = np.pi * (self.waist**2) * n / self.wavelength # Rayleigh range  
        # Beam width w(z) evaluated at beam position z: 
        wz = self.waist * np.sqrt(1. + (z / zR)**2) 
        gouy = np.arctan2(z, zR)
        inv_Rz = z/(z**2 + zR**2)

        amp = (self.waist/wz) * np.exp(-(rho/wz)**2)
        curvature_phase = 0.5 * k * (rho**2) * inv_Rz
        return amp * np.exp(1j * (curvature_phase - gouy)) 


@dataclass(frozen=True, eq=False)
class Laser():
    """ Laser class representing a single monochromatic laser beam  """ 
    wavelength: float
    propagation_vector: Vector
    phase: float
    frequency: float
    polarization: Polarization 
    beam_profile: BeamProfile 
    power: float
    modulation_functions: dict | None=None #e.g. {'phase': Callable, 'amplitude' : Callable, 'frequency' : Callable}
    MODULATION_KEYS = ('amplitude', 'phase', 'frequency')

    def __post_init__(self):
        # Safety checks on propagation vector  
        if hasattr(self.propagation_vector, "__len__"):
            if len(self.propagation_vector) != 3: 
                raise ValueError(f"Specify a 3-component vector for the beam pointing unit vector 'n hat'. Current input has {len(self.propagation_vector)} components.")
        else:
            raise TypeError(f"Propagation vector must be a vector (numpy array), received a {type(self.propagation_vector)}.") 
        assert len(self.propagation_vector) == 3
        
        if np.abs(np.linalg.norm(self.propagation_unit_vector) - 1.) > NUMERICAL_ERROR_THRESHOLD:
            raise IonSimError(f"Propagation unit vector is not normalized! Norm = {np.linalg.norm(self.propagation_unit_vector)}")

        if np.abs(np.dot(self.polarization.vector, self.propagation_unit_vector)) > NUMERICAL_ERROR_THRESHOLD :
            raise ValueError('Laser polarization is not perpendicular to k vector')

        # Check frequency - wavelength relationship 
        # TODO: Check necessary precision for this check to be meaningful 
        # Relative check: optical frequencies are ~1e15 rad/s, so an absolute tolerance would fail on floating-point roundoff.
        light_physics_deviation = np.abs(self.frequency - 2.*np.pi*const.c/self.wavelength) / self.frequency
        if light_physics_deviation > NUMERICAL_EQUIVALENCE_THRESHOLD: 
            raise ValueError(f"Laser frequency and wavelength must satisfy speed of light in vacuum. This is violated with a deviation: {light_physics_deviation}")

    @classmethod
    def from_wavelength(cls, wavelength: float, propagation_vector: Vector, phase: float, polarization: Polarization, beam_profile: Callable,  
                        power: float | None=None, modulation_functions: dict | None=None): 
        """ Constructs laser class from an input wavelength in meters """ 
        frequency = 2 * np.pi * const.c / wavelength  # rad/s 
        return cls(wavelength, propagation_vector, phase, frequency, polarization, beam_profile, power, modulation_functions)


    @classmethod
    def gaussian_from_wavelength(cls, wavelength: float, power: float, waist: float, propagation_vector: Vector, polarization: Polarization, 
                    phase: float, focus: Vector=np.zeros(3), modulation_functions: dict | None=None): 
        profile = GaussianBeam(waist, wavelength, focus)
        return cls.from_wavelength(wavelength, propagation_vector, phase, polarization, profile, power, modulation_functions) 


    @classmethod
    def from_frequency(cls, frequency: float, propagation_vector: Vector, phase: float, polarization: Polarization, beam_profile: Callable,  
                        power: float | None=None, modulation_functions: dict | None=None): 
        """ Constructs laser class from an input frequency in rad/s """ 
        wavelength = 2. * np.pi * const.c / frequency   # meters 
        return cls(wavelength, propagation_vector, phase, frequency, polarization, beam_profile, power, modulation_functions)

    @classmethod
    def gaussian_from_frequency(cls, frequency: float, power: float, waist: float, propagation_vector: Vector, polarization: Polarization, 
                    phase: float, focus: Vector=np.zeros(3), modulation_functions: dict | None=None): 
        wavelength = 2. * np.pi * const.c / frequency   # meters 
        profile = GaussianBeam(waist, wavelength, focus)
        return cls.from_frequency(frequency, propagation_vector, phase, polarization, profile, power, modulation_functions) 

    @property
    def peak_electric_field_magnitude(self) -> float:
        """ Peak electric field magnitude E_0 [V/m] """  
        return self.beam_profile.peak_electric_field_magnitude(self.power) 

    @property
    def peak_intensity(self) -> float:
        """ Peak intensity I_0 [W/m^2] """  
        # Impedence of free space is Z0 = 1/(epsilon0 x speed of light)
        E0 = self.peak_electric_field_magnitude
        return 0.5 * const.c * const.epsilon_0 * (E0**2)

    @property 
    def propagation_unit_vector(self):
        return _unit_vector(self.propagation_vector)

    @property
    def wavevector(self):
        return self.propagation_unit_vector * np.pi * 2. / self.wavelength
        
    ## Helper methods for calculations / AMO simulations  
    def detuning_from_transition_frequency(self, transition_frequency: float) -> float:
        """ Computes detuning defined as laser_frequency - transition_frequency in rad/s of the laser from a transition frequency in rad/s """ 
        return self.frequency - transition_frequency

    def detuning_from_level_transition(self, ground_level: AtomicInternalEnergyLevel, excited_level: AtomicInternalEnergyLevel) -> float:
        """ Computes detuning defined as laser_frequency - transition_frequency in rad/s of the laser from a transition frequency in rad/s """ 
        transition_frequency = excited_level.energy - ground_level.energy
        if transition_frequency <= 0.:
            raise IonSimError(f"Excited level is lower energy than ground level input. Transition frequency is negative or zero: {transition_frequency}")
        return self.detuning_from_transition_frequency(transition_frequency) 
    
    def _modulation_parts(self) -> tuple[Callable | None, Callable | None]:
        """ Validated (envelope, frequency_phase) callables from modulation_functions; either may be None.

            envelope(t) = A(t) exp(i phi(t)) multiplies the |e><g| (absorption) element, like the static laser phase.
            frequency_phase(t) = exp(-i integral_0^t delta(t') dt') multiplies the coupling the same way as the laser
            frequency itself, i.e. as an offset delta(t) added to it.
        """
        if self.modulation_functions is None or all(func is None for func in self.modulation_functions.values()):
            return None, None
        unknown = set(self.modulation_functions) - set(self.MODULATION_KEYS)
        if unknown:
            raise IonSimError(f"Modulation functions must be keyed by {list(self.MODULATION_KEYS)}; got unknown keys {sorted(unknown)}.")
        amplitude_mod = self.modulation_functions.get('amplitude')
        phase_mod = self.modulation_functions.get('phase')
        frequency_mod = self.modulation_functions.get('frequency')
        for key, func in (('amplitude', amplitude_mod), ('phase', phase_mod), ('frequency', frequency_mod)):
            if func is not None and not callable(func):
                raise IonSimError(f"Modulation function '{key}' must be callable, got {type(func).__name__}.")

        envelope = None
        if amplitude_mod is not None or phase_mod is not None:
            def envelope(t: float) -> complex:
                value = 1. + 0j
                if amplitude_mod is not None:
                    value *= amplitude_mod(t)
                if phase_mod is not None:
                    value *= np.exp(1j * phase_mod(t))
                return value

        frequency_phase = None
        if frequency_mod is not None:
            def frequency_phase(t: float) -> complex:
                accumulated_phase, _ = integrate.quad(frequency_mod, 0., t, limit=200)
                return np.exp(-1j * accumulated_phase)

        return envelope, frequency_phase

    @property
    def modulation_function(self) -> Callable | None:
        """ The complex modulation factor f(t) of the |e><g| (absorption) part of a coupling built with the ground level
            first in the basis, or None if unmodulated. Coupling builders use _pair_modulation_function, which handles
            either level order.

            modulation_functions holds optional callables:
                'amplitude': A(t), a real envelope of the Rabi frequency (dimensionless)
                'phase': phi(t), added to the laser phase (rad)
                'frequency': delta(t), added to the laser frequency (rad/s); its phase is integrated numerically
            A constant 'phase' phi is equivalent to adding phi to the laser phase, and a constant 'frequency' delta to
            frequency_shift=delta in the coupling builders.
        """
        return self._pair_modulation_function(ground_index=0, excited_index=1)

    @staticmethod
    def _carrier_sign(ground_index: int, excited_index: int) -> int:
        """ Sign of the oscillation rate for the coupling operator of one (|g>, |e>) pair at these basis indices.

            The Hamiltonian keeps the upper-triangle element of a coupling matrix and multiplies it by
            exp(-i (rate + f_row - f_column) t), where f = rotating_frame_energies. IonSim's interaction picture uses
            f = -E (state energies), which removes H0. In that frame, with omega_0 = E_e - E_g:
                |g> first (|g><e| kept): the physical factor is exp(-i (omega_0 - omega) t) and f_g - f_e = +omega_0,
                                         so rate = -omega.
                |e> first (|e><g| kept): the physical factor is exp(-i (omega - omega_0) t) and f_e - f_g = -omega_0,
                                         so rate = +omega.
            Either way the total rate is +/- the detuning, so a resonant laser is static with H0 removed.
        """
        return -1 if ground_index < excited_index else 1

    def _pair_oscillation_rate(self, ground_index: int, excited_index: int, frequency_shift: float = 0.) -> float:
        """ Oscillation rate for the coupling operator of one (|g>, |e>) pair; see _carrier_sign. """
        return self._carrier_sign(ground_index, excited_index) * (self.frequency + frequency_shift)

    def _pair_modulation_function(self, ground_index: int, excited_index: int) -> Callable | None:
        """ Modulation function for the coupling operator of one (|g>, |e>) pair at these basis indices.

            The Hamiltonian multiplies the upper-triangle element of a coupling matrix by the modulation function and adds
            the Hermitian conjugate. The laser phase sits on |e><g|, so the envelope A(t) exp(i phi(t)) is conjugated when
            |g><e| is the upper-triangle element (ground_index < excited_index). The frequency part follows the sign of the
            laser frequency on that element (_carrier_sign): exp(-i int delta) on |e><g| (rate +omega) and its conjugate on
            |g><e| (rate -omega), so a constant delta matches frequency_shift=delta for either level order.
        """
        envelope, frequency_phase = self._modulation_parts()
        if envelope is None and frequency_phase is None:
            return None
        conjugate_envelope = ground_index < excited_index
        conjugate_frequency = self._carrier_sign(ground_index, excited_index) < 0

        def mod_function(t: float) -> complex:
            value = 1. + 0j
            if envelope is not None:
                value *= np.conj(envelope(t)) if conjugate_envelope else envelope(t)
            if frequency_phase is not None:
                value *= np.conj(frequency_phase(t)) if conjugate_frequency else frequency_phase(t)
            return value

        return mod_function

    # Method for individually-addressing laser  
    def build_individual_atom_laser_coupling_operators(self, basis: Basis, addressed_atom: AtomicStructure, ground_levels: list[AtomicInternalEnergyLevel], 
                                                        excited_levels: list[AtomicInternalEnergyLevel], multipole_order: int, frequency_shift: float=0.) -> list[CouplingOperator]: 
        """ Builds a list of coupling operators corresponding to |g> <-> |e> couplings from laser light upon a particular atom.

            Levels are matched to the atom's own levels by name, so levels taken from another (identical) atom also work.
        """
        if not any(addressed_atom is dof for dof in basis.atomic_structure_DOFs):
            raise ValueError(f"Addressed atom {addressed_atom} has not been included in the basis degrees of freedom.") 
        return self._atom_coupling_operators(basis, addressed_atom, ground_levels, excited_levels, multipole_order, frequency_shift)

    def build_laser_coupling_operators_multiple_atoms(self, basis: Basis, atom_DOFs: list[AtomicStructure] | None, ground_levels: list[AtomicInternalEnergyLevel], 
                                        excited_levels: list[AtomicInternalEnergyLevel], multipole_order: int, frequency_shift: float=0.) -> list[Operator]: 
        """ Builds light-atom coupling operators for several atoms in the basis, for the requested ground and excited levels.

            atom_DOFs: the AtomicStructure DOFs the laser addresses; all AtomicStructure DOFs of the basis if None.
            ground_levels, excited_levels: matched to each atom's own levels by name, so they may come from any one of the
                atoms. Each atom must contain all of them. Each atom's couplings are computed from its own levels.

            The laser couples every addressed atom with the same field (its peak electric field and phase).
        """ 
        if atom_DOFs is None:
            atom_DOFs = basis.atomic_structure_DOFs
        if not atom_DOFs:
            raise ValueError("No atoms to address: the basis has no AtomicStructure degrees of freedom.")
        for atom in atom_DOFs:
            if not any(atom is dof for dof in basis.atomic_structure_DOFs):
                raise ValueError(f"Atom {atom.name or atom} has not been included in the basis degrees of freedom.")

        coupling_operators = []
        for atom in atom_DOFs:
            coupling_operators.extend(self._atom_coupling_operators(basis, atom, ground_levels, excited_levels, multipole_order, frequency_shift))
        return coupling_operators 

    def _atom_coupling_operators(self, basis: Basis, atom: AtomicStructure, ground_levels: list[AtomicInternalEnergyLevel],
                                 excited_levels: list[AtomicInternalEnergyLevel], multipole_order: int, frequency_shift: float) -> list[CouplingOperator]:
        """ Coupling operators for every requested (|g>, |e>) pair of one atom, built from that atom's own level objects. """
        atomic_levels = atom.energy_levels
        level_index = {level.name: index for index, level in enumerate(atomic_levels)}
        missing = [level.name for level in [*ground_levels, *excited_levels] if level.name not in level_index]
        if missing:
            raise ValueError(f"Levels {missing} not found in the atomic structure {atom.name or ''} with levels {list(level_index)}.")

        coupling_operators = []
        # Build coupling operator for each |g>, |e> pairing   
        for ground_name in [level.name for level in ground_levels]:
            for excited_name in [level.name for level in excited_levels]:
                if ground_name == excited_name:
                    raise ValueError(f"Excited level {excited_name} matches the ground level {ground_name}.")
                ground_index, excited_index = level_index[ground_name], level_index[excited_name]
                ground_level, excited_level = atomic_levels[ground_index], atomic_levels[excited_index]

                coupling_matrix = self.build_atom_laser_ge_coupling_matrix(ground_level, excited_level, atomic_levels, multipole_order) 
                # Pass this loop iteration if there are no non-zero couplings for the requested transition  
                if np.count_nonzero(coupling_matrix) == 0 or np.all(np.abs(coupling_matrix) < SMALLEST_ENERGY_SCALE):
                    continue 

                enlarged_matrix = basis.enlarge_matrix(coupling_matrix, [atom]) 
                mod_function = self._pair_modulation_function(ground_index, excited_index)
                rate = self._pair_oscillation_rate(ground_index, excited_index, frequency_shift)
                coupling_operators.append(CouplingOperator.from_matrix(basis, enlarged_matrix, rate, modulation_function=mod_function))
        return coupling_operators

    def build_atom_laser_ge_coupling_matrix(self, ground_level: AtomicInternalEnergyLevel, excited_level: AtomicInternalEnergyLevel, 
                                            atomic_levels: list[AtomicInternalEnergyLevel], multipole_order: int) -> Matrix: 
        """ Builds a coupling matrix representing a laser light-atom coupling """ 

        if multipole_order == 1:
            rabi_frequency = compute_rabi_frequency_between_atomic_levels(ground_level, excited_level, self.polarization.dipole_components(), 
                                                                        atomic_levels, multipole_order, self.peak_electric_field_magnitude) 
        elif multipole_order == 2:
            rabi_frequency = compute_rabi_frequency_between_atomic_levels(ground_level, excited_level, self.polarization.quadrupole_components(), 
                                                                        atomic_levels, multipole_order, self.peak_electric_field_magnitude,
                                                                        wavenumber=np.linalg.norm(self.wavevector)) 
        else:
            raise IonSimError(f"Multipole order must be 1 or 2 for dipole or quadrupole transitions, respectively.")

        # Build coupling operator matrix: 
        single_atom_matrix_size = len(atomic_levels)
        coupling_matrix = np.zeros((single_atom_matrix_size,single_atom_matrix_size), dtype=complex) 

        if np.abs(rabi_frequency) < SMALLEST_ENERGY_SCALE: 
            return coupling_matrix
        
        ground_index = atomic_levels.index(ground_level) 
        excited_index = atomic_levels.index(excited_level) 
        
        if ground_index == excited_index:
            raise ValueError(f"Ground level and excited level should not be the same.")
    
        coupling_matrix[excited_index, ground_index] = 0.5 * rabi_frequency * np.exp(1j*self.phase) 
        coupling_matrix[ground_index, excited_index] = np.conj(coupling_matrix[excited_index,ground_index]) 
        return coupling_matrix


## Helpers for polarization:
def _spherical_basis_vectors(quantization_axis: Vector) -> dict[int, Vector]:
    """ Spherical basis vectors eps +1, 0, -1. See eq. 7.188 from "Quantum and Atom Optics by Daniel Steck""" 
    z = _unit_vector(quantization_axis)
    x, y = _perpendicular_basis(z)
    return {1: -(x + 1j*y)/np.sqrt(2.), 0: z.astype(complex), -1 : (x - 1j*y)/np.sqrt(2.)}

def _project_spherical(v: Vector, quantization_axis: Vector) -> dict[int, complex]:
    """ Spherical components v_+1, v_0, v_-1 of a lab-frame vector "v", relative to the 
        quantization axis."""
    v = np.asarray(v, dtype=complex)
    sph_basis = _spherical_basis_vectors(quantization_axis) 
    # np.vdot(a,b) = sum a*_i b_i 
    # For components via spherical basis dot product, the e_q vectors need conjugation, therefore put in first arg: 
    return {q: np.vdot(sph_basis[q], v) for q in (1, 0, -1)} 


@dataclass(frozen=True, eq=False)
class Polarization:
    """ Complex Cartesian polarization (Jones) Vector in the lab frame. This is set to be perpendicular to a reference propagation direction (e.g. of a laser) """
    # TODO: For naming, Should we use "components" or "vector" to refer to the polarization vector? 
    vector: Vector # input polarization vector; normalized in post init 
    n_hat: Vector  # propagation direction 

    def __post_init__(self):
        """ Check that polarization vector is perpendicular to propagation vector """  
        normalized_n_hat = _unit_vector(self.n_hat) 

        # Check that polarization components (vector) are orthogonal to propagation vector 
        projection = np.dot(self.vector, normalized_n_hat)
        if np.abs(projection) > 1E-9: 
            raise ValueError(f"Polarization vector is not perpendicular to the reference propagation direction. Dot product = {projection}, should be zero.")

        # Normalize the vector if is not normalized 
        norm = np.sqrt(np.vdot(self.vector, self.vector).real)
        # Check normalization is near 1 
        if norm < 1E-9: 
            raise ValueError("Cannot normalize a vector with norm near zero. Computed norm {norm}")

        if np.abs(norm - 1.) > NUMERICAL_EQUIVALENCE_THRESHOLD: 
            normalized_vector = self.vector / norm
            object.__setattr__(self, "vector", normalized_vector)

    def __repr__(self):
        return f"Polarization(vector={self.vector}, n_hat={self.n_hat})" 

    @classmethod
    def linear(cls, propagation_direction: Vector, angle: float = 0., ref_axis: Vector | None=None):
        """ Linear polarization at an angle (radians) from a reference direction in the perpendicular plane. """ 
        e1, e2 = _perpendicular_basis(propagation_direction, ref_axis)
        polarization_vector = np.cos(angle) * e1 + np.sin(angle) * e2
        return cls(polarization_vector, propagation_direction)

    @classmethod
    def circular(cls, propagation_direction: Vector, handedness: str, ref_axis: Vector | None=None):
        """ Circular polarization built in the (e1, e2) plane perpendicular to the laser propagation direction (n hat)

            - handedness is specified by '+' or '-'
            - eps_{+/-} = (e1 +/- ie2)/sqrt(2) 
            - '+' <==> left-circular polarization; '-' <==> right-circular polarization
            - looks at wave from behind. 

            NOTE: Corresponding with atomic raising/lowering angular momentum operators depends on quantization axis. 

        """
        e1, e2 = _perpendicular_basis(propagation_direction, ref_axis)
        if handedness == '+':
            polarization_vector = (e1 + 1j * e2)/np.sqrt(2.)
        elif handedness == '-':
            polarization_vector = (e1 - 1j * e2)/np.sqrt(2.)
        else:
            raise IonSimError("Handedness must be specified either by a string, e.g. '+' or '-'.")

        return cls(polarization_vector, propagation_direction)

    @classmethod
    def from_spherical(cls, epsilon_vector: Vector, propagation_direction: Vector, 
                            quantization_axis: Vector = np.array([0., 0., 1.])):
        """ Builds a Cartesian polarization vector from the specified spherical components (eps_+, eps_0, eps_-), which are 
            relative to the specified quantization axis (defaulting to (0, 0, 1)). 

            epsilon vector specified with 3 componenets: eps_+1, eps_0, eps_-1, 

        """
        # should the input vector be in a dictionary form? {+1: value, 0: value, -1: value}?
        e_q = _spherical_basis_vectors(quantization_axis)
        e_p1 = e_q[1]
        e_0 = e_q[0]
        e_m1 = e_q[-1]

        polarization_vector = e_p1 * epsilon_vector[0] + e_0 * epsilon_vector[1] + e_m1 * epsilon_vector[2] 
        return cls(polarization_vector, propagation_direction)

    def dipole_components(self, quantization_axis: Vector = np.array([0., 0., 1.])) -> dict[int, complex]:
        """ Spherical (dipole) components eps_q, keyed by q = +1, 0, -1, with polarization vector = sum_q eps_q e_q. """
        e_q = _spherical_basis_vectors(quantization_axis)
        eps_p1 = np.vdot(e_q[1], self.vector)
        eps_0 = np.vdot(e_q[0], self.vector)
        eps_m1 = np.vdot(e_q[-1], self.vector)
        components = np.array([eps_p1, eps_0, eps_m1])
        return dict(zip((1, 0, -1), components)) 

    @staticmethod
    def rank2_spherical_basis(quantization_axis: Vector = np.array([0., 0., 1.])) -> dict[int, Matrix]:
        """ Orthonormal rank-2 spherical basis tensors E_q = sum_{m1, m2} <1 m1; 1 m2 | 2 q> e_m1 (x) e_m2, q = -2..2.

            Built from the rank-1 spherical basis vectors e_m (Steck, eq. 7.188); orthonormal in the sense
            sum_ij conj(E_q)_ij (E_q')_ij = delta_qq'.
        """
        e = _spherical_basis_vectors(quantization_axis)
        tensors = {}
        for q in range(-2, 3):
            T = np.zeros((3, 3), dtype=complex)
            for m1 in (-1, 0, 1):
                m2 = q - m1
                if abs(m2) <= 1:
                    T += float(clebsch_gordan(1, 1, 2, m1, m2, q)) * np.outer(e[m1], e[m2])
            tensors[q] = T
        return tensors

    def quadrupole_components(self, quantization_axis: Vector = np.array([0., 0., 1.])) -> dict[int, complex]:
        """ Rank-2 polarization components C_q, q = -2..2, of the E2 geometry tensor eps (x) n_hat.

            C_q = sum_ij conj(E_q)_ij eps_i n_j = sum_{m1, m2} <1 m1; 1 m2 | 2 q> eps_m1 n_m2, where eps_m and n_m are
            the spherical components of the polarization and propagation unit vectors. For an E2 coupling these play
            the role eps_q plays for E1: sum_ij eps_i n_j Q_ij = sum_q C_q Q_q for a symmetric traceless tensor Q.
        """
        n_hat = _unit_vector(self.n_hat)
        tensors = self.rank2_spherical_basis(quantization_axis)
        return {q: complex(np.einsum('ij,i,j->', np.conj(tensors[q]), self.vector, n_hat)) for q in range(-2, 3)}
