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
import warnings

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
from ionsim.degree_of_freedom import DegreeOfFreedom, AtomicStructure, MotionalMode
from ionsim.custom_types import Matrix, Vector
from ionsim.config import NUMERICAL_EQUIVALENCE_THRESHOLD, NUMERICAL_ERROR_THRESHOLD, SMALLEST_ENERGY_SCALE
from ionsim.ionsim_error import IonSimError
from ionsim.operator import Operator, CouplingOperator
from ionsim.named_operators import Fock

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

    @classmethod
    def from_peak_electric_field_magnitude(cls, peak_electric_field_magnitude: float) -> PlaneWave:
        """ Plane wave with peak field |E0| [V/m]: intensity I = c eps0 |E0|^2 / 2 (inverse of peak_electric_field_magnitude). """
        return cls(intensity=0.5 * const.c * const.epsilon_0 * peak_electric_field_magnitude**2)

    def peak_electric_field_magnitude(self, power: float) -> float:
        """ Peak electric field magnitude |E0| """
        return np.sqrt(2. * self.intensity / (const.c * const.epsilon_0))

    def relative_envelope(self, r, n_hat, k) -> complex:
        return 1. + 0j


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

    def power_for_peak_electric_field_magnitude(self, peak_electric_field_magnitude: float) -> float:
        """ Power [W] that gives peak field |E0| [V/m] at the focus (inverse of peak_electric_field_magnitude):
            I0 = c eps0 |E0|^2 / 2,  P = pi w^2 I0 / 2. """
        I0 = 0.5 * const.c * const.epsilon_0 * peak_electric_field_magnitude**2
        return 0.5 * np.pi * self.waist**2 * I0

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

    # ---- constructors from a target Rabi frequency ----
    # The field is chosen so that |rabi_frequency(ground_level, excited_level, multipole_order)| equals the target at the
    # peak field (beam focus) with modulation amplitude 1. Couplings are computed in units where the orbital reduced
    # matrix element <L||r^k||L'> = 1 a0^k, so the derived intensity or power is an effective (calibration) value that
    # absorbs the radial matrix element of this transition. Every other transition of the same L -> L' multiplet (other
    # sublevels, hyperfine and fine-structure components) then follows correctly from the angular factors; transitions
    # with a different radial matrix element (another multiplet) do not. Do not use the derived intensity or power for
    # quantities that need the physical intensity.

    @classmethod
    def from_target_rabi_frequency(cls, target_rabi_frequency: float, ground_level: AtomicInternalEnergyLevel,
                                   excited_level: AtomicInternalEnergyLevel, multipole_order: int, propagation_vector: Vector,
                                   polarization: Polarization, detuning: float = 0., phase: float = 0.,
                                   modulation_functions: dict | None = None) -> Laser:
        """ Plane-wave laser whose Rabi frequency on |ground_level> -> |excited_level> has magnitude target_rabi_frequency.

            target_rabi_frequency: rad/s, positive number.  multipole_order: 1 (E1) or 2 (E2).
            detuning: rad/s; the laser frequency is (E_excited - E_ground) + detuning.
            The beam profile is a PlaneWave with the effective intensity that gives this Rabi frequency (see above).
        """
        frequency, peak_field = cls._frequency_and_peak_field_for_target_rabi_frequency(
            target_rabi_frequency, ground_level, excited_level, multipole_order, propagation_vector, polarization, detuning)
        profile = PlaneWave.from_peak_electric_field_magnitude(peak_field)
        return cls.from_frequency(frequency, propagation_vector, phase, polarization, profile, None, modulation_functions)

    @classmethod
    def gaussian_from_target_rabi_frequency(cls, target_rabi_frequency: float, ground_level: AtomicInternalEnergyLevel,
                                            excited_level: AtomicInternalEnergyLevel, multipole_order: int, waist: float,
                                            propagation_vector: Vector, polarization: Polarization, detuning: float = 0.,
                                            phase: float = 0., focus: Vector = np.zeros(3),
                                            modulation_functions: dict | None = None) -> Laser:
        """ Gaussian laser whose peak Rabi frequency on |ground_level> -> |excited_level> has magnitude target_rabi_frequency.

            Arguments as for from_target_rabi_frequency, plus waist [m] and focus [m]. The power is the effective power
            that gives this Rabi frequency at the focus (see above).
        """
        frequency, peak_field = cls._frequency_and_peak_field_for_target_rabi_frequency(
            target_rabi_frequency, ground_level, excited_level, multipole_order, propagation_vector, polarization, detuning)
        wavelength = 2. * np.pi * const.c / frequency
        profile = GaussianBeam(waist, wavelength, focus)
        power = profile.power_for_peak_electric_field_magnitude(peak_field)
        return cls.from_frequency(frequency, propagation_vector, phase, polarization, profile, power, modulation_functions)

    @classmethod
    def _frequency_and_peak_field_for_target_rabi_frequency(cls, target_rabi_frequency: float, ground_level: AtomicInternalEnergyLevel,
                                                            excited_level: AtomicInternalEnergyLevel, multipole_order: int,
                                                            propagation_vector: Vector, polarization: Polarization,
                                                            detuning: float) -> tuple[float, float]:
        """ Laser frequency (rad/s) and peak field |E0| (V/m, effective) giving |Omega| = target_rabi_frequency.

            Omega is linear in |E0|, so |E0| = target / |Omega(E0 = 1 V/m)|, evaluated with this frequency and
            polarization (the E2 coupling also depends on |k|).
        """
        target = np.asarray(target_rabi_frequency)
        if target.ndim != 0 or np.iscomplexobj(target) or not np.isfinite(target) or target <= 0.:
            raise IonSimError(f"target_rabi_frequency must be a positive real number (rad/s), got {target_rabi_frequency}.")
        transition_frequency = excited_level.energy - ground_level.energy
        if transition_frequency <= 0.:
            raise IonSimError(f"Excited level {excited_level.name} must be above ground level {ground_level.name}; "
                              f"transition frequency is {transition_frequency} rad/s.")
        frequency = transition_frequency + detuning
        if frequency <= 0.:
            raise IonSimError(f"Laser frequency (transition frequency + detuning) must be positive, got {frequency} rad/s.")
        unit_field_laser = cls.from_frequency(frequency, propagation_vector, 0., polarization,
                                              PlaneWave.from_peak_electric_field_magnitude(1.), None)
        rabi_per_unit_field = abs(unit_field_laser.rabi_frequency(ground_level, excited_level, multipole_order))
        if rabi_per_unit_field < SMALLEST_ENERGY_SCALE:
            components = polarization.dipole_components() if multipole_order == 1 else polarization.quadrupole_components()
            raise IonSimError(f"This polarization does not drive {ground_level.name} -> {excited_level.name} "
                              f"(multipole order {multipole_order}); polarization components: {components}.")
        return frequency, float(target) / rabi_per_unit_field

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

    def rabi_frequency(self, ground_level: AtomicInternalEnergyLevel, excited_level: AtomicInternalEnergyLevel,
                       multipole_order: int) -> complex:
        """ Rabi frequency Omega (rad/s) of |ground_level> -> |excited_level> driven by this laser at its peak field (beam
            focus) with modulation amplitude 1, so that the coupling's absorption element is (Omega/2) e^{i phase} |e><g|.

            |Omega| is the Rabi rate. Its complex phase comes from the polarization and angular factors; the laser's static
            phase is not included. multipole_order: 1 (E1) or 2 (E2, which also depends on |k|). Units as for the coupling
            builders: the orbital reduced matrix element <L||r^k||L'> is taken as 1 a0^k.
        """
        if ground_level is excited_level or ground_level.name == excited_level.name:
            raise ValueError(f"Ground level and excited level should not be the same, got {ground_level.name}.")
        levels = [ground_level, excited_level]
        if multipole_order == 1:
            return compute_rabi_frequency_between_atomic_levels(ground_level, excited_level, self.polarization.dipole_components(),
                                                                levels, multipole_order, self.peak_electric_field_magnitude)
        if multipole_order == 2:
            return compute_rabi_frequency_between_atomic_levels(ground_level, excited_level, self.polarization.quadrupole_components(),
                                                                levels, multipole_order, self.peak_electric_field_magnitude,
                                                                wavenumber=np.linalg.norm(self.wavevector))
        raise IonSimError(f"Multipole order must be 1 or 2 for dipole or quadrupole transitions, respectively. Got {multipole_order}.")
    
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
            first in the basis, or None if unmodulated. Coupling builders use _modulation_function_for_level_pair, which handles
            either level order.

            modulation_functions holds optional callables:
                'amplitude': A(t), a real envelope of the Rabi frequency (dimensionless)
                'phase': phi(t), added to the laser phase (rad)
                'frequency': delta(t), added to the laser frequency (rad/s); its phase is integrated numerically
            A constant 'phase' phi is equivalent to adding phi to the laser phase, and a constant 'frequency' delta to
            frequency_shift=delta in the coupling builders.
        """
        return self._modulation_function_for_level_pair(ground_index=0, excited_index=1)

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

    def _oscillation_rate_for_level_pair(self, ground_index: int, excited_index: int, frequency_shift: float = 0.) -> float:
        """ Oscillation rate for the coupling operator of one (|g>, |e>) pair; see _carrier_sign. """
        return self._carrier_sign(ground_index, excited_index) * (self.frequency + frequency_shift)

    def _modulation_function_for_level_pair(self, ground_index: int, excited_index: int) -> Callable | None:
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

    # ---------------------------------------------------------------------------------------------------------------
    # Coupling-operator builders
    #
    # Light-atom interaction for atom j at equilibrium position 0 (Lamb-Dicke expansion of the field's spatial phase):
    #     H_j = (Omega/2) e^{i phi} |e><g| exp(i sum_k eta_jk (a_k + a_k^dag)) e^{-i omega t} + h.c.
    #         ~ (Omega/2) e^{i phi} |e><g| [ 1  +  i sum_k eta_jk a_k  +  i sum_k eta_jk a_k^dag ] e^{-i omega t} + h.c.
    #                                       carrier   red sideband (n -> n-1)   blue sideband (n -> n+1)
    # Every term oscillates at the same laser frequency omega. In the interaction picture
    # (rotating_frame_energies = -E of each basis state, motional energies included) the frame supplies the transition
    # frequency of each element, omega_0 for the carrier and omega_0 -/+ omega_k for the red/blue sideband, so all three
    # builders share the same oscillation-rate and modulation conventions (_carrier_sign, _modulation_function_for_level_pair).
    # ---------------------------------------------------------------------------------------------------------------
    SIDEBANDS = ('red', 'blue')

    def build_carrier_coupling_operators(self, basis: Basis, target_atoms: list[AtomicStructure] | None, ground_levels: list[AtomicInternalEnergyLevel], 
                                               excited_levels: list[AtomicInternalEnergyLevel], multipole_order: int, frequency_shift: float = 0.) -> list[CouplingOperator]:
        """ Carrier couplings |g, n> <-> |e, n> (motional state unchanged) for the requested ground and excited levels.

            target_atoms: addressed AtomicStructure DOFs; all of the basis if None. Levels are matched by name (as for the carrier).
                Pass a single atom, e.g. [atom], for an individually addressing beam.
            ground_levels, excited_levels: matched to each atom's own levels by name, so they may come from any one of the
                atoms. Each atom must contain all of them. Each atom's couplings are computed from its own levels.
            frequency_shift: rad/s added to the laser frequency (e.g. a constant frequency shift from an AOM).

            The laser couples every addressed atom with the same field (its peak electric field and phase). Motional modes in
            the basis are spectators (identity); the carrier's Debye-Waller reduction exp(-eta^2/2) L_n(eta^2) is not included.
        """
        target_atoms = self._parse_atomic_input(basis, target_atoms)
        coupling_operators = []
        for atom in target_atoms:
            coupling_operators.extend(self._atom_coupling_operators(basis, atom, ground_levels, excited_levels, multipole_order, frequency_shift))
        return coupling_operators

    def build_red_sideband_coupling_operators(self, basis: Basis, target_atoms: list[AtomicStructure] | None, target_modes: list[MotionalMode],
                                              ground_levels: list[AtomicInternalEnergyLevel], excited_levels: list[AtomicInternalEnergyLevel],
                                              multipole_order: int, lamb_dicke_parameters: float | dict, frequency_shift: float = 0.,
                                              debye_waller: bool = False) -> list[CouplingOperator]:
        """ First red-sideband couplings |g, n> <-> |e, n-1> (absorption removes one phonon); resonant at
            omega = omega_0 - omega_mode. See build_sideband_coupling_operators for the arguments. """
        return self.build_sideband_coupling_operators('red', basis, target_atoms, target_modes, ground_levels, excited_levels, multipole_order,
                                                      lamb_dicke_parameters, frequency_shift, debye_waller)

    def build_blue_sideband_coupling_operators(self, basis: Basis, target_atoms: list[AtomicStructure] | None, target_modes: list[MotionalMode],
                                               ground_levels: list[AtomicInternalEnergyLevel], excited_levels: list[AtomicInternalEnergyLevel],
                                               multipole_order: int, lamb_dicke_parameters: float | dict, frequency_shift: float = 0.,
                                               debye_waller: bool = False) -> list[CouplingOperator]:
        """ First blue-sideband couplings |g, n> <-> |e, n+1> (absorption adds one phonon); resonant at
            omega = omega_0 + omega_mode. See build_sideband_coupling_operators for the arguments. """
        return self.build_sideband_coupling_operators('blue', basis, target_atoms, target_modes, ground_levels, excited_levels, multipole_order,
                                                      lamb_dicke_parameters, frequency_shift, debye_waller)

    def build_sideband_coupling_operators(self, sideband: str, basis: Basis, target_atoms: list[AtomicStructure] | None, target_modes: list[MotionalMode],
                                          ground_levels: list[AtomicInternalEnergyLevel], excited_levels: list[AtomicInternalEnergyLevel], multipole_order: int, 
                                          lamb_dicke_parameters: float | dict, frequency_shift: float = 0., debye_waller: bool = False) -> list[CouplingOperator]:
        """ First red ('red') or blue ('blue') sideband couplings of the requested levels with each requested motional mode.

            The absorption element of atom j and mode k is (Omega/2) e^{i phi} (i eta_jk) |e><g| (x) a_k (red) or a_k^dag
            (blue), where Omega e^{i phi} is the carrier Rabi frequency of the same transition (E1 or E2). 

            target_atoms: addressed AtomicStructure DOFs; all of the basis if None. Levels are matched by name (as for the carrier).
            lamb_dicke_parameters: signed eta_jk, either one float for every (atom, mode) pair or a dict keyed by
                (atom, mode) DOF objects. Every requested pair must have an entry; an entry of 0 skips that pair. The sign
                (the atom's normal-mode amplitude relative to k) sets the relative phase of different atoms' sideband
                couplings, e.g. for stretch modes. See lamb_dicke_parameter() to compute eta from a mode vector.
            target_modes: the MotionalMode DOFs to couple (required, non-empty). Each must be in the basis; a sideband
                operator is built for every (target atom, target mode) pair.
            frequency_shift: rad/s added to the laser frequency, as for the carrier.
            debye_waller: if True, use the exact Delta n = +/-1 matrix elements of exp(i eta (a + a^dag)) for this mode,
                eta e^{-eta^2/2} L_n^1(eta^2) / sqrt(n+1) instead of eta sqrt(n+1) (Fock.debye_waller_lowering).

            The basis may order atoms and modes in any way: the kept (upper-triangle) element of each coupling and its
            oscillation-rate sign are determined from the basis-state indices (_coupling_operator_from_raising_matrix).
        """
        if sideband not in self.SIDEBANDS:
            raise IonSimError(f"sideband must be one of {self.SIDEBANDS}, got '{sideband}'.")
        target_atoms = self._parse_atomic_input(basis, target_atoms)
        target_modes = self._parse_motional_mode_input(basis, target_modes)
        etas = self._lamb_dicke_table(lamb_dicke_parameters, target_atoms, target_modes)

        coupling_operators = []
        for atom in target_atoms:
            for mode in target_modes:
                eta = etas[(atom, mode)]
                if np.abs(eta) < NUMERICAL_EQUIVALENCE_THRESHOLD:
                    continue
                motional_matrix = basis.enlarge_matrix(self._sideband_motional_matrix(sideband, mode, eta, debye_waller), [mode])
                coupling_operators.extend(self._atom_coupling_operators(basis, atom, ground_levels, excited_levels, multipole_order,
                                                                        frequency_shift, motional_matrix))
        return coupling_operators

    # TODO: replace with functionality from trap 
    def lamb_dicke_parameter(self, mode_frequency: float, atomic_mass: float, mode_participation_vector: Vector) -> float:
        """ Signed Lamb-Dicke parameter eta = (k . b) sqrt(hbar / (2 m omega_mode)) of this laser for one atom and one mode.

            mode_frequency: rad/s.  
            atomic_mass: daltons 
            mode_participation_vector: the atom's 3-component normal-mode participation vector b (the atom's entries of the normalized
                mode eigenvector, e.g. z_hat / sqrt(N) for the axial center-of-mass mode of N identical ions).
            Uses this laser's wavevector, i.e. a single-photon transition. For a two-photon (Raman) transition the
            relevant wavevector is the difference k1 - k2 of the two beams, which a single Laser does not describe.
        """
        mode_participation_vector = np.asarray(mode_participation_vector, dtype=float)
        if mode_participation_vector.shape != (3,):
            raise IonSimError(f"mode_participation_vector must have 3 components, got shape {mode_participation_vector.shape}.")
        if mode_frequency <= 0. or atomic_mass <= 0.:
            raise IonSimError("mode_frequency and atomic_mass must be positive.")
        mass = atomic_mass * const.value('atomic mass constant')
        return float(np.dot(self.wavevector, mode_participation_vector) * np.sqrt(const.hbar / (2. * mass * mode_frequency)))

    # ---- helpers for the builders ----
    @staticmethod
    def _parse_atomic_input(basis: Basis, target_atoms: list[AtomicStructure] | None) -> list[AtomicStructure]:
        if target_atoms is None:
            target_atoms = basis.atomic_structure_DOFs
        if not target_atoms:
            raise ValueError("No atoms to address: the basis has no AtomicStructure degrees of freedom.")
        for atom in target_atoms:
            if not any(atom is dof for dof in basis.atomic_structure_DOFs):
                raise ValueError(f"Atom {atom.name or atom} has not been included in the basis degrees of freedom.")
        return list(target_atoms)

    @staticmethod
    def _parse_motional_mode_input(basis: Basis, target_modes: list[MotionalMode]) -> list[MotionalMode]:
        """ Parses the target mode list input: a non-empty list of distinct MotionalMode DOFs of the basis,
            each with Fock levels 0, 1, ..., N-1 in order. """
        if isinstance(target_modes, MotionalMode) or not isinstance(target_modes, (list, tuple)):
            raise IonSimError(f"target_modes must be a list of MotionalMode DOFs, got {type(target_modes).__name__}; "
                              f"wrap a single mode in a list, e.g. [mode].")
        if not target_modes:
            raise IonSimError("target_modes is empty: specify the motional modes whose sideband couplings to build.")
        not_modes = [type(mode).__name__ for mode in target_modes if not isinstance(mode, MotionalMode)]
        if not_modes:
            raise IonSimError(f"target_modes must contain only MotionalMode DOFs, got {not_modes}.")
        if len({id(mode) for mode in target_modes}) != len(target_modes):
            raise IonSimError("target_modes contains the same mode more than once, which would double its sideband couplings.")
        modes_from_basis = [] if not basis.motional_basis else basis.motional_basis.degrees_of_freedom
        #basis_modes = [dof for dof in basis.degrees_of_freedom if isinstance(dof, MotionalMode)]
        for mode in target_modes:
            if not any(mode is dof for dof in modes_from_basis):
                raise ValueError(f"Motional mode {mode.name or mode} has not been included in the basis degrees of freedom.")
            fock_numbers = [level.fock_number for level in mode.energy_levels]
            if fock_numbers != list(range(len(fock_numbers))):
                raise IonSimError(f"Sideband operators need the mode's levels to be Fock states 0, 1, ..., N-1 in order; got {fock_numbers}.")
        return list(target_modes)

    @staticmethod
    def _lamb_dicke_table(lamb_dicke_parameters: float | dict, atom_dofs: list[AtomicStructure], mode_dofs: list[MotionalMode]) -> dict:
        """ {(atom, mode): eta} for every requested pair, from one float or a dict keyed by (atom, mode) DOF objects. """
        if isinstance(lamb_dicke_parameters, dict):
            # DOFs are frozen dataclasses with eq=False, so dict keys compare by identity.
            missing = [(atom.name or str(i), mode.name or str(k)) for i, atom in enumerate(atom_dofs) for k, mode in enumerate(mode_dofs)
                       if (atom, mode) not in lamb_dicke_parameters]
            if missing:
                raise IonSimError(f"lamb_dicke_parameters has no entry for (atom, mode) pairs {missing}; give every pair "
                                  f"(use 0. to skip a pair).")
            return {(atom, mode): float(lamb_dicke_parameters[(atom, mode)]) for atom in atom_dofs for mode in mode_dofs}
        if np.ndim(lamb_dicke_parameters) != 0 or np.iscomplexobj(lamb_dicke_parameters):
            raise IonSimError("lamb_dicke_parameters must be a real float or a dict keyed by (atom, mode).")
        return {(atom, mode): float(lamb_dicke_parameters) for atom in atom_dofs for mode in mode_dofs}

    @staticmethod
    def _sideband_motional_matrix(sideband: str, mode: MotionalMode, eta: float, debye_waller: bool) -> Matrix:
        """ i eta a (red) or i eta a^dag (blue) on one mode in its isolated Hilbert space """
        dimension = len(mode.energy_levels)
        lowering = Fock.debye_waller_lowering(dimension, eta) if debye_waller else Fock.lowering(dimension)
        motional = lowering if sideband == 'red' else lowering.conj().T
        return 1j * eta * motional

    def _coupling_operator_from_raising_matrix(self, basis: Basis, raising_matrix: Matrix, frequency_shift: float) -> CouplingOperator | None:
        """ The coupling operator for the raising matrix (<excited state| . |ground state>, laser phase included) of one
            light-atom coupling in the full basis, or None if every element is below SMALLEST_ENERGY_SCALE. """
        # Note on conventions with sidebands: The Hamiltonian keeps the upper-triangle element of each Hermitian pair: raising elements below the diagonal have the
            # ground state first in the basis and those above it the excited state first, which need opposite oscillation-rate
            # signs and envelope conjugation (_carrier_sign). For one carrier or one first-sideband coupling every element is on
            # the same side, because the paired states differ only in the target atom (and the target mode for a sideband):
            # carrier, or sideband with the atom before the mode in the basis: the atom's g/e level order decides;
            # sideband with the mode first: blue |g,n> -> |e,n+1> has the ground state first, red |g,n> -> |e,n-1> the excited.
        raising_matrix = np.asarray(raising_matrix, dtype=complex)
        if np.any(np.abs(np.diag(raising_matrix)) > 0.):
            raise IonSimError("Light-atom couplings must be off-diagonal.")
        # Same threshold as CouplingOperator.from_matrix, so the orientation is decided by the elements it keeps.
        rows, columns = np.nonzero(np.abs(raising_matrix) > SMALLEST_ENERGY_SCALE)
        if rows.size == 0:
            return None
        ground_first = rows > columns  # raising element below the diagonal: the ground state has the smaller basis index
        if not (ground_first.all() or not ground_first.any()):
            raise IonSimError("Raising matrix has elements in both triangles; build these couplings as separate operators.")
        ground_index, excited_index = (0, 1) if ground_first[0] else (1, 0)
        rate = self._oscillation_rate_for_level_pair(ground_index, excited_index, frequency_shift)
        mod_function = self._modulation_function_for_level_pair(ground_index, excited_index)
        return CouplingOperator.from_matrix(basis, raising_matrix + raising_matrix.conj().T, rate, modulation_function=mod_function)

    def _atom_coupling_operators(self, basis: Basis, atom: AtomicStructure, ground_levels: list[AtomicInternalEnergyLevel],
                                 excited_levels: list[AtomicInternalEnergyLevel], multipole_order: int, frequency_shift: float,
                                 motional_matrix: Matrix | None = None) -> list[CouplingOperator]:
        """ Coupling operators for every requested (|g>, |e>) pair of one atom, built from that atom's own level objects.

            motional_matrix: optional operator on the full basis acting only on motional DOFs (e.g. i eta a_k for a red
                sideband); the atomic raising operator is multiplied by it. Both are enlarged to the full basis separately,
                so their product is the tensor product for any ordering of the basis DOFs.
        """
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

                raising = np.zeros_like(coupling_matrix)
                raising[excited_index, ground_index] = coupling_matrix[excited_index, ground_index]

                # Enlarge the coupling matrix for the target atom degree of freedom within the total Hilbert space  
                raising = basis.enlarge_matrix(raising, [atom])
                if motional_matrix is not None:
                    raising = raising @ motional_matrix
                coupling_operator = self._coupling_operator_from_raising_matrix(basis, raising, frequency_shift)
                if coupling_operator is not None:
                    coupling_operators.append(coupling_operator)
        return coupling_operators

    def build_atom_laser_ge_coupling_matrix(self, ground_level: AtomicInternalEnergyLevel, excited_level: AtomicInternalEnergyLevel, 
                                            atomic_levels: list[AtomicInternalEnergyLevel], multipole_order: int) -> Matrix: 
        """ Builds a coupling matrix representing a laser light-atom coupling within an ordered list of levels"""
        for level, role in ((ground_level, 'Ground'), (excited_level, 'Excited')):
            if level not in atomic_levels:
                raise ValueError(f"{role} level {level.name} not found in the atomic structure {[l.name for l in atomic_levels]}.")
        rabi_frequency = self.rabi_frequency(ground_level, excited_level, multipole_order)

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
