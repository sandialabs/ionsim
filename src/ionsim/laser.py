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
import numbers
import warnings
import collections 
import numpy as np
from scipy import constants as const
from scipy import integrate
from abc import ABC, abstractmethod
from enum import StrEnum
from numpy.typing import NDArray

from ionsim.basis import Basis 
from ionsim.atomic_internal_energy_level import AtomicInternalEnergyLevel, compute_multipole_amplitude, compute_rabi_frequency_between_atomic_levels
from ionsim.degree_of_freedom import DegreeOfFreedom, AtomicStructure, MotionalMode
from ionsim.custom_types import Matrix, Vector
from ionsim.config import NUMERICAL_EQUIVALENCE_THRESHOLD, NUMERICAL_ERROR_THRESHOLD, SMALLEST_ENERGY_SCALE
from ionsim.ionsim_error import IonSimError
from ionsim.polarization import Polarization, PolarizationInput, _unit_vector
from ionsim.operators import Operator, CouplingOperator
from ionsim.named_operators import Fock
from ionsim.trapped_ion_mode_analysis import TrappedIonCrystal


class SidebandType(StrEnum):
    """ First motional sideband of a light-atom coupling. """
    RED = 'red'     # |g, n> <-> |e, n-1>, resonant at omega = omega_0 - omega_mode
    BLUE = 'blue'   # |g, n> <-> |e, n+1>, resonant at omega = omega_0 + omega_mode


class BeamProfile(ABC):
    """ Beam profile: the relation between power and peak electric field. The couplings use the peak field (atoms at the
        beam peak); there is no position dependence. """
    @abstractmethod
    def peak_electric_field_magnitude_V_per_m(self, power_W: float) -> float:
        """ Peak electric field amplitude |E0| [V/m] given the power [W]. """


class PlaneWave(BeamProfile):
    """ Plane wave exp(i k r) of uniform intensity [W/m^2]. """

    def __init__(self, intensity_W_per_m2: float):
        if not np.isfinite(intensity_W_per_m2) or intensity_W_per_m2 <= 0.:
            raise IonSimError(f"Plane-wave intensity_W_per_m2 must be a positive number (W/m^2), got {intensity_W_per_m2}.")
        self._intensity_W_per_m2 = float(intensity_W_per_m2)

    @property
    def intensity_W_per_m2(self) -> float:
        """ Intensity [W/m^2]. """
        return self._intensity_W_per_m2

    def __repr__(self):
        return f"PlaneWave(intensity_W_per_m2={self.intensity_W_per_m2})"

    @classmethod
    def from_peak_electric_field_magnitude_V_per_m(cls, peak_electric_field_magnitude_V_per_m: float) -> PlaneWave:
        """ Plane wave with peak field |E0| [V/m]: intensity I = c eps0 |E0|^2 / 2 (inverse of peak_electric_field_magnitude_V_per_m). """
        return cls(intensity_W_per_m2=0.5 * const.c * const.epsilon_0 * peak_electric_field_magnitude_V_per_m**2)

    def peak_electric_field_magnitude_V_per_m(self, power_W: float) -> float:
        """ Peak electric field magnitude |E0| [V/m] """
        return np.sqrt(2. * self.intensity_W_per_m2 / (const.c * const.epsilon_0))


class GaussianBeam(BeamProfile):
    """ Gaussian beam with 1/e^2 intensity radius waist_m [m] at the focus. Only the peak (on-axis, at the focus) field is
        used, so the beam's position and wavelength do not enter. """

    def __init__(self, waist_m: float):
        if not np.isfinite(waist_m) or waist_m <= 0.:
            raise IonSimError(f"Gaussian beam waist_m must be a positive length (m), got {waist_m}.")
        self._waist_m = float(waist_m)

    @property
    def waist_m(self) -> float:
        """ 1/e^2 intensity radius at the focus [m]. """
        return self._waist_m

    def __repr__(self):
        return f"GaussianBeam(waist_m={self.waist_m})"

    def peak_electric_field_magnitude_V_per_m(self, power_W: float) -> float:
        """ Peak electric field magnitude |E0| in free space """ 
        if power_W is None:
            raise IonSimError("Power must be set in constructor to compute electric field magnitude when using a Gaussian beam")
        I0 = 2. * power_W / (np.pi * (self.waist_m ** 2)) # intensity, Watts/m^2
        return np.sqrt(2. * I0 / (const.c * const.epsilon_0))

    def power_for_peak_electric_field_magnitude(self, peak_electric_field_magnitude_V_per_m: float) -> float:
        """ Power [W] that gives peak field |E0| [V/m] at the focus (inverse of peak_electric_field_magnitude):
            I0 = c eps0 |E0|^2 / 2,  P = pi w^2 I0 / 2. """
        I0 = 0.5 * const.c * const.epsilon_0 * peak_electric_field_magnitude_V_per_m**2
        return 0.5 * np.pi * self.waist_m**2 * I0


class Laser:
    """ A single monochromatic laser beam.

        frequency_angular: laser frequency omega [rad/s] (positive).
        propagation_direction: real 3-vector (any nonzero length) along which the laser propagates.
        polarization: PolarizationInput (Polarization for an explicit vector, LinearPolarization, CircularPolarization,
            SphericalPolarization), resolved against the propagation direction into laser.polarization, which must be
            transverse to it (n_hat dot eps = 0).
        beam_profile: BeamProfile (GaussianBeam or PlaneWave), giving the peak field for power_W.
        phase: static laser phase phi [rad] of the |e><g| coupling element.
        power_W: optical power [W]; required for a GaussianBeam, unused by a PlaneWave (which carries its intensity).
        amplitude_modulation_function, phase_modulation_function, frequency_modulation_function: optional time-dependent
            modulations of the coupling; see modulation_function.

        Derived (read-only): wavelength_m, propagation_unit_vector, wavevector, peak_electric_field_magnitude_V_per_m, peak_intensity.
    """

    def __init__(self, frequency_angular: float, propagation_direction: Vector, polarization: PolarizationInput,
                 beam_profile: BeamProfile, phase: float = 0., power_W: float | None = None,
                 amplitude_modulation_function: Callable | None = None, phase_modulation_function: Callable | None = None,
                 frequency_modulation_function: Callable | None = None):
        # Check frequency input  
        if not np.isfinite(frequency_angular) or frequency_angular <= 0.:
            raise IonSimError(f"frequency_angular must be a positive number (rad/s), got {frequency_angular}.")
        # Check propagation direction input  
        direction = np.asarray(propagation_direction)
        if direction.shape != (3,) or np.iscomplexobj(direction) or not np.all(np.isfinite(direction)):
            raise IonSimError(f"propagation_direction must be a real 3-component vector, got {propagation_direction}.")
        n_hat = _unit_vector(direction.astype(float))
        # Check polarization input  
        if not isinstance(polarization, PolarizationInput):
            raise IonSimError(f"polarization must be one of Polarization, LinearPolarization, CircularPolarization or "
                              f"SphericalPolarization, got {type(polarization).__name__}.")

        if not isinstance(beam_profile, BeamProfile):
            raise IonSimError(f"beam_profile must be a BeamProfile (e.g. GaussianBeam or PlaneWave), got {type(beam_profile).__name__}.")
        # Check beam profile characteristics 
        if isinstance(beam_profile, GaussianBeam) and (power_W is None or not np.isfinite(power_W) or power_W <= 0.):
            raise IonSimError(f"A Gaussian beam needs a positive power_W (W), got {power_W}.")
        for name, func in (('amplitude', amplitude_modulation_function), ('phase', phase_modulation_function),
                           ('frequency', frequency_modulation_function)):
            if func is not None and not callable(func):
                raise IonSimError(f"{name}_modulation_function must be callable, got {type(func).__name__}.")
        self._frequency_angular = float(frequency_angular)

        # Check polarization - propagation direction constraint 
        _polarization = polarization.resolve(n_hat)
        eps_dot_n = np.dot(_polarization.jones_vector, n_hat)
        if np.abs(eps_dot_n) > NUMERICAL_ERROR_THRESHOLD:
            raise IonSimError(f"{polarization!r} is not perpendicular to the propagation direction {n_hat}: n_hat dot eps = "
                              f"{eps_dot_n}, should be zero.")
        self._n_hat = n_hat
        self._polarization = _polarization 
        self._beam_profile = beam_profile
        self._phase = float(phase)
        self._power_W = None if power_W is None else float(power_W)
        self._amplitude_modulation_function = amplitude_modulation_function
        self._phase_modulation_function = phase_modulation_function
        self._frequency_modulation_function = frequency_modulation_function

    def __repr__(self):
        return (f"Laser(frequency_angular={self.frequency_angular}, wavelength_m={self.wavelength_m}, polarization={self.polarization}, "
                f"beam_profile={self.beam_profile}, phase={self.phase}, power_W={self.power_W})")

    @property
    def frequency_angular(self) -> float:
        """ Laser frequency omega [rad/s]. """
        return self._frequency_angular

    @property
    def polarization(self) -> Polarization:
        """ The polarization resolved against this laser's propagation direction. """
        return self._polarization

    @property
    def beam_profile(self) -> BeamProfile:
        return self._beam_profile

    @property
    def phase(self) -> float:
        """ Static laser phase [rad]. """
        return self._phase

    @property
    def power_W(self) -> float | None:
        """ Optical power [W] (None for a PlaneWave, which carries its intensity). """
        return self._power_W

    @property
    def amplitude_modulation_function(self) -> Callable | None:
        return self._amplitude_modulation_function

    @property
    def phase_modulation_function(self) -> Callable | None:
        return self._phase_modulation_function

    @property
    def frequency_modulation_function(self) -> Callable | None:
        return self._frequency_modulation_function

    @classmethod
    def from_wavelength(cls, wavelength_m: float, propagation_direction: Vector, polarization: PolarizationInput,
                        beam_profile: BeamProfile, phase: float = 0.,
                        power_W: float | None = None, amplitude_modulation_function: Callable | None = None,
                        phase_modulation_function: Callable | None = None,
                        frequency_modulation_function: Callable | None = None) -> Laser:
        """ Laser from its vacuum wavelength [m]; other arguments as for Laser. """
        if not np.isfinite(wavelength_m) or wavelength_m <= 0.:
            raise IonSimError(f"wavelength_m must be a positive length (m), got {wavelength_m}.")
        return cls(2. * np.pi * const.c / wavelength_m, propagation_direction, polarization, beam_profile, phase, power_W,
                   amplitude_modulation_function, phase_modulation_function, frequency_modulation_function)

    @classmethod
    def gaussian_from_wavelength(cls, wavelength_m: float, power_W: float, waist_m: float, propagation_direction: Vector,
                                 polarization: PolarizationInput,
                                 phase: float = 0., amplitude_modulation_function: Callable | None = None,
                                 phase_modulation_function: Callable | None = None,
                                 frequency_modulation_function: Callable | None = None) -> Laser:
        """ Gaussian laser from its vacuum wavelength [m], power [W] and waist [m]. """
        return cls.from_wavelength(wavelength_m, propagation_direction, polarization, GaussianBeam(waist_m), phase,
                                   power_W,
                                   amplitude_modulation_function, phase_modulation_function, frequency_modulation_function)

    @classmethod
    def gaussian_from_frequency(cls, frequency_angular: float, power_W: float, waist_m: float, propagation_direction: Vector,
                                polarization: PolarizationInput,
                                phase: float = 0., amplitude_modulation_function: Callable | None = None,
                                phase_modulation_function: Callable | None = None,
                                frequency_modulation_function: Callable | None = None) -> Laser:
        """ Gaussian laser from its angular frequency [rad/s], power [W] and waist [m]. """
        return cls(frequency_angular, propagation_direction, polarization, GaussianBeam(waist_m), phase, power_W,
                   amplitude_modulation_function, phase_modulation_function, frequency_modulation_function)

    # The field is chosen so that the Rabi frequency on the given transition(s) equals the target at the peak field (beam
    # focus) with modulation amplitude 1. For several ground and/or excited levels the target is the root-sum-square over
    # all (ground, excited) pairs, sqrt(sum |Omega_ge|^2): the light-shift convention, in which Omega^2 / (4 Delta) is the
    # total light shift of a ground level when the detuning Delta is much larger than the splittings within the set
    # (e.g. unresolved hyperfine levels). Its meaning therefore depends on which levels are included.
    # Couplings are computed in units where the orbital reduced matrix element <L||r^k||L'> = 1 a0^k (unless the level
    # carries a tabulated reduced dipole matrix element), so the derived intensity or power is an effective (calibration)
    # value that absorbs the radial matrix element of these transitions. Every other transition of the same L -> L'
    # multiplet (other sublevels, hyperfine and fine-structure components) then follows correctly from the angular
    # factors; transitions with a different radial matrix element (another multiplet) do not. Do not use the derived
    # intensity or power for quantities that need the physical intensity.

    @classmethod
    def from_target_rabi_frequency(cls, rabi_frequency_angular: float, laser_frequency_angular: float,
                                   ground_levels: AtomicInternalEnergyLevel | list[AtomicInternalEnergyLevel],
                                   excited_levels: AtomicInternalEnergyLevel | list[AtomicInternalEnergyLevel],
                                   multipole_order: int, propagation_direction: Vector, polarization: PolarizationInput,
                                   quantization_axis: Vector, phase: float = 0.,
                                   amplitude_modulation_function: Callable | None = None,
                                   phase_modulation_function: Callable | None = None,
                                   frequency_modulation_function: Callable | None = None) -> Laser:
        """ Plane-wave laser at frequency laser_frequency_angular whose carrier Rabi frequency on ground_levels ->
            excited_levels is rabi_frequency_angular.

            rabi_frequency_angular: rad/s, positive. For one ground and one excited level, |Omega_ge|; for lists of levels,
                the root-sum-square over all (ground, excited) pairs (see above).
            laser_frequency_angular: laser frequency omega [rad/s]; e.g. (E_excited - E_ground) + detuning.
                laser.detuning_from_level_transition(g, e) gives the detuning from any transition.
            ground_levels, excited_levels: a level or a list of levels; every excited level must lie above every ground level.
            multipole_order: 1 (E1) or 2 (E2).
            quantization_axis: the atom's quantization axis (magnetic-field direction), as for rabi_frequency.
            The beam profile is a PlaneWave with the effective intensity that gives this Rabi frequency (see above).
        """
        peak_field = cls._peak_field_for_target_rabi_frequency(
            rabi_frequency_angular, laser_frequency_angular, ground_levels, excited_levels, multipole_order,
            propagation_direction, polarization, quantization_axis)
        profile = PlaneWave.from_peak_electric_field_magnitude_V_per_m(peak_field)
        return cls(laser_frequency_angular, propagation_direction, polarization, profile, phase, None,
                   amplitude_modulation_function, phase_modulation_function, frequency_modulation_function)

    @classmethod
    def gaussian_from_target_rabi_frequency(cls, rabi_frequency_angular: float, laser_frequency_angular: float,
                                            ground_levels: AtomicInternalEnergyLevel | list[AtomicInternalEnergyLevel],
                                            excited_levels: AtomicInternalEnergyLevel | list[AtomicInternalEnergyLevel],
                                            multipole_order: int, waist_m: float, propagation_direction: Vector,
                                            polarization: PolarizationInput, quantization_axis: Vector, phase: float = 0.,
                                            amplitude_modulation_function: Callable | None = None,
                                            phase_modulation_function: Callable | None = None,
                                            frequency_modulation_function: Callable | None = None) -> Laser:
        """ Gaussian laser at frequency laser_frequency_angular whose peak Rabi frequency on ground_levels ->
            excited_levels is rabi_frequency_angular.

            Arguments as for from_target_rabi_frequency, plus waist_m [m]. The power is the effective power
            that gives this Rabi frequency at the focus (see above).
        """
        peak_field = cls._peak_field_for_target_rabi_frequency(
            rabi_frequency_angular, laser_frequency_angular, ground_levels, excited_levels, multipole_order,
            propagation_direction, polarization, quantization_axis)
        profile = GaussianBeam(waist_m)
        power_W = profile.power_for_peak_electric_field_magnitude(peak_field)
        return cls(laser_frequency_angular, propagation_direction, polarization, profile, phase, power_W,
                   amplitude_modulation_function, phase_modulation_function, frequency_modulation_function)

    @classmethod
    def _peak_field_for_target_rabi_frequency(cls, rabi_frequency_angular: float, laser_frequency_angular: float,
                                              ground_levels, excited_levels, multipole_order: int,
                                              propagation_direction: Vector, polarization: PolarizationInput,
                                              quantization_axis: Vector) -> float:
        """ Peak field |E0| (V/m, effective) giving a Rabi frequency (root-sum-square over the pairs) of rabi_frequency_angular.

            Every Omega_ge is linear in |E0|, so |E0| = target / sqrt(sum |Omega_ge(E0 = 1 V/m)|^2), evaluated at this
            frequency and polarization (the E2 coupling also depends on |k|).
        """
        target = np.asarray(rabi_frequency_angular)
        if target.ndim != 0 or np.iscomplexobj(target) or not np.isfinite(target) or target <= 0.:
            raise IonSimError(f"rabi_frequency_angular must be a positive real number (rad/s), got {rabi_frequency_angular}.")
        grounds, exciteds = cls._as_level_list(ground_levels), cls._as_level_list(excited_levels)
        for g in grounds:
            for e in exciteds:
                if e.energy - g.energy <= NUMERICAL_EQUIVALENCE_THRESHOLD:
                    raise IonSimError(f"Excited level {e.name} must be above ground level {g.name}; transition frequency is "
                                      f"{e.energy - g.energy} rad/s.")
        unit_field_laser = cls(laser_frequency_angular, propagation_direction, polarization,
                               PlaneWave.from_peak_electric_field_magnitude_V_per_m(1.))
        rabi_per_unit_field = np.sqrt(sum(abs(unit_field_laser.rabi_frequency(g, e, multipole_order, quantization_axis=quantization_axis))**2
                                          for g in grounds for e in exciteds))
        if rabi_per_unit_field < SMALLEST_ENERGY_SCALE:
            _polarization = unit_field_laser.polarization
            components = (_polarization.dipole_components(quantization_axis) if multipole_order == 1
                          else _polarization.quadrupole_components(propagation_direction, quantization_axis))
            raise IonSimError(f"This polarization does not drive {[g.name for g in grounds]} -> {[e.name for e in exciteds]} "
                              f"(multipole order {multipole_order}); polarization components: {components}.")
        return float(target) / rabi_per_unit_field

    @staticmethod
    def _as_level_list(levels) -> list[AtomicInternalEnergyLevel]:
        """ A level or a non-empty collection of levels, as a list. """
        if isinstance(levels, AtomicInternalEnergyLevel):
            return [levels]
        levels = list(levels)
        if not levels or not all(isinstance(level, AtomicInternalEnergyLevel) for level in levels):
            raise IonSimError(f"Expected an atomic energy level or a non-empty list of them, got {levels}.")
        return levels

    @property
    def peak_electric_field_magnitude_V_per_m(self) -> float:
        """ Peak electric field magnitude E_0 [V/m] """
        return self.beam_profile.peak_electric_field_magnitude_V_per_m(self.power_W) 

    @property
    def peak_intensity(self) -> float:
        """ Peak intensity I_0 [W/m^2] """  
        # Impedence of free space is Z0 = 1/(epsilon0 x speed of light)
        E0 = self.peak_electric_field_magnitude_V_per_m
        return 0.5 * const.c * const.epsilon_0 * (E0**2)

    @property
    def wavelength_m(self) -> float:
        """ Vacuum wavelength 2 pi c / omega [m]. """
        return 2. * np.pi * const.c / self.frequency_angular

    @property
    def propagation_unit_vector(self) -> Vector:
        """ Unit propagation direction n_hat (the normalized propagation_direction). """
        return self._n_hat.copy()

    @property
    def wavevector(self) -> Vector:
        """ Lab-frame wavevector k = (omega / c) n_hat [1/m]. """
        return self.propagation_unit_vector * self.frequency_angular / const.c
        
    # Helper methods for calculations / AMO simulations  
    def detuning_from_transition_frequency(self, transition_frequency: float) -> float:
        """ Computes detuning defined as laser_frequency - transition_frequency in rad/s of the laser from a transition frequency in rad/s """ 
        return self.frequency_angular - transition_frequency

    def detuning_from_level_transition(self, ground_level: AtomicInternalEnergyLevel, excited_level: AtomicInternalEnergyLevel) -> float:
        """ Computes detuning defined as laser_frequency - transition_frequency in rad/s of the laser from a transition frequency in rad/s """ 
        transition_frequency = excited_level.energy - ground_level.energy
        if transition_frequency <= NUMERICAL_EQUIVALENCE_THRESHOLD:
            raise IonSimError(f"Excited level is lower energy than ground level input. Transition frequency is negative or nearly zero: {transition_frequency}")
        return self.detuning_from_transition_frequency(transition_frequency)

    def rabi_frequency(self, ground_level: AtomicInternalEnergyLevel, excited_level: AtomicInternalEnergyLevel,
                       multipole_order: int, quantization_axis: Vector, counter_rotating: bool = False) -> complex:
        """ Rabi frequency Omega (rad/s) of |ground_level> -> |excited_level> driven by this laser at its peak field (beam
            focus) with modulation amplitude 1, so that the coupling's raising element is (Omega/2) e^{i phase} |e><g|.

            |Omega| is the Rabi rate. Its complex phase comes from the polarization and angular factors; the laser's static
            phase is not included. multipole_order: 1 (E1) or 2 (E2, which also depends on |k|). Units as for the coupling
            builders: the orbital reduced matrix element <L||r^k||L'> is taken as 1 a0^k.

            counter_rotating=True: the Rabi frequency of the counter-rotating term (the e^{+i omega t} part of the real
            field on |e><g|), which is driven by the conjugate polarization eps* and, for E2, the opposite field gradient
            (-i k instead of i k). See _field_polarization_and_wavenumber.

            quantization_axis: lab-frame direction of the atom's quantization axis (the magnetic-field direction), which
            defines the m (magnetic) quantum numbers of the levels; the polarization's spherical components are taken
            relative to it. Required (no default): the couplings depend strongly on it, so it must be stated explicitly,
            e.g. np.array([0., 0., 1.]) for a field along lab z.
        """
        if ground_level is excited_level or ground_level.name == excited_level.name:
            raise IonSimError(f"Ground level and excited level should not be the same, got {ground_level.name}.")
        levels = [ground_level, excited_level]
        polarization, wavenumber = self._field_polarization_and_wavenumber(counter_rotating)
        if multipole_order == 1:
            return compute_rabi_frequency_between_atomic_levels(ground_level, excited_level, polarization.dipole_components(quantization_axis),
                                                                levels, multipole_order, self.peak_electric_field_magnitude_V_per_m)
        if multipole_order == 2:
            return compute_rabi_frequency_between_atomic_levels(ground_level, excited_level, polarization.quadrupole_components(self.propagation_unit_vector, quantization_axis),
                                                                levels, multipole_order, self.peak_electric_field_magnitude_V_per_m,
                                                                wavenumber=wavenumber)
        raise IonSimError(f"Multipole order must be 1 or 2 for dipole or quadrupole transitions, respectively. Got {multipole_order}.")
    
    def _modulation_parts(self) -> tuple[Callable | None, Callable | None]:
        """ The (envelope, frequency_phase) callables built from the modulation functions; either may be None.

            envelope(t) = A(t) exp(i phi(t)) multiplies the |e><g| (raising) element, like the static laser phase.
            frequency_phase(t) = exp(-i integral_0^t delta(t') dt') multiplies the coupling the same way as the laser
            frequency itself, i.e. as an offset delta(t) added to it.
        """
        amplitude_mod = self.amplitude_modulation_function
        phase_mod = self.phase_modulation_function
        frequency_mod = self.frequency_modulation_function

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
        """ The complex modulation factor f(t) of the |e><g| (raising) part of a coupling built with the ground level
            first in the basis, or None if unmodulated. Coupling builders use _modulation_function_for_level_pair, which handles
            either level order.

            The modulations are optional callables of time t (s):
                amplitude_modulation_function: A(t), a real envelope of the Rabi frequency (dimensionless)
                phase_modulation_function: phi(t), added to the laser phase (rad)
                frequency_modulation_function: delta(t), added to the laser frequency (rad/s); its phase is integrated numerically
            A constant phi(t) = phi is equivalent to adding phi to the laser phase, and a constant delta(t) = delta to
            frequency_shift_angular=delta in the coupling builders.
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

    def _oscillation_rate_for_level_pair(self, ground_index: int, excited_index: int, frequency_shift_angular: float = 0.) -> float:
        """ Oscillation rate for the coupling operator of one (|g>, |e>) pair; see _carrier_sign. """
        return self._carrier_sign(ground_index, excited_index) * (self.frequency_angular + frequency_shift_angular)

    def _field_polarization_and_wavenumber(self, counter_rotating: bool) -> tuple[Polarization, float]:
        """ Polarization and signed wavenumber of the field component that drives the co- or counter-rotating term.

            The real field E = (E0/2) [eps e^{i(k.r - omega t + phi)} + eps* e^{-i(k.r - omega t + phi)}] couples |g> -> |e>
            through both parts: the co-rotating term (eps, e^{-i omega t}, e^{+i phi}, e^{+i k.r}) kept in the rotating-wave
            approximation, and the counter-rotating term (eps*, e^{+i omega t}, e^{-i phi}, e^{-i k.r}). For E2 the field
            gradient of the e^{-i k.r} part carries -i k, i.e. a negative wavenumber in the E2 coupling factor.
        """
        k = np.linalg.norm(self.wavevector)
        if not counter_rotating:
            return self.polarization, k
        return Polarization(np.conj(self.polarization.jones_vector)), -k

    def _modulation_function_for_level_pair(self, ground_index: int, excited_index: int) -> Callable | None:
        """ Modulation function for the coupling operator of one (|g>, |e>) pair at these basis indices.

            The Hamiltonian multiplies the upper-triangle element of a coupling matrix by the modulation function and adds
            the Hermitian conjugate. The laser phase sits on |e><g|, so the envelope A(t) exp(i phi(t)) is conjugated when
            |g><e| is the upper-triangle element (ground_index < excited_index). The frequency part follows the sign of the
            laser frequency on that element (_carrier_sign): exp(-i int delta) on |e><g| (rate +omega) and its conjugate on
            |g><e| (rate -omega), so a constant delta matches frequency_shift_angular=delta for either level order.
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
    # Light-atom interaction for atom j at equilibrium position 0 (Lamb-Dicke expansion of the field's spatial phase), with
    # the ion's displacement along k written in the mode operators as k.dr_j = sum_k (eta_jk a_k + eta_jk^* a_k^dag)
    # (Hermitian; eta_jk complex in general, real for real mode vectors, where k.dr_j = sum_k eta_jk (a_k + a_k^dag)):
    #     H_j = (Omega/2) e^{i phi} |e><g| exp(i sum_k (eta_jk a_k + eta_jk^* a_k^dag)) e^{-i omega t} + h.c.
    #         ~ (Omega/2) e^{i phi} |e><g| [ 1  +  i sum_k eta_jk a_k  +  i sum_k eta_jk^* a_k^dag ] e^{-i omega t} + h.c.
    #                                       carrier   red sideband (n -> n-1)   blue sideband (n -> n+1)
    # Every term oscillates at the same laser frequency omega. In the interaction picture
    # (rotating_frame_energies = -E of each basis state, motional energies included) the frame supplies the transition
    # frequency of each element, omega_0 for the carrier and omega_0 -/+ omega_k for the red/blue sideband, so all three
    # builders share the same oscillation-rate and modulation conventions (_carrier_sign, _modulation_function_for_level_pair).
    #
    # Counter-rotating terms (include_counter_rotating=True, opt-in). The real field also contributes
    #     (Omega~/2) e^{-i phi} |e><g| exp(-i sum_k (eta_jk a_k + eta_jk^* a_k^dag)) e^{+i omega t} + h.c.,
    # with Omega~ the Rabi frequency for the conjugate polarization eps* (and -i k for E2; rabi_frequency(counter_rotating=True)).
    # Relative to the co-rotating coupling of the same pair: the stored element's oscillation rate and the Lamb-Dicke
    # parameters change sign, and the modulation function is the complex conjugate. Each counter-rotating coupling is a
    # separate CouplingOperator. In any frame these terms oscillate at about omega + omega_0, so they are mainly useful for
    # adiabatic elimination (Bloch-Siegert-type corrections to light shifts and Raman couplings) or low-frequency transitions.
    def build_carrier_coupling_operators(self, basis: Basis, target_atoms: list[AtomicStructure] | None, ground_levels: list[AtomicInternalEnergyLevel], 
                                               excited_levels: list[AtomicInternalEnergyLevel], multipole_order: int, quantization_axis: Vector,
                                               frequency_shift_angular: float = 0., include_counter_rotating: bool = False) -> list[CouplingOperator]:
        """ Carrier couplings |g, n> <-> |e, n> (motional state unchanged) for the requested ground and excited levels.

            target_atoms: addressed AtomicStructure DOFs; all of the basis if None. Levels are matched by name (as for the carrier).
                Pass a single atom, e.g. [atom], for an individually addressing beam.
            ground_levels, excited_levels: list of atomic level objects, checked in _atom_coupling_operators() to ensure they are in the atom's structure.  
                Each atom must contain all levels from the list. Each atom's couplings are computed from its own levels.
            frequency_shift_angular: rad/s added to the laser frequency (e.g. a constant frequency shift from an AOM).

            The laser couples every addressed atom with the same field (its peak electric field and phase). Motional modes in
            the basis are spectators (identity); the carrier's Debye-Waller reduction exp(-eta^2/2) L_n(eta^2) is not included.
            include_counter_rotating: also return the counter-rotating couplings. 

            quantization_axis: lab-frame quantization axis (magnetic-field direction) of the atoms, which defines the m
                quantum numbers of their levels. See rabi_frequency.
        """
        target_atoms = self._parse_atomic_input(basis, target_atoms)
        coupling_operators = []
        for counter_rotating in self._rotation_terms(include_counter_rotating):
            for atom in target_atoms:
                coupling_operators.extend(self._atom_coupling_operators(basis, atom, ground_levels, excited_levels, multipole_order,
                                                                        quantization_axis, frequency_shift_angular,
                                                                        counter_rotating=counter_rotating))
        return coupling_operators

    def build_sideband_coupling_operators(self, sideband: SidebandType | str, basis: Basis, target_atoms: list[AtomicStructure] | None,
                                          target_modes: list[MotionalMode], ground_levels: list[AtomicInternalEnergyLevel],
                                          excited_levels: list[AtomicInternalEnergyLevel], multipole_order: int,
                                          lamb_dicke_parameters: Matrix | complex, quantization_axis: Vector, frequency_shift_angular: float = 0.,
                                          debye_waller: bool = False, include_counter_rotating: bool = False) -> list[CouplingOperator]:
        """ First red or blue sideband couplings of the requested levels with each requested motional mode.

            sideband: SidebandType.RED ('red'): |g, n> <-> |e, n-1>, resonant at omega = omega_0 - omega_mode;
                      SidebandType.BLUE ('blue'): |g, n> <-> |e, n+1>, resonant at omega = omega_0 + omega_mode.

            The coupling element of atom j and mode k is (Omega/2) e^{i phi} (i eta_jk) |e><g| (x) a_k (red) or
            (Omega/2) e^{i phi} (i eta_jk^*) |e><g| (x) a_k^dag (blue), where Omega e^{i phi} is the carrier Rabi frequency of
            the same transition (E1 or E2) and eta_jk is the coefficient of a_k in k.dr_j = sum_k (eta_jk a_k + eta_jk^* a_k^dag).

            target_atoms: addressed AtomicStructure DOFs; all of the basis (in basis order) if None. Levels are matched by
                name (as for the carrier).

            target_modes: the MotionalMode DOFs to couple (required, non-empty). Each must be in the basis; a sideband
                operator is built for every (target atom, target mode) pair.

            ground_levels, excited_levels: list of atomic level objects, checked in _atom_coupling_operators() to ensure they are in the atom's structure.  
                This function will build sideband transitions corresponding to each carrier ground <--> excited coupling combination from the lists. 

            lamb_dicke_parameters: Lamb-Dicke number or matrix eta[i, j] for target_atoms[i] and target_modes[j], shape
                (len(target_atoms), len(target_modes)); real (signed) or complex, the coefficient of a_k as above.

            frequency_shift_angular: rad/s added to the laser frequency, as for the carrier.

            debye_waller: if True, use the exact Delta n = +/-1 matrix elements of exp(i (eta a + eta^* a^dag)) for this mode,
                e^{-|eta|^2/2} L_n^1(|eta|^2) / sqrt(n+1) in place of sqrt(n+1) (Fock.debye_waller_lowering with |eta|).

            include_counter_rotating: also return the counter-rotating sideband couplings, (Omega~/2) e^{-i phi} (-i eta)
                |e><g| (x) a_k (red) or (Omega~/2) e^{-i phi} (-i eta^*) |e><g| (x) a_k^dag (blue) at e^{+i omega t}. 

            quantization_axis: lab-frame quantization axis of the atoms, as for the carrier.

            The basis may order atoms and modes in any way: the kept (upper-triangle) element of each coupling and its
            oscillation-rate sign are determined from the basis-state indices (_coupling_operator_from_raising_matrix).
        """
        sideband = self._parse_sideband(sideband)
        target_atoms = self._parse_atomic_input(basis, target_atoms)
        target_modes = self._parse_motional_mode_input(basis, target_modes)
        etas = self._lamb_dicke_table(lamb_dicke_parameters, target_atoms, target_modes)

        coupling_operators = []
        for counter_rotating in self._rotation_terms(include_counter_rotating):
            for atom in target_atoms:
                for mode in target_modes:
                    eta = etas[(atom, mode)]
                    if np.abs(eta) < NUMERICAL_EQUIVALENCE_THRESHOLD:
                        continue
                    # The counter-rotating term carries the field's e^{-i k.r}: eta -> -eta (and so eta^* -> -eta^*).
                    signed_eta = -eta if counter_rotating else eta
                    motional_matrix = basis.enlarge_matrix(self._sideband_motional_matrix(sideband, mode, signed_eta, debye_waller), [mode])
                    coupling_operators.extend(self._atom_coupling_operators(basis, atom, ground_levels, excited_levels, multipole_order,
                                                                            quantization_axis, frequency_shift_angular, motional_matrix,
                                                                            counter_rotating=counter_rotating))
        return coupling_operators

    def build_sideband_coupling_operators_from_trap(self, sideband: SidebandType | str, basis: Basis, target_atoms: list[AtomicStructure] | None,
                                                    target_modes: list[MotionalMode], ground_levels: list[AtomicInternalEnergyLevel],
                                                    excited_levels: list[AtomicInternalEnergyLevel], multipole_order: int,
                                                    ion_trap: TrappedIonCrystal, ion_indices_in_trap: list[int], mode_indices: list[int],
                                                    quantization_axis: Vector, frequency_shift_angular: float = 0., debye_waller: bool = False,
                                                    include_counter_rotating: bool = False) -> list[CouplingOperator]:
        """ Sideband couplings as in build_sideband_coupling_operators, with the Lamb-Dicke matrix computed by the
            trapped-ion mode analysis ion_trap (e.g. an IonChain) for this laser's lab-frame wavevector:
                eta = ion_trap.get_lamb_dicke_parameters_for_ions_and_modes(ion_indices_in_trap, mode_indices, self.wavevector)

            ion_indices_in_trap: the trap ion of each target atom, ion_indices_in_trap[i] for target_atoms[i] (for target_atoms=None, the
                atoms of the basis in basis order). Stored ion indices, i.e. rows of ion_trap.calculate_lamb_dicke_parameters;
                ion_trap.ion_indices_in_trap[k] is the user-supplied index (e.g. position in from_atomic_structure_basis) of stored
                ion k. For an IonChain, stored ions are ordered along the chain axis.
            mode_indices: the trap mode of each target mode, mode_indices[j] for target_modes[j]. Global mode indices 

            As a consistency check, each target mode's frequency must match its trap mode's, and each target atom's mass
            and charge its trap ion's; a mix-up between two identical ions cannot be detected. Uses this laser's own
            wavevector, i.e. a single-photon transition; a two-photon (Raman) transition needs the difference wavevector
            of its two beams, which a single Laser does not describe.
            With degenerate modes, the mode analysis may return complex Lamb-Dicke parameters from a degenerate subspace
            it has not orthonormalized (it warns); these are passed through as given.
        """
        target_atoms = self._parse_atomic_input(basis, target_atoms)
        if not isinstance(ion_trap, TrappedIonCrystal):
            raise IonSimError(f"ion_trap must be a TrappedIonCrystal (e.g. an IonChain), got {type(ion_trap).__name__}.")
        if len(ion_indices_in_trap) != len(target_atoms):
            raise IonSimError(f"Received {len(ion_indices_in_trap)} ion_indices_in_trap for {len(target_atoms)} target atoms; give one trap ion "
                              f"index per target atom, in the same order.")
        if len(mode_indices) != len(target_modes):
            raise IonSimError(f"Received {len(mode_indices)} mode_indices for {len(target_modes)} target modes; give one trap mode "
                              f"index per target mode, in the same order.")
        lamb_dicke_parameters = ion_trap.get_lamb_dicke_parameters_for_ions_and_modes(ion_indices_in_trap, mode_indices, self.wavevector)
        self._check_targets_match_trap(target_atoms, target_modes, ion_trap, ion_indices_in_trap, mode_indices)
        return self.build_sideband_coupling_operators(sideband, basis, target_atoms, target_modes, ground_levels, excited_levels,
                                                      multipole_order, lamb_dicke_parameters, quantization_axis, frequency_shift_angular,
                                                      debye_waller, include_counter_rotating)

    @staticmethod
    def _check_targets_match_trap(target_atoms: list[AtomicStructure], target_modes: list[MotionalMode], ion_trap: TrappedIonCrystal,
                                  ion_indices_in_trap: list[int], mode_indices: list[int], rtol: float = 1e-9):
        """ Raises if a target mode's frequency differs from its trap mode's, or a target atom's mass or charge from its
            trap ion's. """
        trap_frequencies = ion_trap.normal_mode_frequencies
        for mode, mode_index in zip(target_modes, mode_indices):
            if len(mode.energy_levels) < 2:
                continue
            frequency_angular = mode.energy_levels[1].energy - mode.energy_levels[0].energy
            if not np.isclose(frequency_angular, trap_frequencies[mode_index], rtol=rtol, atol=0.):
                raise IonSimError(f"Target mode {mode.name or mode} has frequency {frequency_angular / (2. * np.pi):.6g} Hz, but trap mode "
                                  f"{mode_index} has {trap_frequencies[mode_index] / (2. * np.pi):.6g} Hz; check mode_indices.")
        masses_amu = ion_trap.atomic_masses / const.value('atomic mass constant')
        charges = ion_trap.atomic_charges
        for atom, ion_index in zip(target_atoms, ion_indices_in_trap):
            if not (np.isclose(atom.atomic_mass, masses_amu[ion_index], rtol=rtol, atol=0.)
                    and np.isclose(atom.atomic_charge, charges[ion_index], rtol=rtol, atol=0.)):
                raise IonSimError(f"Target atom {atom.name or atom} (mass {atom.atomic_mass} amu, charge {atom.atomic_charge}) does not "
                                  f"match trap ion {ion_index} (mass {masses_amu[ion_index]:.10g} amu, charge {charges[ion_index]}); "
                                  f"check ion_indices_in_trap.")

    @staticmethod
    def _parse_sideband(sideband: SidebandType | str) -> SidebandType:
        try:
            return SidebandType(sideband)
        except ValueError:
            raise IonSimError(f"sideband must be one of {[str(member) for member in SidebandType]}, got {sideband!r}.") from None

    @staticmethod
    def _rotation_terms(include_counter_rotating: bool) -> tuple[bool, ...]:
        """ Values of counter_rotating to build: the co-rotating terms, and the counter-rotating ones if requested. """
        if not isinstance(include_counter_rotating, (bool, np.bool_)):
            raise IonSimError(f"include_counter_rotating must be a bool, got {type(include_counter_rotating).__name__}.")
        return (False, True) if include_counter_rotating else (False,)

    @staticmethod
    def _parse_atomic_input(basis: Basis, target_atoms: list[AtomicStructure] | None) -> list[AtomicStructure]:
        """ Parses the input list of atoms targeted by the laser. Each atom in target_atoms must match (be the same object in memory) 
                the corresponding AtomicStructure objects found in the basis.  
            Levels in the atom's AtomicStructure will be checked in _atom_coupling_operators(). """ 
        if target_atoms is None:
            target_atoms = basis.atomic_structure_DOFs
        if not target_atoms:
            raise IonSimError("No atoms to address: the basis has no AtomicStructure degrees of freedom.")
        # Check that atom DOF is in the basis 
        assert(all(isinstance(atom, AtomicStructure) for atom in target_atoms))
        for atom in target_atoms:
            if atom not in basis.atomic_structure_DOFs: 
                raise IonSimError(f"Atom {atom.name or atom} has not been included in the basis degrees of freedom.")
        return list(target_atoms)

    @staticmethod
    def _parse_motional_mode_input(basis: Basis, target_modes: list[MotionalMode]) -> list[MotionalMode]:
        """ Parses the target mode list input: a non-empty list of distinct MotionalMode DOFs of the basis,
            each with Fock levels 0, 1, ..., N-1 in order. 

            Each mode in target_modes must match (be the same object in memory) the corresponding motional mode DOFs in basis. 
        """
        if not isinstance(target_modes, collections.abc.Sequence):
            raise IonSimError(f"target_modes must be a list of MotionalMode DOFs, got {type(target_modes).__name__}; "
                              f"wrap a single mode in a list, e.g. [mode].")
        if not target_modes:
            raise IonSimError("target_modes is empty: specify the motional modes whose sideband couplings to build.")
        not_modes = [type(mode).__name__ for mode in target_modes if not isinstance(mode, MotionalMode)]
        if not_modes:
            raise IonSimError(f"target_modes must contain only MotionalMode DOFs, got {not_modes}.")
        # Check that the target modes don't contain duplicate modes 
        if len({id(mode) if mode.name is None else mode.name for mode in target_modes}) != len(target_modes):
            raise IonSimError("target_modes contains the same mode more than once, which would double its sideband couplings.")
        modes_from_basis = [] if not basis.motional_basis else basis.motional_basis.degrees_of_freedom
        for mode in target_modes:
            if mode not in modes_from_basis: 
                raise IonSimError(f"Motional mode {mode.name or mode} has not been included in the basis degrees of freedom.")
            fock_numbers = [level.fock_number for level in mode.energy_levels]
            if fock_numbers != list(range(len(fock_numbers))):
                raise IonSimError(f"Sideband operators need the mode's levels to be Fock states 0, 1, ..., N-1 in order; got {fock_numbers}.")
        return list(target_modes)

    @staticmethod
    def _lamb_dicke_table(lamb_dicke_parameters: Matrix | complex, atom_dofs: list[AtomicStructure], mode_dofs: list[MotionalMode]) -> dict:
        """ {(atom, mode): eta} from the Lamb-Dicke matrix eta[i, j] for atom_dofs[i] and mode_dofs[j], a numpy array of
            shape (len(atom_dofs), len(mode_dofs)), real or complex. A number is accepted for one atom and one mode. """
        expected_shape = (len(atom_dofs), len(mode_dofs))
        if isinstance(lamb_dicke_parameters, numbers.Number):
            if expected_shape != (1, 1):
                raise IonSimError(f"A scalar Lamb-Dicke parameter is only accepted for one target atom and one target mode. Give a "
                                  f"matrix of shape {expected_shape}: rows follow target_atoms, columns follow target_modes.")
            etas = np.array([[lamb_dicke_parameters]])
        elif isinstance(lamb_dicke_parameters, np.ndarray):
            etas = lamb_dicke_parameters
        else:
            raise IonSimError(f"lamb_dicke_parameters must be a number or a matrix (numpy array), got {type(lamb_dicke_parameters).__name__}.")
        if etas.shape != expected_shape:
            raise IonSimError(f"lamb_dicke_parameters has shape {etas.shape}, expected {expected_shape}: rows follow target_atoms "
                              f"({len(atom_dofs)}) and columns follow target_modes ({len(mode_dofs)}).")
        return {(atom, mode): complex(etas[i, j]) for i, atom in enumerate(atom_dofs) for j, mode in enumerate(mode_dofs)}

    @staticmethod
    def _sideband_motional_matrix(sideband: SidebandType, mode: MotionalMode, eta: complex, debye_waller: bool) -> Matrix:
        """ i eta a (red) or i eta^* a^dag (blue) on one mode in its isolated Hilbert space: the Delta n = -1 / +1 parts of
            exp(i (eta a + eta^* a^dag)) to first order in eta. With debye_waller, the exact Delta n = -/+1 matrix elements,
            whose n-dependence depends only on |eta| (Fock.debye_waller_lowering), with the same prefactors i eta / i eta^*. """
        dimension = len(mode.energy_levels)
        lowering = Fock.debye_waller_lowering(dimension, abs(eta)) if debye_waller else Fock.lowering(dimension)
        if sideband == SidebandType.RED:
            return 1j * eta * lowering
        assert sideband == SidebandType.BLUE
        return 1j * np.conj(eta) * lowering.conj().T

    def _coupling_operator_from_raising_matrix(self, basis: Basis, raising_matrix: Matrix, frequency_shift_angular: float,
                                               counter_rotating: bool = False) -> CouplingOperator | None:
        """ The coupling operator for the raising matrix (<excited state| . |ground state>, laser phase included) of one
            light-atom coupling in the full basis, or None if every element is below SMALLEST_ENERGY_SCALE.

            The operator holds one half C of the Hermitian coupling C + C^dagger: the upper-triangle element of each
            (|g>, |e>) pair, with its own oscillation rate and modulation function. The Hamiltonian adds C^dagger; for a
            Lindblad operator or anything that uses the operator as stored, pass [C, adjoint(C)].

            counter_rotating: the raising matrix is a counter-rotating term (e^{+i omega t} on |e><g|): the oscillation
            rate of the stored element is negated and the modulation function conjugated. (The envelope A(t) e^{i phi(t)}
            enters the counter-rotating term as A(t) e^{-i phi(t)} and the frequency offset with the opposite sign, which is
            the complex conjugate of the co-rotating modulation of the same element for either level order.) """
        # Note on conventions with sidebands: The Hamiltonian keeps the upper-triangle element of each Hermitian pair: raising elements below the diagonal have the
            # ground state first in the basis and those above it the excited state first, which need opposite oscillation-rate
            # signs and envelope conjugation (_carrier_sign). For one carrier or one first-sideband coupling every element is on
            # the same side, because the paired states differ only in the target atom (and the target mode for a sideband):
            # carrier, or sideband with the atom before the mode in the basis: the atom's g/e level order decides;
            # sideband with the mode first: blue |g,n> -> |e,n+1> has the ground state first, red |g,n> -> |e,n-1> the excited.
        raising_matrix = np.asarray(raising_matrix, dtype=complex)
        if np.any(np.abs(np.diag(raising_matrix)) > NUMERICAL_EQUIVALENCE_THRESHOLD):
            raise IonSimError("Light-atom couplings must be off-diagonal.")
        # Same threshold as CouplingOperator.from_matrix, so the orientation is decided by the elements it keeps.
        rows, columns = np.nonzero(np.abs(raising_matrix) > SMALLEST_ENERGY_SCALE)
        if rows.size == 0:
            return None
        ground_first = rows > columns  # raising element below the diagonal: the ground state has the smaller basis index
        if not (ground_first.all() or not ground_first.any()):
            raise IonSimError("Raising matrix has elements in both triangles; build these couplings as separate operators.")
        ground_index, excited_index = (0, 1) if ground_first[0] else (1, 0)
        rate = self._oscillation_rate_for_level_pair(ground_index, excited_index, frequency_shift_angular)
        mod_function = self._modulation_function_for_level_pair(ground_index, excited_index)
        if counter_rotating:
            rate = -rate
            if mod_function is not None:
                co_rotating_mod = mod_function
                def mod_function(t: float) -> complex:
                    return np.conj(co_rotating_mod(t))
        # A CouplingOperator has one oscillation rate and one modulation function, so it cannot also hold the other triangle, 
        # whose rate and modulation are the negated rate and the conjugate. 
        stored_matrix = raising_matrix.conj().T if ground_first[0] else raising_matrix
        return CouplingOperator.from_matrix(basis, stored_matrix, rate, modulation_function=mod_function)

    def _atom_coupling_operators(self, basis: Basis, atom: AtomicStructure, ground_levels: list[AtomicInternalEnergyLevel],
                                 excited_levels: list[AtomicInternalEnergyLevel], multipole_order: int, quantization_axis: Vector,
                                 frequency_shift_angular: float, motional_matrix: Matrix | None = None,
                                 counter_rotating: bool = False) -> list[CouplingOperator]:
        """ Coupling operators for every requested (|g>, |e>) pair of one atom, built from that atom's own level objects.

            ground_levels and excited_levels: list of atomic levels, checked here by name to ensure they are in the atom's structure (check by name).  

            motional_matrix: optional operator on the full basis acting only on motional DOFs (e.g. i eta a_k for a red
                sideband); the atomic raising operator is multiplied by it. Both are enlarged to the full basis separately,
                so their product is the tensor product for any ordering of the basis DOFs.

            counter_rotating: build the counter-rotating couplings (the caller supplies the motional matrix with -eta).
        """
        atomic_levels = atom.energy_levels
        # Table of level.name --> level index in list of levels 
        level_indices = {level.name: index for index, level in enumerate(atomic_levels)}
        # Check to ensure the user-supplied ground and excited levels are in the atom's structure 
        missing = [level.name for level in [*ground_levels, *excited_levels] if level.name not in level_indices]
        if missing:
            raise IonSimError(f"Levels {missing} not found in the atomic structure {atom.name or ''} with levels {list(level_indices)}.")

        coupling_operators = []
        # Build coupling operator for each |g>, |e> pairing
        for ground_name in (level.name for level in ground_levels):
            for excited_name in (level.name for level in excited_levels):
                if ground_name == excited_name:
                    raise IonSimError(f"Excited level {excited_name} matches the ground level {ground_name}.")
                ground_index, excited_index = level_indices[ground_name], level_indices[excited_name]
                ground_level, excited_level = atomic_levels[ground_index], atomic_levels[excited_index]

                coupling_matrix = self.build_atom_laser_ge_coupling_matrix(ground_level, excited_level, atomic_levels, multipole_order,
                                                                           counter_rotating=counter_rotating,
                                                                           quantization_axis=quantization_axis)
                # Pass this loop iteration if there are no non-zero couplings for the requested transition
                if np.count_nonzero(coupling_matrix) == 0 or np.all(np.abs(coupling_matrix) < SMALLEST_ENERGY_SCALE):
                    continue

                raising = np.zeros_like(coupling_matrix)
                raising[excited_index, ground_index] = coupling_matrix[excited_index, ground_index]

                # Enlarge the coupling matrix for the target atom degree of freedom within the total Hilbert space  
                raising = basis.enlarge_matrix(raising, [atom])
                if motional_matrix is not None:
                    raising = raising @ motional_matrix
                coupling_operator = self._coupling_operator_from_raising_matrix(basis, raising, frequency_shift_angular, counter_rotating)
                if coupling_operator is not None:
                    coupling_operators.append(coupling_operator)
        return coupling_operators

    def build_atom_laser_ge_coupling_matrix(self, ground_level: AtomicInternalEnergyLevel, excited_level: AtomicInternalEnergyLevel, 
                                            atomic_levels: list[AtomicInternalEnergyLevel], multipole_order: int,
                                            quantization_axis: Vector, counter_rotating: bool = False) -> Matrix: 
        """ Builds a coupling matrix representing a laser light-atom coupling within an ordered list of ground (g) and excited (e) levels.

            counter_rotating: the counter-rotating term, (Omega~/2) e^{-i phase} |e><g| with Omega~ from the conjugate
            polarization (rabi_frequency(counter_rotating=True)). quantization_axis: as for rabi_frequency. """
        for level, role in ((ground_level, 'Ground'), (excited_level, 'Excited')):
            if level not in atomic_levels:
                raise IonSimError(f"{role} level {level.name} not found in the atomic structure {[l.name for l in atomic_levels]}.")
        rabi_frequency = self.rabi_frequency(ground_level, excited_level, multipole_order, counter_rotating=counter_rotating,
                                             quantization_axis=quantization_axis)
        phase = -self.phase if counter_rotating else self.phase

        # Build coupling operator matrix: 
        single_atom_matrix_size = len(atomic_levels)
        coupling_matrix = np.zeros((single_atom_matrix_size,single_atom_matrix_size), dtype=complex) 

        if np.abs(rabi_frequency) < SMALLEST_ENERGY_SCALE: 
            return coupling_matrix
        
        ground_index = atomic_levels.index(ground_level) 
        excited_index = atomic_levels.index(excited_level) 
        
        if ground_index == excited_index:
            raise IonSimError(f"Ground level and excited level should not be the same.")
    
        coupling_matrix[excited_index, ground_index] = 0.5 * rabi_frequency * np.exp(1j*phase) 
        coupling_matrix[ground_index, excited_index] = np.conj(coupling_matrix[excited_index,ground_index]) 
        return coupling_matrix
