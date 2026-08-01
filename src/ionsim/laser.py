#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

import numpy as np
from scipy import constants as const
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
#from scipy import integrate as _int
#from scipy import interpolate as _interp
#from numpy import linalg as lin
#from ionsim.tools import amotools as amo
#from ionsim.tools import constants as const
#from scipy.special import comb 
from numpy.typing import NDArray
import sympy 
from sympy.physics.wigner import wigner_3j, wigner_6j 

from ionsim.basis import Basis 
from ionsim.atomic_internal_energy_level import AtomicInternalEnergyLevel, compute_multipole_amplitude
from ionsim.degree_of_freedom import DegreeOfFreedom, AtomicStructure
from ionsim.custom_types import Matrix, Vector
from ionsim.config import NUMERICAL_EQUIVALENCE_THRESHOLD, NUMERICAL_ERROR_THRESHOLD, SMALLEST_ENERGY_SCALE 
from ionsim.ionsim_error import IonSimError
from ionsim.operator import Operator, CouplingOperator


## Vector helper operations
def _unit_vector(vec: Vector) -> Vector:
    """ Returns a real unit vector of an input vector """ 
    v = np.asarray(vec)

    norm = np.linalg.norm(v)
    if norm < 1E-9:
        raise ValueError("Cannot normalize a vector with norm near zero. Computed norm {norm}")

    return v/norm


def _perpendicular_basis(n_hat: Vector, ref_axis: Vector | None=None) -> tuple[Vector, Vector]:
    """ Finds orthonormal vectors to a vector "n hat". Option to include a reference axis """ 

    n_hat = _unit_vector(n_hat)
    if ref_axis is None:
        # TODO: Decide whether this convention is best.
        ref_axis = np.array([0., 0., 1.]) if np.abs(n_hat[2]) < (1. - 1E-5)  else np.array([1., 0., 0.]) 

    e1 = np.cross(ref_axis, n_hat)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n_hat, e1) # by definition, n x e1 gives you e2. 
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

    def __post_init__(self):
        # Safety checks on propagation vector  
        if hasattr(self.propagation_vector, "__len__"):
            if len(self.propagation_vector) != 3: 
                raise ValueError(f"Specify a 3-component vector for the beam pointing unit vector 'n hat'. Current input has {len(propagation_unit_vector)} components.")
        else:
            raise TypeError(f"Propagation vector must be a vector (numpy array), received a {type(propagation_vector)}.") 
        assert len(self.propagation_vector) == 3
        
        if np.abs(np.linalg.norm(self.propagation_unit_vector) - 1.) > NUMERICAL_ERROR_THRESHOLD:
            raise IonSimError(f"Propagation unit vector is not normalized! Norm = {np.linalg.norm(self.propagation_unit_vector)}")

        if np.abs(np.dot(self.polarization.vector,self.propagation_unit_vector)) > NUMERICAL_ERROR_THRESHOLD :
            raise ValueError('Laser polarization is not perpendicular to k vector')

        # Check frequency - wavelength relationship 
        # TODO: Check necessary precision for this check to be meaningful 
        light_physics_deviation = np.abs(self.frequency - 2.*np.pi*const.c/self.wavelength)
        if light_physics_deviation > NUMERICAL_ERROR_THRESHOLD: 
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
        profile = Gaussian(waist, focus, wavelength)
        return cls.from_frequency(frequency, propagation_vector, phase, polarization, profile, power, modulation_functions) 


    @property
    def peak_field_amplitude(self) -> float:
        """ Peak E0 [V/m], e.g. at beam focus for a Gaussian beam """ 
        if isinstance(self.profile, PlaneWave):
            return self.profile.peak_field(self.power) 
        return self.profile.peak_field(self.power)

    @property
    def peak_intensity(self) -> float:
        """ Peak intensity [W/m^2] """  
        # Impedence of free space is Z0 = 1/(epsilon0 x speed of light)
        E0 = self.peak_field_amplitude
        return 0.5 * const.c * const.epsilon_0 * (E0**2)

    @property
    def peak_electric_field_magnitude(self) -> float:
        return self.beam_profile.peak_electric_field_magnitude(self.power) 

    @property 
    def propagation_unit_vector(self):
        return _unit_vector(self.propagation_vector)


    @property
    def wavevector(self):
        return self.propagation_unit_vector * np.pi * 2. / self.wavelength
        

    ## Helper methods for calculations / AMO simulations  
    def detuning_from(self, transition_frequency: float) -> float:
        """ Computes detuning defined as laser_frequency - transition_frequency in rad/s of the laser from a transition frequency in rad/s """ 
        return self.frequency - transition_frequency
    

    @property
    def modulation_function(self) -> Callable: 
        """ Returns the laser's modulation function f(t) of the laser profile """ 
        if self.modulation_functions is None: 
            return None

        if all(func is None for func in self.modulation_functions.values()):
            return None 

        # Safety checks 
        allowed_keys = ['amplitude', 'phase', 'frequency']
        if self.modulation_functions.keys() != allowed_keys:
            raise ValueError("Modulation functions should be specified as callables for the allowed keys: {allowed_keys}. Received keys {self.modulation_functions.keys()} instead.")

 #                    if self.mod_functions is not None and None not in self.mod_functions.values():
        # Unpack modulation functions and check which exist 
        amplitude_mod = self.modulation_functions['amplitude']
        phase_mod = self.modulation_functions['phase']
        frequency_mod = self.modulation_functions['phase']
        # Handle all combinations of the modulation functions and combine usage  
        # There are several combinations of None/not None to handle: 
        mod_function = None
        if phase_mod is None and frequency_mod is None and amplitude_mod is not None:
            def mod_function(t: float):
                return lambda t: amplitude_mod(t) 
        elif phase_mod is None and amplitude_mod is None and frequency_mod is not None:
            def mod_function(t: float):
                return lambda t: np.exp(1j * frequency_mod(t) * t) 

        elif phase_mod is None and amplitude_mod is None and frequency_mod is not None:
            def mod_function(t: float):
                return lambda t: np.exp(1j * frequency_mod(t) * t) 
        else:
            # IonSimError(f"Modulation function must be w.r.t to amplitude, phase, or frequency.")
            # TODO: Decide, do we fail loudly in this case? 
            mod_function = None

        return mod_function



    # Method for IA laser  
    def build_individual_atom_laser_coupling_operators(self, basis: Basis, addressed_atom: AtomicStructure, ground_levels: list[AtomicInternalEnergyLevel], 
                                                        excited_levels: list[AtomicInternalEnergyLevel], multipole_order: int): 
        if addressed_atom not in basis.atomic_structure_DOFs:
            raise ValueError(f"Addressed atom {addressed_atom} has not been included in the basis degrees of freedom.") 

        atomic_levels = addressed_atom.energy_levels 
        coupling_operators = []
        # Build coupling operator for each |g>, |e> pairing   
        for ground_level in ground_levels:
            if ground_level not in atomic_levels:
                raise ValueError(f"Ground level {ground_level.name} not found in the atomic structure {atomic_levels}.")

            for excited_level in excited_levels: 
                # Build coupling operator for this |g>, |e> pair and append 
                if excited_level not in atomic_levels:
                    raise ValueError(f"Excited level {excited_level.name} not found in the atomic structure {atomic_levels}.")

                if ground_level.name == excited_level.name:
                    raise ValueError(f"Excited level {excited_level.name} matches the ground level {ground_level.name}.")
                
                coupling_matrix = self.build_atom_laser_ge_coupling_matrix(ground_level, excited_level, atomic_levels, multipole_order) 

                # Pass this loop iteration if there are no non-zero couplings for the requested transition  
                if np.count_nonzero(coupling_matrix) == 0 or np.all(np.abs(coupling_matrix) < SMALLEST_ENERGY_SCALE):
                    continue 

                enlarged_matrix = basis.enlarge_matrix(coupling_matrix, [addressed_atom]) 
                coupling_operators.append(CouplingOperator.from_matrix(basis, enlarged_matrix, self.frequency, modulation_function = self.modulation_function))

        return coupling_operators 

 #    def build_all_atom_laser_coupling_operators(self, basis: Basis, ground_levels: list[AtomicInternalEnergyLevel], excited_levels: list[AtomicInternalEnergyLevel], multipole_order: int): 
 #
        # Find which atoms are the same, handle each list separately;
        # within each list, check if level structures are the same

        # If all atoms are the same, user only needs to specify the ground levels and excited levels for that atom since all structures are the same 
        # If atoms are different, user needs to specify a dictionary for which ground - excited couplings they want for each atom. 

 #        atomic_DOFs = basis.atomic_structure_DOFs
 #        atomic_species = [atom.species for atom in atomic_DOFs]
 #        num_unique_species = 1

    def build_laser_coupling_operators_multiple_atoms(self, basis: Basis, atom_DOFs: list[AtomicStructure], ground_levels: list[AtomicInternalEnergyLevel], 
                                        excited_levels: list[AtomicInternalEnergyLevel], multipole_order: int, all_atoms_are_same: bool = True) -> list[Operator]: 
        """ Builds light-atom coupling operators for all atoms in the basis using AMO physics details for requested ground levels and excited levels via Atomic Structure details """ 
        # TODO: Generalize for varied-species atom ensembles 
        coupling_operators = []
        for atom in atom_DOFs:        
            # Build coupling operator for each |g>, |e> pairing   
            atomic_levels = atom.energy_levels 

            for ground_level in ground_levels:
                if ground_level not in atomic_levels:
                    raise ValueError(f"Ground level {ground_level.name} not found in the atomic structure {atomic_levels}.")

                for excited_level in excited_levels: 
                    # Build coupling operator for this |g>, |e> pair and append 
                    if excited_level not in atomic_levels:
                        raise ValueError(f"Excited level {excited_level.name} not found in the atomic structure {atomic_levels}.")

                    if ground_level.name == excited_level.name:
                        raise ValueError(f"Excited level {excited_level.name} matches the ground level {ground_level.name}.")
                    
                    coupling_matrix = self.build_atom_laser_ge_coupling_matrix(ground_level, excited_level, atomic_levels, multipole_order) 
                    if np.count_nonzero(coupling_matrix) == 0 or np.all(np.abs(coupling_matrix) < SMALLEST_ENERGY_SCALE):
                        continue 
            
                    mod_function = self.modulation_function 
                    if not all_atoms_are_same:
                        enlarged_matrix = basis.enlarge_matrix(coupling_matrix, [atom]) 
                        coupling_operators.append(CouplingOperator.from_matrix(basis, enlarged_matrix, self.frequency, modulation_function=mod_function))
                    else:
                        enlarged_coupling_matrices = [basis.enlarge_matrix(coupling_matrix, [atom]) for atom in basis.atomic_structure_DOFs]
                        coupling_matrices = [large_matrix for large_matrix in enlarged_coupling_matrices] 
                        for matrix in coupling_matrices:
                            coupling_operators.append(CouplingOperator.from_matrix(basis, matrix, self.frequency, modulation_function=mod_function))

            # Break from the loop if all the Atomic Structure DOFs are the same 
            if all_atoms_are_same:
                break 

        return coupling_operators 


    # TODO: Naming, remove "ge"? more readable but less precisely informative  
    def build_atom_laser_ge_coupling_matrix(self, ground_level: AtomicInternalEnergyLevel, excited_level: AtomicInternalEnergyLevel, 
                                            atomic_levels: list[AtomicInternalEnergyLevel], multipole_order: int) -> Matrix: 
        """ Builds a coupling matrix representing a laser light-atom coupling """ 
        if ground_level not in atomic_levels:
            raise ValueError(f"Ground level {ground_level.name} not found in the atomic structure {atomic_levels}.")
        if excited_level not in atomic_levels:
            raise ValueError(f"Excited level {excited_level.name} not found in the atomic structure {atomic_levels}.")

        q = list(np.arange(-multipole_order, multipole_order+1))
        if multipole_order != 1 and multipole_order != 2:
            raise ValueError(f"Multipole order be either 1 or 2, corresponding to E1 dipole or E2 quadrupole transitions. Received {multipole_order}.")

        # Estimate rabi frequency from laser polarization and multipole amplitude components  
        # Compute dot product w.r.t q of spherical polarization components and multipole amplitude components 
        coupling_amplitudes = {}
        for _q in q: 
            coupling_amplitudes[_q] = compute_multipole_amplitude(ground_level, excited_level, multipole_order, _q) 
        
        # Compute dot product with laser field polarization vector 
        # TODO: should we use vdot? 
        # TODO: do we need hbar?  
        # TODO: do we normalize the spherical polarization components? 
        polarization = self.polarization.spherical_components()
        rabi_frequency = 0. + 1j*0.
        rabi_frequency = 2. * self.peak_electric_field_magnitude * np.dot(polarization, np.array(list(coupling_amplitudes.values()))) 
        #rabi_frequency = 2. * self.peak_electric_field_magnitude * np.dot(polarization, np.array(list(coupling_amplitudes.values()))) / const.hbar 
        #print(f"Rabi frequency: {rabi_frequency}")
    
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


    #==============================================================================================
    #==============================================================================================
    #==============================================================================================

 #    def __init__(self):
 #        super(Laser, self).__init__()
 #        self.circular_basis = {
 #             1: array([-1., 1.j,0])/sqrt(2.),
 #             0: array([0,0,1.]),
 #            -1: array([1., 1.j,0])/sqrt(2.)}
 #        self.circular_tensor = {q: sum( [sqrt(10./3)*(-1)**q * self.Wigner3j(1,m1,1,m2,2,-q) * array((mat(self.circular_basis[m1]).T * mat(self.circular_basis[m2])).tolist()) for m1 in [1,0,-1] for m2 in [1,0,-1]], 0) for q in [2,1,0,-1,-2] }
 #
 #        self.label = None
 #
 #    # TODO: get rid of this laser type, replace with two logical lasers
 #    def define_logical_raman(self, single_photon_frequency, nhat, spin_states,
 #                             rabi_rate, detuning=0, gamma=0, phi=0, atom=None,
 #                             counter_prop=True, frequency_jump_times=None):
 #
 #        self.raman = True
 #
 #        if not callable(rabi_rate):
 #            r0 = rabi_rate
 #            rabi_rate = lambda x: r0
 #        if not callable(detuning):
 #            d0 = detuning
 #            detuning = lambda x: d0
 #        if not callable(phi):
 #            p0 = phi
 #            phi = lambda x: p0
 #
 #        # single-photon parameters
 #        self.single_photon_frequency = single_photon_frequency
 #        self.single_photon_wavelength = const.SPEED_OF_LIGHT / self.single_photon_frequency
 #        self.single_photon_wavenumber = 2*_np.pi/self.single_photon_wavelength
 #        
 #        # two-photon parameters
 #        self.nhat = _np.array(nhat)/lin.norm(nhat)
 #        self.spin_states = spin_states
 #        self.rabi_rate = rabi_rate
 #        self.detuning = detuning
 #        self.gamma = gamma
 #        self.phi = phi
 #        self.frequency = None
 #        self.frequency_0 = None
 #        self.frequency_jump_times = frequency_jump_times
 #
 #        self.atom = atom
 #        if self.atom is not None:
 #            self.frequency = lambda t: self.atom.energy_difference(self.spin_states[0], self.spin_states[1]) + self.detuning(t)
 #            self.frequency_0 = self.frequency(0)
 #
 #        # TODO: generalize for two beams propagating in arbitrary directions
 #        self.counter_prop = counter_prop
 #        if self.counter_prop:
 #            self.wavenumber = 2 * self.single_photon_wavenumber
 #        else:
 #            self.wavenumber = 0.0
 #
 #        return self
 #
 #    def define_logical_raman_tones(self, atom, hyperfine_label, hyperfine_label_p,
 #           raman_label, raman_rabi_rate, single_photon_detuning, states, 
 #           rabi_rate_0=None, phase=0, detuning=0, tone_efield_ratio=1, 
 #           polarization=None, nhat=None, waist=None,
 #           polarization_p=None, nhat_p=None, waist_p=None,
 #           hyperfine_states=None, choose_sidebands=None, verbose=False, frequency_jump_times=None):
 #        """Define two logical laser objects that yield the specified raman_rabi_rate."""
 #
 #        self.rabi_rate_0 = rabi_rate_0
 #        self.define_modulation_functions(raman_rabi_rate, 0, 0, logical=True)
 #        raman_rabi_rate_0 = self.rabi_rate_0
 #
 #        temp0 = Laser().define_logical(atom, hyperfine_label,
 #                                       raman_label, 1.0,
 #                                       rabi_rate_0=None, phase=0, detuning=0,
 #                                       polarization=polarization, nhat=nhat,
 #                                       waist=waist, states=states, frequency_jump_times=frequency_jump_times)
 #        temp1 = Laser().define_logical(atom, hyperfine_label_p,
 #                                       raman_label, 1.0,
 #                                       rabi_rate_0=None, phase=0, detuning=0,
 #                                       polarization=polarization, nhat=nhat,
 #                                       waist=waist, states=states, frequency_jump_times=frequency_jump_times)
 #
 #        total = 0     
 #        for i, element0 in enumerate(temp0.elements):
 #            element1, pair0, pair1 = temp1.elements[i], temp0.hf_pairs[i], temp1.hf_pairs[i]
 #            if pair0[1] != pair1[1]:
 #                raise ValueError('Intermediate states do not match!', pair0[1], pair1[1])
 #            upper_label = pair0[1]
 #            splitting = 2*_np.pi*single_photon_detuning - atom.energy_difference(raman_label, upper_label, 'rad/s')
 #            total += element0*element1/splitting
 #
 #
 #        e_field = _np.sqrt(2*_np.pi*raman_rabi_rate_0/abs(2*total))
 #        target_rabi0_0 = 1/_np.sqrt(tone_efield_ratio)*2*e_field*abs(temp0.target_element)/(2*_np.pi)
 #        target_rabi1_0 = _np.sqrt(tone_efield_ratio)*2*e_field*abs(temp1.target_element)/(2*_np.pi)
 #
 #        target_rabi0 = lambda t: target_rabi0_0 * _np.sqrt(self.ham_scale(t))
 #        target_rabi1 = lambda t: target_rabi1_0 * _np.sqrt(self.ham_scale(t))
 #        
 #        # print(target_rabi0_0, target_rabi1_0, target_rabi0_0*target_rabi1_0/(2*single_photon_detuning))
 #
 #        d0 = single_photon_detuning
 #        d1 = lambda t: single_photon_detuning + detuning(t)
 #
 #        laser0 = Laser().define_logical(atom, hyperfine_label, raman_label, target_rabi0,
 #               rabi_rate_0=target_rabi0_0, phase=0, detuning=d0,
 #               polarization=polarization, nhat=nhat, waist=waist, states=states)
 #
 #        laser1 = Laser().define_logical(atom, hyperfine_label_p, raman_label, target_rabi1,
 #               rabi_rate_0=target_rabi1_0, phase=phase, detuning=d1,
 #               polarization=polarization_p, nhat=nhat_p, waist=waist_p, states=states,
 #               hyperfine_states=hyperfine_states, choose_sidebands=choose_sidebands)
 #        
 #        return laser0, laser1
 #
 #
 #    def define_logical(self, atom, hyperfine_label, hyperfine_label_p, rabi_rate,
 #                       rabi_rate_0=None, phase=0, detuning=0, polarization=None,
 #                       nhat=None, waist=None, states=None, hyperfine_states=None,
 #                       choose_sidebands=None, frequency_jump_times=None, verbose=False):
 #        """
 #        Define a laser in terms of a particular transition's Rabi rate.
 #        Only works for carrier transitions (no sidebands).
 #
 #        Example:
 #        laser.define_logical(Ca, '4S1/2',4,4, '3D5/2',5,5, 100.e-6, frequency, polarization_vector, k_vector, states = ['4S1/2','3D5/2'] )
 #
 #        Optional inputs:
 #        states - Which fine structure levels to couple
 #        """
 #
 #        #TODO: remove self.raman from all laser types
 #        self.raman = False
 #        self.frequency_jump_times = frequency_jump_times
 #
 #        self.rabi_rate_0 = rabi_rate_0
 #        self.states = states
 #        if self.states is None:
 #            self.states = [[atom[hyperfine_label].fine_label, atom[hyperfine_label_p].fine_label]]
 #        self.hyperfine_states = hyperfine_states
 #        if atom.eliminate_states is not None:
 #            self.raman_states = [hyperfine_label, hyperfine_label_p]
 #        else:
 #            self.raman_states = None
 #        self.choose_sidebands = choose_sidebands
 #
 #        temp_freq = atom.energy_difference(hyperfine_label, hyperfine_label_p, 'Hz')
 #        if callable(detuning):
 #            frequency = lambda t: temp_freq + detuning(t)
 #        else:
 #            frequency = lambda t: temp_freq + detuning
 #        self.define_modulation_functions(rabi_rate, frequency, phase, logical=True)
 #
 #        self.pi_time = _np.pi/(2*_np.pi*self.rabi_rate_0) # s  
 #
 #        mf = atom[hyperfine_label].mf
 #        mfp = atom[hyperfine_label_p].mf
 #        f = atom[hyperfine_label].f
 #        fp = atom[hyperfine_label_p].f
 #
 #        if polarization is not None and nhat is None:
 #            raise ValueError("Must specify nhat!")
 #        if polarization is None and nhat is not None:
 #            raise ValueError("Must sepcify polarization!")
 #        if (polarization is None) or (nhat is None):
 #            dq = -mfp + mf
 #            self.dipole_polarizations = {q:float(q==dq) for q in [1,0,-1]}
 #            sqs = 1./_np.sqrt(6.)
 #            if abs(dq) == 2:
 #                self.quadrupole_polarizations = {2:sqs, 1:0., 0:0, -1:0., -2:sqs}
 #            if abs(dq) == 1:
 #                self.quadrupole_polarizations = {q:1./_np.sqrt(3.) * float((q == dq)) for q in [2,1,0,-1,-2]}
 #            if dq == 0:
 #                self.quadrupole_polarizations = {2:sqs/2., 1:0., 0:sqs, -1:0., -2:sqs/2.}
 #        else:
 #            # Check that polarization is perpendicular to k-vector:
 #            self.polarization = _np.array(polarization)/lin.norm(polarization) # complex 3-vector
 #            self.nhat = _np.array(nhat)/lin.norm(nhat) # real 3-vector
 #            if abs(_np.dot(self.polarization,self.nhat)) > 1.e-6:
 #                raise ValueError('Laser polarization is not perpendicular to k vector')
 #            self.dipole_polarizations = {q:_np.dot(self.polarization, amo.circular_basis[q]) for q in [1,0,-1]}      
 #            # self.quadrupole_polarizations = {q:abs(_np.dot(self.polarization, _np.dot(amo.circular_tensor[q], self.nhat))) for q in [2,1,0,-1,-2]}
 #            self.quadrupole_polarizations = {q:_np.dot(self.polarization, _np.dot(amo.circular_tensor[q], self.nhat)) for q in [2,1,0,-1,-2]}
 #            
 #        self.wavelength = const.c/ self.frequency_0 # meters
 #        self.wavenumber = 2*_np.pi/self.wavelength # inverse meters
 #
 #        # Computing electric field
 #        coupling = atom.coupling_strength(hyperfine_label, hyperfine_label_p, self, known_electric_field=False)
 #        self.target_element = coupling # save target transition's electric dipole matrix element
 #        self.electric_field = 1/(4 * self.pi_time * coupling) * 2*_np.pi
 #        if verbose:
 #            print("Coupling strength from {} to {}is: {}".format(hyperfine_label, hyperfine_label_p, _np.real(coupling)))
 #            print("Computed electric field is: {}".format(self.electric_field))
 #
 #        # Computing the electric dipole maxtrix elements for each pair of hyperfine states in raman transition
 #        # Matrix elements defined by: rabi_rate/2 = element * electric_field, in rad/sec
 #        if self.raman_states is not None:
 #            sublabel_low = self.raman_states[0]
 #            self.elements = []
 #            self.hf_pairs = []
 #            for label, label_p in self.states:
 #                for sublevel_p in atom[label_p].sublevels.values():
 #                    if atom.keep_hyperfine is None or sublevel_p.label in atom.keep_hyperfine:
 #                        sublabel_high = sublevel_p.label
 #                        self.elements += [atom.coupling_strength(sublabel_low, sublabel_high, self, known_electric_field=False)]
 #                        self.hf_pairs += [[sublabel_low, sublabel_high]]
 #    
 #        # Computed quantities
 #        self.intensity = self.electric_field**2 / (2 * const.IMPEDENCE_OF_FREE_SPACE) # Watts/meter^2
 #        if waist is not None:
 #            self.waist = waist
 #            self.power_0 = ( _np.pi * self.intensity * waist**2 ) / 2 # Watts
 #            if verbose:
 #                print("Calculated power is: {} W".format(self.power))
 #        # if power_0 is not None:
 #        #     self.power_0 = power_0
 #        #     self.waist = _np.sqrt( 2 * power_0 / (_np.pi * self.intensity ))  # Meters
 #        #     if verbose:
 #        #         print("Calculated waist is: {} m".format(self.waist))
 #
 #        # # Compute Lamb-Dicke parameters
 #        # if atom.secular_frequencies is not None:
 #        #     self.lamb_dicke_parameters = [0,0,0]
 #        #     for ind in range(3):
 #        #         self.lamb_dicke_parameters[ind] = self.wavenumber * abs(_np.dot(self.nhat, atom.motional_unit_vectors[ind])) * _np.sqrt(const.HBAR /
 #        #                                                 (2 * atom.mass * self.amu * 2 * _np.pi * atom.secular_frequencies[ind]))
 #
 #        # Allow for method chaining
 #        return self
 #
 #    def polarization_check(self, pol, nhat=None):
 #        pol = _np.array(pol)/lin.norm(pol) # complex 3-vector
 #        dipole_pols = {q:abs(_np.dot(pol, amo.circular_basis[q])) for q in [1,0,-1]}
 #
 #        if nhat is not None:
 #            # quadrupole_pols = {q:abs(_np.dot(pol, _np.dot(amo.circular_tensor[q], self.nhat))) for q in [2,1,0,-1,-2]}
 #            quadrupole_pols = {q:_np.dot(pol, _np.dot(amo.circular_tensor[q], self.nhat)) for q in [2,1,0,-1,-2]}
 #            return dipole_pols, quadrupole_pols
 #        else:
 #            return dipole_pols
 #
 #    def define_modulation_functions(self, xxx, frequency, phase, logical=False):
 #        """Turn laser input parameters into time-dependent functions."""
 #        if not logical:
 #            if not callable(xxx):
 #                self.power = lambda t: xxx
 #                if self.power_0 is None:
 #                    self.power_0 = xxx
 #            else:
 #                self.power = xxx
 #                if self.power_0 is None:
 #                    self.power_0 = xxx(0)
 #
 #            # if abs(self.power_0) < 1e-14:
 #            #     print('Warning: Setting power_0 to 1 W. Some derived laser attributes will reflect this.')
 #            #     self.power_0 = 1 # W
 #
 #            if abs(self.power_0) < 1e-14:
 #                print('Warning: Laser power is close to zero. Turning off laser Hamiltonian.',
 #                     f'To keep Hamiltonian on, try setting value of power_0 in laser {self}.',
 #                      sep='\n')
 #                self.power_0 = 0
 #                self.ham_scale = lambda t: 1
 #            else:
 #                self.ham_scale = lambda t: _np.sqrt(self.power(t)/self.power_0)
 #
 #        else:
 #            if not callable(xxx):
 #                self.rabi_rate = lambda t: xxx
 #                self.rabi_rate_0 = xxx
 #            else:
 #                self.rabi_rate = xxx
 #                if self.rabi_rate_0 is None:
 #                    # print('Warning: Setting rabi_rate_0 to 1 Hz. Some derived laser attributes will reflect this.')
 #                    self.rabi_rate_0 = 1 # Hz
 #            self.ham_scale = lambda t: self.rabi_rate(t)/self.rabi_rate_0
 #
 #        if not callable(frequency):
 #            self.frequency = lambda t: frequency
 #            self.freq_shift_int = lambda t: 0
 #        else:
 #            self.frequency = frequency
 #            freq_shift = lambda t: self.frequency(t) - self.frequency(0)
 #            def integrate_freq_shift(t0, t1):
 #                times = _np.linspace(t0, t1, 1000 * int((t1-t0)/1e-6))
 #                vals = [freq_shift(t) for t in times]
 #                ints = _int.cumtrapz(vals, times, initial=0)
 #                spline = _interp.CubicSpline(times, ints, bc_type='natural')
 #                return spline
 #            tgo = 100e-6 # max integration time in seconds
 #            if self.frequency_jump_times is not None:  
 #                # starts = [0] + self.frequency_jump_times
 #                # stops = self.frequency_jump_times + [tgo]
 #                # splines = [integrate_freq_shift(start, stop) for start, stop in zip(starts, stops)]
 #                # stop_vals = [spline(stop) for spline,stop in zip(splines, stops)]
 #                # def freq_shift_int(t):
 #                #     total = 0
 #                #     for stop, spline, stop_val in zip(stops, splines, stop_vals):
 #                #         if t >= stop:
 #                #             total += 2*_np.pi * stop_val
 #                #         else:
 #                #             total += 2*_np.pi * spline(t)
 #                #     return total
 #                # # TODO: rename to omega_shift_int
 #                # self.freq_shift_int = freq_shift_int
 #
 #                starts = [0] + self.frequency_jump_times
 #                stops = self.frequency_jump_times + [tgo]
 #                # # TODO: rename to omega_shift_int
 #                def freq_shift_int(t):
 #                    total = 0
 #                    for start, stop in zip(starts, stops):
 #                        if t >= stop:
 #                            total += freq_shift((stop+start)/2)*(stop - start)
 #                        else:
 #                            total += freq_shift(t)*(t - start)
 #                    return 2*_np.pi*total
 #                self.freq_shift_int = freq_shift_int
 #
 #                # spline = integrate_freq_shift(0, tgo)
 #                # self.freq_shift_int = lambda t: 2*_np.pi * spline(t)
 #
 #            else:
 #                spline = integrate_freq_shift(0, tgo)
 #                self.freq_shift_int = lambda t: 2*_np.pi * spline(t)
 #
 #        self.frequency_0 = self.frequency(0)
 #
 #        if not callable(phase):
 #            self.phase = lambda t: phase
 #        else:
 #            self.phase = phase
 #        self.phase_0 = self.phase(0)
 #
 #    def define_physical(self, power, waist, frequency, polarization, nhat, power_0=None,
 #                        phase=0., states=None, hyperfine_states=None, raman_states=None,
 #                        choose_sidebands=None, add_opposite_tone=False, frequency_jump_times=None):
 #
 #        self.raman = False
 #
 #        if states is not None:
 #            if states[0].__class__ != list:
 #                states = [states]
 #
 #        # Specified quantities
 #        self.waist = waist # meters
 #        self.polarization = _np.array(polarization)/lin.norm(polarization) # complex 3-vector
 #        self.nhat = _np.array(nhat)/lin.norm(nhat) # real 3-vector
 #        self.power_0 = power_0
 #        self.states = states # Pairs of fine structure labels to consider when computing couplings
 #                             # (ex. [['4S1/2', '4P1/2'],['4S1/2','4P3/2']] )
 #                             # this ex. neglects the laser's effect on the 4S1/2 <--> 3D3/2 and 3D5/2 coupling
 #        self.hyperfine_states = hyperfine_states
 #        self.raman_states = raman_states
 #        self.choose_sidebands = choose_sidebands
 #        self.frequency_jump_times = frequency_jump_times
 #
 #        # define modulation functions
 #        self.define_modulation_functions(power, frequency, phase)
 #
 #        # Check that polarization is perpendicular to k-vector:
 #        if abs(_np.dot(self.polarization,self.nhat)) > 1.e-6:
 #            raise ValueError('Laser polarization is not perpendicular to k vector')
 #
 #        # Computed quantities
 #        self.dipole_polarizations = {q:_np.dot(self.polarization, amo.circular_basis[q]) for q in [1,0,-1]}
 #        # self.quadrupole_polarizations = {q:abs(_np.dot(self.polarization, _np.dot(amo.circular_tensor[q], self.nhat))) for q in [2,1,0,-1,-2]}
 #        self.quadrupole_polarizations = {q:_np.dot(self.polarization, _np.dot(amo.circular_tensor[q], self.nhat)) for q in [2,1,0,-1,-2]}
 #
 #        self.wavelength = const.SPEED_OF_LIGHT / self.frequency_0
 #        self.wavenumber = 2*_np.pi/self.wavelength # Wavenumber, Inverse meters (often inverse centimeters, but we work in mks)
 #        
 #        self.intensity = 2*self.power_0/(_np.pi * self.waist**2) # Watts/meter^2
 #        self.electric_field = _np.sqrt( 2 * self.intensity * const.IMPEDENCE_OF_FREE_SPACE ) # Volts/meter
 #        #self.intensity = self.electric_field**2 / (2 * const.IMPEDENCE_OF_FREE_SPACE) # Watts/meter^2
 #
 #        # Allow for method chaining
 #        return self
 #
 #    def plot_quadrupole_selection_rules(self):
 #        nhat = _np.array([0,1,0])
 #        polx = _np.array([0,0,1])
 #        poly = _np.array([1,0,0])
 #        thetas = linspace(0,_np.pi/2,100)
 #        phases = exp(1.j*linspace(0,2*_np.pi,90))
 #        x, y = meshgrid(thetas, phases)
 #        for q in range(-2,2+1):
 #            rhl = _np.dot(amo.circular_tensor[q], nhat)
 #            # print q, rhl
 #            print(q, 'RHL', rhl)
 #            z = empty([len(thetas), len(phases)], dtype='complex')
 #            for tind, theta in enumerate(thetas):
 #                for pind, phase in enumerate(phases):
 #                    pol = polx * cos(theta) + poly * sin(theta) * phase
 #                    zval = _np.dot(pol, rhl)
 #                    z[tind,pind] = zval
 #            # print 'x', x
 #            # print 'y', y
 #            # print 'z', z
 #            print(q)
 #            print(imag(z))
 #            print(real(z))
 #            if abs(z).max() > 0:
 #                cs = plt.contourf(abs(z),20)
 #                plt.colorbar()
 #                plt.show()
 #            # cs = plt.contour(imag(z), 20)
 #            # plt.colorbar()
 #            # plt.show()
 #            # cs = plt.contour(real(z), 20)
 #            # plt.colorbar()
 #            # plt.show()
 #
 #
 #

    #==============================================================================================
    #==============================================================================================
    #==============================================================================================

#TODO: Should we factor these methods out of this class since it may be used elsewhere in ionsim?
def _spherical_basis_vectors(quantization_axis: Vector) -> Dict[int, Vector]:
    z = _unit_vector(quantization_axis)
    x, y = _perpendicular_basis(z)

    return {1: -(x + 1j*y)/np.sqrt(2.), 0: z.astype(complex), -1 : (x - 1j*y)/np.sqrt(2.)}


def _project_spherical(v: Vector, quantization_axis: Vector) -> dict[int, complex]:
    """ Spherical components v_+1, v_0, v_-1 of a lab-frame vector "v", relative to the 
        quantization axis."""

    v = np.asarray(v, dtype=complex)
    sph_basis = _spherical_basis_vectors(quantization_axis) 
    return {q: np.vdot(basis[q], vec) for q in (1, 0, -1)} 


@dataclass(frozen=True, eq=False)
class Polarization:
    """ Complex Cartesian polarization (Jones) Vector in the lab frame. This is set to be perpendicular to a reference propagation direction (e.g. of a laser) """
    # TODO: make quantization axis an attribute; make resulting basis vectors a property? 
    # TODO: For naming, Should we use "components" or "vector" to refer to the polarization vector? 
    vector: Vector # input polarization vector; normalized in post init 
    EM_field_propagation_direction: Vector

    def __post_init__(self):
        """ Check that polarization vector is perpendicular to propagation vector """  
        #self.normalized_EM_field_propagation_direction = _unit_vector(self.EM_field_propagation_direction) 
        normalized_EM_field_propagation_direction = _unit_vector(self.EM_field_propagation_direction) 

        # Safety check: 
        projection = np.dot(self.vector, normalized_EM_field_propagation_direction)
        if np.abs(projection) > 1E-9: 
            raise ValueError(f"Polarization vector is not perpendicular to the reference propagation direction. Dot product = {projection}, should be zero.")

        # Normalize the vector if is not normalized 
        norm = np.sqrt(np.vdot(self.vector, self.vector).real)
        if norm < 1E-9: 
            raise ValueError("Cannot normalize a vector with norm near zero. Computed norm {norm}")

        if np.abs(norm - 1.) > NUMERICAL_EQUIVALENCE_THRESHOLD: 
            normalized_vector = self.vector / norm
            object.__setattr__(self, "vector", normalized_vector)

    def __repr__(self):
        return f"Polarization(vec = {self.vec}, propagation_direction={self.propagation_direction})" 


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
        # TODO: factor this using _spherical_basis_vectors method 
        z = _unit_vector(quantization_axis)
        x, y = _perpendicular_basis(z)
        # See eq. 7.188 from "Quantum and Atom Optics by Daniel Steck"  
        e_p1 = -(x + 1j*y)/np.sqrt(2.)
        e_0 = z.astype(complex) 
        e_m1 = (x - 1j*y)/np.sqrt(2.)
        #sph_basis_vectors = _spherical_basis_vectors(quantization_axis)

        #polarization_vector = np.array([epsilon_vector, np.array(list(sph_basis_vectors.values())) for  ]) 
        polarization_vector = e_p1 * epsilon_vector[0] + e_0 * epsilon_vector[1] + e_m1 * epsilon_vector[2] 
        return cls(polarization_vector, propagation_direction)

    # TODO: make quantization axis an attribute of the class?     
    def spherical_components(self, quantization_axis: Vector = np.array([0., 0., 1.])) -> Vector: 
        """ Projects the polarization vector's components along a specified quantization axis. """ 
        # TODO Decide - return as a tuple or a Vector? 
        z = _unit_vector(quantization_axis)
        x, y = _perpendicular_basis(z)
        e_p1 = -(x + 1j*y)/np.sqrt(2.)
        e_0 = z.astype(complex) 
        e_m1 = (x - 1j*y)/np.sqrt(2.)
        
        eps_p1 = np.vdot(e_p1, self.vector)
        eps_0 = np.vdot(e_0, self.vector)
        eps_m1 = np.vdot(e_m1, self.vector)
        return np.array([eps_p1, eps_0, eps_m1])

 #    def quadrupole_tensor_components(self, quantization_axis: Vector) -> dict[int, complex]:
 #        """ Rank-2 spherical tensor T_q(n_hat, eps), q = -2, -1, 0, 1, 2, which is built
 #            by the Clebsch-Gordan coefficients coupling of the spherical components of the polarization
 #            vector and the propagation vector. 
 #
 #            This represents the geometric factor governing E2 transitions. 
 #        """ 
    # TODO: Do these actually get used? 
    def rank2_spherical_basis(self, quantization_axis: Vector) -> dict: 
        # TODO: Add docstring w. reference ideally 
        eps = self.spherical_components(quantization_axis)
        propagation_in_spherical = _project_spherical(self.propagation_direction, quantization_axis)

        cq_ij = {}
        sph_basis_vectors = _spherical_basis_vectors(quantization_axis)
        for q in (-2, -1, 0, 1, 2):
            T = np.zeros((3,3),dtype=complex) 
            for m1 in (-1, 0, 1):
                for m2 in (-1, 0, 1):
                    w = complex(sp.N(wigner_3j(1, 1, 2, m1, m2, -q)) )
                    T += w * np.outer(sph_basis_vectors[m1], spherical_basis_vectors[m2])

                cq_ij[q] = np.sqrt(10/3) * ((-1)**q) * T

        return cq_ij

    def e2_q_components(self, quantization_axis: Vector) -> dict:
        # TODO: add doc strings 
        cq_ij = self.rank2_spherical_basis(quantization_axis)
        eps = self.vector
        n_hat = self.EM_field_propagation_direction
        return {q: np.einsum('ij,i,j->', cq_ij[q], self.vector, self.EM_field_propagation_direction) for q in range(-2,3)}

     
