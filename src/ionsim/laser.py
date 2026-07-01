#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

import numpy as _np
from scipy import integrate as _int
from scipy import interpolate as _interp
from numpy import linalg as lin
from ionsim.tools import amotools as amo
from ionsim.tools import constants as const
import sys

class Laser():
    """Laser type to store the data associated with a particular laser"""
    def __init__(self):
        super(Laser, self).__init__()
        self.circular_basis = {
             1: array([-1., 1.j,0])/sqrt(2.),
             0: array([0,0,1.]),
            -1: array([1., 1.j,0])/sqrt(2.)}
        self.circular_tensor = {q: sum( [sqrt(10./3)*(-1)**q * self.Wigner3j(1,m1,1,m2,2,-q) * array((mat(self.circular_basis[m1]).T * mat(self.circular_basis[m2])).tolist()) for m1 in [1,0,-1] for m2 in [1,0,-1]], 0) for q in [2,1,0,-1,-2] }

        self.label = None

    # TODO: get rid of this laser type, replace with two logical lasers
    def define_logical_raman(self, single_photon_frequency, nhat, spin_states,
                             rabi_rate, detuning=0, gamma=0, phi=0, atom=None,
                             counter_prop=True, frequency_jump_times=None):

        self.raman = True

        if not callable(rabi_rate):
            r0 = rabi_rate
            rabi_rate = lambda x: r0
        if not callable(detuning):
            d0 = detuning
            detuning = lambda x: d0
        if not callable(phi):
            p0 = phi
            phi = lambda x: p0

        # single-photon parameters
        self.single_photon_frequency = single_photon_frequency
        self.single_photon_wavelength = const.SPEED_OF_LIGHT / self.single_photon_frequency
        self.single_photon_wavenumber = 2*_np.pi/self.single_photon_wavelength
        
        # two-photon parameters
        self.nhat = _np.array(nhat)/lin.norm(nhat)
        self.spin_states = spin_states
        self.rabi_rate = rabi_rate
        self.detuning = detuning
        self.gamma = gamma
        self.phi = phi
        self.frequency = None
        self.frequency_0 = None
        self.frequency_jump_times = frequency_jump_times

        self.atom = atom
        if self.atom is not None:
            self.frequency = lambda t: self.atom.energy_difference(self.spin_states[0], self.spin_states[1]) + self.detuning(t)
            self.frequency_0 = self.frequency(0)

        # TODO: generalize for two beams propagating in arbitrary directions
        self.counter_prop = counter_prop
        if self.counter_prop:
            self.wavenumber = 2 * self.single_photon_wavenumber
        else:
            self.wavenumber = 0.0

        return self

    def define_logical_raman_tones(self, atom, hyperfine_label, hyperfine_label_p,
           raman_label, raman_rabi_rate, single_photon_detuning, states, 
           rabi_rate_0=None, phase=0, detuning=0, tone_efield_ratio=1, 
           polarization=None, nhat=None, waist=None,
           polarization_p=None, nhat_p=None, waist_p=None,
           hyperfine_states=None, choose_sidebands=None, verbose=False, frequency_jump_times=None):
        """Define two logical laser objects that yield the specified raman_rabi_rate."""

        self.rabi_rate_0 = rabi_rate_0
        self.define_modulation_functions(raman_rabi_rate, 0, 0, logical=True)
        raman_rabi_rate_0 = self.rabi_rate_0

        temp0 = Laser().define_logical(atom, hyperfine_label,
                                       raman_label, 1.0,
                                       rabi_rate_0=None, phase=0, detuning=0,
                                       polarization=polarization, nhat=nhat,
                                       waist=waist, states=states, frequency_jump_times=frequency_jump_times)
        temp1 = Laser().define_logical(atom, hyperfine_label_p,
                                       raman_label, 1.0,
                                       rabi_rate_0=None, phase=0, detuning=0,
                                       polarization=polarization, nhat=nhat,
                                       waist=waist, states=states, frequency_jump_times=frequency_jump_times)

        total = 0     
        for i, element0 in enumerate(temp0.elements):
            element1, pair0, pair1 = temp1.elements[i], temp0.hf_pairs[i], temp1.hf_pairs[i]
            if pair0[1] != pair1[1]:
                raise ValueError('Intermediate states do not match!', pair0[1], pair1[1])
            upper_label = pair0[1]
            splitting = 2*_np.pi*single_photon_detuning - atom.energy_difference(raman_label, upper_label, 'rad/s')
            total += element0*element1/splitting


        e_field = _np.sqrt(2*_np.pi*raman_rabi_rate_0/abs(2*total))
        target_rabi0_0 = 1/_np.sqrt(tone_efield_ratio)*2*e_field*abs(temp0.target_element)/(2*_np.pi)
        target_rabi1_0 = _np.sqrt(tone_efield_ratio)*2*e_field*abs(temp1.target_element)/(2*_np.pi)

        target_rabi0 = lambda t: target_rabi0_0 * _np.sqrt(self.ham_scale(t))
        target_rabi1 = lambda t: target_rabi1_0 * _np.sqrt(self.ham_scale(t))
        
        # print(target_rabi0_0, target_rabi1_0, target_rabi0_0*target_rabi1_0/(2*single_photon_detuning))

        d0 = single_photon_detuning
        d1 = lambda t: single_photon_detuning + detuning(t)

        laser0 = Laser().define_logical(atom, hyperfine_label, raman_label, target_rabi0,
               rabi_rate_0=target_rabi0_0, phase=0, detuning=d0,
               polarization=polarization, nhat=nhat, waist=waist, states=states)

        laser1 = Laser().define_logical(atom, hyperfine_label_p, raman_label, target_rabi1,
               rabi_rate_0=target_rabi1_0, phase=phase, detuning=d1,
               polarization=polarization_p, nhat=nhat_p, waist=waist_p, states=states,
               hyperfine_states=hyperfine_states, choose_sidebands=choose_sidebands)
        
        return laser0, laser1


    def define_logical(self, atom, hyperfine_label, hyperfine_label_p, rabi_rate,
                       rabi_rate_0=None, phase=0, detuning=0, polarization=None,
                       nhat=None, waist=None, states=None, hyperfine_states=None,
                       choose_sidebands=None, frequency_jump_times=None, verbose=False):
        """
        Define a laser in terms of a particular transition's Rabi rate.
        Only works for carrier transitions (no sidebands).

        Example:
        laser.define_logical(Ca, '4S1/2',4,4, '3D5/2',5,5, 100.e-6, frequency, polarization_vector, k_vector, states = ['4S1/2','3D5/2'] )

        Optional inputs:
        states - Which fine structure levels to couple
        """

        #TODO: remove self.raman from all laser types
        self.raman = False
        self.frequency_jump_times = frequency_jump_times

        self.rabi_rate_0 = rabi_rate_0
        self.states = states
        if self.states is None:
            self.states = [[atom[hyperfine_label].fine_label, atom[hyperfine_label_p].fine_label]]
        self.hyperfine_states = hyperfine_states
        if atom.eliminate_states is not None:
            self.raman_states = [hyperfine_label, hyperfine_label_p]
        else:
            self.raman_states = None
        self.choose_sidebands = choose_sidebands

        temp_freq = atom.energy_difference(hyperfine_label, hyperfine_label_p, 'Hz')
        if callable(detuning):
            frequency = lambda t: temp_freq + detuning(t)
        else:
            frequency = lambda t: temp_freq + detuning
        self.define_modulation_functions(rabi_rate, frequency, phase, logical=True)

        self.pi_time = _np.pi/(2*_np.pi*self.rabi_rate_0) # s  

        mf = atom[hyperfine_label].mf
        mfp = atom[hyperfine_label_p].mf
        f = atom[hyperfine_label].f
        fp = atom[hyperfine_label_p].f

        if polarization is not None and nhat is None:
            raise ValueError("Must specify nhat!")
        if polarization is None and nhat is not None:
            raise ValueError("Must sepcify polarization!")
        if (polarization is None) or (nhat is None):
            dq = -mfp + mf
            self.dipole_polarizations = {q:float(q==dq) for q in [1,0,-1]}
            sqs = 1./_np.sqrt(6.)
            if abs(dq) == 2:
                self.quadrupole_polarizations = {2:sqs, 1:0., 0:0, -1:0., -2:sqs}
            if abs(dq) == 1:
                self.quadrupole_polarizations = {q:1./_np.sqrt(3.) * float((q == dq)) for q in [2,1,0,-1,-2]}
            if dq == 0:
                self.quadrupole_polarizations = {2:sqs/2., 1:0., 0:sqs, -1:0., -2:sqs/2.}
        else:
            # Check that polarization is perpendicular to k-vector:
            self.polarization = _np.array(polarization)/lin.norm(polarization) # complex 3-vector
            self.nhat = _np.array(nhat)/lin.norm(nhat) # real 3-vector
            if abs(_np.dot(self.polarization,self.nhat)) > 1.e-6:
                raise ValueError('Laser polarization is not perpendicular to k vector')
            self.dipole_polarizations = {q:_np.dot(self.polarization, amo.circular_basis[q]) for q in [1,0,-1]}      
            # self.quadrupole_polarizations = {q:abs(_np.dot(self.polarization, _np.dot(amo.circular_tensor[q], self.nhat))) for q in [2,1,0,-1,-2]}
            self.quadrupole_polarizations = {q:_np.dot(self.polarization, _np.dot(amo.circular_tensor[q], self.nhat)) for q in [2,1,0,-1,-2]}
            
        self.wavelength = const.SPEED_OF_LIGHT / self.frequency_0 # meters
        self.wavenumber = 2*_np.pi/self.wavelength # inverse meters

        # Computing electric field
        coupling = atom.coupling_strength(hyperfine_label, hyperfine_label_p, self, known_electric_field=False)
        self.target_element = coupling # save target transition's electric dipole matrix element
        self.electric_field = 1/(4 * self.pi_time * coupling) * 2*_np.pi
        if verbose:
            print("Coupling strength from {} to {}is: {}".format(hyperfine_label, hyperfine_label_p, _np.real(coupling)))
            print("Computed electric field is: {}".format(self.electric_field))

        # Computing the electric dipole maxtrix elements for each pair of hyperfine states in raman transition
        # Matrix elements defined by: rabi_rate/2 = element * electric_field, in rad/sec
        if self.raman_states is not None:
            sublabel_low = self.raman_states[0]
            self.elements = []
            self.hf_pairs = []
            for label, label_p in self.states:
                for sublevel_p in atom[label_p].sublevels.values():
                    if atom.keep_hyperfine is None or sublevel_p.label in atom.keep_hyperfine:
                        sublabel_high = sublevel_p.label
                        self.elements += [atom.coupling_strength(sublabel_low, sublabel_high, self, known_electric_field=False)]
                        self.hf_pairs += [[sublabel_low, sublabel_high]]
    
        # Computed quantities
        self.intensity = self.electric_field**2 / (2 * const.IMPEDENCE_OF_FREE_SPACE) # Watts/meter^2
        if waist is not None:
            self.waist = waist
            self.power_0 = ( _np.pi * self.intensity * waist**2 ) / 2 # Watts
            if verbose:
                print("Calculated power is: {} W".format(self.power))
        # if power_0 is not None:
        #     self.power_0 = power_0
        #     self.waist = _np.sqrt( 2 * power_0 / (_np.pi * self.intensity ))  # Meters
        #     if verbose:
        #         print("Calculated waist is: {} m".format(self.waist))

        # # Compute Lamb-Dicke parameters
        # if atom.secular_frequencies is not None:
        #     self.lamb_dicke_parameters = [0,0,0]
        #     for ind in range(3):
        #         self.lamb_dicke_parameters[ind] = self.wavenumber * abs(_np.dot(self.nhat, atom.motional_unit_vectors[ind])) * _np.sqrt(const.HBAR /
        #                                                 (2 * atom.mass * self.amu * 2 * _np.pi * atom.secular_frequencies[ind]))

        # Allow for method chaining
        return self

    def polarization_check(self, pol, nhat=None):
        pol = _np.array(pol)/lin.norm(pol) # complex 3-vector
        dipole_pols = {q:abs(_np.dot(pol, amo.circular_basis[q])) for q in [1,0,-1]}

        if nhat is not None:
            # quadrupole_pols = {q:abs(_np.dot(pol, _np.dot(amo.circular_tensor[q], self.nhat))) for q in [2,1,0,-1,-2]}
            quadrupole_pols = {q:_np.dot(pol, _np.dot(amo.circular_tensor[q], self.nhat)) for q in [2,1,0,-1,-2]}
            return dipole_pols, quadrupole_pols
        else:
            return dipole_pols

    def define_modulation_functions(self, xxx, frequency, phase, logical=False):
        """Turn laser input parameters into time-dependent functions."""
        if not logical:
            if not callable(xxx):
                self.power = lambda t: xxx
                if self.power_0 is None:
                    self.power_0 = xxx
            else:
                self.power = xxx
                if self.power_0 is None:
                    self.power_0 = xxx(0)

            # if abs(self.power_0) < 1e-14:
            #     print('Warning: Setting power_0 to 1 W. Some derived laser attributes will reflect this.')
            #     self.power_0 = 1 # W

            if abs(self.power_0) < 1e-14:
                print('Warning: Laser power is close to zero. Turning off laser Hamiltonian.',
                     f'To keep Hamiltonian on, try setting value of power_0 in laser {self}.',
                      sep='\n')
                self.power_0 = 0
                self.ham_scale = lambda t: 1
            else:
                self.ham_scale = lambda t: _np.sqrt(self.power(t)/self.power_0)

        else:
            if not callable(xxx):
                self.rabi_rate = lambda t: xxx
                self.rabi_rate_0 = xxx
            else:
                self.rabi_rate = xxx
                if self.rabi_rate_0 is None:
                    # print('Warning: Setting rabi_rate_0 to 1 Hz. Some derived laser attributes will reflect this.')
                    self.rabi_rate_0 = 1 # Hz
            self.ham_scale = lambda t: self.rabi_rate(t)/self.rabi_rate_0

        if not callable(frequency):
            self.frequency = lambda t: frequency
            self.freq_shift_int = lambda t: 0
        else:
            self.frequency = frequency
            freq_shift = lambda t: self.frequency(t) - self.frequency(0)
            def integrate_freq_shift(t0, t1):
                times = _np.linspace(t0, t1, 1000 * int((t1-t0)/1e-6))
                vals = [freq_shift(t) for t in times]
                ints = _int.cumtrapz(vals, times, initial=0)
                spline = _interp.CubicSpline(times, ints, bc_type='natural')
                return spline
            tgo = 100e-6 # max integration time in seconds
            if self.frequency_jump_times is not None:  
                # starts = [0] + self.frequency_jump_times
                # stops = self.frequency_jump_times + [tgo]
                # splines = [integrate_freq_shift(start, stop) for start, stop in zip(starts, stops)]
                # stop_vals = [spline(stop) for spline,stop in zip(splines, stops)]
                # def freq_shift_int(t):
                #     total = 0
                #     for stop, spline, stop_val in zip(stops, splines, stop_vals):
                #         if t >= stop:
                #             total += 2*_np.pi * stop_val
                #         else:
                #             total += 2*_np.pi * spline(t)
                #     return total
                # # TODO: rename to omega_shift_int
                # self.freq_shift_int = freq_shift_int

                starts = [0] + self.frequency_jump_times
                stops = self.frequency_jump_times + [tgo]
                # # TODO: rename to omega_shift_int
                def freq_shift_int(t):
                    total = 0
                    for start, stop in zip(starts, stops):
                        if t >= stop:
                            total += freq_shift((stop+start)/2)*(stop - start)
                        else:
                            total += freq_shift(t)*(t - start)
                    return 2*_np.pi*total
                self.freq_shift_int = freq_shift_int

                # spline = integrate_freq_shift(0, tgo)
                # self.freq_shift_int = lambda t: 2*_np.pi * spline(t)

            else:
                spline = integrate_freq_shift(0, tgo)
                self.freq_shift_int = lambda t: 2*_np.pi * spline(t)

        self.frequency_0 = self.frequency(0)

        if not callable(phase):
            self.phase = lambda t: phase
        else:
            self.phase = phase
        self.phase_0 = self.phase(0)

    def define_physical(self, power, waist, frequency, polarization, nhat, power_0=None,
                        phase=0., states=None, hyperfine_states=None, raman_states=None,
                        choose_sidebands=None, add_opposite_tone=False, frequency_jump_times=None):

        self.raman = False

        if states is not None:
            if states[0].__class__ != list:
                states = [states]

        # Specified quantities
        self.waist = waist # meters
        self.polarization = _np.array(polarization)/lin.norm(polarization) # complex 3-vector
        self.nhat = _np.array(nhat)/lin.norm(nhat) # real 3-vector
        self.power_0 = power_0
        self.states = states # Pairs of fine structure labels to consider when computing couplings
                             # (ex. [['4S1/2', '4P1/2'],['4S1/2','4P3/2']] )
                             # this ex. neglects the laser's effect on the 4S1/2 <--> 3D3/2 and 3D5/2 coupling
        self.hyperfine_states = hyperfine_states
        self.raman_states = raman_states
        self.choose_sidebands = choose_sidebands
        self.frequency_jump_times = frequency_jump_times

        # define modulation functions
        self.define_modulation_functions(power, frequency, phase)

        # Check that polarization is perpendicular to k-vector:
        if abs(_np.dot(self.polarization,self.nhat)) > 1.e-6:
            raise ValueError('Laser polarization is not perpendicular to k vector')

        # Computed quantities
        self.dipole_polarizations = {q:_np.dot(self.polarization, amo.circular_basis[q]) for q in [1,0,-1]}
        # self.quadrupole_polarizations = {q:abs(_np.dot(self.polarization, _np.dot(amo.circular_tensor[q], self.nhat))) for q in [2,1,0,-1,-2]}
        self.quadrupole_polarizations = {q:_np.dot(self.polarization, _np.dot(amo.circular_tensor[q], self.nhat)) for q in [2,1,0,-1,-2]}

        self.wavelength = const.SPEED_OF_LIGHT / self.frequency_0
        self.wavenumber = 2*_np.pi/self.wavelength # Wavenumber, Inverse meters (often inverse centimeters, but we work in mks)
        
        self.intensity = 2*self.power_0/(_np.pi * self.waist**2) # Watts/meter^2
        self.electric_field = _np.sqrt( 2 * self.intensity * const.IMPEDENCE_OF_FREE_SPACE ) # Volts/meter

        # Allow for method chaining
        return self

    def plot_quadrupole_selection_rules(self):
        nhat = _np.array([0,1,0])
        polx = _np.array([0,0,1])
        poly = _np.array([1,0,0])
        thetas = linspace(0,_np.pi/2,100)
        phases = exp(1.j*linspace(0,2*_np.pi,90))
        x, y = meshgrid(thetas, phases)
        for q in range(-2,2+1):
            rhl = _np.dot(amo.circular_tensor[q], nhat)
            # print q, rhl
            print(q, 'RHL', rhl)
            z = empty([len(thetas), len(phases)], dtype='complex')
            for tind, theta in enumerate(thetas):
                for pind, phase in enumerate(phases):
                    pol = polx * cos(theta) + poly * sin(theta) * phase
                    zval = _np.dot(pol, rhl)
                    z[tind,pind] = zval
            # print 'x', x
            # print 'y', y
            # print 'z', z
            print(q)
            print(imag(z))
            print(real(z))
            if abs(z).max() > 0:
                cs = plt.contourf(abs(z),20)
                plt.colorbar()
                plt.show()
            # cs = plt.contour(imag(z), 20)
            # plt.colorbar()
            # plt.show()
            # cs = plt.contour(real(z), 20)
            # plt.colorbar()
            # plt.show()
