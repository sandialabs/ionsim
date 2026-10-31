#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

import unittest
import warnings

import numpy as np
from scipy import constants as const

from ionsim.atomic_internal_energy_level import reduced_dipole_matrix_element
from ionsim.basis import StandardBasis
from ionsim.degree_of_freedom import AtomicStructure
from ionsim.frequency_comb import FrequencyComb, comb_raman_couplings, frequency_comb_overlap
from ionsim.hamiltonian import Hamiltonian
from ionsim.ionsim_error import IonSimError
from ionsim.laser import Laser, Polarization

TPI = 2. * np.pi
Z, X = np.array([0., 0., 1.]), np.array([1., 0., 0.])


class TestFrequencyComb(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.atom = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2', 'P3/2'], magnetic_field=5.)
        cls.ground = [l for l in cls.atom.energy_levels if l.term_symbol == 'S1/2']
        cls.excited = [l for l in cls.atom.energy_levels if l.term_symbol != 'S1/2']
        cls.names = [g.name for g in cls.ground]
        cls.i00, cls.i10 = cls.names.index('S1/2,0,0'), cls.names.index('S1/2,1,0')
        cls.omega_q = cls.ground[cls.i10].energy - cls.ground[cls.i00].energy

    def test_tooth_weights_and_overlap(self):
        """ Normalized sech teeth; the harmonic overlap is sum_j sech(jx) sech((j+l)x) / sum sech^2 ~ l x / sinh(l x) """
        laser = Laser.gaussian_from_wavelength(355e-9, 1e-3, 10e-6, Z, Polarization.circular(Z, '+'), 0.)
        comb = FrequencyComb(laser, 120e6, 14e-12)
        self.assertAlmostEqual(np.sum(comb.tooth_weights**2), 1., places=12)
        x = comb.envelope_argument
        for l in (0, 1, 105, 300):
            expected = 1. if l == 0 else l * x / np.sinh(l * x)
            self.assertAlmostEqual(frequency_comb_overlap(comb, comb, [l])[0] / expected, 1., delta=1e-3)
        # a CW beam is a Laser, not a comb: one tooth, overlap 1 at l = 0 only
        np.testing.assert_allclose(frequency_comb_overlap(laser, laser, [-1, 0, 1]), [0., 1., 0.])
        for bad in ((-1., 1e-12), (120e6, None), (120e6, 0.)):
            with self.assertRaises(IonSimError):
                FrequencyComb(laser, *bad)
        with self.assertRaises(IonSimError):
            comb_raman_couplings(self.ground, self.excited, [comb.laser.frequency])

    def test_cw_limit_matches_adiabatic_elimination(self):
        """ Two CW tones (Lasers; lin-perp-lin, along B, at the two-photon resonance): the resonant Raman element equals
            IonSim's rotating-wave adiabatic elimination (which also includes the P hyperfine structure, ~2e-4 here) """
        A = Laser.gaussian_from_wavelength(355e-9, 20e-3, 20e-6, Z, Polarization.linear(Z, 0., ref_axis=X), 0.3)
        B = Laser.gaussian_from_frequency(A.frequency - self.omega_q, 20e-3, 20e-6, Z, Polarization.linear(Z, np.pi / 2, ref_axis=X), 0.)
        result = comb_raman_couplings(self.ground, self.excited, [A, B],
                                      atomic_levels=self.atom.energy_levels)
        basis = StandardBasis([self.atom])
        frame = []
        for state in basis.states:
            level = state.components[0]
            frame.append(-(self.ground[self.i00].energy + self.omega_q) if level.term_symbol == 'S1/2' and level.f == 1 else -level.energy)
        operators = []
        for laser in (A, B):
            operators += laser.build_carrier_coupling_operators(basis, [self.atom], self.ground, self.excited, 1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            eliminated = Hamiltonian(basis, operators, frame).adiabatically_eliminate(self.excited, rwa_rate_cutoff=TPI * 1e9)
        H = np.asarray(eliminated.hamiltonian.hamiltonian_function(0.))
        drive = result.resonant_coupling(self.i10, self.i00)
        self.assertAlmostEqual(abs(drive) / abs(H[self.i10, self.i00]), 1., delta=5e-4)
        self.assertAlmostEqual(np.angle(drive), np.angle(H[self.i10, self.i00]), places=6)
        # second-order differential shift: the closed form -omega_q sum_arms sum_J Omega_J^2 / (12 Delta_J^2)
        s = self.ground[0]
        expected = 0.
        for laser in (A, B):
            for J in ('P1/2', 'P3/2'):
                e = next(l for l in self.excited if l.term_symbol == J)
                rabi = laser.peak_electric_field_magnitude * const.e * const.value('Bohr radius') * reduced_dipole_matrix_element(s, e) / const.hbar
                expected -= self.omega_q * rabi**2 / (12. * (laser.frequency - (e.fine_energy - s.fine_energy))**2)
        so = result.second_order_shifts
        self.assertAlmostEqual((so[self.i10] - so[self.i00]) / expected, 1., delta=1e-3)

    def test_cw_tone_with_comb(self):
        """ A CW tone resonant with tooth 0 of a comb: the drive is the CW-tone drive times that tooth's amplitude w(0) """
        pol_a, pol_b = Polarization.linear(Z, 0., ref_axis=X), Polarization.linear(Z, np.pi / 2, ref_axis=X)
        B = Laser.gaussian_from_wavelength(355e-9, 20e-3, 20e-6, Z, pol_b, 0.)
        A = Laser.gaussian_from_frequency(B.frequency + self.omega_q, 20e-3, 20e-6, Z, pol_a, 0.)
        comb = FrequencyComb(B, 120.125e6, 3.9e-12)
        mixed = comb_raman_couplings(self.ground, self.excited, [A, comb], atomic_levels=self.atom.energy_levels)
        cw = comb_raman_couplings(self.ground, self.excited, [A, B], atomic_levels=self.atom.energy_levels)
        w0 = comb.tooth_weights[comb.tooth_indices == 0][0]
        ratio = mixed.resonant_coupling(self.i10, self.i00) / cw.resonant_coupling(self.i10, self.i00)
        self.assertAlmostEqual(ratio.real / w0, 1., delta=1e-3)      # per-tooth detunings differ by ~1e-4 only
        self.assertAlmostEqual(ratio.imag, 0., places=9)

    def lee_shifts(self, k, polarization, f_rep):
        laser = Laser.gaussian_from_wavelength(355e-9, 20e-3, 3e-6, k, polarization, 0.)
        return comb_raman_couplings(self.ground, self.excited, [FrequencyComb(laser, f_rep, 14e-12)],
                                    atomic_levels=self.atom.energy_levels)

    def test_lee_et_al_benchmark(self):
        """ A. C. Lee et al., PRA 94, 042308 (2016): one comb, 20 mW at a 3 um waist, 14 ps pulses. The paper's 247 kHz
            (sigma+-) and 132 kHz (beta-hat) are reproduced to 3% and 5% for a clock two-photon detuning of 29.7 MHz
            (f_rep = 120.125 MHz, the repetition rate quoted for this laser by Yale et al. 2025); the paper's own closed
            forms (rho = 2, sech overlap) give 260 and 136 kHz. Level structure as in the paper: sigma+- shifts |0,0> and
            |1,0> oppositely and leaves |1,+-1> alone; beta-hat leaves |1,0> unshifted. """
        sigma = self.lee_shifts(Z, Polarization.circular(Z, '+'), 120.125e6)
        beta = self.lee_shifts(X, Polarization.circular(X, '+'), 120.125e6)
        self.assertAlmostEqual(sigma.differential_fourth_order_shift(self.i10, self.i00) / (TPI * 247e3), 1., delta=0.05)
        self.assertAlmostEqual(beta.differential_fourth_order_shift(self.i10, self.i00) / (TPI * 132e3), 1., delta=0.08)
        shifts = sigma.fourth_order_shifts
        self.assertAlmostEqual(shifts[self.i10] / -shifts[self.i00], 1., delta=1e-3)
        self.assertLess(abs(shifts[self.names.index('S1/2,1,1')]), 1e-3 * abs(shifts[self.i00]))
        self.assertLess(abs(beta.fourth_order_shifts[self.i10]), 1e-3 * abs(beta.fourth_order_shifts[self.i00]))
        self.assertLess(sigma.max_perturbation_ratio, 0.1)

    def test_against_lee_closed_form(self):
        """ The tooth sum over all harmonics reproduces Lee et al.'s Eqs. (8)-(10) for sigma+- (with this atom's reduced
            matrix elements and the exact tooth overlap l x / sinh(l x)) at f_rep = 120 MHz """
        result = self.lee_shifts(Z, Polarization.circular(Z, '+'), 120e6)
        laser = Laser.gaussian_from_wavelength(355e-9, 20e-3, 3e-6, Z, Polarization.circular(Z, '+'), 0.)
        s = self.ground[0]
        g, D = {}, {}
        for J in ('P1/2', 'P3/2'):
            e = next(l for l in self.excited if l.term_symbol == J)
            g[J] = laser.peak_electric_field_magnitude * const.e * const.value('Bohr radius') * reduced_dipole_matrix_element(s, e) / const.hbar
            D[J] = laser.frequency - (e.fine_energy - s.fine_energy)
        Omega0 = g['P1/2']**2 / (6. * D['P1/2']) - g['P3/2']**2 / (12. * D['P3/2'])
        x = TPI * 120e6 * 14e-12
        j = int(round(self.omega_q / (TPI * 120e6)))
        delta = self.omega_q - j * TPI * 120e6
        k = np.arange(-8000, 8001)
        l = j + k
        overlap = np.where(l == 0, 1., l * x / np.sinh(np.where(l == 0, 1., l * x)))
        C = np.sum(overlap**2 / (1. - k * TPI * 120e6 / delta))
        expected = C * Omega0**2 / (2. * delta)
        self.assertAlmostEqual(result.differential_fourth_order_shift(self.i10, self.i00) / expected, 1., delta=0.01)


if __name__ == '__main__':
    unittest.main()
