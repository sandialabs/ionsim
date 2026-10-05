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
import sympy
from sympy.physics.wigner import wigner_3j, wigner_6j
import scipy.constants as const

from ionsim.degree_of_freedom import AtomicStructure
from ionsim.atomic_internal_energy_level import (
    compute_dipole_amplitude, LSFineLevel, LSHyperfineLevel, LSBackGoudsmitLevel, LSPaschenBackLevel,
    J1L2HyperfineLevel,
)
from ionsim.ionsim_error import IonSimError


def _half_integers(x):
    return np.arange(-x, x + 1)


def _rational(x):
    return sympy.Rational(x).limit_denominator(2)

"""Tests for quantum-number level specification, |mJ, mI> (Back-Goudsmit) levels, and dipole amplitudes."""

class TestQuantumNumberSpecification(unittest.TestCase):
    """from_species(quantum_numbers=...) with the 87Rb, 40Ca+ and 171Yb+ configs."""

    def setUp(self):
        self.qns = [
            {'n': 5, 'l': 0, 's': 0.5, 'j': 1/2, 'f': 1, 'mf': 0},
            {'n': 5, 'l': 0, 's': 0.5, 'j': 1/2, 'f': 2, 'mf': 0},
            {'n': 6, 'l': 1, 's': 0.5, 'j': 3/2, 'f': 3, 'mf': -1},
            {'n': 53, 'l': 0, 's': 0.5, 'j': 1/2, 'mj': 1/2, 'mi': 1/2},
            {'n': 53, 'l': 0, 's': 0.5, 'j': 1/2, 'mj': -1/2, 'mi': 1/2},
        ]
        self.B = 13.6 # Gauss 
        self.atom = AtomicStructure.from_species(species='87Rb', term_symbols=['S1/2', '6 P3/2', '53 S1/2'],
                                                 quantum_numbers=self.qns, magnetic_field=self.B)

    def test_level_types_and_order(self):
        types = [type(level) for level in self.atom.energy_levels]
        self.assertEqual(types, [LSHyperfineLevel] * 3 + [LSBackGoudsmitLevel] * 2)
        names = [level.name for level in self.atom.energy_levels]
        self.assertEqual(names, ['S1/2,1,0', 'S1/2,2,0', '6 P3/2,3,-1', '53 S1/2,1/2,1/2', '53 S1/2,-1/2,1/2'])

    def test_clock_state_quadratic_zeeman(self):
        """5S1/2 |F=1,0> and |F=2,0> shift by -/+ K B^2 / 2, with K = 575.15 Hz/G^2 (Steck)."""
        shift_f1 = self.atom.energy_levels[0].external_energy_shift / (2 * np.pi)
        shift_f2 = self.atom.energy_levels[1].external_energy_shift / (2 * np.pi)
        expected = 575.15 * (self.B)**2 / 2
        self.assertAlmostEqual(shift_f2, expected, delta=0.01 * expected)
        self.assertAlmostEqual(shift_f1, -expected, delta=0.01 * expected)

    def test_minimal_dicts_and_string_fractions(self):
        atom = AtomicStructure.from_species(species='87Rb',
                                            quantum_numbers=[{'n': 53, 'mj': '-1/2', 'mi': '3/2'},
                                                             {'n': 5, 'l': 0, 'f': 2, 'mf': -2}],
                                            magnetic_field=1.)
        self.assertEqual([level.name for level in atom.energy_levels], ['53 S1/2,-1/2,3/2', 'S1/2,2,-2'])

    def test_invalid_inputs_raise(self):
        bad_calls = [
            dict(quantum_numbers=[{'l': 0, 'j': 0.5, 'f': 1, 'mf': 0}]),                        # ambiguous: 5S vs 53S
            dict(quantum_numbers=[{'n': 5, 'l': 0, 'f': 3, 'mf': 0}]),                           # F out of range
            dict(quantum_numbers=[{'n': 5, 'l': 0, 'f': 1, 'mf': 2}]),                           # |mF| > F
            dict(quantum_numbers=[{'n': 53, 'mj': 0.5, 'mi': 0.5}]),                             # |mJ, mI> at zero field
            dict(quantum_numbers=[{'n': 53, 'mj': 0.5, 'mi': 0.5, 'f': 1}], magnetic_field=1.),  # mixed basis
            dict(quantum_numbers=[{'n': 53, 'mj': 0.5, 'mi': 2.5}], magnetic_field=1.),          # |mI| > I
            dict(quantum_numbers=[{'n': 7, 'f': 1, 'mf': 0}]),                                   # no matching manifold
            dict(quantum_numbers=[{'n': 5, 'l': 0, 'f': 1, 'mF': 0}]),                           # unknown key
            dict(quantum_numbers=[{'n': 5, 'l': 0, 'f': 1, 'mf': 0}] * 2),                       # duplicate level
            dict(quantum_numbers=[{'n': 53, 'ml': 0, 'ms': 0.5, 'mi': 0.5}], magnetic_field=1.), # Paschen-Back
            dict(quantum_numbers=[{'n': 53, 'mj': 0.5, 'mi': 0.5}], magnetic_field=1.,
                 approximation='weak field'),                                                     # |mJ, mI> needs exact solver
            dict(term_symbols=['S1/2'], level_names=['S1/2,1,0'], quantum_numbers=[{'f': 1, 'mf': 0}]),
            dict(term_symbols=['S1/2'], level_names=['S1/2,3,0']),                                  # unknown level name
            dict(term_symbols=['S9/2']),                                                            # unknown manifold
            dict(term_symbols=['S1/2'], level_aliases=['a']),                                       # aliases need a level list
        ]
        for kwargs in bad_calls:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(IonSimError):
                    AtomicStructure.from_species(species='87Rb', **kwargs)

    def test_low_overlap_warning(self):
        """|mJ, mI> labels at a weak field are nominal and should warn."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            AtomicStructure.from_species(species='87Rb', quantum_numbers=[{'n': 53, 'mj': -0.5, 'mi': 0.5}],
                                         magnetic_field=0.01)
        self.assertTrue(any('overlap' in str(w.message) for w in caught))

    def test_level_names_path(self):
        atom = AtomicStructure.from_species(species='87Rb', term_symbols=['S1/2', '6 P3/2', '53 S1/2'],
                                            level_names=['S1/2,1,0', 'S1/2,2,0', '6 P3/2,3,-1', '53 S1/2,1,0'],
                                            magnetic_field=0.01)
        self.assertEqual([level.name for level in atom.energy_levels],
                         ['S1/2,1,0', 'S1/2,2,0', '6 P3/2,3,-1', '53 S1/2,1,0'])

    def test_whole_manifolds(self):
        """No level filter: every sublevel of the selected manifolds (8 for 5S, 16 for 6P3/2)."""
        atom = AtomicStructure.from_species(species='87Rb', term_symbols=['S1/2', '6 P3/2'], magnetic_field=1.)
        self.assertEqual(len(atom.energy_levels), 24)

    def test_zero_nuclear_spin_at_nonzero_field(self):
        """40Ca+ (I = 0) previously hit a division by zero in the solver's lande_gi."""
        atom = AtomicStructure.from_species(species='40Ca+', quantum_numbers=[{'n': 4, 'l': 0, 'mj': 0.5},
                                                                              {'n': 4, 'l': 0, 'mj': -0.5}],
                                            magnetic_field=1.)
        up, down = atom.energy_levels
        self.assertIsInstance(up, LSFineLevel)
        splitting = (up.energy - down.energy) / (2 * np.pi)
        gs = np.abs(const.physical_constants['electron g factor'][0])  # electron spin g factor. 
        MU_B_HZ_PER_GAUSS = 1.39962449e6  # Bohr magneton / h, Hz per gauss
        self.assertAlmostEqual(splitting, gs * MU_B_HZ_PER_GAUSS, places = 2) 

    def test_j1l2_gj_independent_of_field(self):
        """171Yb+ [3/2]1/2 gets the same computed gJ at zero and nonzero field."""
        levels = [AtomicStructure.from_species(species='171Yb+', term_symbols=['[3/2]1/2'], magnetic_field=b).energy_levels[0]
                  for b in (0., 1.)]
        self.assertIsNotNone(levels[0].gj)
        self.assertAlmostEqual(levels[0].gj, levels[1].gj, places=12)


class TestDipoleAmplitude(unittest.TestCase):
    """compute_dipole_amplitude against Steck's conventions (Rb87 D1 and D2)."""

    I = 1.5
    J = 0.5

    def _hf(self, j, f, mf, term_symbol='S1/2', l=0):
        return LSHyperfineLevel(n=5, j=j, term_symbol=term_symbol, fine_energy=0., hyperfine_A=0.,
                                l=l, s=0.5, i=self.I, f=f, mf=mf)

    def _bg(self, j, mj, mi, term_symbol='S1/2', l=0):
        return LSBackGoudsmitLevel(n=5, j=j, term_symbol=term_symbol, fine_energy=0., hyperfine_A=0.,
                                   l=l, s=0.5, i=self.I, mj=mj, mi=mi)

    def _f_range(self, j):
        return np.arange(abs(j - self.I), j + self.I + 1)

    def _steck_closed_form(self, f, mf, jp, fp, mfp, q):
        """Steck Eqs. 35-36: 3j x 6j, with the (2F'+1) prefactor."""
        f, mf, jp, fp, mfp = map(_rational, (f, mf, jp, fp, mfp))
        j, i = _rational(self.J), _rational(self.I)
        three_j = (-1)**(fp - 1 + mf) * sympy.sqrt(2*f + 1) * wigner_3j(fp, 1, f, mfp, q, -mf)
        six_j = (-1)**(fp + j + 1 + i) * sympy.sqrt((2*fp + 1) * (2*j + 1)) * wigner_6j(j, jp, 1, fp, f, i)
        return float(three_j * six_j)

    def test_matches_steck_closed_form(self):
        for jp, term_symbol in ((0.5, 'P1/2'), (1.5, 'P3/2')):
            for f in self._f_range(self.J):
                for mf in _half_integers(f):
                    for fp in self._f_range(jp):
                        for mfp in _half_integers(fp):
                            for q in (-1, 0, 1):
                                with self.subTest(jp=jp, f=f, mf=mf, fp=fp, mfp=mfp, q=q):
                                    amplitude = compute_dipole_amplitude(self._hf(self.J, f, mf),
                                                                         self._hf(jp, fp, mfp, term_symbol, 1), q)
                                    self.assertAlmostEqual(amplitude,
                                                           self._steck_closed_form(f, mf, jp, fp, mfp, q), places=12)

    def test_sum_rule_all_bases(self):
        """Total strength out of every ground sublevel is 1, for |F,mF>, |mJ,mI>, and mixed pairs."""
        for jp, term_symbol in ((0.5, 'P1/2'), (1.5, 'P3/2')):
            excited_hf = [self._hf(jp, fp, mfp, term_symbol, 1) for fp in self._f_range(jp) for mfp in _half_integers(fp)]
            excited_bg = [self._bg(jp, mjp, mip, term_symbol, 1) for mjp in _half_integers(jp) for mip in _half_integers(self.I)]
            ground_hf = [self._hf(self.J, f, mf) for f in self._f_range(self.J) for mf in _half_integers(f)]
            ground_bg = [self._bg(self.J, mj, mi) for mj in _half_integers(self.J) for mi in _half_integers(self.I)]
            for label, grounds, excited in (('hf-hf', ground_hf, excited_hf), ('bg-bg', ground_bg, excited_bg),
                                            ('hf-bg', ground_hf, excited_bg), ('bg-hf', ground_bg, excited_hf)):
                for ground in grounds:
                    with self.subTest(jp=jp, basis=label, ground=ground.name):
                        total = sum(compute_dipole_amplitude(ground, e, q)**2 for e in excited for q in (-1, 0, 1))
                        self.assertAlmostEqual(total, 1.0, places=10)

    def test_d2_cycling_transition(self):
        amplitude = compute_dipole_amplitude(self._hf(self.J, 2, 2), self._hf(1.5, 3, 3, 'P3/2', 1), -1)
        self.assertAlmostEqual(amplitude**2, 0.5, places=12)

    def test_fine_structure_levels(self):
        ground = LSFineLevel(n=5, j=0.5, term_symbol='S1/2', fine_energy=0., hyperfine_A=0., l=0, s=0.5, mj=0.5)
        excited = LSFineLevel(n=5, j=1.5, term_symbol='P3/2', fine_energy=0., hyperfine_A=0., l=1, s=0.5, mj=1.5)
        self.assertAlmostEqual(compute_dipole_amplitude(ground, excited, -1)**2, 0.5, places=12)

    def test_j1l2_uses_level_j(self):
        """171Yb+ [3/2]1/2 has J = 1/2; the old code used k + s2. S1/2 -> [3/2]1/2 sum rule must be 1."""
        ground = [LSHyperfineLevel(n=6, j=0.5, term_symbol='S1/2', fine_energy=0., hyperfine_A=0., l=0, s=0.5,
                                   i=0.5, f=f, mf=mf) for f in (0, 1) for mf in _half_integers(f)]
        excited = [J1L2HyperfineLevel(n=4, j=0.5, term_symbol='[3/2]1/2', fine_energy=0., hyperfine_A=0., j1=3.5,
                                      l2=2., k=1.5, s2=1., i=0.5, f=fp, mf=mfp, gj=1.)
                   for fp in (0, 1) for mfp in _half_integers(fp)]
        for g in ground:
            with self.subTest(ground=g.name):
                total = sum(compute_dipole_amplitude(g, e, q)**2 for e in excited for q in (-1, 0, 1))
                self.assertAlmostEqual(total, 1.0, places=10)

    def test_paschen_back_level_raises(self):
        pb = LSPaschenBackLevel(n=5, j=0.5, term_symbol='S1/2', fine_energy=0., hyperfine_A=0., l=0, s=0.5,
                                i=self.I, mi=0.5, ml=0, ms=0.5)
        with self.assertRaises(IonSimError):
            compute_dipole_amplitude(pb, self._hf(1.5, 3, 3, 'P3/2', 1), -1)


if __name__ == '__main__':
    unittest.main()
