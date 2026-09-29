#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************
"""Tests for DissipatorSpontaneousEmission.

    The total decay rate out of a basis state |e> is sum_i sum_g |<g|L_i|e>|^2 over all Lindblad operators L_i. With the
    exact branching fractions, decay out of any excited sublevel into a complete ground manifold occurs at
    (branching ratio) / (lifetime), independent of basis choice and of which other levels are included.
"""

import unittest
from dataclasses import replace

import numpy as np

from ionsim.basis import StandardBasis
from ionsim.degree_of_freedom import AtomicStructure, levels_in_manifold
from ionsim.lindbladian import DissipatorSpontaneousEmission, Lindbladian
from ionsim.ionsim_error import IonSimError


def decay_rate_matrix(dissipator) -> np.ndarray:
    """R[i, j] = sum over Lindblad operators of |L[i, j]|^2: the decay rate from basis state j to basis state i."""
    rates = np.zeros((dissipator.size, dissipator.size))
    for lindblad_function in dissipator.lindblad_matrix_functions:
        matrix = lindblad_function(0.)
        matrix = matrix.toarray() if hasattr(matrix, 'toarray') else np.asarray(matrix)
        rates += np.abs(matrix)**2
    return rates


def decay_by_manifold(structure, dissipator, excited_level):
    """Decay rate out of excited_level (single-DOF basis), summed per ground manifold."""
    rates = decay_rate_matrix(dissipator)
    column = structure.energy_levels.index(excited_level)
    totals = {}
    for row, level in enumerate(structure.energy_levels):
        if rates[row, column] > 0:
            totals[level.term_symbol] = totals.get(level.term_symbol, 0.) + rates[row, column]
    return totals


class TestSpontaneousEmissionRates(unittest.TestCase):

    def test_multiple_ground_manifolds(self):
        """40Ca+ P1/2 -> {S1/2, D3/2}: total rate 1/tau, split by the config branching ratios."""
        ca = AtomicStructure.from_species(species='40Ca+', manifolds=['S1/2', 'D3/2', 'P1/2'])
        dissipator = DissipatorSpontaneousEmission.from_atomic_structure_data(
            StandardBasis([ca]), levels_in_manifold(ca, 'S1/2') + levels_in_manifold(ca, 'D3/2'), levels_in_manifold(ca, 'P1/2'))
        tau = 6.904e-9
        for excited in levels_in_manifold(ca, 'P1/2'):
            with self.subTest(excited=excited.name):
                by_manifold = decay_by_manifold(ca, dissipator, excited)
                self.assertAlmostEqual(by_manifold['S1/2'] * tau, 0.93565, places=10)
                self.assertAlmostEqual(by_manifold['D3/2'] * tau, 0.06435, places=10)

    def test_three_ground_manifolds_neutral_yb(self):
        """171Yb 3D1 -> 3P0, 3P1, 3P2 in the hyperfine basis, with branching ratios keyed by term symbol."""
        yb = AtomicStructure.from_species(species='171Yb', manifolds=['P0', 'P1', 'P2', 'D1'])
        ground = levels_in_manifold(yb, 'P0') + levels_in_manifold(yb, 'P1') + levels_in_manifold(yb, 'P2')
        dissipator = DissipatorSpontaneousEmission.from_atomic_structure_data(StandardBasis([yb]), ground, levels_in_manifold(yb, 'D1'))
        tau = 0.0527
        for excited in levels_in_manifold(yb, 'D1'):
            with self.subTest(excited=excited.name):
                by_manifold = decay_by_manifold(yb, dissipator, excited)
                for term_symbol, ratio in {'P0': 0.638, 'P1': 0.352, 'P2': 0.01}.items():
                    self.assertAlmostEqual(by_manifold[term_symbol] * tau, ratio, places=10)

    def test_single_manifold_without_branching_ratios(self):
        """87Rb 5P3/2 -> 5S1/2 (no branching ratios in config): every excited sublevel decays at 1/tau."""
        rb = AtomicStructure.from_species(species='87Rb', manifolds=['S1/2', 'P3/2'], magnetic_field=1.)
        excited = [level for level in levels_in_manifold(rb, 'P3/2') if level.mf >= 0]
        dissipator = DissipatorSpontaneousEmission.from_atomic_structure_data(StandardBasis([rb]), levels_in_manifold(rb, 'S1/2'), excited)
        tau = 26.24e-9
        for level in excited:
            with self.subTest(excited=level.name):
                self.assertAlmostEqual(decay_by_manifold(rb, dissipator, level)['S1/2'] * tau, 1., places=10)

    def test_cycling_transition(self):
        """87Rb |F'=3, mF'=3> decays only to |F=2, mF=2>."""
        rb = AtomicStructure.from_species(species='87Rb', manifolds=['S1/2', 'P3/2'])
        cycling = next(level for level in rb.energy_levels if level.name == 'P3/2,3,3')
        dissipator = DissipatorSpontaneousEmission.from_atomic_structure_data(StandardBasis([rb]), levels_in_manifold(rb, 'S1/2'), [cycling])
        self.assertEqual(len(dissipator.operators), 1)
        rates = decay_rate_matrix(dissipator)[:, rb.energy_levels.index(cycling)]
        target = next(i for i, level in enumerate(rb.energy_levels) if level.name == 'S1/2,2,2')
        self.assertAlmostEqual(rates[target] * 26.24e-9, 1., places=10)

    def test_truncated_ground_manifold_is_not_renormalized(self):
        """87Rb |F'=1, mF'=0> with only the F=2 ground levels: rate is the true 1/6 of 1/tau, not 1/tau."""
        rb = AtomicStructure.from_species(species='87Rb', manifolds=['S1/2', 'P3/2'])
        excited = next(level for level in rb.energy_levels if level.name == 'P3/2,1,0')
        ground = [level for level in levels_in_manifold(rb, 'S1/2') if level.f == 2]
        dissipator = DissipatorSpontaneousEmission.from_atomic_structure_data(StandardBasis([rb]), ground, [excited])
        self.assertAlmostEqual(decay_by_manifold(rb, dissipator, excited)['S1/2'] * 26.24e-9, 1/6, places=10)

    def test_mixed_bases(self):
        """87Rb 53S1/2 |mJ, mI> Rydberg level -> complete 6P3/2 |F, mF> manifold: total rate 1/tau."""
        six_p = [{'n': 6, 'l': 1, 'f': f, 'mf': mf} for f in (0, 1, 2, 3) for mf in np.arange(-f, f + 1)]
        rydberg = [{'n': 53, 'mj': 0.5, 'mi': 1.5}, {'n': 53, 'mj': -0.5, 'mi': 0.5}]
        rb = AtomicStructure.from_species(species='87Rb', manifolds=['6 P3/2', '53 S1/2'], quantum_numbers=six_p + rydberg,
                                          magnetic_field=13.6)
        dissipator = DissipatorSpontaneousEmission.from_atomic_structure_data(
            StandardBasis([rb]), levels_in_manifold(rb, '6 P3/2'), levels_in_manifold(rb, '53 S1/2'))
        for excited in levels_in_manifold(rb, '53 S1/2'):
            with self.subTest(excited=excited.name):
                self.assertAlmostEqual(decay_by_manifold(rb, dissipator, excited)['6 P3/2'] * 72.0e-6, 1., places=10)

    def test_sparse_matches_dense(self):
        ca = AtomicStructure.from_species(species='40Ca+', manifolds=['S1/2', 'D3/2', 'P1/2'])
        args = (StandardBasis([ca]), levels_in_manifold(ca, 'S1/2') + levels_in_manifold(ca, 'D3/2'), levels_in_manifold(ca, 'P1/2'))
        dense = DissipatorSpontaneousEmission.from_atomic_structure_data(*args)
        sparse = DissipatorSpontaneousEmission.from_atomic_structure_data(*args, sparse=True)
        np.testing.assert_allclose(decay_rate_matrix(sparse), decay_rate_matrix(dense), rtol=1e-12)


class TestSpontaneousEmissionMultipleIons(unittest.TestCase):

    def setUp(self):
        # Two separately built (distinct but identical) ions.
        self.ion_a = AtomicStructure.from_species(species='40Ca+', manifolds=['S1/2', 'P1/2'], name='a')
        self.ion_b = AtomicStructure.from_species(species='40Ca+', manifolds=['S1/2', 'P1/2'], name='b')
        self.basis = StandardBasis([self.ion_a, self.ion_b])
        self.rate = 0.93565 / 6.904e-9  # P1/2 -> S1/2 only

    def _total_decay_out_of(self, dissipator):
        """Total decay rate out of each two-ion basis state, keyed by (ion a excited, ion b excited)."""
        totals = decay_rate_matrix(dissipator).sum(axis=0)
        result = {}
        for state, total in zip(self.basis.states, totals):
            a_excited, b_excited = (component.term_symbol == 'P1/2' for component in state.components)
            result.setdefault((a_excited, b_excited), set()).add(round(total / self.rate, 10))
        return result

    def test_decay_added_to_every_ion(self):
        dissipator = DissipatorSpontaneousEmission.from_atomic_structure_data(
            self.basis, levels_in_manifold(self.ion_a, 'S1/2'), levels_in_manifold(self.ion_a, 'P1/2'))
        self.assertEqual(len(dissipator.operators), 8)  # 4 paths per ion
        totals = self._total_decay_out_of(dissipator)
        self.assertEqual(totals[(False, False)], {0.})
        self.assertEqual(totals[(True, False)], {1.})
        self.assertEqual(totals[(False, True)], {1.})
        self.assertEqual(totals[(True, True)], {2.})

    def test_select_dofs(self):
        dissipator = DissipatorSpontaneousEmission.from_atomic_structure_data(
            self.basis, levels_in_manifold(self.ion_a, 'S1/2'), levels_in_manifold(self.ion_a, 'P1/2'), select_DOFs=[self.ion_b])
        totals = self._total_decay_out_of(dissipator)
        self.assertEqual(totals[(True, False)], {0.})
        self.assertEqual(totals[(False, True)], {1.})


class TestSpontaneousEmissionErrors(unittest.TestCase):

    def assert_raises_building(self, species, manifolds, ground_terms, excited_terms, **kwargs):
        structure = AtomicStructure.from_species(species=species, manifolds=manifolds)
        ground = [level for term in ground_terms for level in levels_in_manifold(structure, term)]
        excited = [level for term in excited_terms for level in levels_in_manifold(structure, term)]
        with self.assertRaises(IonSimError):
            DissipatorSpontaneousEmission.from_atomic_structure_data(StandardBasis([structure]), ground, excited, **kwargs)

    def test_no_branching_ratios_with_several_manifolds(self):
        """87Rb 6P3/2 has no branching ratios; decay to both 5S1/2 and 5P3/2 would be ambiguous."""
        self.assert_raises_building('87Rb', ['S1/2', 'P3/2', '6 P3/2'], ['S1/2', 'P3/2'], ['6 P3/2'])

    def test_missing_lifetime(self):
        """40Ca+ S1/2 has lifetime null."""
        self.assert_raises_building('40Ca+', ['S1/2', 'D3/2'], ['D3/2'], ['S1/2'])

    def test_non_dipole_channel(self):
        """171Yb+ D5/2 -> S1/2 is an E2 decay with Delta J = 2: no dipole paths, so it must not vanish silently."""
        self.assert_raises_building('171Yb+', ['S1/2', 'F7/2', 'D5/2'], ['S1/2', 'F7/2'], ['D5/2'])

    def test_ground_level_above_excited_level(self):
        """87Rb 5P1/2 lies below 5P3/2, so it cannot decay to it."""
        self.assert_raises_building('87Rb', ['P1/2', 'P3/2'], ['P3/2'], ['P1/2'])

    def test_level_not_in_basis(self):
        ca = AtomicStructure.from_species(species='40Ca+', manifolds=['S1/2', 'P1/2'])
        rb = AtomicStructure.from_species(species='87Rb', manifolds=['S1/2', 'P3/2'])
        with self.assertRaises(IonSimError):
            DissipatorSpontaneousEmission.from_atomic_structure_data(StandardBasis([ca]), levels_in_manifold(rb, 'S1/2'), levels_in_manifold(rb, 'P3/2'))

    def test_select_dofs_not_in_basis(self):
        ca = AtomicStructure.from_species(species='40Ca+', manifolds=['S1/2', 'P1/2'])
        other = AtomicStructure.from_species(species='40Ca+', manifolds=['S1/2', 'P1/2'])
        with self.assertRaises(IonSimError):
            DissipatorSpontaneousEmission.from_atomic_structure_data(
                StandardBasis([ca]), levels_in_manifold(ca, 'S1/2'), levels_in_manifold(ca, 'P1/2'), select_DOFs=[other])

    def test_lindbladian_needs_hamiltonian_or_dissipator(self):
        with self.assertRaises(IonSimError):
            Lindbladian(None, None)


class TestSpontaneousEmissionSink(unittest.TestCase):
    """decay_to_sink=True: decay that does not reach the given ground levels goes to the SinkLevel."""

    @staticmethod
    def _rydberg_structure(branching_ratios):
        """87Rb complete 6P3/2 |F, mF> manifold, one 53S1/2 |mJ, mI> level with the given branching ratios, and a sink."""
        six_p = [{'n': 6, 'l': 1, 'f': f, 'mf': mf} for f in (0, 1, 2, 3) for mf in np.arange(-f, f + 1)]
        rb = AtomicStructure.from_species(species='87Rb', manifolds=['6 P3/2', '53 S1/2'], magnetic_field=13.6,
                                          quantum_numbers=six_p + [{'n': 53, 'mj': 0.5, 'mi': 1.5}], include_sink=True)
        return AtomicStructure([replace(level, branching_ratios=branching_ratios) if getattr(level, 'n', None) == 53 else level
                                for level in rb.energy_levels])

    def test_include_sink(self):
        ca = AtomicStructure.from_species(species='40Ca+', manifolds=['S1/2', 'P1/2'], include_sink=True)
        self.assertEqual([level.name for level in ca.energy_levels], ['S1/2,-1/2', 'S1/2,1/2', 'P1/2,-1/2', 'P1/2,1/2', 'sink'])

    def test_rydberg_remainder_goes_to_sink(self):
        """53S1/2 with 10% branching to 6P3/2: 0.1/tau to 6P3/2, 0.9/tau to the sink, 1/tau in total."""
        rb = self._rydberg_structure({'6 P3/2': 0.1})
        rydberg = levels_in_manifold(rb, '53 S1/2')
        dissipator = DissipatorSpontaneousEmission.from_atomic_structure_data(
            StandardBasis([rb]), levels_in_manifold(rb, '6 P3/2'), rydberg, decay_to_sink=True)
        by_manifold = decay_by_manifold(rb, dissipator, rydberg[0])
        tau = 72.0e-6
        self.assertAlmostEqual(by_manifold['6 P3/2'] * tau, 0.1, places=10)
        sink_rate = decay_rate_matrix(dissipator)[rb.energy_levels.index(rb.energy_levels[-1]), rb.energy_levels.index(rydberg[0])]
        self.assertAlmostEqual(sink_rate * tau, 0.9, places=10)

    def test_without_sink_excited_level_decays_slower(self):
        """Same structure without decay_to_sink: only the modeled 0.1/tau, no decay to the sink."""
        rb = self._rydberg_structure({'6 P3/2': 0.1})
        rydberg = levels_in_manifold(rb, '53 S1/2')
        dissipator = DissipatorSpontaneousEmission.from_atomic_structure_data(StandardBasis([rb]), levels_in_manifold(rb, '6 P3/2'), rydberg)
        total = decay_rate_matrix(dissipator)[:, rb.energy_levels.index(rydberg[0])].sum()
        self.assertAlmostEqual(total * 72.0e-6, 0.1, places=10)

    def test_truncated_manifold_remainder_goes_to_sink(self):
        """87Rb |F'=1, mF'=0> with only F=2 ground levels: 1/6 to F=2, 5/6 to the sink."""
        rb = AtomicStructure.from_species(species='87Rb', manifolds=['S1/2', 'P3/2'], include_sink=True)
        excited = next(level for level in rb.energy_levels if level.name == 'P3/2,1,0')
        ground = [level for level in levels_in_manifold(rb, 'S1/2') if level.f == 2]
        dissipator = DissipatorSpontaneousEmission.from_atomic_structure_data(StandardBasis([rb]), ground, [excited],
                                                                              decay_to_sink=True)
        column = decay_rate_matrix(dissipator)[:, rb.energy_levels.index(excited)] * 26.24e-9
        self.assertAlmostEqual(column[-1], 5/6, places=10)
        self.assertAlmostEqual(column.sum(), 1., places=10)

    def test_complete_channels_add_no_sink_operator(self):
        """40Ca+ P1/2 with both of its decay manifolds included: nothing is left over for the sink."""
        ca = AtomicStructure.from_species(species='40Ca+', manifolds=['S1/2', 'D3/2', 'P1/2'], include_sink=True)
        args = (StandardBasis([ca]), levels_in_manifold(ca, 'S1/2') + levels_in_manifold(ca, 'D3/2'), levels_in_manifold(ca, 'P1/2'))
        with_sink = DissipatorSpontaneousEmission.from_atomic_structure_data(*args, decay_to_sink=True)
        without_sink = DissipatorSpontaneousEmission.from_atomic_structure_data(*args)
        self.assertEqual(len(with_sink.operators), len(without_sink.operators))
        self.assertAlmostEqual(decay_rate_matrix(with_sink)[-1].sum(), 0.)

    def test_all_decay_to_sink(self):
        """No ground levels: the excited level decays entirely to the sink at 1/tau."""
        ca = AtomicStructure.from_species(species='40Ca+', manifolds=['S1/2', 'P1/2'], include_sink=True)
        excited = levels_in_manifold(ca, 'P1/2')
        dissipator = DissipatorSpontaneousEmission.from_atomic_structure_data(StandardBasis([ca]), [], excited, decay_to_sink=True)
        rates = decay_rate_matrix(dissipator)
        for level in excited:
            self.assertAlmostEqual(rates[-1, ca.energy_levels.index(level)] * 6.904e-9, 1., places=10)

    def test_every_ion_uses_its_own_sink(self):
        ion_a = AtomicStructure.from_species(species='40Ca+', manifolds=['S1/2', 'P1/2'], include_sink=True)
        ion_b = AtomicStructure.from_species(species='40Ca+', manifolds=['S1/2', 'P1/2'], include_sink=True)
        basis = StandardBasis([ion_a, ion_b])
        dissipator = DissipatorSpontaneousEmission.from_atomic_structure_data(
            basis, levels_in_manifold(ion_a, 'S1/2'), levels_in_manifold(ion_a, 'P1/2'), decay_to_sink=True)
        totals = decay_rate_matrix(dissipator).sum(axis=0) * 6.904e-9
        for state, total in zip(basis.states, totals):
            n_excited = sum(getattr(component, 'term_symbol', None) == 'P1/2' for component in state.components)
            with self.subTest(state=state.name):
                self.assertAlmostEqual(total, n_excited, places=10)

    def test_sink_required(self):
        ca = AtomicStructure.from_species(species='40Ca+', manifolds=['S1/2', 'P1/2'])
        with self.assertRaises(IonSimError):
            DissipatorSpontaneousEmission.from_atomic_structure_data(
                StandardBasis([ca]), levels_in_manifold(ca, 'S1/2'), levels_in_manifold(ca, 'P1/2'), decay_to_sink=True)

    def test_no_branching_ratios_means_single_channel(self):
        """87Rb 53S1/2 has no branching ratios in the config: it is assumed to decay only to 6P3/2, so nothing reaches the sink."""
        rb = self._rydberg_structure(None)
        dissipator = DissipatorSpontaneousEmission.from_atomic_structure_data(
            StandardBasis([rb]), levels_in_manifold(rb, '6 P3/2'), levels_in_manifold(rb, '53 S1/2'), decay_to_sink=True)
        by_manifold = decay_by_manifold(rb, dissipator, levels_in_manifold(rb, '53 S1/2')[0])
        self.assertAlmostEqual(by_manifold['6 P3/2'] * 72.0e-6, 1., places=10)
        self.assertNotIn('sink', by_manifold)

if __name__ == '__main__':
    unittest.main()
