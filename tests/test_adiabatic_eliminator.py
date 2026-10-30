#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

import dataclasses
import unittest
import warnings

import numpy as np
from scipy import constants as const
from scipy.integrate import solve_ivp

from ionsim.adiabatic_eliminator import adiabatically_eliminate
from ionsim.atomic_internal_energy_level import reduced_dipole_matrix_element
from ionsim.basis import StandardBasis
from ionsim.degree_of_freedom import AtomicStructure
from ionsim.hamiltonian import Hamiltonian
from ionsim.ionsim_error import IonSimError
from ionsim.laser import Laser, Polarization
from ionsim.lindbladian import DissipatorSpontaneousEmission
from ionsim.operator import CouplingOperator, EnergyShiftOperator, GeneralOperator

TPI = 2. * np.pi
X, Y, Z = np.eye(3)


def evolve_density_matrix(hamiltonian, dissipator, rho0, times):
    """ Lindblad master equation with tight tolerances. Each Lindblad operator's frame-shifted matrices are computed once. """
    parts = []
    for op in (dissipator.operators if dissipator is not None else []):
        static = np.zeros((dissipator.size,) * 2, dtype=complex)
        if isinstance(op, GeneralOperator) and op.energy_shifts:
            static += op.energy_shift_operator_contribution.static_matrix.toarray()
        elif isinstance(op, EnergyShiftOperator):
            static += op.static_matrix.toarray()
        L, R = np.zeros_like(static), np.zeros(static.shape)
        if isinstance(op, (CouplingOperator, GeneralOperator)) and op.couplings:
            L, R = (m.toarray() for m in dissipator._frame_shifted_coupling_matrix_and_rate_from_operator(op))
        parts.append((static, L, R))
    n = rho0.shape[0]
    H_of_t = hamiltonian.hamiltonian_function

    def rhs(t, y):
        rho = y.reshape(n, n)
        H = np.asarray(H_of_t(t))
        drho = -1j * (H @ rho - rho @ H)
        for static, L, R in parts:
            Lt = static + L * np.exp(-1j * R * t)
            LdL = Lt.conj().T @ Lt
            drho += Lt @ rho @ Lt.conj().T - 0.5 * (LdL @ rho + rho @ LdL)
        return drho.ravel()

    solution = solve_ivp(rhs, (times[0], times[-1]), rho0.astype(complex).ravel(), t_eval=times, method='DOP853', rtol=1e-9, atol=1e-11)
    assert solution.success, solution.message
    return [y.reshape(n, n) for y in solution.y.T]


class TestLambdaSystem(unittest.TestCase):
    """ Hand-built Lambda system: two kept S1/2 clock states and one eliminated P1/2 sublevel of 171Yb+. """

    def setUp(self):
        self.atom = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2'],
                                                 level_names=['S1/2,0,0', 'S1/2,1,0', 'P1/2,1,1'])
        self.g0, self.g1, self.e = self.atom.energy_levels
        self.basis = StandardBasis([self.atom])
        self.E = [state.energy for state in self.basis.states]
        self.Delta, self.delta = TPI * 1e9, TPI * 2e6            # single-photon detuning (blue), two-photon detuning
        self.O1, self.O2 = TPI * 20e6, TPI * 30e6 * np.exp(0.7j)  # Rabi frequencies; couplings are Omega/2
        # Laser frequencies: |g0> -> |e> at Delta, |g1> -> |e> at Delta + delta
        self.w1 = self.E[2] - self.E[0] + self.Delta
        self.w2 = self.E[2] - self.E[1] + self.Delta + self.delta

    def operator(self, ground_index, rabi, frequency):
        """ (Omega/2) e^{-i w t} |e><g| + h.c., in IonSim's convention (stored |g><e| element with rate -w). """
        matrix = np.zeros((3, 3), dtype=complex)
        matrix[2, ground_index] = rabi / 2.
        matrix[ground_index, 2] = np.conj(rabi) / 2.
        return CouplingOperator.from_matrix(self.basis, matrix, -frequency)

    def hamiltonian(self, frame):
        return Hamiltonian(self.basis, [self.operator(0, self.O1, self.w1), self.operator(1, self.O2, self.w2)], frame)

    def test_effective_couplings_and_light_shifts(self):
        """ Light shifts |Omega|^2/(4 Delta) (positive for blue detuning) and the two-photon coupling
            conj(Omega_1) Omega_2 / 8 (1/Delta_1 + 1/Delta_2) oscillating at the two-photon detuning. """
        frame = [-energy for energy in self.E]
        result = self.hamiltonian(frame).adiabatically_eliminate([self.e])
        self.assertEqual([s.name for s in result.reduced_basis.states], ['S1/2,0,0', 'S1/2,1,0'])
        Delta1, Delta2 = self.Delta, self.Delta + self.delta
        np.testing.assert_allclose(result.light_shifts, [abs(self.O1)**2 / (4 * Delta1), abs(self.O2)**2 / (4 * Delta2)], rtol=1e-10)
        expected = np.conj(self.O1) * self.O2 / 8. * (1. / Delta1 + 1. / Delta2)
        for t in (0., 1.3e-7, 4.1e-7):
            H = np.asarray(result.hamiltonian.hamiltonian_function(t))
            # <g0|H|g1> = <g0|H|e><e|H|g1> / Delta oscillates at the two-photon detuning (the rates are differences of
            # optical frequencies, so the phase is accurate to ~1e-7 at these times)
            np.testing.assert_allclose(H[0, 1], expected * np.exp(-1j * self.delta * t), rtol=1e-6)
            full = np.asarray(self.hamiltonian(frame).hamiltonian_function(t))
            np.testing.assert_allclose(H[0, 1], full[0, 2] * full[2, 1] / 2. * (1. / Delta1 + 1. / Delta2), rtol=1e-6)
            np.testing.assert_allclose(H, H.conj().T, atol=1e-6)
        self.assertAlmostEqual(result.max_perturbation_ratio, abs(self.O2) / 2. / Delta2, places=12)
        self.assertEqual(result.dissipator, None)

    def test_frame_independence(self):
        """ The effective Hamiltonian is the same physical operator in any frame: compare after transforming frames. """
        lab = adiabatically_eliminate(self.hamiltonian([0.] * 3), [self.e])
        rotating = adiabatically_eliminate(self.hamiltonian([-energy for energy in self.E]), [self.e])
        np.testing.assert_allclose(lab.light_shifts, rotating.light_shifts, rtol=1e-9)
        t = 2.7e-7
        H_lab = np.asarray(lab.hamiltonian.hamiltonian_function(t))
        H_rot = np.asarray(rotating.hamiltonian.hamiltonian_function(t))
        # The rotating frame f = -E is the lab frame transformed by U = exp(i diag(E) t): off-diagonals map as U H U^dagger.
        U = np.diag(np.exp(1j * np.array(self.E[:2]) * t))
        np.testing.assert_allclose((U @ H_lab @ U.conj().T)[0, 1], H_rot[0, 1], rtol=1e-6)

    def test_oscillating_shifts_drop_and_keep(self):
        """ Two lasers on the same |g> -> |e> transition give a diagonal beat term a e^{-iwt} + c.c.: dropped by default,
            kept as a modulated energy shift on request. """
        frame = [-energy for energy in self.E]
        beat = TPI * 5e6
        H = Hamiltonian(self.basis, [self.operator(0, self.O1, self.w1), self.operator(0, self.O1, self.w1 + beat)], frame)
        dropped = H.adiabatically_eliminate([self.e])
        self.assertEqual(len(dropped.dropped_oscillating_shifts), 1)
        state, amplitude, rate = dropped.dropped_oscillating_shifts[0]
        self.assertAlmostEqual(rate / beat, 1., places=6)  # rates are differences of optical frequencies
        kept = H.adiabatically_eliminate([self.e], oscillating_shifts='keep')
        self.assertEqual(kept.dropped_oscillating_shifts, [])
        for t in (0., 3.3e-8, 1.1e-7):
            diag_drop = np.real(np.asarray(dropped.hamiltonian.hamiltonian_function(t))[0, 0])
            diag_keep = np.real(np.asarray(kept.hamiltonian.hamiltonian_function(t))[0, 0])
            self.assertAlmostEqual((diag_keep - diag_drop) / abs(amplitude), 2. * np.cos(rate * t - np.angle(amplitude)), places=6)
            # and it is the beat between the two drives, |O1|^2/(4 Delta) e^{-i beat t} + c.c. (for beat << Delta)
            self.assertAlmostEqual(abs(amplitude) / (abs(self.O1)**2 / (4. * self.Delta)), 1., delta=2. * beat / self.Delta)

    def test_cross_couplings_and_rwa_cutoff(self):
        """ With each laser also driving the other leg, the effective Hamiltonian gains cross terms at about twice the
            ground splitting; rwa_rate_cutoff drops (and reports) them, leaving the resonant Raman coupling. """
        frame = [-energy for energy in self.E]
        operators = [self.operator(0, self.O1, self.w1), self.operator(1, self.O2, self.w2),
                     self.operator(1, self.O1, self.w1), self.operator(0, self.O2, self.w2)]
        H = Hamiltonian(self.basis, operators, frame)
        full = H.adiabatically_eliminate([self.e])
        cut = H.adiabatically_eliminate([self.e], rwa_rate_cutoff=TPI * 1e9)
        self.assertEqual(full.dropped_fast_terms, [])
        splitting = self.E[1] - self.E[0]
        fast_rates = sorted(abs(rate) for *_, rate in cut.dropped_fast_terms)
        self.assertGreater(len(fast_rates), 0)
        for rate in fast_rates:
            self.assertTrue(abs(rate / splitting - 1.) < 0.01 or abs(rate / (2. * splitting) - 1.) < 0.01, rate / splitting)
        expected = np.conj(self.O1) * self.O2 / 8. * (1. / self.Delta + 1. / (self.Delta + self.delta))
        for t in (0., 2.1e-7):
            np.testing.assert_allclose(np.asarray(cut.hamiltonian.hamiltonian_function(t))[0, 1], expected * np.exp(-1j * self.delta * t), rtol=1e-6)
            self.assertGreater(abs(np.asarray(full.hamiltonian.hamiltonian_function(t))[0, 1] - expected * np.exp(-1j * self.delta * t)),
                               1e-3 * abs(expected))
        # light shifts now include the off-resonant leg of each laser: |O|^2/(4 Delta') with Delta' shifted by the splitting
        expected_shift_0 = abs(self.O1)**2 / (4 * self.Delta) + abs(self.O2)**2 / (4 * (self.Delta + self.delta - splitting))
        self.assertAlmostEqual(cut.light_shifts[0] / expected_shift_0, 1., places=6)

    def test_input_validation(self):
        frame = [-energy for energy in self.E]
        H = self.hamiltonian(frame)
        other = AtomicStructure.from_species(species='171Yb+', term_symbols=['P3/2'], level_names=['P3/2,2,2'])
        with self.assertRaises(IonSimError):
            H.adiabatically_eliminate(other.energy_levels)
        with self.assertRaises(IonSimError):
            H.adiabatically_eliminate([self.g0, self.g1, self.e])
        with self.assertRaises(IonSimError):
            H.adiabatically_eliminate([self.e], oscillating_shifts='average')
        # couplings among eliminated states
        matrix = np.zeros((3, 3)); matrix[0, 1] = matrix[1, 0] = TPI * 1e5
        with self.assertRaises(NotImplementedError):
            Hamiltonian(self.basis, [self.operator(0, self.O1, self.w1), CouplingOperator.from_matrix(self.basis, matrix, 0.)],
                        frame).adiabatically_eliminate([self.g0, self.e])
        # modulated drives
        modulated = dataclasses.replace(self.operator(0, self.O1, self.w1), modulation_function=lambda t: 1.)
        with self.assertRaises(NotImplementedError):
            Hamiltonian(self.basis, [modulated], frame).adiabatically_eliminate([self.e])
        # dissipator in a different frame
        D = DissipatorSpontaneousEmission.from_atomic_structure_data(self.basis, [self.g0, self.g1], [self.e], [0.] * 3)
        with self.assertRaises(IonSimError):
            H.adiabatically_eliminate([self.e], D)
        # validity warning
        strong = Hamiltonian(self.basis, [self.operator(0, self.Delta, self.w1)], frame)
        with self.assertWarns(UserWarning):
            strong.adiabatically_eliminate([self.e])


class TestFullVersusEffective(unittest.TestCase):
    """ Neutral 171Yb, 556 nm Raman transition between the 1S0 nuclear-spin states through 3P1, detuned by 1 GHz from
        F' = 3/2. The 3P1 lifetime is shortened 1000x (Gamma = 2 pi x 182 MHz) so that scattering is significant within
        the simulated time and the complex denominators (Gamma/Delta ~ 0.2) are tested. """

    def test_master_equation_dynamics(self):
        base = AtomicStructure.from_species(species='171Yb', term_symbols=['S0', 'P1'])
        levels = [dataclasses.replace(l, lifetime=l.lifetime / 1000.) if l.term_symbol == 'P1' else l for l in base.energy_levels]
        atom = dataclasses.replace(base, energy_levels=levels)
        ground = [l for l in atom.energy_levels if l.term_symbol == 'S0']
        excited = [l for l in atom.energy_levels if l.term_symbol == 'P1']
        basis = StandardBasis([atom])
        frame = [-state.energy for state in basis.states]
        F32 = next(l for l in excited if l.name == 'P1,3/2,3/2')
        frequency = F32.energy - ground[0].energy + TPI * 1e9
        sigma = Laser.gaussian_from_frequency(frequency, 0.1e-3, 20e-6, Z, Polarization.circular(Z, '+'), 0.)
        pi = Laser.gaussian_from_frequency(frequency, 0.1e-3, 20e-6, X, Polarization.linear(X, 0., ref_axis=Z), 0.3)
        operators = []
        for laser in (sigma, pi):
            operators += laser.build_carrier_coupling_operators(basis, [atom], ground, excited, 1)
        H = Hamiltonian(basis, operators, frame)
        D = DissipatorSpontaneousEmission.from_atomic_structure_data(basis, ground, excited, frame, coherent_channels=True)
        result = H.adiabatically_eliminate(excited, D)
        self.assertLess(result.max_perturbation_ratio, 0.05)
        self.assertGreater(result.scattering_rate(0) * 0.5e-6, 0.1)  # > 0.1 scattering events in the simulated time

        times = np.linspace(0., 0.5e-6, 6)
        rho0 = np.zeros((len(basis.states),) * 2, dtype=complex); rho0[0, 0] = 1.
        full = evolve_density_matrix(H, D, rho0, times)
        effective = evolve_density_matrix(result.hamiltonian, result.dissipator, rho0[:2, :2], times)
        kept = [basis.states.index(state) for state in result.kept_states]
        self.assertEqual(kept, [0, 1])
        # the dynamics are nontrivial: population transfer and decoherence
        self.assertGreater(effective[-1][1, 1].real, 0.1)
        for rho_full, rho_eff in zip(full, effective):
            np.testing.assert_allclose(rho_full[np.ix_(kept, kept)], rho_eff, atol=2e-3)  # excited population ~5e-4
            self.assertAlmostEqual(np.trace(rho_eff).real, 1., places=7)


class TestYb355(unittest.TestCase):
    """ 171Yb+ at 355 nm (33 THz above P1/2, 67 THz below P3/2), pi-polarized beam, all S1/2, P1/2 and P3/2 sublevels.
        Detuned far beyond the hyperfine structure, the sublevel-averaged results follow from the reduced elements alone. """

    @classmethod
    def setUpClass(cls):
        cls.atom = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2', 'P3/2'])
        cls.ground = [l for l in cls.atom.energy_levels if l.term_symbol == 'S1/2']
        cls.excited = [l for l in cls.atom.energy_levels if l.term_symbol != 'S1/2']
        cls.basis = StandardBasis([cls.atom])
        cls.frame = [-state.energy for state in cls.basis.states]
        cls.frequency = TPI * const.c / 355e-9
        cls.laser = Laser.gaussian_from_frequency(cls.frequency, 1., 20e-6, X, Polarization.linear(X, 0., ref_axis=Z), 0.)
        cls.H = Hamiltonian(cls.basis, cls.laser.build_carrier_coupling_operators(cls.basis, [cls.atom], cls.ground, cls.excited, 1), cls.frame)
        cls.results = {}
        for coherent in (True, False):
            D = DissipatorSpontaneousEmission.from_atomic_structure_data(cls.basis, cls.ground, cls.excited, cls.frame,
                                                                         coherent_channels=coherent)
            cls.results[coherent] = cls.H.adiabatically_eliminate(cls.excited, D)

    def closed_forms(self):
        """ Scalar light shift sum_J Omega_J^2 / (12 Delta_J) and scattering rate sum_J Gamma_J Omega_J^2 / (12 Delta_J^2),
            with Omega_J = E0 <S||er||P_J> / hbar (1/3 of the line strength for pi light out of J = 1/2). """
        E0 = self.laser.peak_electric_field_magnitude
        s = self.ground[0]
        shift = scatter = 0.
        for J in ('P1/2', 'P3/2'):
            e = next(l for l in self.excited if l.term_symbol == J)
            rabi = E0 * const.e * const.value('Bohr radius') * reduced_dipole_matrix_element(s, e) / const.hbar
            detuning = self.frequency - (e.fine_energy - s.fine_energy)
            shift += rabi**2 / 12. / detuning
            scatter += e.branching_ratios['S1/2'] / e.lifetime * rabi**2 / 12. / detuning**2
        return shift, scatter

    def test_light_shift_and_scattering_rate(self):
        shift, scatter = self.closed_forms()
        for result in self.results.values():
            self.assertLess(result.max_perturbation_ratio, 1e-3)
            # the hyperfine corrections average out over the S1/2 sublevels at first order in omega_HF / Delta
            self.assertAlmostEqual(np.mean(result.light_shifts) / shift, 1., delta=1e-4)
            self.assertAlmostEqual(np.mean([result.scattering_rate(i) for i in range(4)]) / scatter, 1., delta=1e-4)

    def test_coherent_channels_suppress_rayleigh_dephasing(self):
        """ The total scattering rate does not depend on the decay-channel representation, but the Rayleigh/Raman split
            does: with coherent channels the elastic amplitudes of the clock states are nearly equal and the elastic
            dephasing nearly vanishes; per-path (secular) operators give which-path information and spurious dephasing. """
        coherent, secular = self.results[True], self.results[False]
        names = [s.name for s in coherent.reduced_basis.states]
        a, b = names.index('S1/2,0,0'), names.index('S1/2,1,0')
        self.assertAlmostEqual(coherent.scattering_rate(a) / secular.scattering_rate(a), 1., places=10)
        self.assertLess(coherent.elastic_dephasing_rate(a, b), 1e-5 * coherent.scattering_rate(a))
        self.assertGreater(secular.elastic_dephasing_rate(a, b), 0.1 * secular.scattering_rate(a))
        np.testing.assert_allclose(coherent.light_shifts, secular.light_shifts, rtol=1e-12)


class TestDecayChannels(unittest.TestCase):

    def test_coherent_channels_reproduce_decay_rates(self):
        atom = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2', 'P3/2'])
        basis = StandardBasis([atom])
        ground = [l for l in atom.energy_levels if l.term_symbol == 'S1/2']
        excited = [l for l in atom.energy_levels if l.term_symbol != 'S1/2']
        def total(dissipator):
            return sum(op.static_matrix.toarray().conj().T @ op.static_matrix.toarray() for op in dissipator.operators)
        secular = DissipatorSpontaneousEmission.from_atomic_structure_data(basis, ground, excited)
        coherent = DissipatorSpontaneousEmission.from_atomic_structure_data(basis, ground, excited, coherent_channels=True)
        self.assertEqual(len(coherent.operators), 3)  # q = -1, 0, +1 into S1/2
        np.testing.assert_allclose(np.diag(total(coherent)), np.diag(total(secular)), rtol=1e-12)
        # complete ground manifold: the channels are orthogonal on the excited states
        T = total(coherent)
        np.testing.assert_allclose(T - np.diag(np.diag(T)), 0., atol=1e-9 * np.max(np.abs(T)))

    def test_derived_reduced_dipole_elements(self):
        """ Without tabulated values, <S1/2||er||P_J> follows from lifetime and branching ratio (Steck Eq. 38); E2 decays
            are not derived, and tabulated values take precedence. """
        atom = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2', 'D3/2'])
        manifold = {}
        for level in atom.energy_levels:
            manifold.setdefault(level.term_symbol, level)
        for lower in ('S1/2', 'D3/2'):
            g, e = manifold[lower], manifold['P1/2']
            reduced = reduced_dipole_matrix_element(g, e) * const.e * const.value('Bohr radius')
            omega_0 = e.fine_energy - g.fine_energy
            rate = omega_0**3 / (3. * np.pi * const.epsilon_0 * const.hbar * const.c**3) * (2. * g.j + 1.) / (2. * e.j + 1.) * reduced**2
            self.assertAlmostEqual(rate * e.lifetime / e.branching_ratios[lower], 1., places=10)
            self.assertGreater(reduced, 0.)
        self.assertIsNone(reduced_dipole_matrix_element(manifold['S1/2'], manifold['D3/2']))
        tabulated = dataclasses.replace(manifold['P1/2'], reduced_dipole_matrix_elements={'S1/2': 1.234})
        self.assertEqual(reduced_dipole_matrix_element(manifold['S1/2'], tabulated), 1.234)


if __name__ == '__main__':
    unittest.main()
