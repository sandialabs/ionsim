#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

""" Density-matrix evolution of the Lindblad master equation, and the ODE solver layer. """

import unittest
from unittest import mock

import numpy as np
from scipy.sparse import issparse

from ionsim.basis import StandardBasis
from ionsim.custom_math import solve_ode, solve_time_evolution_equation
from ionsim.degree_of_freedom import AtomicStructure, MotionalMode
from ionsim.hamiltonian import Hamiltonian
from ionsim.ionsim_error import IonSimError
from ionsim.lindbladian import Dissipator, DissipatorSpontaneousEmission, Lindbladian
from ionsim.multi_operator import MultiOperator
from ionsim.named_operators import Pauli
from ionsim.operators import Coupling, CouplingOperator, EnergyShift, EnergyShiftOperator, GeneralOperator, adjoint
from ionsim.process import Gate
from ionsim.state import State

TPI = 2. * np.pi
ACCURATE = {'ode_solver': 'solve_ivp', 'method': 'DOP853', 'rtol': 1e-10, 'atol': 1e-12}

FULL_LEVELS = ('S1/2,0,0', 'S1/2,1,0', 'P1/2,1,0', 'P1/2,1,1')
SMALL_LEVELS = ('S1/2,0,0', 'P1/2,1,1')

def build_model(sparse: bool = False, levels: tuple = FULL_LEVELS, fock_dimension: int = 3):
    """ 171Yb+ levels x a motional mode (2 MHz), in the interaction picture (frame -E).

        Hamiltonian: drives S -> P detuned by a few MHz (one amplitude-modulated) and a modulated energy shift.
        Dissipator: spontaneous emission (one operator per path: a single rate each), a GeneralOperator with a diagonal
        element and motional couplings |n+1> -> |n> (frame rates of MHz, so it is time-dependent in the frame), and a
        pure-dephasing EnergyShiftOperator. All frame rates are MHz-scale, so the ODE is not stiff. """
    atom = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2', 'P1/2'], level_names=list(levels))
    ground = [l for l in atom.energy_levels if l.term_symbol == 'S1/2']
    excited = [l for l in atom.energy_levels if l.term_symbol == 'P1/2']
    if fock_dimension < 2:
        raise ValueError("The test model needs a motional mode for its time-dependent Lindblad operator.")
    mode = MotionalMode.from_frequency(TPI * 2e6, fock_dimension)
    basis = StandardBasis([atom, mode])
    n = len(basis.states)
    frame = [-state.energy for state in basis.states]

    def drive(g, e, rabi, detuning, modulation=None):
        matrix = np.zeros((len(levels), len(levels)), dtype=complex)
        i, j = atom.energy_levels.index(e), atom.energy_levels.index(g)
        matrix[i, j], matrix[j, i] = rabi / 2., np.conj(rabi) / 2.
        # stored element |g><e| (g first): rate -omega, so that the total rate in the frame is -(omega - omega_0)
        return CouplingOperator.from_matrix(basis, matrix, -(e.energy - g.energy + detuning), current_dofs=[atom], modulation_function=modulation)

    operators = [drive(ground[0], excited[-1], TPI * 4e6, TPI * 3e6, modulation=lambda t: 0.7 + 0.3 * np.cos(TPI * 1e6 * t)),
                 EnergyShiftOperator.from_vector(basis, np.linspace(0., TPI * 0.5e6, n), modulation_function=lambda t: np.sin(TPI * 2e6 * t))]
    if len(ground) > 1:
        operators.append(drive(ground[1], excited[-1], TPI * 3e6 * np.exp(0.4j), TPI * 2.5e6))
    hamiltonian = Hamiltonian(basis, operators, frame, sparse)

    emission = DissipatorSpontaneousEmission.from_atomic_structure_data(basis, ground, excited, frame)
    states = basis.states
    general = GeneralOperator(basis, [EnergyShift(row_state=states[0], column_state=states[0], strength=np.sqrt(TPI * 0.2e6)),
                                      Coupling(row_state=states[0], column_state=states[1],
                                               strength=np.sqrt(TPI * 0.3e6) * np.exp(0.2j), oscillation_rate=0.),
                                      Coupling(row_state=states[n - 2], column_state=states[n - 1],
                                               strength=np.sqrt(TPI * 0.1e6), oscillation_rate=0.)])
    dephasing = EnergyShiftOperator.from_vector(basis, np.sqrt(TPI * 0.05e6) * np.linspace(-1., 1., n))
    dissipator = Dissipator(basis, list(emission.operators) + [general, dephasing], frame, sparse)
    return basis, hamiltonian, dissipator


def mixed_density_matrix(n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    rho = A @ A.conj().T
    return rho / np.trace(rho)


class TestDensityMatrixEvolution(unittest.TestCase):

    TIMES = np.linspace(0., 0.25e-6, 5)

    def compare(self, lindbladian: Lindbladian, basis: StandardBasis, rho0: np.ndarray, atol: float = 1e-8):
        _, supervectors = lindbladian.evolve_supervector(basis.compute_supervector_from_density_matrix(rho0), self.TIMES[-1],
                                                         self.TIMES, **ACCURATE)
        times, rhos = lindbladian.evolve_density_matrix(rho0, self.TIMES[-1], self.TIMES, **ACCURATE)
        np.testing.assert_allclose(times, self.TIMES)
        for supervector, rho in zip(supervectors, rhos):
            np.testing.assert_allclose(rho, basis.compute_density_matrix_from_supervector(supervector), atol=atol)
        return rhos

    def test_matches_supervector_dense_and_sparse(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                basis, hamiltonian, dissipator = build_model(sparse)
                rhos = self.compare(Lindbladian(hamiltonian, dissipator), basis, mixed_density_matrix(len(basis.states)))
                self.assertAlmostEqual(np.trace(rhos[-1]).real, 1., places=8)
                np.testing.assert_allclose(rhos[-1], rhos[-1].conj().T, atol=1e-10)
                self.assertGreater(np.max(np.abs(rhos[-1] - rhos[0])), 1e-2)  # nontrivial dynamics

    def test_hamiltonian_only_and_dissipator_only(self):
        basis, hamiltonian, dissipator = build_model()
        rho0 = mixed_density_matrix(len(basis.states))
        for lindbladian in (Lindbladian(hamiltonian, None), Lindbladian(None, dissipator)):
            with self.subTest(hamiltonian=lindbladian.hamiltonian is not None):
                self.compare(lindbladian, basis, rho0)

    def test_non_hermitian_initial_operator(self):
        """ |i><j| inputs (used to build process matrices) evolve linearly; no Hermiticity is assumed. """
        basis, hamiltonian, dissipator = build_model()
        n = len(basis.states)
        rho0 = np.zeros((n, n), dtype=complex); rho0[0, 1] = 1.
        self.compare(Lindbladian(hamiltonian, dissipator), basis, rho0)

    def test_operator_classification(self):
        """ Unmodulated single-rate Lindblad operators form one group (precomputed as static); operators mixing rates in
            the frame have several groups. """
        _, hamiltonian, dissipator = build_model()
        parts = dissipator.lindblad_operator_parts
        self.assertTrue(all(p.is_single_group and p.groups[0][0] is None for p in parts[:-2]))  # emission paths
        self.assertEqual(len(parts[-2].groups), 2)  # GeneralOperator: diagonal element (rate 0) and couplings (2 MHz)
        self.assertTrue(parts[-1].is_single_group)  # diagonal dephasing
        self.assertEqual(parts[-1].groups[0][1], 0.)
        # The cached parts reproduce the Lindblad matrix functions at any time. Elements whose rates are equal up to
        # floating-point rounding (e.g. the decay paths |S, n><P, n| for every n: E_P - E_S from frame energies ~5e15 rad/s,
        # equal to ~1 rad/s) share one group and one rate, so their optical phases (~1e9 rad here) agree with the per-element
        # reference to ~(1 rad/s) t, not to machine precision.
        for t in (0., 1.7e-7):
            for p, L_of_t in zip(parts, dissipator.lindblad_matrix_functions):
                np.testing.assert_allclose(p.matrix(t), L_of_t(t), rtol=1e-5)

    def test_state_propagation_representations(self):
        basis, hamiltonian, dissipator = build_model(levels=SMALL_LEVELS, fock_dimension=2)
        lindbladian = Lindbladian(hamiltonian, dissipator)
        state = State.from_density_matrix(basis, mixed_density_matrix(len(basis.states)))
        final = {r: state.propagate_using_master_equation(lindbladian, 0.15e-6, representation=r, **ACCURATE)
                 for r in ('supervector', 'density_matrix')}
        np.testing.assert_allclose(final['density_matrix'].density_matrix, final['supervector'].density_matrix, atol=1e-8)
        series = state.propagate_using_master_equation(lindbladian, 0.15e-6, np.linspace(0., 0.15e-6, 4), representation='density_matrix', **ACCURATE)
        self.assertEqual(len(series), 4)
        np.testing.assert_allclose(series[-1].density_matrix, final['supervector'].density_matrix, atol=1e-8)
        with self.assertRaises(IonSimError):
            state.propagate_using_master_equation(lindbladian, 0.15e-6, representation='matrix')

    def test_gate_from_lindbladian(self):
        """ The representation option passes through Gate.from_lindbladian (which evolves every |i><j|). """
        basis, hamiltonian, dissipator = build_model(levels=SMALL_LEVELS, fock_dimension=2)
        lindbladian = Lindbladian(hamiltonian, dissipator)
        default = Gate.from_lindbladian(basis, lindbladian, 0.1e-6, **ACCURATE)
        matrix_form = Gate.from_lindbladian(basis, lindbladian, 0.1e-6, representation='density_matrix', **ACCURATE)
        np.testing.assert_allclose(matrix_form.process_matrix, default.process_matrix, atol=1e-8)


class TestPrecomputedSuperoperators(unittest.TestCase):
    """ The dissipator superoperator is precomputed as S_0 + sum_c S_c f_a(t) conj(f_b(t)) e^{-i w_c t}, and the Lindbladian
        evaluates H(t) once. """

    TIMES = (0., 3.7e-8, 1.9e-7, 4.4e-7)

    def test_matches_per_step_construction(self):
        for sparse in (False, True):
            basis, hamiltonian, dissipator = build_model(sparse)
            static, oscillating = dissipator.dissipator_superoperator_components
            # only the GeneralOperator mixes rates (rate 0 for its diagonal element, 2 MHz for its couplings)
            self.assertEqual(sorted(round(abs(w) / TPI / 1e6, 6) for w, *_ in oscillating), [2., 2.])
            for t in self.TIMES:
                with self.subTest(sparse=sparse, t=t):
                    reference = sum(Dissipator.lindblad_matrix_to_superoperator(L(t)) for L in dissipator.lindblad_matrix_functions)
                    D = dissipator.get_matrix(t)
                    self.assertEqual(issparse(D), sparse)
                    reference = reference.toarray() if issparse(reference) else reference
                    D = D.toarray() if issparse(D) else D
                    np.testing.assert_allclose(D, reference, atol=1e-6 * np.max(np.abs(reference)))

    def test_single_rate_operators_are_static(self):
        basis, hamiltonian, dissipator = build_model()
        single_rate = Dissipator(basis, dissipator.operators[:-2] + dissipator.operators[-1:], dissipator.rotating_frame_energies)
        static, oscillating = single_rate.dissipator_superoperator_components
        self.assertEqual(oscillating, [])
        np.testing.assert_allclose(single_rate.get_matrix(2.3e-7), static)

    def test_no_construction_per_evaluation(self):
        """ After the first evaluation, evaluating the Lindbladian builds no superoperators and evaluates H(t) once. """
        basis, hamiltonian, dissipator = build_model()
        lindbladian = Lindbladian(hamiltonian, dissipator)
        lindbladian.get_matrix(0.)
        calls = {'pair': 0, 'H': 0}
        original_get_matrix = Hamiltonian.get_matrix

        def counted_hamiltonian(self, t):
            calls['H'] += 1
            return original_get_matrix(self, t)

        original_pair = Dissipator.lindblad_pair_to_superoperator

        def counted_pair(*args):
            calls['pair'] += 1
            return original_pair(*args)

        fresh = Lindbladian(hamiltonian, dissipator)
        with mock.patch.object(Hamiltonian, 'get_matrix', counted_hamiltonian), \
             mock.patch.object(Dissipator, 'lindblad_pair_to_superoperator', staticmethod(counted_pair)):
            for t in self.TIMES:
                fresh.get_matrix(t)
        self.assertEqual(calls['pair'], 0)
        self.assertEqual(calls['H'], len(self.TIMES))


def build_modulated_model(sparse: bool = False):
    """ 171Yb+ clock qubit in its interaction frame, with modulated Lindblad operators (MultiOperator, from branch 58):
            L1 = sqrt(kappa) (C + C^dagger), C a resonant drive with amplitude and phase modulation f(t) (two groups:
                 C with f, adjoint(C) with conj(f), so time-dependent),
            L2 = sqrt(gamma) g(t) |0><1| (one group, modulated; its 12.6 GHz frame phase cancels in the dissipator),
            L3 = sqrt(gamma_phi) h(t) Z (diagonal, one modulated group).
        The Hamiltonian is the drive C + C^dagger. """
    qubit = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
    basis = StandardBasis([qubit])
    frame = [-state.energy for state in basis.states]
    qubit_frequency = qubit.energy_levels[1].energy - qubit.energy_levels[0].energy
    rabi = TPI * 1e6

    def f(t):
        return (1. + 0.3 * np.sin(TPI * 0.5e6 * t)) * np.exp(1j * 0.7 * np.sin(TPI * 0.8e6 * t))

    C = CouplingOperator.from_matrix(basis, 0.5 * rabi * basis.enlarge_matrix(Pauli.plus, [qubit]), qubit_frequency, modulation_function=f)
    hamiltonian = Hamiltonian(basis, [C, adjoint(C)], frame, sparse)
    L1 = MultiOperator(basis, [C, adjoint(C)], frame, sparse, prefactor=np.sqrt(0.05 / rabi))
    L2 = CouplingOperator.from_matrix(basis, np.sqrt(TPI * 0.1e6) * basis.enlarge_matrix(Pauli.minus, [qubit]), 0.,
                                      modulation_function=lambda t: 0.5 + 0.5 * np.cos(TPI * 0.3e6 * t))
    L3 = EnergyShiftOperator.from_matrix(basis, np.sqrt(TPI * 0.05e6) * basis.enlarge_matrix(Pauli.Z, [qubit]),
                                         modulation_function=lambda t: np.sin(TPI * 0.2e6 * t))
    return basis, hamiltonian, Dissipator(basis, [L1, L2, L3], frame, sparse)


class TestModulatedLindbladOperators(unittest.TestCase):
    """ Modulation functions and prefactors of Lindblad operators (MultiOperator) in both precomputed paths. """

    TIMES = (0., 1.3e-7, 4.1e-7, 9.7e-7)

    def test_parts_reproduce_lindblad_operators(self):
        for sparse in (False, True):
            _, _, dissipator = build_modulated_model(sparse)
            parts = dissipator.lindblad_operator_parts
            self.assertEqual([len(p.groups) for p in parts], [2, 1, 1])
            for t in self.TIMES:
                for p, L_of_t in zip(parts, dissipator.lindblad_matrix_functions):
                    with self.subTest(sparse=sparse, t=t):
                        L, reference = p.matrix(t), L_of_t(t)
                        L = L.toarray() if issparse(L) else L
                        reference = reference.toarray() if issparse(reference) else reference
                        np.testing.assert_allclose(L, reference, atol=1e-9 * np.max(np.abs(reference)) + 1e-14)

    def test_superoperator_matches_per_step_construction(self):
        for sparse in (False, True):
            _, _, dissipator = build_modulated_model(sparse)
            static, components = dissipator.dissipator_superoperator_components
            self.assertGreater(len(components), 0)
            for t in self.TIMES:
                with self.subTest(sparse=sparse, t=t):
                    reference = sum(Dissipator.lindblad_matrix_to_superoperator(L(t)) for L in dissipator.lindblad_matrix_functions)
                    D = dissipator.get_matrix(t)
                    self.assertEqual(issparse(D), sparse)
                    reference = reference.toarray() if issparse(reference) else reference
                    D = D.toarray() if issparse(D) else D
                    np.testing.assert_allclose(D, reference, atol=1e-9 * np.max(np.abs(reference)))

    def test_density_matrix_matches_supervector(self):
        times = np.linspace(0., 1e-6, 5)
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                basis, hamiltonian, dissipator = build_modulated_model(sparse)
                lindbladian = Lindbladian(hamiltonian, dissipator)
                rho0 = mixed_density_matrix(len(basis.states))
                _, supervectors = lindbladian.evolve_supervector(basis.compute_supervector_from_density_matrix(rho0), times[-1], times, **ACCURATE)
                _, rhos = lindbladian.evolve_density_matrix(rho0, times[-1], times, **ACCURATE)
                for supervector, rho in zip(supervectors, rhos):
                    np.testing.assert_allclose(rho, basis.compute_density_matrix_from_supervector(supervector), atol=1e-8)
                self.assertGreater(np.max(np.abs(rhos[-1] - rhos[0])), 1e-2)


class TestGateTimeIndependentPaths(unittest.TestCase):
    """ Gate.from_lindbladian's lindbladian_time_independent and lindbladian_commutes_at_later_times paths use
        Lindbladian.get_matrix (they called the removed Lindbladian.matrix_function). """

    def test_time_independent_paths_match_ode(self):
        qubit = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        basis = StandardBasis([qubit])
        frame = [-state.energy for state in basis.states]
        qubit_frequency = qubit.energy_levels[1].energy - qubit.energy_levels[0].energy
        C = CouplingOperator.from_matrix(basis, 0.5 * TPI * 1e6 * basis.enlarge_matrix(Pauli.plus, [qubit]), qubit_frequency)
        decay = CouplingOperator.from_matrix(basis, np.sqrt(TPI * 0.1e6) * basis.enlarge_matrix(Pauli.minus, [qubit]), 0.)
        lindbladian = Lindbladian(Hamiltonian(basis, [C, adjoint(C)], frame), Dissipator(basis, [decay], frame))
        reference = Gate.from_lindbladian(basis, lindbladian, 0.4e-6, **ACCURATE).process_matrix
        for option in ('lindbladian_time_independent', 'lindbladian_commutes_at_later_times'):
            with self.subTest(option=option):
                gate = Gate.from_lindbladian(basis, lindbladian, 0.4e-6, **{option: True})
                np.testing.assert_allclose(gate.process_matrix, reference, atol=1e-7)


class TestSolverLayer(unittest.TestCase):
    """ dy/dt = -i Omega/2 sigma_x y: y(t) = (cos(Omega t/2), -i sin(Omega t/2)). """

    OMEGA = TPI * 1e6
    H = np.array([[0., 0.5], [0.5, 0.]]) * OMEGA

    def exact(self, t):
        return np.array([np.cos(self.OMEGA * t / 2.), -1j * np.sin(self.OMEGA * t / 2.)])

    def test_solvers_accept_options(self):
        times = np.linspace(0., 1e-6, 5)
        for solver, options in (('odeintz', {'rtol': 1e-11, 'atol': 1e-12}), ('solve_ivp', {'method': 'DOP853', 'rtol': 1e-11, 'atol': 1e-12}),
                                ('zvode', {'rtol': 1e-12, 'atol': 1e-14, 'method': 'bdf'})):
            with self.subTest(solver=solver):
                ts, ys = solve_time_evolution_equation(lambda t: self.H, np.array([1., 0.]), times[-1], times, ode_solver=solver, **options)
                np.testing.assert_allclose(ts, times)
                for t, y in zip(ts, ys):
                    np.testing.assert_allclose(y, self.exact(t), atol=1e-8)

    def test_default_times(self):
        for solver, count in (('odeintz', 4), ('zvode', 3)):
            with self.subTest(solver=solver):
                ts, ys = solve_time_evolution_equation(lambda t: self.H, np.array([1., 0.]), 1e-6, ode_solver=solver)
                self.assertEqual(len(ts), count)
                self.assertEqual(len(ys), count)
                self.assertAlmostEqual(ts[-1], 1e-6)
                np.testing.assert_allclose(ys[-1], self.exact(1e-6), atol=1e-5)  # default (loose) solver tolerances

if __name__ == '__main__':
    unittest.main()
