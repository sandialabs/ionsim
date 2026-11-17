#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************
""" Polarization of light for IonSim's light-atom couplings.

    Polarization: an explicit complex Cartesian (Jones) vector in the lab frame.
    LinearPolarization, CircularPolarization: specified relative to the propagation direction of the light.
    SphericalPolarization: specified by spherical components relative to a quantization axis.
    PolarizationInput: any of these, as accepted by Laser. Each has resolve(propagation_direction) -> Polarization.

    Spherical components follow q = m_excited - m_ground (sigma+ <-> q = +1) throughout IonSim; the quantization axis is
    always an explicit argument.
"""
from __future__ import annotations

import numpy as np
from sympy.physics.wigner import clebsch_gordan

from ionsim.custom_types import Matrix, Vector
from ionsim.config import NUMERICAL_EQUIVALENCE_THRESHOLD, NUMERICAL_ERROR_THRESHOLD
from ionsim.ionsim_error import IonSimError


## Vector helper operations
def _unit_vector(vec: Vector) -> Vector:
    """ Returns a unit vector of an input vector """ 
    v = np.asarray(vec)
    norm = np.linalg.norm(v)
    if norm < NUMERICAL_ERROR_THRESHOLD:
        raise IonSimError(f"Cannot normalize a vector with norm near zero. Computed norm {norm}")
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

def _spherical_basis_vectors(quantization_axis: Vector) -> dict[int, Vector]:
    """ Spherical basis vectors eps +1, 0, -1. See eq. 7.188 from "Quantum and Atom Optics by Daniel Steck""" 
    axis = np.asarray(quantization_axis)
    if axis.shape != (3,) or np.iscomplexobj(axis):
        raise IonSimError(f"The quantization axis must be a real 3-component vector, got {quantization_axis}.")
    z = _unit_vector(axis.astype(float))
    x, y = _perpendicular_basis(z)
    return {1: -(x + 1j*y)/np.sqrt(2.), 0: z.astype(complex), -1 : (x - 1j*y)/np.sqrt(2.)}


# Ways to give a Laser its polarization (PolarizationInput, defined after these classes). Each has
#     resolve(propagation_direction) -> Polarization, which the Laser calls with its own propagation direction n_hat; the Laser then checks n_hat . eps = 0.

class Polarization:
    """ Complex Cartesian polarization (Jones) vector eps in the lab frame.

        polarization_vector: complex 3-component vector (Cartesian coordinates), any nonzero length; it is normalized.
            The overall phase is kept and acts like the laser phase.

        Derived attribute:
            jones_vector: the unit complex polarization vector, sum_i |eps_i|^2 = 1

        The propagation direction belongs to the (Laser) light, which requires n_hat dot eps = 0. Given to a Laser, a
        Polarization resolves to itself; LinearPolarization, CircularPolarization and SphericalPolarization build one from
        a description relative to the propagation direction or the quantization axis.
    """

    def __init__(self, polarization_vector: Vector):
        vector = np.asarray(polarization_vector, dtype=complex)
        if vector.shape != (3,):
            raise IonSimError(f"polarization_vector must be a 3-component vector, got shape {vector.shape}.")
        norm = np.sqrt(np.vdot(vector, vector).real)
        if norm < NUMERICAL_ERROR_THRESHOLD:
            raise IonSimError(f"Cannot normalize a polarization vector with norm near zero. Computed norm {norm}")
        self._jones_vector = vector / norm

    @property
    def jones_vector(self) -> Vector:
        """ Unit complex polarization (Jones) vector in the lab frame. """
        return (self._jones_vector).copy()

    def __repr__(self):
        return f"Polarization(jones_vector={self.jones_vector})"

    def resolve(self, propagation_direction: Vector) -> Polarization:
        """ An explicit polarization does not depend on the propagation direction (the Laser checks transversality). """
        return self

    def dipole_components(self, quantization_axis: Vector) -> dict[int, complex]:
        """ Spherical (dipole) components eps_q, keyed by q = +1, 0, -1, with polarization vector = sum_q eps_q e_q.

            q is the change of m on absorption, q = m_excited - m_ground (sigma+ <-> q = +1, pi <-> 0, sigma- <-> -1), the
            convention used throughout IonSim. """
        e_q = _spherical_basis_vectors(quantization_axis)
        eps_p1 = np.vdot(e_q[1], self.jones_vector)
        eps_0 = np.vdot(e_q[0], self.jones_vector)
        eps_m1 = np.vdot(e_q[-1], self.jones_vector)
        components = np.array([eps_p1, eps_0, eps_m1])
        return dict(zip((1, 0, -1), components)) 

    @staticmethod
    def rank2_spherical_basis(quantization_axis: Vector) -> dict[int, Matrix]:
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

    def quadrupole_components(self, propagation_direction: Vector, quantization_axis: Vector) -> dict[int, complex]:
        """ Rank-2 polarization components C_q, q = -2..2 (q = m_excited - m_ground, as for dipole_components), of the
            E2 geometry tensor eps (x) n_hat, for light with this polarization propagating along propagation_direction.

            C_q = sum_ij conj(E_q)_ij eps_i n_j = sum_{m1, m2} <1 m1; 1 m2 | 2 q> eps_m1 n_m2, where eps_m and n_m are
            the spherical components of the polarization and propagation unit vectors. For an E2 coupling these play
            the role eps_q plays for E1: sum_ij eps_i n_j Q_ij = sum_q C_q Q_q for a symmetric traceless tensor Q.
        """
        n_hat = _unit_vector(np.asarray(propagation_direction, dtype=float))
        tensors = self.rank2_spherical_basis(quantization_axis)
        return {q: complex(np.einsum('ij,i,j->', np.conj(tensors[q]), self._jones_vector, n_hat)) for q in range(-2, 3)}


class LinearPolarization:
    """ Linear polarization at an angle [radians] in the plane perpendicular to the propagation direction:
            eps = cos(angle) e1 + sin(angle) e2, with (e1, e2, n_hat) right-handed. 
        e1 is the component of ref_axis perpendicular to n_hat (if ref_axis is given),
        otherwise the default transverse basis of _perpendicular_basis. 
    """

    def __init__(self, angle: float = 0., ref_axis: Vector | None = None):
        if not np.isfinite(angle):
            raise IonSimError(f"angle must be a finite number (rad), got {angle}.")
        self._angle = float(angle)
        self._ref_axis = None if ref_axis is None else np.array(ref_axis, dtype=float)

    @property
    def angle(self) -> float:
        return self._angle

    @property
    def ref_axis(self) -> Vector | None:
        return None if self._ref_axis is None else self._ref_axis.copy()

    def __repr__(self):
        return f"LinearPolarization(angle={self.angle}, ref_axis={self.ref_axis})"

    def resolve(self, propagation_direction: Vector) -> Polarization:
        e1, e2 = _perpendicular_basis(propagation_direction, self._ref_axis)
        return Polarization(np.cos(self.angle) * e1 + np.sin(self.angle) * e2)


class CircularPolarization:
    """ Circular polarization in the (e1, e2) plane perpendicular to the propagation direction n_hat, (e1, e2, n_hat)
        right-handed:

            - handedness is specified by '+' or '-'
            - eps_{+/-} = (e1 +/- ie2)/sqrt(2)
            - '+' <==> left-circular polarization; '-' <==> right-circular polarization
            - looks at wave from behind.

        The handedness is relative to the propagation direction (helicity), not to the atom's quantization axis: '+' drives
        sigma+ (q = +1) for light along +quantization axis and sigma- for light along -quantization axis.
        ref_axis (optional) only sets the orientation of e1, i.e. an overall phase of eps, which acts like the laser phase
        (it matters only for relative phases between beams).
    """
    HANDEDNESS = ('+', '-')

    def __init__(self, handedness: str, ref_axis: Vector | None = None):
        if handedness not in self.HANDEDNESS:
            raise IonSimError(f"Handedness must be specified by a string, '+' or '-', got {handedness!r}.")
        self._handedness = handedness
        self._ref_axis = None if ref_axis is None else np.array(ref_axis, dtype=float)

    @property
    def handedness(self) -> str:
        return self._handedness

    @property
    def ref_axis(self) -> Vector | None:
        return None if self._ref_axis is None else self._ref_axis.copy()

    def __repr__(self):
        return f"CircularPolarization(handedness={self.handedness!r}, ref_axis={self.ref_axis})"

    def resolve(self, propagation_direction: Vector) -> Polarization:
        e1, e2 = _perpendicular_basis(propagation_direction, self._ref_axis)
        sign = 1. if self.handedness == '+' else -1.
        return Polarization((e1 + sign * 1j * e2) / np.sqrt(2.))


class SphericalPolarization:
    """ Polarization from its spherical components relative to a quantization axis, eps = sum_q eps_q e_q, with
        components = {+1: eps_+1, 0: eps_0, -1: eps_-1} (q = m_excited - m_ground; the inverse of
        Polarization.dipole_components). The Laser requires the resulting vector to be perpendicular to its propagation
        direction: e.g. pure sigma+ ({+1: 1, 0: 0, -1: 0}) needs light along the quantization axis, and pi ({0: 1}) light perpendicular to it.
        Components not given are zero. """

    def __init__(self, components: dict[int, complex], quantization_axis: Vector):
        if not isinstance(components, dict) or not set(components) <= {-1, 0, 1} or not components:
            raise IonSimError(f"components must be a dict keyed by q in (+1, 0, -1), e.g. {{1: 1.}} for sigma+, got {components}.")
        self._components = {q: complex(components.get(q, 0.)) for q in (1, 0, -1)}
        self._quantization_axis = np.array(quantization_axis, dtype=float)
        _spherical_basis_vectors(self._quantization_axis)   # validates the axis

    @property
    def components(self) -> dict[int, complex]:
        return dict(self._components)

    @property
    def quantization_axis(self) -> Vector:
        return self._quantization_axis.copy()

    def __repr__(self):
        return f"SphericalPolarization(components={self.components}, quantization_axis={self.quantization_axis})"

    def resolve(self, propagation_direction: Vector) -> Polarization:
        e_q = _spherical_basis_vectors(self._quantization_axis)
        return Polarization(sum(self._components[q] * e_q[q] for q in (1, 0, -1)))


# The accepted ways for a user to give a Laser its polarization
PolarizationInput = Polarization | LinearPolarization | CircularPolarization | SphericalPolarization
