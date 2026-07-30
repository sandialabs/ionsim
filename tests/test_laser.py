#***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE.md file in the root IonSim directory.
#***************************************************************************************************

import unittest

import numpy as np

from ionsim.process import Gate, Circuit
from ionsim.degree_of_freedom import AtomicStructure
from ionsim.basis import StandardBasis
from ionsim.laser import Laser, Polarization, BeamProfile 
from ionsim.named_operators import Unitary
from ionsim.noise import Noise

class TestProcess(unittest.TestCase):

    def setUp(self):
        """Set up the necessary objects for testing and test constructors."""
        self.atom_a = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.atom_b = AtomicStructure.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
        self.basis = StandardBasis([self.spin_a, self.spin_b])

        propagation_vector = np.array([np.cos(np.pi/4.), np.sin(np.pi/4.), 0.])
        phase = np.pi 
        # Create polarization 
        laser_polarization = Polarization.circular(propagation_vector, '+')

        # Create Gaussian beam profile  
        wavelength = 355*1E-9 # nm -> meters 
        beam_waist = 20 * 1E-6 # µm -> meters 
        laser_power = 1e-3 # mWatt -> Watt  
        self.laser = Laser.gaussian_from_wavelength(wavelength, laser_power, beam_waist, propagation_vector, laser_polarization, phase) 






    def test_laser_coupling_builder(self):
        """Test the process fidelity of the extra noisy gate."""
        print('test')
 #        extra_noisy_gate = Gate.from_process_matrix_function(
 #            self.basis, self.noisy_phi_gate.process_matrix_function, {'phi': 0, 'theta': np.pi/2}, [self.spin_a], self.theta_noise,
 #        )
 #        fidelity = extra_noisy_gate.compute_process_fidelity(self.Sx.process_matrix)
 #        self.assertAlmostEqual(fidelity, 0.9306176541502549, places=14)

if __name__ == '__main__':
    unittest.main()
