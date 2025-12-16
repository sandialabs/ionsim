""" Physical constants in SI units to be used in IonSim"""

# Fundamental constants
SPEED_OF_LIGHT = 299792458 # m/s 
PLANCK = 6.62607015E-34 # J / Hz  
HBAR = 1.054571817E-34 # J / Hz  
CHARGE = 1.602176634E-19 # Coulomb 
PI = 3.1415926535897932384626433832795028841971693

# Particle masses
mu_N = 5.050783739316E-27 # J/T , Nuclear magneton
mu_B = 9.274010065729E-24 # J/T , Bohr magneton 
ELECTRON_MASS = 9.109383713928E-31 # mass of electron in kg 

# Magnetic constants
NUCLEAR_MAGNETON = 5.050783739316E-27 # J/T , Nuclear magneton
BOHR_MAGNETON = 9.274010065729E-24 # J/T , Bohr magneton
SPIN_G_FACTOR = 2.0023193043609236 # Lande gS factor (electron spin g factor)


#######---- Example usage-----#######

# import physical_constants 

# wavelength = 600 * 1E-9 # meters  
# frequency = physical_constants.SPEED_OF_LIGHT / wavelength


# import physical_constants.SPEED_OF_LIGHT
# frequency = SPEED_OF_LIGHT / wavelength


