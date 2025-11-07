import numpy as np


# TODO: Add tests, verifications for certain atoms and states  
def Lande_gJ(gL: float, gS: float, J: float, L: float, S: float) -> None | float:
  JJp1 = J*(J+1)
  if JJp1 == 0:
    # 2 Cases for J = 0, either L and S are zero, or L and S cancel.
    if (L == 0 and S == 0) or (L == S):
      # Use limit procedure: limit of JJp1/JJP1 is 1
      return None 
    else:
      raise ValueError('Division by zero error. Do not use this function if J = 0.') 
  else:
    LLp1 = L*(L+1)
    SSp1 = S*(S+1)
    return (gL*(JJp1 - SSp1 + LLp1) + gS*(JJp1 + SSp1 - LLp1) )*0.5/JJp1 


def Lande_gF(gI: float, F: float, I: float, J: float, gJ: float | None=None):
  ffp1 = F*(F+1)
  if ffp1 == 0:
    raise ValueError('Division by zero error. Do not use this function if F = 0.') 

  if gJ is None:
    return gI
  IIp1 = I*(I+1)
  JJp1 = J*(J+1)
  return ((gJ*0.5*(ffp1 - IIp1 + JJp1)) + gI*0.5*(ffp1 + IIp1 - JJp1))/ffp1


def compute_Zeeman_prefactor_from_species(i: float, j: float, l: float, s: float): 
  ''' Helper function to compute the linear Zeeman prefactor for an atomic species with quantum numbers I, J, L, S'''
  g_S = 2.0023193043609236 # electron spin G-factor 
  m_e = 9.109383713928E-31 # kg 
  # Convert to Daltons 
  m_e /= 1.66054E-27 # kg  
  atomic_mass = 170.936323 # Daltons 
  Z = 70 # for Yb  
  nuclear_mass = atomic_mass - Z*m_e
  
  # Compute Lande g factor for orbital angular momentum: 
  g_L = 1. - (m_e / nuclear_mass)

  # Nuclear and Bohr magnetons, req'd for g_I factor: 
  mu = 0.49234 # \times mu_N, for Yb171
  # Nuclear magneton
  mu_N = 5.050783739316E-27 # J/T 
  # Bohr magneton
  mu_B = 9.274010065729E-24 # J/T 
  ratio_mu_N_to_mu_B = mu_N/mu_B 
  if(i == 0.5 and l == 1):
    g_I = -(mu/i)*ratio_mu_N_to_mu_B + -0.000142549 
  else:
    g_I = -(mu/i)*ratio_mu_N_to_mu_B 
  # TODO: specifiy which F state we are using if J non-zero 
  F = i + j 
  print('F = ' + str(F))
  # Compute Lande_G factors for other quantum numbers 
  g_J = Lande_gJ(g_L, g_S, j, l, s)
  g_F = Lande_gF(g_I, F, i, j, g_J) # this matches g_I for J = 0 

  # Planck's constant
  h = 6.626E-34
  Zeeman_prefactor = g_F * mu_B * 1E-3 * 1E-4 / h   # J kHz / Gauss  
  print('Zeeman prefactor = ' + str(Zeeman_prefactor))  
  return Zeeman_prefactor 



def compute_qubit_frequency(B_field: float, Zeeman_prefactor: float) -> float:
  ''' Qubit frequency defined via the linear Zeeman shift. '''
  ''' B_field is in Gauss, Zeeman prefactor is in units of kHz / G ''' 
  # Low field regime, Zeeman shift given by \DeltaE_{Z} = \pm \mu_{B} g_{F}/h |B| / 2
  omega_qubit = B_field * Zeeman_prefactor # prefactor is negative, so +1/2 spin is lower energy 
  print('Computing qubit frequency with applied ' + str(B_field) + ' Gauss magnetic field. \n')
  print('Qubit frequency: ' + str(omega_qubit) + ' kHz ') 
  return omega_qubit



# Define neutral atom qubit(s). 
# Qubit = ground levels of F = 1/2 hyperfine manifold of the 6s^2 orbital. In the neutral atom, this orbital has a singlet configuration. 
print('Testing script for Neutral atom 171Yb')

# Nuclear spin qubit in J = 0 manifold has degenerate levels naturally. Therefore, apply a bias magnetic field to set the qubit frequency. 
# Assuming all qubits have same frequency 
bias_magnetic_field = 500 # Gauss, relatively weak  

# H_{B} = (\mu_{B} / \hbar) * g_F \hat{F}_z B_z 
# H_{B} = \mu_{B}/\hbar (g_S S + g_L L + g_I I) \cdot B; consider B_z only and  

# \mu_{B} g_{f}/h 
Zeeman_prefactor_1S0 = -0.75056 # kHz/G . valid for 1S_0 state in low field regime . TODO: Verified, send to yaml file  
Zeeman_prefactor_3P0 = -1.14966 # kHz/G . valid for 3P_0 state in low field regime . TODO: send to yaml file  


# Idea for doing this in IonSim's AtomicSpecies class 

# if user_specified: 
#   return zeeman_prefactor (via user)
# else:
#   return compute_Zeeman_prefactor_from_species(i, j, l, s):

# Examples  
# 0S_1
i = 0.5
j = 0.
l = 0.
s = 0.
prefactor = compute_Zeeman_prefactor_from_species(i, j, l, s) 
#print('Zeeman prefactor estimate : ' + str(Zeeman_prefactor_estimate))

#3P_0 state: 
i = 0.5
j = 0.
l = 1.
s = 1.
#prefactor = compute_Zeeman_prefactor_from_species(i, j, l, s) 
prefactor = compute_Zeeman_prefactor_from_species(0.5, 0, 1, 1) 

compute_qubit_frequency(500, Zeeman_prefactor_1S0)



