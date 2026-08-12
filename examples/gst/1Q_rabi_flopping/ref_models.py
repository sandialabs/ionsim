import numpy as np
import ionsim as ism

N = 2
spins = [
    ism.AtomicSpin.from_species(species='171Yb+', term_symbols=['S1/2'], level_names=['S1/2,0,0', 'S1/2,1,0'])
    for _ in range(N)
]

basis = ism.StandardBasis([*spins])

""" Example containing gate simulators that take in a state and return a state. """
def idle(theta):
    """ Returns d^2 x d^2 process matrix in standard basis for Z-rotation by theta """  
    # Build identity matrix with Z rotation by theta:
    I = np.eye(2,dtype=complex)
    I[0,0] = np.exp( - 1j * theta ) 
    I[1,1] = np.exp( 1j * theta ) 
    I = np.kron(I, I)  

    # Promote to a d^2 x d^2 superoperator 
    return basis.compute_superoperator_from_unitary_operator(I)



def X_pi2(X_rot, Z_rot):
    """ Returns Gate from the d^2 x d^2 process matrix function in standard basis 

        X_pi2 = exp( -i [ (pi/2 + X_rot) X  + (Z_rot)Z ] )

        - X_rot is an additional X_rotation parameter (over/under rotation).
        - Z_rot is a Z_rotation parameter, e.g. from a detuned laser. 

    """  
    x_angle = np.pi/2. + X_rot
    Rxpi2 = ism.Unitary.R_bloch([x_angle/2., 0./2., Z_rot/2.]) 
    
    # Promote to a d^2 x d^2 superoperator 
    #return basis.compute_superoperator_from_unitary_operator(Rxpi2) # superoperator 
    return Rxpi2 


def Y_pi2(Y_rot):
    """ Returns Gate from the d^2 x d^2 process matrix function in standard basis 

        Y_pi2 = exp( -i [ (pi/2 + Y_rot) Y  + (Z_rot)Z ] )

        - Y_rot is an additional Y_rotation parameter (over/under rotation).
        - Z_rot is a Z_rotation parameter, e.g. from a detuned laser. 

    """  
    y_angle = np.pi/2. + Y_rot
    Rypi2 = ism.Unitary.R_bloch([0./2., y_angle/2., 0./2.]) 

    # Promote to a d^2 x d^2 superoperator 
    #return basis.compute_superoperator_from_unitary_operator(Rypi2)
    return Rypi2



def X_pi2_q0(X_rot, Z_rot):
    I = np.eye(2)
    return basis.compute_superoperator_from_unitary_operator(np.kron(X_pi2(X_rot, Z_rot), I)) 

def X_pi2_q1(X_rot, Z_rot):
    I = np.eye(2)
    return basis.compute_superoperator_from_unitary_operator(np.kron(I, X_pi2(X_rot, Z_rot)))

def Y_pi2_q0(Y_rot):
    I = np.eye(2)
    return basis.compute_superoperator_from_unitary_operator(np.kron(Y_pi2(Y_rot), I)) 

def Y_pi2_q1(Y_rot):
    I = np.eye(2)
    return basis.compute_superoperator_from_unitary_operator(np.kron(I, Y_pi2(Y_rot))) 


def cnot(d_theta_MS):
    # Jx MS unitary with under/over rotation  
    # Assume perfect 1Q gate rotations for example  
    return basis.compute_superoperator_from_unitary_operator(ism.Unitary.CNOT_from_MS(0., d_theta_MS + np.pi/2.))
 
def MS_pm(d_theta_MS: float, phi: float):
    # Jx MS unitary with under/over rotation  
    # Assume perfect 1Q gate rotations for example  
    return basis.compute_superoperator_from_unitary_operator(ism.Unitary.MS(phi, d_theta_MS + np.pi/2.))
