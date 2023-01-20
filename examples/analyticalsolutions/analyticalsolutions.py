# Some analytical solutions

import math
import numpy as np
from scipy.special import hankel2, jv

#-------------------------#
# Preliminary definitions
#-------------------------#

######
# 2D Fourier transforms of radial symmetry

fn_IntFraunhofer_delta = lambda k: 1
fn_IntFraunhofer_circ  = lambda R: lambda k: 2*jv(1,k*R)/(k*R) #usage: fn_R = fn_IntFraunhofer_circ(R) is the Fraunhofer integral function for a circular source of radius R
fn_IntFraunhofer_gauss = lambda sigma: lambda k: np.exp(-1/2*(k*sigma)**2) #fn_s = fn_IntFraunhofer_gauss(s) -> gaussian source of standard deviation 'sigma'

int_Fraunhofer_2D = {'none': fn_IntFraunhofer_delta, 'delta': fn_IntFraunhofer_delta, 'circ': fn_IntFraunhofer_circ, 'gaussian': fn_IntFraunhofer_gauss}

#-------------------------#
#      2D Full Space
#-------------------------#
 
######
# SH

# Green(r,w): space - frequency domain
def green_2D_SH_rw(r, w, rho=1, mu=1, fn_IntFraunhofer=None):
    """Green function of a 2D full, homogeneous space, for a Shear Horizontal load, in the frequency domain
    
    -- Input --
       r: distance to source
       w: angular frequency
       rho: density
       mu : shear modulus
       
       fn_IntFraunhofer: callable (optional) to account for finite size sources of radial symmetry, in the Fraunhofer approximation
          (2D) Fourier transform of the source. Takes the radial wavenumber 'k' as unique argument
          ex: fn_IntFraunhofer = lambda k: 1 #for a delta(r) source
              fn_IntFraunhofer = lambda k: 2*jv(1,k*a)/(k*a) #for a normalized circ(r/a) source, i.e. a uniform source of radius 'a'
              fn_IntFraunhofer = lambda k: np.exp(-1/2*(k*sigma)**2) #for a normalized gaussian source of standard deviation 'sigma'
    """
    if fn_IntFraunhofer is None:
        fn_IntFraunhofer = fn_IntFraunhofer_delta
    k = w*np.sqrt(rho/mu)
    return -1J/(4*mu)*hankel2(0, k*r) * fn_IntFraunhofer(k)

# U(r,t): space - time domain after convolution with a source
def u_2D_SH_rt(r, src, rho=1, mu=1, dt=1, fn_IntFraunhofer=None, eps=1e-8):
    """Displacement response to a SH line load with a given time-dependency, of a full, homogeneous space, in the time domain
    
    -- Input --
       r: distance to source
       src: array_like; time dependency of the source
       rho: density
       mu : shear modulus
       dt : time step
       
       fn_IntFraunhofer: callable (optional) to account for finite size sources of radial symmetry, in the Fraunhofer approximation
          (2D) Fourier transform of the source. Takes the radial wavenumber 'k' as unique argument
          ex: fn_IntFraunhofer = lambda k: 1 #for a delta(r) source
              fn_IntFraunhofer = lambda k: 2*jv(1,k*a)/(k*a) #for a circ(r/a) source, i.e. a uniform source of radius 'a'
              fn_IntFraunhofer = lambda k: np.exp(-1/2*(k*sigma)**2) #for a normalized gaussian source of standard deviation 'sigma'

       eps: small number to avoid NaN at r=0 or w=0
    """
    r = np.maximum(eps, r)
    w = 2*np.pi*np.fft.rfftfreq(np.size(src))/dt
    w[0] = eps #cheat zero frequency to avoid NaN in the green function
    Gxw = green_2D_SH_rw(r[:,np.newaxis], w[np.newaxis,:], rho, mu, fn_IntFraunhofer)
    return np.fft.irfft(Gxw * np.fft.rfft(src)[np.newaxis,:], axis=-1) #output: [x, t]


######
# P-SV

#Green(r,w): space - frequency domain
def green_2D_PSV_xw(x, w, rho=1, lambda_=2, mu=1, fn_IntFraunhofer=None, eps=1e-8):
    """Green function of a 2D full, homogeneous space, for an in-plane load, in the frequency domain
    
    -- Input --
       x: shape=(nbx, 3) coordinates of reception relative to the source
       w: shape=(nbw,) angular frequency
       rho: density
       lambda_: lame's modulus
       mu : shear modulus
       
       eps: small number to avoid NaN at r=0 or w=0

    -- Output --
       shape=(nbx,2,2,nbw)
    """
    #nb : x.shape = [N, 3, newaxis]
    #nb : w.shape = [,, M]
    N, M = x.shape[0], w.size
    green_xw = np.zeros((N,2,2,M), dtype=np.complex128)
    c_P = np.sqrt((lambda_+2*mu)/rho)#P-wave velocity   
    c_S = np.sqrt(mu/rho) #S-wave velocity
    r = np.linalg.norm(x, axis=1)# np.sqrt(x[:,0]**2+x[:,1]**2)
    r = np.maximum(eps, r)
    if fn_IntFraunhofer is None:
        fn_IntFraunhofer = fn_IntFraunhofer_delta
    fraunhofer_kP = fn_IntFraunhofer( w/c_P )
    fraunhofer_kS = fn_IntFraunhofer( w/c_S )
    O_P = w*r/c_P
    O_S = w*r/c_S
    Psi=1J/4*( (hankel2(1, O_S)/O_S*fraunhofer_kS - (c_S/c_P)**2*hankel2(1, O_P)/O_P)*fraunhofer_kP - hankel2(0, O_S)*fraunhofer_kS)
    Chi=1J/4*( (c_S/c_P)**2*hankel2(2, O_P)*fraunhofer_kP - hankel2(2, O_S)*fraunhofer_kS )
    green_xw[:,0,0,:]=1/mu*(Psi+Chi*(x[:,0]/r)**2)
    green_xw[:,0,1,:]=1/mu*(Chi*(x[:,0]/r)*(x[:,1]/r))
    green_xw[:,1,0,:]=green_xw[:,0,1,:]
    green_xw[:,1,1,:]=1/mu*(Psi+Chi*(x[:,1]/r)**2)
    return green_xw

#Green(r,w): space - frequency domain
def green_2D_PSV_half_S_xw(x, w, rho=2.719, lambda_=49.1, mu=26, fn_IntFraunhofer=None, eps=1e-8):
    """Green function of a 2D full, homogeneous space, for an in-plane load, in the frequency domain
    
    -- Input --
       x: shape=(nbx, 3) coordinates of reception relative to the source
       w: shape=(nbw,) angular frequency
       rho: density
       lambda_: lame's modulus
       mu : shear modulus
       
       eps: small number to avoid NaN at r=0 or w=0

    -- Output --
       shape=(nbx,2,2,nbw)
    """
    #nb : x.shape = [N, 3, newaxis]
    #nb : w.shape = [,, M]
    N, M = x.shape[0], w.size
    green_xw = np.zeros((N,2,2,M), dtype=np.complex128)
    ups=lambda_/(2*(lambda_+mu))
    Q= -1
    c_P = np.sqrt((lambda_+2*mu)/rho)#P-wave velocity   
    c_S = np.sqrt(mu/rho) #S-wave velocity
    c_R=((0.862+1.14*ups)/(1+ups))*c_S
    r = np.linalg.norm(x, axis=1)# np.sqrt(x[:,0]**2+x[:,1]**2)
    r = np.maximum(eps, r)
    if fn_IntFraunhofer is None:
        fn_IntFraunhofer = fn_IntFraunhofer_delta
    kP = w/c_P
    kS = w/c_S
    kR = w/c_R
    O_P = w*r/c_P
    O_S = w*r/c_S
    O_R = w*r/c_R
    F_prime=8*kR*(2*kR**2-kS**2)-4*kR**3*(kR**2-kP**2)**0.5*(kR**2-kS**2)**(-0.5)-(8*kR*(kR**2-kP**2)**0.5+4*kR**3*(kR**2-kP**2)**(-0.5))*(kR**2-kS**2)**0.5
    HH=-(kR*(2*kR**2-kS**2-2*np.sqrt(kR**2-kP**2)*np.sqrt(kR**2-kS**2)))/F_prime
    KK=-(kS**2*(kR**2-kP**2))/F_prime
    #
    wR, wP, wS = True, True , True
    #
    green_xw[:,0,1,:]=-wR*(Q/mu)*HH*np.exp(-1J*kR*x[:,0]) \
    +wS*(Q/mu)*np.sqrt(2/np.pi) * (1-(kP/kS)**2) * np.exp(-1J*(kS*x[:,0]+np.pi/4))/(kS*x[:,0])**(1.5) \
    -wP*(Q/mu)*np.sqrt(2/np.pi) * kP**3*kS**2*np.sqrt(kS**2-kP**2)/(kS**2-2*kP**2)**3 * 1J*np.exp(-1J*(kP*x[:,0]+np.pi/4))/(kP*x[:,0])**(1.5)
    
    green_xw[:,1,1,:] = -1J*(Q/mu)*KK*np.exp(-1J*kR*x[:,0]) \
    + (2*Q/mu)*np.sqrt(2/np.pi)*(1-(kP/kS)**2)*(1J*np.exp(-1J*(kS*x[:,0]+np.pi/4))/(kS*x[:,0])**(3/2)) \
    - (Q/2*mu)*np.sqrt(2/np.pi)*((kP**3*kS**2)/(kS**2-kP**2)**2)*((1J*np.exp(-1J*(kP*x[:,0]+np.pi/4)))/((kP*x[:,0])**(3/2)))
    
    green_xw[:,1,1,:] *= 0
    return green_xw

# U(r,t): space - time domain after convolution with a source
def u_2D_PSV_rt(x, src, F_=(1,0), rho=2.719, lambda_=49.1, mu=26, dt=1, fn_IntFraunhofer=None, eps=1e-8):
    """Displacement response to an in-plane line load with a given time-dependency, of a full, homogeneous space, in the time domain
    
    -- Input --
       x: shape=(nbx, 3) coordinates of reception relative to the source
       src: array_like of shape (nbt,); time dependency of the source
       F_: F_=(F_1, F_2) force vector
       rho: density
       lambda_: lame's modulus
       mu : shear modulus
       dt : time step
       
       eps: small number to avoid NaN at r=0 or w=0
       
    -- Output --
       shape=(nbx,2,nbt)
    """
    x  = np.asarray(x)
    F_ = np.asarray(F_)
    w = 2*np.pi*np.fft.rfftfreq(np.size(src))/dt
    w[0] = eps #cheat zero frequency to avoid NaN in the green function
    Gxw = green_2D_PSV_xw(x[:,:,np.newaxis], w[np.newaxis,np.newaxis,:], rho,lambda_, mu, fn_IntFraunhofer, eps) #out: [Nx, 2, 2, Nw]
    U_xw=np.tensordot(Gxw, F_, axes=(2,0)) #out: [Nx, 2, Nw]
    return np.fft.irfft(U_xw * np.fft.rfft(src)[np.newaxis,np.newaxis,:], axis=-1) #output: [Nx, 2, Nt]
    
# U(r,t): space - time domain after convolution with a source
def u_2D_PSV_half_S_rt(x, src, F_=(1,0), rho=2.719, lambda_=49.1, mu=26, dt=1, fn_IntFraunhofer=None, eps=1e-8):
    """Displacement response to an in-plane line load with a given time-dependency, of a full, homogeneous space, in the time domain
    
    -- Input --
       x: shape=(nbx, 3) coordinates of reception relative to the source
       src: array_like of shape (nbt,); time dependency of the source
       F_: F_=(F_1, F_2) force vector
       rho: density
       lambda_: lame's modulus
       mu : shear modulus
       dt : time step
       
       eps: small number to avoid NaN at r=0 or w=0
       
    -- Output --
       shape=(nbx,2,nbt)
    """
    x  = np.asarray(x)
    F_ = np.asarray(F_)
    w = 2*np.pi*np.fft.rfftfreq(np.size(src))/dt
    w[0] = eps #cheat zero frequency to avoid NaN in the green function
    Gxw = green_2D_PSV_half_S_xw(x[:,:,np.newaxis], w[np.newaxis,np.newaxis,:], rho,lambda_, mu, fn_IntFraunhofer, eps) #out: [Nx, 2, 2, Nw]
    U_xw=np.tensordot(Gxw, F_, axes=(2,0)) #out: [Nx, 2, Nw]
    src_w = np.fft.rfft(src)
    return np.fft.irfft(U_xw * src_w[np.newaxis,np.newaxis,:], axis=-1) #output: [Nx, 2, Nt]


def _test():
    import matplotlib.pyplot as plt
    
    ### -> Time function
    #
    f0 = 1 #central frequency of the source
    T0 = 1/f0 #period
    d0 = 2*T0 #duration of source
    #
    src_t = lambda t: np.sin(2*np.pi*f0 * t) * np.sin(np.pi*t/d0)**2 * (t<d0) * (t>0) #source(t)
    
    #
    tstart = 0 # Start time
    tmax   = 4*d0 *1 # Final time
    num_steps = 1000 *1
    dt = (tmax-tstart) / num_steps # time step size
    
    #
    x = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])
    rho, lambda_, mu = 1, 2, 1
    #
    F_0 = np.array([0,1])
    signals_at_points_exact = u_2D_PSV_half_S_rt(x, src_t(dt*np.arange(num_steps)), F_0,rho,lambda_, mu, dt, None)
    #
    fig, ax = plt.subplots(1,1)
    t = dt*np.arange(num_steps)
    ax.set_title('Signals at few points')
    for i in range(len(signals_at_points_exact)):
        ax.plot(t, signals_at_points_exact[i,0,:], c='C'+str(i), ls='--') #exact
    ax.set_xlabel('Time')
    
    fig, ax = plt.subplots(2,1)
    f = np.fft.rfftfreq(t.size)/dt
    ax[0].plot( t, src_t(dt*np.arange(num_steps)), c='k' )
    ax[1].plot( f, np.abs(np.fft.rfft(src_t(t))), c='k' )
    ax[1].plot( f, np.abs(np.fft.rfft(signals_at_points_exact[0,0,:])) )
    
    plt.show()

if __name__ == "__main__" :
    _test()

