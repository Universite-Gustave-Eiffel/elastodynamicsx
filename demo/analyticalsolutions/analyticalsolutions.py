# Copyright (C) 2023 Pierric Mora, Massina Fengal
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

"""
Some analytical solutions.

See:
    Kausel, E. (2006). Fundamental solutions in elastodynamics: a compendium. Cambridge University Press.
"""

import math
import numpy as np
from scipy.special import hankel2, jv


# ------------------------- #
# Preliminary definitions
# ------------------------- #

# #####
# some 2D Fourier transforms of radial symmetry

fn_IntFraunhofer_delta = lambda k: 1
# fn_R = fn_IntFraunhofer_circ(R) is the Fraunhofer integral function for a circular source of radius R
fn_IntFraunhofer_circ  = lambda R: lambda k: 2*jv(1,k*R)/(k*R)
# fn_s = fn_IntFraunhofer_gauss(s) -> gaussian source of standard deviation 'sigma'
fn_IntFraunhofer_gauss = lambda sigma: lambda k: np.exp(-1/2*(k*sigma)**2)

int_Fraunhofer_2D = {'none': fn_IntFraunhofer_delta, 'delta': fn_IntFraunhofer_delta,
                     'circ': fn_IntFraunhofer_circ, 'gaussian': fn_IntFraunhofer_gauss}


# ------------------------- #
#      2D Full Space
# ------------------------- #
 
# #####
# SH

# Green(r, w): space - frequency domain
def green_2D_SH_rw(r, w, rho, mu, fn_IntFraunhofer=None):
    """
    Green function of a 2D full, homogeneous space, for a Shear Horizontal load,
    in the frequency domain
    
    -- Input --
       r: distance to source
       w: angular frequency
       rho: density
       mu : shear modulus
       
       fn_IntFraunhofer: callable (optional) to account for finite size sources
           of radial symmetry, in the Fraunhofer approximation. (2D) Fourier transform
           of the source. Takes the radial wavenumber 'k' as unique argument
           ex: fn_IntFraunhofer = lambda k: 1  # delta(r)
               fn_IntFraunhofer = lambda k: 2*jv(1,k*a)/(k*a)  # circ(r/a): uniform source of radius 'a'
               fn_IntFraunhofer = lambda k: np.exp(-1/2*(k*sigma)**2)  # gaussian source, std: 'sigma'
    """
    if hasattr(w, '__iter__'):
        w = np.asarray(w)
    else:
        w = np.array([w])
    if len(w.shape) == 1:
        w = w[np.newaxis,:]
    if hasattr(r, '__iter__'):
        r = np.asarray(r)
    else:
        r = np.array([r])
    if len(r.shape) == 1:
        r = r[:,np.newaxis]
        
    if fn_IntFraunhofer is None:
        fn_IntFraunhofer = fn_IntFraunhofer_delta
    k = w*np.sqrt(rho/mu)
    return -1J/(4*mu)*hankel2(0, k*r) * fn_IntFraunhofer(k)


# U(r, t): space - time domain after convolution with a source
def u_2D_SH_rt(r, src, rho, mu, dt=1, fn_IntFraunhofer=None, eps=1e-8):
    """
    Displacement response to a SH line load with a given time-dependency,
    of a full, homogeneous space, in the time domain.
    
    -- Input --
       r: distance to source
       src: array_like; time dependency of the source
       rho: density
       mu : shear modulus
       dt : time step
       
       fn_IntFraunhofer: callable (optional) to account for finite size sources
           of radial symmetry, in the Fraunhofer approximation
           (2D) Fourier transform of the source. Takes the radial wavenumber 'k' as unique argument
           ex: fn_IntFraunhofer = lambda k: 1 # delta(r)
               fn_IntFraunhofer = lambda k: 2*jv(1,k*a)/(k*a)  #circ(r/a): uniform source of radius 'a'
               fn_IntFraunhofer = lambda k: np.exp(-1/2*(k*sigma)**2)  # gaussian source, std: 'sigma'

       eps: small number to avoid NaN at r=0 or w=0
    """
    r = np.maximum(eps, r)
    w = 2*np.pi*np.fft.rfftfreq(np.size(src))/dt
    w[0] = eps  # cheat zero frequency to avoid NaN in the green function
    Gxw = green_2D_SH_rw(r[:,np.newaxis], w[np.newaxis,:], rho, mu, fn_IntFraunhofer)
    return np.fft.irfft(Gxw * np.fft.rfft(src)[np.newaxis,:], axis=-1)  # output: [x, t]


# #####
# P-SV

# Green(r, w): space - frequency domain
def green_2D_PSV_xw(x, w, rho, lambda_, mu, fn_IntFraunhofer=None, eps=1e-8):
    """
    Green function of a 2D full, homogeneous space, for an in-plane load, in the frequency domain
    
    -- Input --
       x: shape=(nbx, 3) or (nbx, 3, np.newaxis),        coordinates of reception
       w: shape=(nbw,) or (np.newaxis, np.newaxis, nbw), angular frequency
       rho: density
       lambda_: lame's modulus
       mu : shear modulus
       
       eps: small number to avoid NaN at r=0 or w=0

    -- Output --
       shape=(nbx, 2, 2, nbw)
    """
    # nb : x.shape = [N, 3, newaxis]
    # nb : w.shape = [,, M]
    x = np.asarray(x)
    w = np.asarray(w)
    if len(x.shape) == 2:
        x = x[:,:,np.newaxis]
    if len(w.shape) == 1:
        w = w[np.newaxis,np.newaxis,:]
    assert len(x.shape)==3, "Wrong format for x; expected 2D or 3D array"
    assert len(w.shape)==3, "Wrong format for w; expected scalar or 1D or 3D array"
    
    N, M = x.shape[0], w.size
    green_xw = np.zeros((N,2,2,M), dtype=np.complex128)
    c_P = np.sqrt((lambda_+2*mu)/rho)  # P-wave velocity   
    c_S = np.sqrt(mu/rho)  # S-wave velocity
    r = np.linalg.norm(x, axis=1)
    r = np.maximum(eps, r)
    if fn_IntFraunhofer is None:
        fn_IntFraunhofer = fn_IntFraunhofer_delta
    fraunhofer_kP = fn_IntFraunhofer( w/c_P )
    fraunhofer_kS = fn_IntFraunhofer( w/c_S )
    O_P = w*r/c_P
    O_S = w*r/c_S
    Psi=1J/4*( (hankel2(1, O_S)/O_S*fraunhofer_kS
                - (c_S/c_P)**2*hankel2(1, O_P)/O_P)*fraunhofer_kP
                - hankel2(0, O_S)*fraunhofer_kS)
    Chi=1J/4*( (c_S/c_P)**2*hankel2(2, O_P)*fraunhofer_kP
                - hankel2(2, O_S)*fraunhofer_kS )
    green_xw[:,0,0,:]=1/mu*(Psi+Chi*(x[:,0]/r)**2)
    green_xw[:,0,1,:]=1/mu*(Chi*(x[:,0]/r)*(x[:,1]/r))
    green_xw[:,1,0,:]=green_xw[:,0,1,:]
    green_xw[:,1,1,:]=1/mu*(Psi+Chi*(x[:,1]/r)**2)
    return green_xw


# U(x,w): space - frequency domain after product with a source vector
def u_2D_PSV_xw(x, w, F_, rho, lambda_, mu, fn_IntFraunhofer=None, eps=1e-8):
    """
    Displacement response to an in-plane line load of a 2D full, homogeneous space,
    in the frequency domain
    
    -- Input --
       x: shape=(nbx, 3) or (nbx, 3, np.newaxis),        coordinates of reception
       w: shape=(nbw,) or (np.newaxis, np.newaxis, nbw), angular frequency
       F_: F_=(F_1, F_2) force vector
       rho: density
       lambda_: lame's modulus
       mu : shear modulus
       
       eps: small number to avoid NaN at r=0 or w=0

    -- Output --
       shape=(nbx,2,nbw)
    """
    if hasattr(w, '__iter__'):
        w = np.asarray(w)
    else:
        w = np.array([w])
    if hasattr(x, '__iter__'):
        x = np.asarray(x)
    else:
        x = np.array([x])

    # Gxw: [Nx, 2, 2, Nw]
    Gxw = green_2D_PSV_xw(x[:,:,np.newaxis], w[np.newaxis,np.newaxis,:], rho,lambda_, mu, fn_IntFraunhofer, eps)
    # Uxw: [Nx, 2, Nw]
    Uxw = np.tensordot(Gxw, F_, axes=(2,0))
    return Uxw


# U(x,t): space - time domain after convolution with a source
def u_2D_PSV_xt(x, src, F_, rho, lambda_, mu, dt=1, fn_IntFraunhofer=None, eps=1e-8):
    """
    Displacement response to an in-plane line load with a given time-dependency,
    of a full, homogeneous space, in the time domain
    
    -- Input --
       x: shape=(nbx, 3) coordinates of reception
       src: array_like of shape (nbt,); time dependency of the source
       F_: F_=(F_1, F_2) force vector
       rho: density
       lambda_: lame's modulus
       mu : shear modulus
       dt : time step

       eps: small number to avoid NaN at r=0 or w=0
       
    -- Output --
       shape=(nbx, 2, nbt)
    """
    x  = np.asarray(x)
    F_ = np.asarray(F_)
    w = 2*np.pi*np.fft.rfftfreq(np.size(src))/dt
    w[0] = eps  # cheat zero frequency to avoid NaN in the green function
    # Gxw: [Nx, 2, 2, Nw]
    Gxw = green_2D_PSV_xw(x[:,:,np.newaxis], w[np.newaxis,np.newaxis,:],
                          rho,lambda_, mu, fn_IntFraunhofer, eps)
    # Uxw: [Nx, 2, Nw]
    Uxw = np.tensordot(Gxw, F_, axes=(2,0))
    # output: [Nx, 2, Nt]
    return np.fft.irfft(Uxw * np.fft.rfft(src)[np.newaxis,np.newaxis,:], axis=-1)


# ------------------------- #
#      2D Half Space
# ------------------------- #

# #####
# P-SV

# Green(r, w): space - frequency domain
def green_2D_PSV_half_S_xw(x, w, rho, lambda_, mu, fn_IntFraunhofer=None, eps=1e-8):
    """
    Green function of a 2D half, homogeneous space, for an in-plane load, in the frequency domain
    
    -- Input --
       x: shape=(nbx, 3) or (nbx, 3, np.newaxis),        coordinates of reception
       w: shape=(nbw,) or (np.newaxis, np.newaxis, nbw), angular frequency
       rho: density
       lambda_: lame's modulus
       mu : shear modulus
       
       eps: small number to avoid NaN at r=0 or w=0

    -- Output --
       shape=(nbx,2,2,nbw)
    """
    import warnings
    warnings.warn('Incorrect results')  # TODO: fixme

    #nb : x.shape = [N, 3, newaxis]
    #nb : w.shape = [,, M]
    x = np.asarray(x)
    w = np.asarray(w)
    if len(x.shape) == 2:
        x = x[:,:,np.newaxis]
    if len(w.shape) == 1:
        w = w[np.newaxis,np.newaxis,:]
    assert len(x.shape)==3, "Wrong format for x; expected 2D or 3D array"
    assert len(w.shape)==3, "Wrong format for w; expected scalar or 1D or 3D array"

    N, M = x.shape[0], w.size
    green_xw = np.zeros((N,2,2,M), dtype=np.complex128)
    ups=lambda_/(2*(lambda_+mu))
    Q= -1
    c_P = np.sqrt((lambda_+2*mu)/rho)  # P-wave velocity   
    c_S = np.sqrt(mu/rho)  # S-wave velocity
    c_R=((0.862+1.14*ups)/(1+ups))*c_S  # approx R-wave velocity
    r = np.linalg.norm(x, axis=1)
    r = np.maximum(eps, r)

    if fn_IntFraunhofer is None:
        fn_IntFraunhofer = fn_IntFraunhofer_delta
        # TODO: use fn_IntFraunhofer

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

    green_xw[:,0,1,:]=-wR*(Q/mu)*HH*np.exp(-1J*kR*x[:,0]) \
    + wS*(Q/mu)*np.sqrt(2/np.pi) * (1-(kP/kS)**2) * np.exp(-1J*(kS*x[:,0]+np.pi/4)) / (kS*x[:,0])**(1.5) \
    - wP*(Q/mu)*np.sqrt(2/np.pi) * kP**3*kS**2*np.sqrt(kS**2-kP**2)/(kS**2-2*kP**2)**3 * 1J*np.exp(-1J*(kP*x[:,0]+np.pi/4)) / (kP*x[:,0])**(1.5)

    green_xw[:,1,1,:] = -1J*wR*(Q/mu)*KK*np.exp(-1J*kR*x[:,0]) \
    + wS*(2*Q/mu)*np.sqrt(2/np.pi)*(1-(kP/kS)**2)*(1J*np.exp(-1J*(kS*x[:,0]+np.pi/4)) / (kS*x[:,0])**(1.5)) \
    - wP*(Q/2*mu)*np.sqrt(2/np.pi)*((kP**3*kS**2)/(kS**2-kP**2)**2)*((1J*np.exp(-1J*(kP*x[:,0]+np.pi/4))) / ((kP*x[:,0])**(1.5)))

    return green_xw


# U(r,t): space - time domain after convolution with a source
def u_2D_PSV_half_S_rt(x, src, F_, rho, lambda_, mu, dt=1, fn_IntFraunhofer=None, eps=1e-8):
    """
    Displacement response to an in-plane line load with a given time-dependency,
    of a full, homogeneous space, in the time domain
    
    -- Input --
       x: shape=(nbx, 3) coordinates of reception
       src: array_like of shape (nbt,); time dependency of the source
       F_: F_=(F_1, F_2) force vector
       rho: density
       lambda_: lame's modulus
       mu : shear modulus
       dt : time step
       
       eps: small number to avoid NaN at r=0 or w=0
       
    -- Output --
       shape=(nbx, 2, nbt)
    """
    x  = np.asarray(x)
    F_ = np.asarray(F_)
    w = 2*np.pi*np.fft.rfftfreq(np.size(src))/dt
    w[0] = eps  # cheat zero frequency to avoid NaN in the green function
    # Gxw: [Nx, 2, 2, Nw]
    Gxw  = green_2D_PSV_half_S_xw(x[:,:,np.newaxis], w[np.newaxis,np.newaxis,:],
                                  rho, lambda_, mu, fn_IntFraunhofer, eps)
    # Uxw: [Nx, 2, Nw]
    Uxw = np.tensordot(Gxw, F_, axes=(2,0))
    src_w = np.fft.rfft(src)
    # output: [Nx, 2, Nt]
    return np.fft.irfft(Uxw * src_w[np.newaxis,np.newaxis,:], axis=-1)


# ------------------------- #
#          TESTS
# ------------------------- #

def _test():
    import matplotlib.pyplot as plt

    # ## -> Time function
    #
    fc  = 14.5 # Central frequency
    sig = np.sqrt(2)/(2*np.pi*fc)  # Gaussian standard deviation
    t0  = 4*sig

    def src_t(t):
        return (1 - ((t-t0)/sig)**2) * np.exp(-0.5*((t-t0)/sig)**2)  # Source(t)

    #
    sizefactor = 0.5
    tilt = 10  # tilt angle (degrees)
    y_surf = lambda x: 2 * sizefactor + np.tan(np.radians(tilt)) * x
    X0_src = np.array([1.720*sizefactor, y_surf(1.720*sizefactor), 0])  # Center

    tstart = 0 # Start time
    dt     = 0.25e-3 # Time step
    num_steps = int(6000*sizefactor)

    # -----------------------------------------------------
    #                 Material parameters
    # -----------------------------------------------------
    rho     = 2.2  # density
    cP, cS  = 3.2, 1.8475  # P- and S-wave velocities
    c11, c44= rho * cP**2, rho * cS**2
    mu      = c44
    lambda_ = c11-2*c44

    # ---------
    #  FE Data
    # ---------
    dat = np.load('seismogram_weq_2D-PSV_HalfSpace_Lamb_KomatitschVilotte_BSSA1998.npz')
    t = dat['t']
    x = dat['x'] - X0_src[np.newaxis,:]
    x = np.linalg.norm(x, axis=1) * np.sign(x[:,0])
    x = np.array([x, 0*x, 0*x]).T
    x = x[-10::3]
    sigs_FE = dat['signals'][-10::3]
    #
    F_0 = np.array([0, -1])
    src = src_t(dt*np.arange(num_steps))
    src = src - np.mean(src)
    sigs_analytical = u_2D_PSV_half_S_rt(x, src, F_0, rho, lambda_, mu, dt)
    #
    fig, ax = plt.subplots(1,1)
    ax.set_title('Signals at few points')
    icomp = 0
    for i in range(len(sigs_analytical)):
        ax.plot(t, sigs_FE[i,icomp,:], c='C'+str(i), ls='-')  # FE
        ax.plot(t, sigs_analytical[i,icomp,:], c='C'+str(i), ls='--')  # analytical
    ax.set_xlabel('Time')

    fig, ax = plt.subplots(2,1)
    f = np.fft.rfftfreq(t.size)/dt
    ax[0].plot( t, src, c='k' )
    ax[1].plot( f, np.abs(np.fft.rfft(src)), c='k' )
    ax[1].plot( f, np.abs(np.fft.rfft(sigs_FE[0,icomp,:])), c='C0', ls='-' )
    ax[1].plot( f, np.abs(np.fft.rfft(sigs_analytical[0,icomp,:])), c='C0', ls='--' )

    plt.show()


if __name__ == "__main__" :
    _test()
