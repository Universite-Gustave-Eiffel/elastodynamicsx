# Some analytical solutions

import math
import numpy as np
from scipy.special import hankel2

#-------------------------#
#      2D Full Space
#-------------------------#

######
# SH

# Green(r,w): space - frequency domain
def green_2D_SH_rw(r, w, rho=1, mu=1):
    """Green function of a 2D full, homogeneous space, for a Shear Horizontal load, in the frequency domain
    
    -- Input --
       r: distance to source
       w: angular frequency
       rho: density
       mu : shear modulus
    """
    return -1J/(4*mu)*hankel2(0, w*r*math.sqrt(rho/mu))

# U(r,t): space - time domain after convolution with a source
def u_2D_SH_rt(r, src, rho=1, mu=1, dt=1):
    """Displacement response to a SH line load with a given time-dependency, of a full, homogeneous space, in the time domain
    
    -- Input --
       r: distance to source
       src: array_like; time dependency of the source
       rho: density
       mu : shear modulus
       dt : time step
    """
    w = 2*np.pi*np.fft.rfftfreq(np.size(src))/dt
    w[0] = 1e-8 #cheat zero frequency to avoid NaN in the green function
    Gxw = green_2D_SH_rw(r[:,np.newaxis], w[np.newaxis,:], rho, mu)
    return np.fft.irfft(Gxw * np.fft.rfft(src)[np.newaxis,:], axis=-1) #output: [x, t]


######
# P-SV
supW = lambda r: np.maximum(1e-8, r)

#Green(r,w): space - frequency domain
def green_2D_PSV_xw(x, w, rho=1, lambda_=2, mu=1): #TODO
    #nb : x.shape = [N, 3, newaxis]
    #nb : w.shape = [,, M]
    N, M = x.shape[0], w.size
    green_xw = np.zeros((N,2,2,M), dtype=np.complex128)
    c_P = np.sqrt((lambda_+2*mu)/rho)#P-wave velocity   
    c_S = np.sqrt(mu/rho) #S-wave velocity
    r = np.linalg.norm(x, axis=1)# np.sqrt(x[:,0]**2+x[:,1]**2)
    Psi=1J/4*((hankel2(1, supW(w*r/c_S))/c_S-(c_S/c_P)**2*hankel2(1, supW(w*r/c_P))/c_P)-hankel2(1, supW(w*r/c_S)))
    Chi=1J/4*((c_S/c_P)**2*hankel2(2, supW(w*r/c_P))-hankel2(2, supW(w*r/c_S)))
    green_xw[:,0,0,:]=1/mu*(Psi+Chi*(x[:,0]/r)**2)
    green_xw[:,0,1,:]=1/mu*(Chi*(x[:,0]/r)*(x[:,1]/r))
    green_xw[:,1,0,:]=green_xw[:,0,1,:]
    green_xw[:,1,1,:]=1/mu*(Psi+Chi*(x[:,1]/r)**2)
    return green_xw
       	

# U(r,t): space - time domain after convolution with a source
def u_2D_PSV_rt(x, src, F_=(1,0), rho=1, lambda_=2, mu=1, dt=1):
    w = 2*np.pi*np.fft.rfftfreq(np.size(src))/dt
    w[0] = 1e-8 #cheat zero frequency to avoid NaN in the green function
    Gxw = green_2D_PSV_xw(x[:,:,np.newaxis], w[np.newaxis,np.newaxis,:], rho,lambda_, mu) #out: [Nx, 2, 2, Nw]
    U_xw=np.tensordot(Gxw,F_, axes=(2,0)) #out: [Nx, 2, Nw]
    return np.fft.irfft(U_xw * np.fft.rfft(src)[np.newaxis,np.newaxis,:], axis=-1) #output: [Nx, 2, Nt]
    #return np.zeros((r.size, 2, src.size)) #output: [x, 2, t] -> ux_i(t)=out[i,0,:]; uz_i(t)=out[i,1,:] #TODO
    
