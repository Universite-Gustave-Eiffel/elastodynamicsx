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

# Green(r,w): space - frequency domain
#def green_2D_PSV_rw(r, w, rho=1, lambda_=2, mu=1): #TODO

# U(r,t): space - time domain after convolution with a source
def u_2D_PSV_rt(r, src, F_=(1,0), rho=1, lambda_=2, mu=1, dt=1):
    return np.zeros((r.size, 2, src.size)) #output: [x, 2, t] -> ux_i(t)=out[i,0,:]; uz_i(t)=out[i,1,:] #TODO
