"""Determine the ability for VLTI to have simultaneous near-infrared fringes in 
several bandpasses"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import pdb
import sys
plt.ion()
np.set_printoptions(precision=5)
if not '..' in sys.path:
    sys.path.insert(0,'..')
import opticstools as ot
import scipy.optimize as op

def v2_loss(x, wn, nm1_air, n_glass, wl_los=[1.48,1.63,1.95,2.16], wl_his=[1.63,1.81,2.16,2.40], n_sub=4):
    """Find the approximate loss in V^2 in the quadratic
    approximation per m of vacuum
    
    FIXME: comment and add defaults"""
    phase = 2*np.pi*1e6*(x[0]*nm1_air + x[1]*n_glass + (x[0] - 1.0))*wn
    N_wn = len(wn)
    if wl_los is None:
        ix_los = N_wn//n_sub*np.arange(n_sub)
        ix_his = N_wn//n_sub*(np.arange(n_sub)+1)
    else:
        #Wavelength and wavenumber are back to front, so this is 
        #slightly confusing.
        ix_his = [np.where(1./wn <= wl_lo)[0][0] for wl_lo in wl_los][::-1]
        ix_los = [np.where(1./wn <= wl_hi)[0][0] for wl_hi in wl_his][::-1]
    rms = 0.
    for ix_lo,ix_hi in zip(ix_los, ix_his):
        phase_sub = phase[ix_lo:ix_hi]
        rms += np.var(phase_sub)
    rms = np.sqrt(rms/n_sub)
    return rms

#Air properties. Note that this formula isn't supposed to work at longer wavelengths
#Then H or K.
t_air = 5.0 #InC
p_air = 750e2 #In Pascals
h_air = 0.0 #humidity: between 0 and 1
xc_air = 400.
glass = 'si'
delta = 175.0
N_wn = 100
wl_los=np.array([1.48,1.63,1.95,2.16])
wl_his=np.array([1.63,1.81,2.16,2.40])

#The following has a 16% SNR loss just due to dispersion.
#wl_los=[1.25,1.48,1.63,1.95]
#wl_his=[1.33,1.63,1.81,2.40]


#Wave-number in um^-1
wn = np.linspace(1/wl_his[3],1/wl_los[0],N_wn)
mn_wn = 0.5*(wn[1:] + wn[:-1])
del_wn = wn[1:] - wn[:-1]

nm1_air = ot.nm1_air(1./wn,t_air,p_air,h_air,xc_air)
n_glass = ot.nglass(1./wn, glass=glass)

#Derivatives evaluated everywhere but endpoints.
d1_air = (nm1_air[2:]-nm1_air[:-2])/(wn[2:] - wn[:-2]) 
d2_air = (nm1_air[2:]-2*nm1_air[1:-1]+nm1_air[:-2])/(0.5*(wn[2:] - wn[:-2]))**2
b1_air = 1.0 + nm1_air[1:-1] + wn[1:-1] * d1_air 
b2_air = d1_air + 1/2. * wn[1:-1] * d2_air

d1_glass = (n_glass[2:]-n_glass[:-2])/(wn[2:] - wn[:-2]) 
d2_glass = (n_glass[2:]-2*n_glass[1:-1]+n_glass[:-2])/(0.5*(wn[2:] - wn[:-2]))**2
b1_glass = n_glass[1:-1] + wn[1:-1] * d1_glass
b2_glass = d1_glass + 1/2. * wn[1:-1] * d2_glass

b_arrs = np.array([[b1_air,b1_glass],[b2_air,b2_glass]])
b_arrs = b_arrs.transpose( (2,0,1) )
x0s = np.zeros( (len(b1_air),2) )
x0s[:,0] = 1.0
x_matsolve = np.linalg.solve(b_arrs, x0s)

#Unfortunately, that was just a guess. Now we need a least-squares about this
#to optimise the amount of glass
x0 = x_matsolve[N_wn//2]
print(v2_loss(x0, wn, nm1_air, n_glass))
#best_x = op.minimize(v2_loss, x0, args=(wn, nm1_air, n_glass), options={'eps':1e-13, 'gtol':1e-4}, tol=1e-6, method='bfgs')
best_x = op.minimize(v2_loss, x0, args=(wn, nm1_air, n_glass, wl_los, wl_his), tol=1e-8, method='Nelder-Mead') 
print(v2_loss(best_x.x, wn, nm1_air, n_glass))

phase = 2*np.pi*delta*1e6*(best_x.x[0]*(nm1_air+1.0) + best_x.x[1]*n_glass - 1.0)*wn
fig1=plt.figure(1)
fig1.clf()
ax1 = fig1.add_subplot(111)
ax1.plot(1/wn, phase-np.mean(phase))
for wl_lo, wl_hi in zip(wl_los, wl_his):
    ax1.add_patch(patches.Rectangle((wl_lo,-5), wl_hi-wl_lo, 10.0,alpha=0.1,edgecolor="grey"))
#Need to neaten this
#plt.plot(1/wn[[25,50,75]], phase[[25,50,75]] - np.mean(phase),'o')

plt.xlabel('Wavelength')
plt.ylabel('Phase (radians)')
plt.title('Fringe Phase at {0:5.1f}m of air path'.format(delta))

#Original test plotting code
if False:
    #Rather than picking a single wavelength, lets take a few
    #key wavelengths and see the result
    plt.clf()
    for ix in [25,50,75]:
        x_air = x[ix,0]
        x_glass = x[ix,1]

        #Now compute the fringe phase as a function of wavenumber
        phase = 1e6*(x_air*nm1_air[1:-1] + x_glass*n_glass[1:-1] - delta)*wn[1:-1]

        plt.plot(1./wn[1:-1], phase - np.mean(phase))