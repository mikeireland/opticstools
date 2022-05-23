"""Determine the ability for VLTI to have simultaneous near-infrared fringes in 
several bandpasses

#For K-band, by using delta/n_air_group of air to compensate for
#delta vacuum delay, we have +/- pi radians of phase as a worst case.
#This reduces visibility by 2/np.pi, which is significant but maybe not 
#enough to justify atmospheric dispersion correction.

dwn = wn[1:]-wn[:-1]
cwn = cwn = 0.5*(wn[1:]+wn[:-1])
dnm1 = (nm1_air[1:] - nm1_air[:-1])
cnm1 = 0.5*(nm1_air[1:] + nm1_air[:-1])
group_index = 1.0 + cnm1 + cwn*dnm1/dwn
phase = 175e6*2*np.pi*cwn*(1-(1+cnm1)/group_index[22])
plt.clf()
plt.plot(1/cwn, phase - phase[22])
plt.axis([1.95,2.35,-4,0.5])
plt.xlabel('Wavelength (microns)')
plt.ylabel('Phase (radians)')


#Sensitivity at f/20 in 24 micron pixels.
#wave = 1.9 + np.arange(61)/100
#flux_mode = 2/(np.exp(6.626e-34*3e8/wave/1e-6/288/1.38e-23)-1)*3e8/2.2e-6*.01/2
#flux_pixel = flux_mode * (24/2.3)**2 * np.pi*(1/40)**2
"""

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

def vis_loss(x, wn, nm1_air, n_glass, wl_los=[1.48,1.63,1.95,2.16], wl_his=[1.63,1.81,2.16,2.40], n_sub=4):
    """Find the approximate loss in V^2 in the quadratic
    approximation per 100m of vacuum
    
    FIXME: comment and add defaults"""
    if len(x)>1:
        phase = 2*np.pi*1e6*(x[0]*nm1_air + x[1]*n_glass + (x[0] - 1.0))*wn
    else:
        phase = 2*np.pi*1e6*(x[0]*nm1_air + (x[0] - 1.0))*wn
    N_wn = len(wn)
    if wl_los is None:
        ix_los = N_wn//n_sub*np.arange(n_sub)
        ix_his = N_wn//n_sub*(np.arange(n_sub)+1)
    else:
        #Wavelength and wavenumber are back to front, so this is 
        #slightly confusing.
        ix_his = [np.where(1./wn <= wl_lo)[0][0] for wl_lo in wl_los][::-1]
        ix_los = [np.where(1./wn <= wl_hi)[0][0] for wl_hi in wl_his][::-1]
    mnsq = 0.
    for ix_lo,ix_hi in zip(ix_los, ix_his):
        phase_sub = phase[ix_lo:ix_hi]
        mnsq += np.var(phase_sub)
    return 100**2*mnsq/n_sub

#Air properties. Note that this formula isn't supposed to work at longer wavelengths
#Then H or K.
plot_extra=False
t_air = 5.0 #InC
p_air = 750e2 #In Pascals
h_air = 0.0 #humidity: between 0 and 1
xc_air = 400.
glass = 'si' 
glass = 'znse' 
delta = 100.0
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
print(vis_loss(x0, wn, nm1_air, n_glass))
#best_x = op.minimize(vis_loss, x0, args=(wn, nm1_air, n_glass), options={'eps':1e-13, 'gtol':1e-4}, tol=1e-6, method='bfgs')
best_x = op.minimize(vis_loss, x0, args=(wn, nm1_air, n_glass, wl_los, wl_his), tol=1e-8, method='Nelder-Mead') 
print(vis_loss(best_x.x, wn, nm1_air, n_glass))

wn = np.linspace(1/wl_his[3],1/wl_los[0],N_wn)

wn = np.linspace(1/2.5,1/1.4,N_wn)
nm1_air = ot.nm1_air(1./wn,t_air,p_air,h_air,xc_air)
n_glass = ot.nglass(1./wn, glass=glass)


phase = 2*np.pi*delta*1e6*(best_x.x[0]*(nm1_air+1.0) + best_x.x[1]*n_glass - 1.0)*wn
fig1=plt.figure(1)
fig1.clf()
ax1 = fig1.add_subplot(111)
ax1.plot(1/wn, phase-np.mean(phase), 'k', label='Phase')
ax1.axis([1.4,2.5,-0.3,0.3])
if not plot_extra:
    for wl_lo, wl_hi in zip(wl_los, wl_his):
        ax1.add_patch(patches.Rectangle((wl_lo,-5), wl_hi-wl_lo, 10.0,alpha=0.1,edgecolor="grey"))
#Need to neaten this
#plt.plot(1/wn[[25,50,75]], phase[[25,50,75]] - np.mean(phase),'o')

plt.xlabel('Wavelength')
plt.ylabel(r'Fringe Phase (radians)')
plt.title('{0:5.1f}m of air path and 2.3mm PWV'.format(delta))

print('Glass thickness: {:5.2f}mm'.format(best_x.x[1]*delta*1e3))

#Now for K only
wl_los_k=np.array([1.95,2.16])
wl_his_k=np.array([2.16,2.40])
best_x_k = op.minimize(vis_loss, [1.0], args=(wn, nm1_air, n_glass, wl_los_k, wl_his_k), tol=1e-8, method='Nelder-Mead') 
print(vis_loss(best_x_k.x, wn, nm1_air, n_glass,wl_los_k, wl_his_k))
phase_k = 2*np.pi*delta*1e6*(best_x_k.x[0]*(nm1_air+1.0) - 1.0)*wn
fig2=plt.figure(2)
fig2.clf()
ax2 = fig2.add_subplot(111)
ax2.plot(1/wn, phase_k-np.min(phase_k)-2, 'k', label='Phase')
ax2.axis([1.6,2.5,-3,20])
plt.xlabel('Wavelength')
plt.ylabel(r'Fringe Phase (radians)')
plt.title('{0:5.1f}m of air path and 2.3mm PWV'.format(delta))
if not plot_extra:
    for wl_lo, wl_hi in zip(wl_los_k, wl_his_k):
        ax2.add_patch(patches.Rectangle((wl_lo,-5), wl_hi-wl_lo, 10.0,alpha=0.1,edgecolor="grey"))

if plot_extra:
    dir = '/Users/mireland/Google Drive/LE19_Heimdallr/Dewar Technical Documents/'
    lfilt = 1400 + np.arange(2200)/2
    d1 = np.loadtxt(dir + 'SmallDichroic1.csv', delimiter=',', skiprows=1)
    d2 = np.loadtxt(dir + 'SmallDichroic2.csv', delimiter=',', skiprows=1)
    d3 = np.loadtxt(dir + 'SmallDichroic3.csv', delimiter=',', skiprows=1)
    d4 = np.loadtxt(dir + 'SmallDichroic4.csv', delimiter=',', skiprows=1)
    filt = np.loadtxt(dir + 'SmallFilter.csv', delimiter=',', skiprows=1)
    ldich = np.loadtxt(dir + 'LargeDichroic.csv', delimiter=',', skiprows=1)
    atm = np.loadtxt(dir + 'cptrans_zm_23_10.dat')
    d1f = np.interp(lfilt, d1[::-1,0], d1[::-1,1])/1e2
    d2f = np.interp(lfilt, d2[::-1,0], d2[::-1,1])/1e2
    d3f = np.interp(lfilt, d3[::-1,0], d3[::-1,1])/1e2
    d4f = np.interp(lfilt, d4[::-1,0], d4[::-1,1])/1e2
    filtf = np.interp(lfilt, filt[::-1,0], filt[::-1,1])/1e2
    ldichf = np.interp(lfilt, ldich[::-1,0], ldich[::-1,1])/1e2
    atmf = np.interp(lfilt, atm[:,0]*1e3, atm[:,1])
    atmf = np.convolve(atmf, np.ones(9)/9, mode='same')
    ax2 = ax1.twinx()
    ax2.plot(lfilt/1e3, atmf*(1-ldichf)*filtf*(1-d1f)*(1-d2f)*(1-d3f), label='H1')
    ax2.plot(lfilt/1e3, atmf*(1-ldichf)*filtf*(1-d1f)*(1-d2f)*d3f*(1-d4f), label='H2')
    ax2.plot(lfilt/1e3, atmf*(1-ldichf)*filtf*(1-d1f)*d2f, label='K1')
    ax2.plot(lfilt/1e3, atmf*(1-ldichf)*filtf*d1f, label='K2')
    #ax2.plot(atm[:,0], np.convolve(atm[:,1], np.ones(41)/41, mode='same'), '--')
    ax2.axis([1.4,2.5,0,1])
    ax2.set_ylabel('Throughput')
    fig1.tight_layout()
    #ax1.legend(loc='lower right', framealpha=.95)
    ax2.legend(loc='lower right', framealpha=.9)

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