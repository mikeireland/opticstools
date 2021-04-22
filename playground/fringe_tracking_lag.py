"""
Fringe tracking effects can be computed using the Glindemann textbook formulae, as used
in a similar manner in Ireland and Woillez (2021).
"""

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import pdb
plt.ion()

#Atmospheric Parameters
r_0 = 0.98*.5e-6/np.radians(0.86/3600)  #Visible fried parameter in m.
wave = 0.5e-6                           #Wavelength that this parameter applies at.
v_mn = 9.4                              #9.4 for median seeing. Actually vbar
Bs = np.array([[130.,0], [0,130.]])     #2-dimensional baselines in m
diam = 8                                #telescope diameter in m

#Labels...
labels = [r'$B \parallel v$', r'$B \perp v$']
colours = ['b','g']

#Now for outer scale...
lines = ['-'] #['-', '--']
L_0s =  [100.0]  #[np.inf, 60.0]

#Servo parameters
Ki = 0.6
tau = 0.003
Tint = 0.003

#Spatial frequency in in cycles per mm
ky = np.linspace(-5,5,2048)
nf = 4096  
minf = .9e-3
maxf = 1.1e2  
fs = np.exp(np.linspace(np.log(minf), np.log(maxf), nf))
s = 2j*np.pi*fs

#Gain Laplace transform for a standard integral servo loop. From a textbook
#on adaptive optics (which one???)
G = Ki*np.exp(-tau*s)*(1-np.exp(-Tint*s))/Tint**2/s**2
G[fs > .5/Tint] = 0
error = np.abs(1/(1+G))
Th_f = np.empty( (len(Bs), len(fs)) )

#For plotting... (avoid log(0) and smooth the plot)
g = np.exp(-np.arange(-15,15)**2/2./5**2)
g /= np.sum(g)

#Create the figure and loop over outer scales.
plt.figure(1)
plt.clf()
ax = plt.axes([.15,.15,.8,.8])
for line, L_0 in zip(lines, L_0s):
    print("Outer Scale: {:5.1f}".format(L_0))
    for i, (colour, B, label) in enumerate(zip(colours, Bs, labels)):
        for j, f in enumerate(fs):
            kx = f/v_mn
            k_abs = np.sqrt(kx**2 + ky**2)
            bdotkappa = B[0]*kx + B[1]*ky
            Phi_opd = 0.0229/np.pi**2 * wave**2*r_0**(-5/3.) * (L_0**(-2) + k_abs**2)**(-11/6.)*\
                (2*sp.jv(1,np.pi*diam*k_abs)/(np.pi*diam*k_abs))**2 * np.sin(np.pi*bdotkappa)**2
            Th_f[i,j] = np.trapz(Phi_opd, ky)/v_mn*1e12
        if L_0 != L_0s[0]:
            label=None
        plt.loglog(fs, np.convolve(Th_f[i], g, mode='same'), colour+line, label=label)

        #Now lets take into account a servo loop. "Error" is the fraction of the amplitude that
        #remains after the servo loop.
        plt.loglog(fs, np.convolve(Th_f[i]*error**2, g, mode='same'), colour+':', label=label + ' corrected')
        print("RMS differential error per ms: {:5.3f} microns".format(np.sqrt(np.trapz(fs**2*Th_f[i], fs))*2*np.pi*1e-3))
        print("Fringe Tracker error: {:5.3f} microns".format(np.sqrt(np.trapz(Th_f[i]*error**2, fs))))
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'Power ($\mu$m$^2$/Hz)')
plt.axis([1e-2,1e2, 1e-11,1e3])

