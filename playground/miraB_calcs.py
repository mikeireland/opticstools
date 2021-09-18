import numpy as np
import matplotlib.pyplot as plt
import sys
if not '..' in sys.path:
    sys.path.insert(0,'..')
import opticstools as ot
    
eta_w = 0.2 #VLTI efficiency at K
diam = 1.8
strehl = 0.7
seeing = 0.7
sep = 0.48

#Photon rate for target and background after the warm optics
photons_s_B = eta_w*strehl*np.pi*(diam/2)**2*4.31e9*.4*10**(-10*0.4)
bg_phot = 2*3e8/2.2e-6*.4/2.2/(np.exp(6.626e-34*3e8/2.2e-6/1.38e-23/290)-1)

difflim = 2.2e-6/diam*203e3

residual_seeing_coupling = (1-strehl)*(difflim/seeing)**2*np.exp(-(sep/seeing*2.355)**2)

x = np.linspace(-4,4,400)
xy = np.meshgrid(x,x)
rr = np.sqrt(xy[0]**2 + xy[1]**2)
gg = np.exp(-1.3*rr**2)
e_airy = ot.airy(rr, obstruction_sz=0.1)
frac_coupling = np.abs(np.sum(np.roll(gg,100)*e_airy)/np.sum(gg*e_airy))**2

#Mira A flux from Whitelock
#https://academic.oup.com/mnras/article/319/3/728/1073962
photons_s_A = eta_w*strehl*np.pi*(diam/2)**2*4.31e9*.4*10**(2.45*0.4)*(frac_coupling + residual_seeing_coupling)

print('Photon rate from Mira B, Mira A and Background: {:1e} {:.1e} {:.1e}'.format(photons_s_B, photons_s_A, bg_phot))
print('SNR decrease due to A: {:1f}'.format(np.sqrt(photons_s_A/photons_s_B)))