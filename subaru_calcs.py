"""Some diffraction calculations for the RHEA slit feed. Speak to Mike about details..."""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import opticstools as ot
import pdb
plt.ion()

#Firstly, define a fiber beam
wave = 0.65e-6
m_pix = 0.2e-6
core_diam = 3.5e-6
numerical_aperture = 0.13
sz = 512
llet_f = 4.64 * 1.1 #Lenslet focal length in mm
llet_w = 1.0  #Lenslet width in mm
nf = 20
f_ratios = 700 + np.arange(nf)*50
obstruct = 0.25
offset = 0.0e-6; label = 'Perfect Alignment'
#offset = 1.0e-6; label = '1 micron offset'
#offset = 2.0e-6; label = '2 microns offset'

llet_offset=0.67
#----

rad_pix = wave/(sz*m_pix)
m_pix_llet = rad_pix*llet_f/1e3
V = ot.compute_v_number(wave, core_diam/2, numerical_aperture)
fib_mode = ot.mode_2d(V, core_diam/2, j=0, n=0, sampling=m_pix,  sz=sz)
fib_angle = np.real(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(fib_mode))))

llet = np.roll(ot.square(sz, llet_w/rad_pix/llet_f), int(llet_offset*llet_w/rad_pix/llet_f))
fib_mode = np.roll(fib_mode,int(offset/m_pix), axis=0)

mode = llet * fib_angle

couplings1 = []
couplings2 = []
for f_ratio in f_ratios:
    l_d_pix = f_ratio*wave/m_pix_llet 
    pup_diam_pix = sz/l_d_pix
    pup = ot.circle(sz, pup_diam_pix) - ot.circle(sz, pup_diam_pix*obstruct)
    psf = np.real(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pup))))
    psf_trunc = psf * llet
    llet_loss = np.sum(psf_trunc**2)/np.sum(psf**2)
    psf_fiber = np.real(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(psf_trunc))))
    couplings1.append(np.sum(psf*mode)**2/np.sum(psf**2)/np.sum(mode**2))
    couplings2.append(np.sum(psf_fiber*fib_mode)**2/np.sum(psf_fiber**2)/np.sum(fib_mode**2)*llet_loss)

#plt.clf()    

#plt.plot(f_ratios,couplings1,label='Total Coupling')
plt.plot(f_ratios,couplings2,label=label)
plt.xlabel('Input focal ratio')
plt.ylabel('Central Fiber Coupling')
plt.axis([700,1650,0,.7])
