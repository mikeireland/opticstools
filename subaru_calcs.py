"""Some diffraction calculations for the RHEA slit feed. Speak to Mike about details...

Central lenslet mean coupling = 0.292
Edge/top lenslet mean couplings = [0.012,0.028,0.012,0.028]. Sum=0.080
Corner lenslet mean coupling = [0.002,0.010,0.010,0.010]. Sum=0.032
Total coupling = 0.404.

In the lab, with a smaller "pupil" from the SM28 fiber:
0.064
0.049 * 4
0.0369 * 4
Total Coupling = 0.407

"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import opticstools as ot
import pdb
from scipy.ndimage.interpolation import shift
plt.ion()

#Firstly, define a fiber beam
wave = 0.65e-6
m_pix = 0.1e-6
core_diam = 3.5e-6
numerical_aperture = 0.13
sz = 1024
llet_f = 4.64 * 1.1 #Lenslet focal length in mm
llet_w = 1.0  #Lenslet width in mm
nf = 20
nf = 1
f_ratios = np.linspace(1150,1150,nf)
obstruct = 0.25
offset = 0.0e-6; label = 'Perfect Alignment'
#offset = 1.0e-6; label = '1 micron offset'
#offset = 2.0e-6; label = '2 microns offset'

#Offset of the lenslet in mm
llet_offsets=np.array( [[0,0]])

nx = 10
x = (np.arange(nx) + 0.5)/20.0          #Single-sided
x = (np.arange(nx) + 0.5 - nx//2)/10.0  #Dual-sided

xy = np.meshgrid(x,x)
llet_offsets = np.rollaxis(np.array([xy[0],xy[1]]),0,3).reshape(nx*nx,2)

plotit = False

#Now a calculation that mimics the 
pup_size_microns_physical_mm = 1.45/300*7.2
pup_size_lab = 50e-3#9e-3

#Set non-None for this "special" calculation.
lab_pup_scale = pup_size_lab/pup_size_microns_physical_mm
#lab_pup_scale = None

#----

rad_pix = wave/(sz*m_pix)

#Metres per pixel in the lenslet plane.
m_pix_llet = rad_pix*llet_f/1e3

V = ot.compute_v_number(wave, core_diam/2, numerical_aperture)
fib_mode = ot.mode_2d(V, core_diam/2, j=0, n=0, sampling=m_pix,  sz=sz)
fib_angle = np.real(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(fib_mode))))

fib_mode = shift(fib_mode.real,(offset/m_pix,0), order=1)

llet = ot.square(sz, llet_w/rad_pix/llet_f)
mode = llet * fib_angle
fib_llet_loss = np.sum(mode**2)/np.sum(fib_angle**2)

couplings1 = []
couplings2 = []
for llet_offset in llet_offsets:
    for f_ratio in f_ratios:
        l_d_pix = f_ratio*wave/m_pix_llet 
        pup_diam_pix = sz/l_d_pix
        
        #The input pupil, which changes its size dependent on focal ratio.
        pup = ot.circle(sz, pup_diam_pix) - ot.circle(sz, pup_diam_pix*obstruct)
        
        #"Special" calculation of lab pupil...
        if lab_pup_scale:
            pup = ot.circle(sz, pup_diam_pix*lab_pup_scale) 
        
        #Create a psf, shift it by the offset and truncate.
        psf = np.real(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pup))))
        psf = shift(psf, llet_offset*llet_w/rad_pix/llet_f, order=1)
        psf_trunc = psf * llet
        
        #Compute the loss associated with this truncation.
        llet_loss = np.sum(psf_trunc**2)/np.sum(psf**2)
        
        #The PSF at the fiber is complex in general
        psf_fiber = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(psf_trunc)))
        
        #Couplings1 is coupling "at the microlens array", not taking into account lenslet loss.
        couplings1.append(np.sum(psf*mode)**2/np.sum(psf**2)/np.sum(mode**2)*fib_llet_loss)
        
        #Couplings2 is coupling "at the fiber", taking into account the lenslet loss.
        couplings2.append(np.abs(np.sum(psf_fiber*fib_mode))**2/np.sum(np.abs(psf_fiber)**2)/np.sum(fib_mode**2)*llet_loss)

#plt.clf()    

print(np.mean(couplings1))

#plt.plot(f_ratios,couplings1,label='Total Coupling')
if plotit:
    plt.plot(f_ratios,couplings2,label=label)
    plt.xlabel('Input focal ratio')
    plt.ylabel('Central Fiber Coupling')
    plt.axis([700,1650,0,.7])
