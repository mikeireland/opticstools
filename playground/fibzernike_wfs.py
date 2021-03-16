from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pdb
import sys
plt.ion()
if not '..' in sys.path:
    sys.path.insert(0,'..')
import opticstools as ot

#First, design a wavefront sensor based on optimising a combination of 
#pupil plane sensitivity in the readout-noise limited regime with throughput to
#BIFROST, with some consideration given to photon-noise. Photon rate is proportional 
#to the square of the real part of E plus the square of the imaginary part of E, so 
#we simply have to (roughly) maximise the imaginary part of E.
#
#In the small signal, photon-limited regime, it doesn't matter how much light is 
#reflected from the fiber. We'll work with 50% electric field (25% reflectivity)
#at the key wavelength.

#Approximately 300 photons are needed. For a 1.8m aperture at 15%
#efficiency in H and at 1 kHz, this is a magnitude of H=10.4.

reflectivity = 0.25
width=6.5

#Preliminaries
sz = 512
diam = 128
obst = 16
psf_diam = sz/diam
x = np.arange(sz)-sz//2
x_frac = 2*x/diam
xy = np.meshgrid(x,x)
rr = np.sqrt(xy[0]**2 + xy[1]**2)

#Create a Gaussian beam as an approximation for a fiber mode.
sig = psf_diam/2.35*0.75
gg = np.exp(-(rr/2/sig)**2)
gg /= np.sqrt(np.sum(gg**2))

#Create a pupil and ADD CUSTOM ABERRATION
pup_outer = ot.utils.circle(sz,diam)
rpup = pup_outer - ot.utils.circle(sz,obst)
lyot_outer = ot.utils.circle(sz,diam*2)
pup = rpup*ot.zernike_wf(sz, coeffs=[0, 0,0, 0,0,0, 0], diam=diam)

#Create the image electric field.
im_E = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pup)))
im_E_orig = im_E.copy()
im_E = im_E*(1-ot.circle(512,width, interp_edge=True))+ 1j*im_E*ot.circle(512,width, interp_edge=True)*np.sqrt(reflectivity)

#Same for the reference image (perfect beam)
rim_E = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(rpup)))
rim_E = rim_E*(1-ot.circle(512,width, interp_edge=True))+ 1j*rim_E*ot.circle(512,width, interp_edge=True)*np.sqrt(reflectivity)

#Calculate the coupling to the fiber.
coupling = np.sum(im_E_orig*ot.circle(512,width, interp_edge=True)*np.conj(gg))*np.sqrt(1-reflectivity)/np.sqrt(np.sum(im_E_orig**2))
print("Coupling: {:.2f}".format(np.abs(coupling)**2))

#Now for the "Lyot" stop
pup_lyot = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(im_E)))
rpup_lyot = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(rim_E)))

plt.figure(1)
plt.clf()
plt.plot(x_frac, np.abs(pup[sz//2]), label='|E| (Pupil)')
plt.plot(x_frac, np.abs(pup_lyot[sz//2]), label='|E| (Lyot)')
plt.plot(x_frac, pup_lyot[sz//2].real, label='Re(E) (Lyot)')
plt.plot(x_frac, pup_lyot[sz//2].imag, label='Im(E) (Lyot)')
plt.legend()
plt.axis([-1.1,1.1,-1.2,1.2])

plt.figure(3)
plt.clf()
plt.plot(x_frac, np.abs(pup_lyot[sz//2])**2, label='I (Lyot)')
plt.plot(x_frac, np.abs(rpup_lyot[sz//2])**2, label='I (ref)')
plt.plot(x_frac, np.abs(pup_lyot[sz//2])**2 - np.abs(rpup_lyot[sz//2])**2, label='I (diff)')
plt.plot(x_frac, (np.abs(pup_lyot[sz//2])**2 - np.abs(rpup_lyot[sz//2])**2)/rpup_lyot[sz//2].imag,\
 label='Phase (linear)')
plt.legend()
plt.axis([-1.1,1.1,-0.5,1.5])

im_outer = np.fft.fftshift(np.abs(np.fft.fft2(np.fft.fftshift(pup_lyot*(1-pup_outer))))**2)
plt.figure(2)
plt.clf()
plt.imshow(im_outer[sz//2-20:sz//2+20,sz//2-20:sz//2+20])

plt.figure(4)
plt.imshow(np.abs(pup_lyot))