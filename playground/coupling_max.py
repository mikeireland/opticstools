import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
plt.ion()
if not '..' in sys.path:
    sys.path.insert(0,'..')
import opticstools as ot

#Fibre core radius in mm
core_radius = 0.015

#Wavelenghth in mm
wavelength_in_mm = np.linspace(0.004,0.012,17)

#Fibre numerical aperture
na = 0.1

#mm per pixel in the image plane.
mm_pix = 0.002

#Focal ratio ***Optimise this for max coupling at a given wavelength***
f_rat = 6.45

#Simulation size
sz = 512

#For an apodized pupil, here is the number of 1/e^2 diameters for apodization
apod_scale = 1.25
#--------Automatic from here-------------
psize = sz/(wavelength_in_mm*f_rat/mm_pix) #Pupil size.
couplings = []
acouplings = []
xy = np.meshgrid(np.arange(sz)-sz//2, np.arange(sz)-sz//2)
rr2 = xy[1]**2 + xy[0]**2

#Now go through 1 wavelength at a time
for ix, wave in enumerate(wavelength_in_mm):
    pup = ot.circle(sz,psize[ix], interp_edge=True)
    pup_apod = ot.circle(sz,psize[ix]*apod_scale, interp_edge=True)*np.exp(-rr2/(psize[ix]*apod_scale/2)**2)
    im_e = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pup)))
    im_e_apod = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pup_apod)))
    V = ot.compute_v_number(wave, core_radius, na)
    mode = ot.mode_2d(V, core_radius, sampling=mm_pix, sz=sz).real
    couplings += [np.abs(np.sum(im_e*mode.conj()))**2/np.sum(np.abs(im_e)**2)/np.sum(np.abs(mode)**2)]
    acouplings += [np.abs(np.sum(im_e_apod*mode.conj()))**2/np.sum(np.abs(im_e_apod)**2)/np.sum(np.abs(mode)**2)]
    print("Wave (mm): {:.4f} V: {:.2f} Coupling: {:.3f} Apod coupling: {:.3f}".format(wave, V, couplings[-1], acouplings[-1]))

#Couping versus wavelength figure.
plt.figure(1)
plt.clf()
plt.plot(wavelength_in_mm*1e3, couplings, label='Airy Disk')
plt.plot(wavelength_in_mm*1e3, acouplings, label=r'1/e$^2$ PIAA')
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel('Coupling')
plt.title(r'Optimisation: 6$\mu$m')
plt.tight_layout()
plt.legend()
plt.show()

x = (np.arange(sz)-sz//2)*mm_pix*1e3
plt.figure(2)
plt.clf()
plt.plot(x, mode[256]/np.max(mode[256]))
plt.plot([-1e3*core_radius,-1e3*core_radius],[0,1],'r')
plt.plot([1e3*core_radius,1e3*core_radius],[0,1],'r')
plt.xlabel(r'Offset ($\mu$m)')
plt.ylabel('Field Amplitude')
plt.tight_layout()