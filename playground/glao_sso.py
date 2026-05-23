"""
Simulate a ground-laser adaptive optics system,
using the Fresnel propagation from opticstools.

Bid picture: for a 5m telescope with 2800 actuators, there
is a 10cm spacing between actuators. At a 
35m altitude, 5 arcmin is about 5cm, so GLAO will
correct this well in principle.

This is just a first pass assuming only servo lag and bright stars.
But given that ground layer seeing is slow, there should be a
reasonable sky coverage near the galactic plane where this works.
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
plt.ion()
if not '..' in sys.path:
    sys.path.insert(0,'..')
import opticstools as ot

sz = 2048
m_per_pix = 0.02
wave = 0.5e-6
imsz = np.degrees(wave/m_per_pix)*3600
actuator_sep = 0.10
mask = np.zeros((sz, sz))
actuator_du = int(m_per_pix/actuator_sep*sz)
mask[sz//2-actuator_du//2:sz//2+actuator_du//2, sz//2-actuator_du//2:sz//2+actuator_du//2] = 1
mask = np.fft.fftshift(mask)
print('Image size is %.1f arcsec' % imsz)
layer_alts = np.array([37.5, 250, 1000, 3000, 6000, 9000, 13500])
fturb = np.array([0.7575, 0.1185, 0.0489, 0.0447, 0.0095, 0.0088, 0.0122])
layer_alts = layer_alts[::-1]
fturb = fturb[::-1]
constellation_dx = np.radians(np.array([-2,2,0,0,0])/60)
constellation_dy = np.radians(np.array([0,0,-2,2,0])/60)
# Convert to pixel offsets at each layer.
dxy = np.array([constellation_dx, constellation_dy])*layer_alts[:,None,None]/m_per_pix

seeing = 1.2427
r0 = ot.utils.r0(seeing, wave)
print('r0 = %.2f cm' % (r0*1e2))

# Set up the turbulence layers. The fractional turbulence is the wavefront
# variance in each layer, so we can use this to set the r0 in each layer, 
# noting the formula for the wavefront variance in each layer is 
# proportional to (r0)**(-5/3), so r0 is proportional to (fturb)**(-3/5)
layer_r0s = r0*(fturb)**(-3/5)
print('Layer r0s (cm):', layer_r0s*1e2)

# Aperture size is 5m
aperture = ot.utils.circle(sz, 5.0/m_per_pix, interp_edge=True)
aperture = np.fft.fftshift(aperture)

# Initialise the Fresnel Propagators.
propagators = []
for alt0, alt1 in zip(layer_alts[:-1], layer_alts[1:]):
    d = alt0 - alt1
    propagators.append(ot.FresnelPropagator(sz,m_per_pix, d, wave))
propagators.append(ot.FresnelPropagator(sz,m_per_pix, layer_alts[-1], wave))
nsample = 16
# Create the layers as phase screens.
phase_screens = []
nstars = dxy.shape[2]
im_uncorr = np.zeros((nstars, sz, sz))
im_corr = np.zeros((sz, sz))
for _ in range(nsample):
    wf = np.ones((nstars, sz, sz), dtype=complex)
    for ii in range(len(layer_alts)):
        print("Simulating layer %d" % ii)
        screen = np.exp(1j*ot.kmf(sz, r_0_pix = layer_r0s[ii]/m_per_pix))
        for jj in range(nstars):
            screen_shifted = np.roll(screen, int(dxy[ii, 0, jj]), axis=1)
            screen_shifted = np.roll(screen_shifted, int(dxy[ii, 1, jj]), axis=0)
            wf[jj] *= screen_shifted
            if ii < len(layer_alts)-1:
                wf[jj] = propagators[ii].propagate(wf[jj])
            else:
                im_uncorr[jj] += np.abs(np.fft.fftshift(np.fft.fft2(wf[jj]*aperture)))**2
        
    wfcorr = np.sum(wf[:4], axis=0)
    wfcorr = np.fft.ifft2(mask * np.fft.fft2(wfcorr))
    wfcorr = wfcorr/np.abs(wfcorr)
    im_corr += np.abs(np.fft.fftshift(np.fft.fft2(wf[jj]*wfcorr.conj()*aperture)))**2
    print('Done with sample %d' % _)
    #phase_screens.append(wf)

print('Finding azimuthally averaged profiles')
bin_centers,prof_corr = ot.azimuthalAverage(im_corr, returnradii=True, binsize=3)
bin_centers,prof_uncorr = ot.azimuthalAverage(im_uncorr[-1], returnradii=True, binsize=3)