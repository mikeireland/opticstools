import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
plt.ion()
if not '..' in sys.path:
    sys.path.insert(0,'..')
import opticstools as ot
from scipy.ndimage import shift
from scipy.linalg import eigh

"""
1.2 microns
15 micron pixels
2.15mm warm stop
Phase mask diameter is 1.2 lambda/D
Pupil is 19 pixels accross
DM is 12x12, with the pupil being 10 actuators across.
"""

sim = 'Nice H'
sim = 'Sydney'

if sim == 'Sydney':
	wave = 1.2e-3
	pix = 15e-3
	psize = 19 #in pixels
	pmask_ld = 1.2
else:
	#Do a H-band Baldr sim
	wave = 1.65e-3
	pix = 24e-3
	psize = 12 #in pixels
	pmask_ld = 1.4

stop_dist = 39.0 #distance to stop
stop = 2.15

oversamp = 2
sz = 128
poke_amp = 0.2
awidth = 0.6 #The actuator influence function width. I think 0.6 is about right for BM.

#Firstly, lets go without Fresnel diffraction. Compute the stop size in lambda/D
stop_diam_lD = stop/(wave/(pix * psize) * stop_dist)

#actuator spacing in pixels
actuator_spacing = psize/10*oversamp
actuator_x = (np.arange(10)-4.5)*actuator_spacing
actuator_xy = np.meshgrid(actuator_x, actuator_x)
actuator_r = np.sqrt(actuator_xy[0]**2 + actuator_xy[1]**2)
actuator_xy = (actuator_xy[0][actuator_r < 5*actuator_spacing], actuator_xy[1][actuator_r < 5*actuator_spacing])
nact = len(actuator_xy[0])

#centered actuator_footprint
x = np.arange(sz) - sz//2 + 0.5
rr = np.meshgrid(x,x)
rr = np.sqrt(rr[0]**2 + rr[1]**2)
gg = np.exp(-(rr/(actuator_spacing*awidth))**2)

#All actuator footprints
acts = np.zeros((nact,sz,sz))
for i in np.arange(nact):
	acts[i] = shift(gg,(actuator_xy[0][i], actuator_xy[1][i]))
	
#The phase mask
pmask = ot.circle(sz, sz/(psize*oversamp)*pmask_ld, interp_edge=True)
pmask = np.exp(1j*np.pi/2*pmask)
pmask *= ot.circle(sz, stop_diam_lD*sz/(psize*oversamp), interp_edge=True)
pmask_ftshift = np.fft.fftshift(pmask)

#Create the input pupil, and the perfect image
pup = ot.circle(sz, psize*oversamp, interp_edge=True)
det_Ep = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.fftshift(pup)) * pmask_ftshift))
det_Ip = ot.rebin(np.abs(det_Ep)**2,(sz//oversamp,sz//oversamp))

#For the interaction matrix, we will need to define which pixels we are using.
pix_to_use = np.where(ot.circle(sz//oversamp, psize*1.05) > 0)
npix = len(pix_to_use[0])

Imat = np.zeros((nact,npix))

#Compute and plot the interaction matrix
plt.clf()
plt.imshow(np.abs(det_Ip))
plt.title('Perfect Pupil')
plt.pause(1)	

#Now poke the actuators!
for i in np.arange(nact):
	pup_act = pup * np.exp(1j*acts[i]*poke_amp)
	det_E = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.fftshift(pup_act)) * pmask_ftshift))
	det_I1 = ot.rebin(np.abs(det_E)**2,(sz//oversamp,sz//oversamp))
	pup_act = pup * np.exp(-1j*acts[i]*poke_amp)
	det_E = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.fftshift(pup_act)) * pmask_ftshift))
	det_I2 = ot.rebin(np.abs(det_E)**2,(sz//oversamp,sz//oversamp))
	Imat[i] = ((det_I1 - det_I2)/det_Ip)[pix_to_use]
	plt.clf()
	plt.imshow(det_I1 - det_I2)
	plt.title('Opposite Poke difference')
	plt.pause(.01)
	
print("Computing eigenfunctions")
W,V = eigh(Imat.T @ Imat)
plt.figure(1)
plt.clf()
plt.semilogy(W[::-1], '.')
plt.axis([0,nact,.01,10])
plt.xlabel('Eigenvalue index')
plt.ylabel('Eigenvalue')

#Now show the eigenvectors
plt.figure(2)
for i in np.arange(nact):
	mode = np.zeros( (sz//oversamp, sz//oversamp) )
	mode[pix_to_use] = V[:,-nact+i]
	plt.clf()
	plt.imshow(mode)
	plt.pause(.02)