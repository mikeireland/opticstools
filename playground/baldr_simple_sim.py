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
from scipy.linalg import svd


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
det_Inorm = np.sum(det_Ip)

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
	#Divide by 2 on the next line as we are averaging the +ve and -ve components.
	#Divide by the poke_amp because we want the intensity per radian of phase
	Imat[i] = ((det_I1 - det_I2)/det_Inorm/2/poke_amp)[pix_to_use]
	plt.clf()
	plt.imshow(det_I1 - det_I2)
	plt.title('Opposite Poke difference')
	plt.pause(.01)
	
print("Computing SVD")
#W,V = eigh(Imat.T @ Imat) #Here was Mike's intuitive way to start this.
Us, s, Vs = svd(Imat, full_matrices=False)
plt.figure(1)
plt.clf()
plt.semilogy(s, '.', label="Raw")
plt.axis([0,nact,1e-3,1])
plt.xlabel('Index')
plt.ylabel('Singular Value')


#Next normalise by the amplitude of the DM mode (NB something we don't actually know
#for the real system!)
dm_modes = np.tensordot(Us, acts, axes=[[0],[0]])
dm_mode_rms = np.zeros(nact)
wpup = np.where(pup > 0.5)
for i in np.arange(nact):
	dm_mode_rms[i] = np.std(dm_modes[i,wpup[0], wpup[1]])

plt.semilogy(s/dm_mode_rms, '.', label="Normalised")
plt.legend()
	
#Now show the eigenvectors in sensor space
plt.figure(2)
for i in np.arange(nact):
	mode = np.zeros( (sz//oversamp, sz//oversamp) )
	mode[pix_to_use] = Vs[i]
	plt.clf()
	plt.imshow(mode)
	plt.pause(.02)

#Now do a final test to make sure we understand...
#Note that the natural units we're working with here in defining 
#Orthogonality etc is a readout noise limit with 1 electron RMS per pixel.
dm_mode0 = dm_modes[0]/dm_mode_rms[0]
#Put 1/10th of this on the DM.
pup_act = pup * np.exp(1j*dm_mode0 * 0.1)
det_E = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.fftshift(pup_act)) * pmask_ftshift))
det_I1 = ot.rebin(np.abs(det_E)**2,(sz//oversamp,sz//oversamp))
mode0_rms = np.std((det_I1-det_Ip)[pix_to_use]/det_Inorm/0.1)
test = 0
#plt.imshow((det_I1 - det_Ip)/det_Inorm / 0.1)

#pup_act = pup * np.exp(-1j*acts[i]*poke_amp)
#det_E = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.fftshift(pup_act)) * pmask_ftshift))
#det_I2 = ot.rebin(np.abs(det_E)**2,(sz//oversamp,sz//oversamp))

#Divide by 2 on the next line as we are averaging the +ve and -ve components.
#Divide by the poke_amp because we want the intensity per radian of phase
#Imat[i] = ((det_I1 - det_I2)/det_Inorm/2/poke_amp)[pix_to_use]
