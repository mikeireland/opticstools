"""
In this script, we attempt to back-propagate a mode from LIFE onto a spacecraft with an
annular emission pattern. This idea originally came from Thomas in Zurich.

The final incoherent coupling is just the integral of the intensity pattern over the 
mask.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
if not '..' in sys.path:
    sys.path.insert(0,'..')
import opticstools as ot
import astropy.constants as c, astropy.units as u
plt.ion()

#Key geometric parameters of the array.
mirror_diam = 3.0
spacecraft_inner_diam = 3.5
spacecraft_diam = 4.54
shield_inner_diam = 4.52
dist = 1200.
dist_combiner = 1.5

#Metres per pixel at the entrance to the BC spacecraft
m_pix = 0.003

#Magnify by this factor when getting to the combiner spacecraft
#combiner_mag = 16 

#Size of the large array is m_pix * sz
sz = 2048
field_tel_f = 1/(1/dist_combiner + 1/dist)

entrance_diam = 0.3

#Maximum angle we want to consider (1D for 2D approximation)
max_angle = 2.5/1200.
wave = 18.5e-6

#Maximum angle we want to consider for the full 2D computation.
max_angle_complete = 10*wave/mirror_diam
dm_diam = dist_combiner / dist * mirror_diam

#Do we want to try a diffractive pupil? This doesn't seem to work for a fixed
#maximum diameter of the input mirror
try_diffractive_pupil=False

#How about an apodized pupil?
try_apodized_pupil=False
nspikes=7
frac_rad_diffract = 0.33
#-------
m_pix_dm = wave/(m_pix*sz)*dist_combiner

#First, create the DM (perfect reverse output of PIAA - imperfect has less coupling!)
dm = np.sqrt(ot.circle(sz, dm_diam/m_pix_dm, interp_edge=True))

#Create the efield at the entrance, and normalise here
efield_entrance = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(dm)))
efield_entrance /= np.sqrt(np.sum(np.abs(efield_entrance)**2))
entrance = np.sqrt(ot.circle(sz, entrance_diam/m_pix, interp_edge=True))
efield_toprop = efield_entrance * entrance

#Now focus and propagate!
prop = ot.FresnelPropagator(sz,m_pix, dist, wave)
telescope = ot.curved_wf(sz,m_pix,f_length=dist,wave=wave)
efield_telescope = prop.propagate(efield_toprop*telescope)

#Overlap with the warm spacecraft and the thin part of the sheild
spacecraft = ot.circle(sz, spacecraft_diam/m_pix, interp_edge=True) - \
	ot.circle(sz, spacecraft_inner_diam/m_pix, interp_edge=True)
shield = ot.circle(sz, spacecraft_diam/m_pix, interp_edge=True) - \
	ot.circle(sz, shield_inner_diam/m_pix, interp_edge=True)
shield[sz//2:]=0

#Find the coupling
total_coupling = np.sum(np.abs(efield_telescope)**2 * spacecraft)
print("Total modal coupling from spacecraft: {:.2e}".format(total_coupling))
total_coupling = np.sum(np.abs(efield_telescope)**2 * shield)
print("Total modal coupling from shield: {:.2e}".format(total_coupling))

plt.clf()
plt.imshow(np.abs(efield_telescope)*spacecraft, extent=[-sz//2*m_pix, sz//2*m_pix, -sz//2*m_pix, sz//2*m_pix])
plt.xlabel('Offset (m)')
plt.ylabel('Offset (m)')

theta = np.linspace(0,2*np.pi,100)
plt.plot(mirror_diam/2*np.cos(theta), mirror_diam/2*np.sin(theta),'w')
plt.plot(spacecraft_inner_diam/2*np.cos(theta), spacecraft_inner_diam/2*np.sin(theta),'C1')
plt.plot(spacecraft_diam/2*np.cos(theta), spacecraft_diam/2*np.sin(theta),'r')

if try_diffractive_pupil:
	#Now lets experiment with a diffractive aperture just in the ~300mm entrance.
	x = np.arange(sz)-sz//2
	xy = np.meshgrid(x,x)
	rr = np.sqrt(xy[0]**2 + xy[1]**2)
	theta = np.arctan2(xy[0],xy[1])
	spikes = np.abs( ((theta * nspikes) + np.pi)% (2*np.pi) - np.pi )
	ap = (entrance_diam/m_pix/2.0) - rr + 0.5 - spikes/2/np.pi*entrance_diam/m_pix*frac_rad_diffract 
	ap = np.maximum(np.minimum(ap, 1),0)
	efield_toprop = efield_entrance * ap
	
	#Now focus and propagate!
	efield_telescope_ap = prop.propagate(efield_toprop*telescope)
	
	#Find the coupling
	total_coupling = np.sum(np.abs(efield_telescope_ap)**2 * spacecraft)
	print("Total modal coupling (diffractive): {:.2e}".format(total_coupling))
	
if try_apodized_pupil:
	#Now lets experiment with an apodized aperture.
	x = np.arange(sz)-sz//2
	xy = np.meshgrid(x,x)
	rr = np.sqrt(xy[0]**2 + xy[1]**2)
	theta = np.arctan2(xy[0],xy[1])
	import pdb; pdb.set_trace()
	#!!! Up to here.
	ap = np.maximum(np.minimum(ap, 1),0)
	efield_toprop = efield_entrance * ap
	
	#Now focus and propagate!
	efield_telescope_ap = prop.propagate(efield_toprop*telescope)
	
	#Find the coupling
	total_coupling = np.sum(np.abs(efield_telescope_ap)**2 * spacecraft)
	print("Total modal coupling (apodized): {:.2e}".format(total_coupling))

	
