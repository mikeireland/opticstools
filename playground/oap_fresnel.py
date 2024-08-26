import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
plt.ion()
if not '..' in sys.path:
    sys.path.insert(0,'..')
import opticstools as ot
import scipy.ndimage as nd

ddir = '/Users/mireland/pyxis/telescope/PyxisTel/HeimdallrMirorTesting/'

file = '17July23/BigOAP4_comastfocsub.csv'

#Full diameter in mm
full_diam = 50.8 

#Beam size in mm
beam_diam = 18.0

crop = 1

sz = 256

waves = np.linspace(0.9,1.3,10)*1e-6 #in m
#waves = np.linspace(0.633,0.633,1)*1e-6 #in m
wave_units = 0.633e-6 #Wave units of the solved wavefront surface.

eff_OAP1 = 1363.12/(1 + np.cos(np.radians(3.765)))
eff_OAP2 = 903.669/(1 + np.cos(np.radians(9.268)))
OAP1_to_DM = 825.6 / np.cos(np.radians(3.765))
OAP2_to_DM = (884.381 - 573.664)/np.cos(np.radians(3.765))
DM_virtual_dist = 1/(1/eff_OAP1 - 1/OAP1_to_DM)
#-------
wave = np.mean(waves)
s1 = np.genfromtxt(ddir + file, delimiter=',')
s1_filt = nd.median_filter(s1, 3)
bad = np.where(np.abs(s1_filt - s1) > 0.1)
s1[bad]=s1_filt[bad]
sz1 = 2*(np.minimum(s1.shape[0], s1.shape[1])//2) - crop
s1 = s1[s1.shape[0]//2 - sz1//2:s1.shape[0]//2 + sz1//2,s1.shape[1]//2 - sz1//2:s1.shape[1]//2 + sz1//2]

mm_pix = full_diam / sz1

s_crop = s1[sz1//2-sz//2:sz1//2+sz//2,sz1//2-sz//2:sz1//2+sz//2]
wl_pup = np.zeros((sz,sz))
for w in waves:
	prop1 = ot.FresnelPropagator(sz,mm_pix/1e3, DM_virtual_dist/1e3, w)
	pup = prop1.propagate(ot.circle(sz, beam_diam/mm_pix, interp_edge=True)) * \
		np.exp(2j*np.pi*s_crop*wave_units/w)
	prop2 = ot.FresnelPropagator(sz,mm_pix/1e3, -DM_virtual_dist/1e3, w)
	pup_on_dm = prop2.propagate(pup)
	wl_pup += np.abs(pup_on_dm)**2

im = np.fft.fftshift(np.abs(np.fft.fft2(pup)))**2