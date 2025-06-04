import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
plt.ion()
if not '..' in sys.path:
    sys.path.insert(0,'..')
import opticstools as ot
import scipy.ndimage as nd
from matplotlib.colors import LogNorm

ddir = '/Users/mireland/pyxis/telescope/PyxisTel/HeimdallrMirorTesting/'

#Choose a file
file = '17July23/BigOAP4.csv'
#file = '17July23/BigOAP3.csv'
#file = '17July23/BigOAP2.csv'
#file = '17July23/BigOAP1.csv'
#file = '17July23/BigOAP4_comastfocsub.csv'
#file = '17July23/SmallOAP1_comafocsub.csv'
#file = 'Jun24/Stefan_OAP1_1.csv'
#file = 'Jun24/Stefan_OAP1_2.csv'
#file = 'Sep24/sep24_OAP1_2_50mm.csv'
#file = 'Sep24/sep24_OAP1_1_50mm.csv'
#file = 'Sep24/sep24_OAP1_5_50mm_unmount.csv'
#file = 'Sep24/sep24_OAP1_4_50mm_unmount.csv'
#file = 'Sep24/sep24_OAP1_3_50mm_unmount.csv'
#file = 'Sep24/sep24_OAP1_2_50mm_unmount.csv'
#file = 'Sep24/sep24_OAP1_1_50mm_unmount.csv'
#file = 'Sep24/oct24_OAP1_4_nosub.csv'
#file = 'Sep24/oct24_OAP1_3_nosub.csv'
#file = 'Sep24/oct24_OAP2_1_nosub.csv'
#file = 'Sep24/oct24_OAP2_2_nosub.csv'
file = 'Jun24/Baldr_OAP1.csv'
#file = '2024Feb07/thorlabs_OAP_30deg.csv'

#file = 'Aug24/spherical_aug24_1.csv'
#file = 'Aug24/spherical_aug24_2.csv'
#file = 'Aug24/spherical_aug24_3.csv'
#file = 'Aug24/spherical_aug24_4.csv'
#file = 'Aug24/spherical_aug24_5.csv'
#sz = 60
#mult = 2

#file = 'May25/OAP1_1_May25.csv'
#file = 'May25/OAP1_2_May25.csv'
#file = 'May25/OAP1_0_May25.csv'
#file = 'May25/OAP1_4_May25.csv'


beam_diam = 12.0
full_diam = 25.8
sz = 192
mult = 1

full_diam = 50.8 	#Full diameter in mm
beam_diam = 18.0	#Beam size in mm
sz = 256

sz = 240

crop = 1

#waves = np.linspace(1.1,1.3,5)*1e-6 #in m
#waves = np.linspace(1.5,1.8,5)*1e-6 #in m
waves = np.linspace(0.633,0.633,1)*1e-6 #in m
#waves = np.linspace(0.9,1.15,5)*1e-6 #in m
wave_units = 0.633e-6 #Wave units of the solved wavefront surface.

eff_OAP1 = 1363.12/(1 + np.cos(np.radians(3.765)))
eff_OAP2 = 903.669/(1 + np.cos(np.radians(9.268)))
OAP1_to_DM = 825.6 / np.cos(np.radians(3.765))
OAP2_to_DM = (884.381 - 573.664)/np.cos(np.radians(3.765))
DM_virtual_dist = 1/(1/eff_OAP1 - 1/OAP1_to_DM)
#DM_virtual_dist = 1/(1/eff_OAP2 - 1/OAP2_to_DM)
DM_virtual_dist = -600
imcrop = 128
#-------
wave = np.mean(waves)
s1 = np.genfromtxt(ddir + file, delimiter=',')
s1_filt = nd.median_filter(s1, 3)
bad = np.where(np.abs(s1_filt - s1) > 0.1)
s1[bad]=s1_filt[bad]
s1 *= mult
sz1 = 2*(np.minimum(s1.shape[0], s1.shape[1])//2) - crop
s1 = s1[s1.shape[0]//2 - sz1//2:s1.shape[0]//2 + sz1//2,s1.shape[1]//2 - sz1//2:s1.shape[1]//2 + sz1//2]
mm_pix = full_diam / sz1

#First figure
plt.figure(1)
plt.clf()
plt.imshow(s1, vmax=1.5, vmin=0)
plt.colorbar()

s_crop = s1[sz1//2-sz//2:sz1//2+sz//2,sz1//2-sz//2:sz1//2+sz//2]
wl_pup = np.zeros((sz,sz))
pcirc = ot.circle(sz, beam_diam/mm_pix, interp_edge=True)
for w in waves:
	prop1 = ot.FresnelPropagator(sz,mm_pix/1e3, DM_virtual_dist/1e3, w)
	pup = prop1.propagate(pcirc) * \
		np.exp(2j*np.pi*s_crop*wave_units/w)
	prop2 = ot.FresnelPropagator(sz,mm_pix/1e3, -DM_virtual_dist/1e3, w)
	pup_on_dm = prop2.propagate(pup)
	wl_pup += np.abs(pup_on_dm)**2


#Now try to fit astig and focus
cropped_pup = s_crop * pcirc
x = (np.arange(sz)-sz//2)/(beam_diam/mm_pix)*2
xy = np.meshgrid(x,x)
ww = np.where(pcirc)
x_in_circ = xy[0][ww]
y_in_circ = xy[1][ww]
s_in_circ = s_crop[ww]
xx = np.array([np.ones(len(x_in_circ)), x_in_circ**2-y_in_circ**2, 2*x_in_circ*y_in_circ, x_in_circ**2 + y_in_circ**2])
fit = np.linalg.inv(xx.dot(xx.T)).dot(xx).dot(s_in_circ)
s_sub = np.zeros_like(s_crop)
s_sub[ww] = s_crop[ww] - np.dot(fit, xx)
print(f"Astig 1: {fit[1]*wave_units*1e9:.1f} nm")
print(f"Astig 2: {fit[2]*wave_units*1e9:.1f} nm")
fit_foc = fit
fit_foc[:-1] = 0
s_crop[ww] -= np.dot(fit_foc, xx)


#Make an image
pup_big = np.zeros((1024,1024), dtype=complex)
pup_big[1024//2-sz//2:1024//2+sz//2,1024//2-sz//2:1024//2+sz//2]= np.exp(2j*np.pi*s_crop*wave_units/w)*pcirc
im = np.fft.fftshift(np.abs(np.fft.fft2(pup_big)))**2
pup_big[1024//2-sz//2:1024//2+sz//2,1024//2-sz//2:1024//2+sz//2] = pcirc
imp = np.fft.fftshift(np.abs(np.fft.fft2(pup_big)))**2
strehl = np.max(im)/np.max(imp) * 100
im = im[1024//2-imcrop:1024//2+imcrop,1024//2-imcrop:1024//2+imcrop]
print(f"Strehl: {strehl:.1f}%")

ww = np.where(ot.circle(sz, beam_diam/mm_pix))
pup_phasor = np.mean(pup_on_dm[ww])
phasor_av = np.abs(pup_phasor) #NB, this is actually the key metric...
pup_phasor /= phasor_av
print( f'RMS (rad) after Propagation: {np.std(np.angle(pup_on_dm[ww] / pup_phasor)):.3f}')
print( f'Initial RMS (rad): {np.std(s_crop[ww]*wave_units/wave*2*np.pi):.3f}')

plt.figure(2)
plt.clf()
plt.imshow(np.abs(pup_on_dm)**2)

plt.figure(3)
plt.clf()
im /= np.max(im)
plt.imshow(im, norm=LogNorm(vmin=1e-3,vmax=1))
#plt.imshow(im)
plt.colorbar()

