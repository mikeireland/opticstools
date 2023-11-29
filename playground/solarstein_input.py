"""
Need to diffract through pinhole up to a pupil image that is much, much larger than 
the pinhole.

So the Fraunhofer approximation is appropriate. 

For BIFROST, we have beams on an input pupil that are focussed to a spot, where 
beams are truncated.

So the input beam can be made up of a bunch of independent modes (orthogonal functions on
a circle). Then the pinhole and aperture stop together becomes a modal filter.

The set of final modes can then be diagonalised, forming a set of independent modes
with differing amplitudes.


"""
import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
plt.ion()
if not '..' in sys.path:
    sys.path.insert(0,'..')
import opticstools as ot

f_before = 2*25.4		#Focal length before the pinhole
f_after = 444.5			#Focal length after the pinhole
pinhole_diam = 20e-3 	#Pinhole diameter in mm
wave = 0.63e-3			#Wavelength for single-mode fibre
psize = 18				#Pupil size after pinhole in mm
obs_size = psize * 0.13	#Obstruction size
wave_bif = 1.8e-6		#Bifrost wavelength
g_1e2_width_mm = 2 		#Gaussian 1/e^2 width in mm (for laser)
b_width_mm = 3			#BIFROST width in mm (to check!!!)

sz = 512				#size of wavefront computation : a numerical computation only
before_mm_pix = 0.03	#mm per pixel: a numerical parameter only.
#--------------------------------------------------

image_angular_rad_pix = wave/(sz*before_mm_pix)
image_linear_mm_pix = image_angular_rad_pix * f_before


x = (np.arange(sz)-sz//2) * before_mm_pix
xy_in = np.meshgrid(x,x)
r_in = np.hypot(*xy_in)
intensity_in = np.exp(-(r_in/(g_1e2_width_mm/2))**2)
E_in = np.sqrt(intensity_in)
E_implane = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(E_in)))

E_implane *= ot.circle(sz, pinhole_diam/image_linear_mm_pix, interp_edge=True)

#Now reverse for the outgoing beam.
output_angular_rad_pix = wave/(sz*image_linear_mm_pix)
output_linear_mm_pix = output_angular_rad_pix * f_after

E_out = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(E_implane)))
out_mask = ot.circle(sz, psize/output_linear_mm_pix, interp_edge=True) - \
	ot.circle(sz, obs_size/output_linear_mm_pix, interp_edge=True)
edge_mm = sz//2*output_linear_mm_pix
plt.clf()
plt.imshow(np.abs(E_out * out_mask), extent=[-edge_mm, edge_mm, -edge_mm, edge_mm])
plt.title('Laser electric field')
plt.xlabel('Pupil x distance (mm)')
plt.ylabel('Pupil y distance (mm)')
plt.colorbar()
plt.axis([-15,15,-15,15])

loss = np.sum(np.abs(E_out * out_mask)**2 / np.sum(intensity_in))
print("Loss to mask: {:.3f}".format(loss))

#Now, lets see what happens with BIFROST
wave_long = 1.8e-6