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
pinhole_diam = 50e-3 	#Pinhole diameter in mm
wave = 0.63e-3			#Wavelength for single-mode fibre
psize = 18				#Pupil size after pinhole in mm
obs_size = psize * 0.13	#Obstruction size
wave_bif = 1.0e-3		#Bifrost wavelength
g_1e2_width_mm = 2 		#Gaussian 1/e^2 width in mm (for laser)
b_width_mm = 2*.2*7		#BIFROST width in mm (based on 0.2 NA and 7mm focal length)

sz = 512				#size of wavefront computation : a numerical computation only
before_mm_pix = 0.05	#mm per pixel: a numerical parameter only.
nzernikes = 15			#Number of zernikes
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
I_implane_laser = np.abs(E_implane)**2

#Now reverse for the outgoing beam.
output_angular_rad_pix = wave/(sz*image_linear_mm_pix)
output_linear_mm_pix = output_angular_rad_pix * f_after

E_out = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(E_implane)))
out_mask = ot.circle(sz, psize/output_linear_mm_pix, interp_edge=True) - \
	ot.circle(sz, obs_size/output_linear_mm_pix, interp_edge=True)
edge_mm = sz//2*output_linear_mm_pix

plt.figure(1)
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
wave_long = 1.0e-6


image_angular_rad_pix_bif = wave_bif/(sz*before_mm_pix)
image_linear_mm_pix_bif = image_angular_rad_pix_bif * f_before
output_angular_rad_pix_bif = wave_bif/(sz*image_linear_mm_pix_bif)
output_linear_mm_pix_bif = output_angular_rad_pix_bif * f_after
out_mask = ot.circle(sz, psize/output_linear_mm_pix_bif, interp_edge=True) - \
	ot.circle(sz, obs_size/output_linear_mm_pix_bif, interp_edge=True)
edge_mm = sz//2*output_linear_mm_pix_bif

I_out = np.zeros((sz,sz))
E_outs = np.zeros((nzernikes,sz,sz), dtype=complex)
for i in range(nzernikes):
	coeffs = np.zeros(nzernikes)
	coeffs[i] = 1
	E_in = ot.zernike(sz, coeffs=coeffs, diam=b_width_mm/before_mm_pix, rms_norm=True) * \
		ot.circle(sz, b_width_mm/before_mm_pix, interp_edge=True)
	E_implane = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(E_in)))
	E_implane *= ot.circle(sz, pinhole_diam/image_linear_mm_pix_bif, interp_edge=True)
	E_outs[i] = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(E_implane)))  * out_mask
	I_out += np.abs(E_outs[i])**2
	
#Now, lets compute the overlap integrals
overlaps = np.zeros((nzernikes, nzernikes), dtype=complex)
for i in range(nzernikes):
	for j in range(nzernikes):
		overlaps[i,j] = np.sum(E_outs[i]*E_outs[j].conj())
W, V = np.linalg.eigh(overlaps)
single_mode_frac = np.abs(W[-1])**2/np.sum(np.abs(W)**2)
print("Fractional energy in fundamental mode: {:.3f}".format(single_mode_frac))
modes = E_outs.T.dot(V.dot(np.diag(W)))

#To find coupling for 2 different alignments, if we have 2 modes:
#M1 = a11 E1 + a12 E2
#M2 = a21 E1 + a22 E2
#Fringes have a complex visibility:
#<M1 | M2*> = a11 a21* + a12 a22*
#Lets have a 

#M1 = np.sqrt(intensity_in) * np.exp(1j*xy_in[0]/psize*np.pi)
	
plt.figure(2)
plt.clf()
plt.imshow(np.abs(I_out), extent=[-edge_mm, edge_mm, -edge_mm, edge_mm])
plt.title('Intensity')
plt.xlabel('Pupil x distance (mm)')
plt.ylabel('Pupil y distance (mm)')
plt.colorbar()
plt.axis([-15,15,-15,15])

plt.figure(4)
plt.clf()
plt.imshow(np.abs(modes[:,:,-1])**2, extent=[-edge_mm, edge_mm, -edge_mm, edge_mm])
plt.title('Mode 0 Intensity')
plt.xlabel('Pupil x distance (mm)')
plt.ylabel('Pupil y distance (mm)')
plt.colorbar()
plt.axis([-15,15,-15,15])

plt.figure(5)
plt.clf()
plt.imshow(np.abs(modes[:,:,-2])**2, extent=[-edge_mm, edge_mm, -edge_mm, edge_mm])
plt.title('Mode 1 Intensity')
plt.xlabel('Pupil x distance (mm)')
plt.ylabel('Pupil y distance (mm)')
plt.colorbar()
plt.axis([-15,15,-15,15])

plt.figure(6)
plt.clf()
plt.imshow(np.abs(modes[:,:,-6]), extent=[-edge_mm, edge_mm, -edge_mm, edge_mm])
plt.title('Mode 5 Intensity')
plt.xlabel('Pupil x distance (mm)')
plt.ylabel('Pupil y distance (mm)')
plt.colorbar()
plt.axis([-15,15,-15,15])
	
print("microns per pixel in image plane (laser): {:.1f}".format(image_linear_mm_pix*1000))
print("microns per pixel in image plane (BIFROST): {:.1f}".format(image_linear_mm_pix_bif*1000))
print("Output microns per pixel (laser): {:.1f}".format(output_linear_mm_pix*1000))
print("Output microns per pixel (BIFROST): {:.1f}".format(output_linear_mm_pix_bif*1000))

plt.figure(3)
plt.clf()
edge_mm = sz//2*image_linear_mm_pix_bif*1000
plt.imshow(np.abs(I_implane_laser), extent=[-edge_mm, edge_mm, -edge_mm, edge_mm])
plt.title('Laser Efield')
plt.xlabel('Image x distance (um)')
plt.ylabel('Image y distance (um)')
plt.colorbar()
plt.axis([-25,25,-25,25])


