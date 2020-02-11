from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import sys
if not '..' in sys.path:
    sys.path.insert(0,'..')
import opticstools as ot
import scipy.ndimage as nd

#Lets start with some definitions. Put everything in length units of mm
mm_pix = 0.0025
sz = 4096 #Size in pixels.
ulens_diameter = 0.25
d_to_lens1 = 2.0
wave=800e-6
flength1 = 20.
cyl_flength = 1
flength2 = 20.
#-----
one_aperture = ot.utils.circle(sz, ulens_diameter/mm_pix, interp_edge=True)
lens1 = ot.curved_wf(sz, mm_pix, wave=wave, f_length=flength1)
initial_wf = nd.interpolation.shift(one_aperture, [-3*ulens_diameter/mm_pix,0]) + \
    nd.interpolation.shift(one_aperture, [3*ulens_diameter/mm_pix,0]) + nd.interpolation.shift(one_aperture, [1*ulens_diameter/mm_pix,0])
wf_at_sph_lens = ot.propagate_by_fresnel(initial_wf, mm_pix, d_to_lens1, wave)

wf_at_cyl_lens = ot.propagate_by_fresnel(wf_at_sph_lens*lens1, mm_pix, flength1-cyl_flength, wave)
print("Made it to cylindrical lens!")

#Now the painful bit: create a cylindrical lens.
x = np.arange(sz) - sz//2
xy = np.meshgrid(x,x)
power = 1.0/cyl_flength
phase = 0.5*mm_pix**2/wave*power*xy[0]**2 
cyl_wf=np.exp(2j*np.pi*phase)

#Find the final fringes!
image_plane_E = ot.propagate_by_fresnel(wf_at_cyl_lens*cyl_wf, mm_pix, cyl_flength, wave)
print("Made it to image plane!")
plt.figure(1)
plt.clf()
plt.imshow(np.abs(image_plane_E))

#for i in range(20):
#    plt.clf()
#    wf = ot.propagate_by_fresnel(wf_at_sph_lens*lens1, mm_pix, i*0.5, wave)
#   plt.imshow(np.abs(wf))
#    plt.pause(.01)

wf_at_lens1 = ot.propagate_by_fresnel(image_plane_E, mm_pix, flength2, wave)
lens2 = ot.curved_wf(sz, mm_pix, wave=wave, f_length=flength2)
wf_at_grating = ot.propagate_by_fresnel(wf_at_lens1*lens2, mm_pix, flength2, wave)

plt.figure(2)
plt.clf()
plt.imshow(np.abs(wf_at_grating))