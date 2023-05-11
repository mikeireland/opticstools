"""
Diameter of 94mm, with 18mm secondary obstruction. 5:1 magnification
"""
import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
plt.ion()
if not '..' in sys.path:
    sys.path.insert(0,'..')
import opticstools as ot
beam_diam = 94
sec_obs = 18
mag = 5
V = 2.2
r = 1.75e-3
sz = 1024
wave = 0.63e-3
f_length = 1.4*100 / 6 * 4.5 #effective focal length
sampling = 0.2e-3

beam_diam /= mag
sec_obs /= mag


#start with the fiber in the image plane. 
fibre_mode = ot.mode_2d(V, r, j=0, n=0, sampling=sampling,  sz=sz)
fibre_far_field = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(fibre_mode)))

#1 pix in far field angle is lambda/(sampling*size)
#1 pix in far field after collimating is f_length * lambda/(sampling*size)
mm_pix_pupil = f_length * wave/(sampling*sz)

pupil_beam = ot.circle(sz, beam_diam/mm_pix_pupil, interp_edge='true') - ot.circle(sz, sec_obs/mm_pix_pupil, interp_edge='true')

mode_overlap = np.abs(np.sum(pupil_beam*fibre_far_field))**2/np.sum(np.abs(np.sum(pupil_beam**2)))/np.sum(np.abs(np.sum(fibre_far_field**2)))
print("Mode Overlap: {:.2f}".format(mode_overlap))

x = np.arange(-sz//2,sz//2)*mm_pix_pupil

plt.clf()
plt.plot(x, np.abs(fibre_far_field[sz//2]/fibre_far_field[sz//2,sz//2]))
plt.plot(x, pupil_beam[sz//2])
plt.axis([-12,12,-0.05,1.05])