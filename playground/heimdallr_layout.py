"""Layout the HEIMDALLR instrument. Inspired by the way Antoine Merand
laid out PAVO all those years ago (in yorick).

Goal: take this output and input in to a multi-configuration zemax file."""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import pdb
import sys
plt.ion()
np.set_printoptions(precision=5)
if not '..' in sys.path:
    sys.path.insert(0,'..')
import opticstools as ot
import scipy.optimize as op
import scipy.ndimage as nd

BEAM_HEIGHT = 200. #In mm, GUESSED
BEAM_SEP = 240.
BEAM_DIAM = 18.
M1_F = 2500.
M1_THETA = np.radians(6.0) #Angle of beam off M1, in radians.
WAVE_SHORT = 1.48e-3    #Shortest wavelength in mm
WAVE_LONG = 2.4e-3      #Longest wavelength in mm
PIXEL_PITCH = 24e-3     #Pixel size in mm
Z_M1 = 1000.0
X_M3 = 1000.0
M3_TO_FOCUS = np.array([450.,500.,550.,600.])
Z_FOCUS=250.
NTEL=4

#Beam locations in the pupil.
P_OFFSET = BEAM_DIAM*1.8
S32 = np.sqrt(3)/2
PUPIL_LOCATIONS = [[-S32*P_OFFSET, -0.5*P_OFFSET], [S32*P_OFFSET, -0.5*P_OFFSET], [0,0], [0,P_OFFSET]]
PUPIL_LOCATIONS  = np.array(PUPIL_LOCATIONS)

def display_error(string, is_good):
    """Error string display. Good for checking without
    having to plot. """
    if is_good:
        print(string + " [OK] ")
    else:
        print(string + " [ERROR]")

def path_resid(m2_z, m1_xyz, m2_xyz, m3_xyz, m1_m3_path, M1_THETA, return_m2=False):
    """Adjust m2 x and z to match the path length.
    
    Returns
    -------
    Difference between distance and target
    """
    m2_xyz[2] = m2_z
    m2_xyz[0] = m1_xyz[0] + (m2_xyz[2] - m1_xyz[2])*np.tan(M1_THETA)
    if return_m2:
        return m2_xyz
    else:
        return np.sqrt(np.sum((m1_xyz-m2_xyz)**2)) + \
               np.sqrt(np.sum((m2_xyz-m3_xyz)**2)) - m1_m3_path

#*** Some checks ***

#Is Coma OK?
linear_coma = 3.*M1_THETA/16/M1_F*BEAM_DIAM**2
linear_coma_frac = linear_coma/(M1_F/BEAM_DIAM)/WAVE_SHORT
display_error("Linear coma is {0:6.3f} times the diffraction limit.".format(linear_coma_frac), linear_coma_frac<0.5)

#Is the pixel scale OK?
pupil_xsize = P_OFFSET * np.sqrt(3) + BEAM_DIAM
nyquist_pixel = WAVE_SHORT/pupil_xsize*M1_F/2
display_error("Pixel size is {0:6.3f} times nyquist requirement.".format(PIXEL_PITCH/nyquist_pixel), PIXEL_PITCH/nyquist_pixel<1)

#The only thing we actually solve for is the Z location of M2. Everything else is fixed.
#Lets calculate some of these other quantities
m3_xyz = np.zeros( (4,3) )
m3_xyz[:,2] = Z_FOCUS + M3_TO_FOCUS
for i in range(NTEL):
    m3_xyz[i,0] = PUPIL_LOCATIONS[i,0]*M3_TO_FOCUS[i]/M1_F + X_M3
    m3_xyz[i,1] = PUPIL_LOCATIONS[i,1]*M3_TO_FOCUS[i]/M1_F + BEAM_HEIGHT

m1_xyz = np.zeros( (4,3) )
m1_xyz[:,0] = BEAM_SEP*np.arange(NTEL)
m1_xyz[:,1] = BEAM_HEIGHT
m1_xyz[:,2] = Z_M1

#Optical path from M1 to M3
m1_m3_path = M1_F-M3_TO_FOCUS
m2_xyz = np.zeros( (4,3) )
m2_xyz[:,1] = BEAM_HEIGHT

#Now to solve for the other two axes
for i in range(NTEL):
    m2z = op.fsolve(path_resid, 0., args=(m1_xyz[i], m2_xyz[i], m3_xyz[i], m1_m3_path[i], M1_THETA))
    m2_xyz[i] = path_resid(m2z, m1_xyz[i], m2_xyz[i], m3_xyz[i], m1_m3_path[i], M1_THETA, return_m2=True)

#More checks
min_z = np.min(m2_xyz[:,2])
display_error("Minimum M2 z location is {0:6.3f}.".format(PIXEL_PITCH/nyquist_pixel), min_z>0)

#More checks
min_xoffset = np.min(np.abs(m2_xyz[:,0] - m1_xyz[:,0]))
display_error("Minimum M2 beam x offset {0:6.3f}mm.".format(min_xoffset), min_xoffset>1.5*BEAM_DIAM)


#Plot...
plt.figure(1)
plt.clf()
for i in range(NTEL):
    plt.plot([m1_xyz[i,0], m1_xyz[i,0], m2_xyz[i,0], m3_xyz[i,0], X_M3], \
        [0, m1_xyz[i,2], m2_xyz[i,2], m3_xyz[i,2], Z_FOCUS])
    plt.text(m1_xyz[i,0]-5, 5, 'B{0:d}'.format(i+1))
    plt.text(m1_xyz[i,0]-5, m1_xyz[i,2]+5, 'M1')
    plt.text(m2_xyz[i,0]-5, m2_xyz[i,2]+5, 'M2')
plt.axis([1100,-200,0,1100])
plt.axes().set_aspect('equal')
plt.xlabel('X Axis')
plt.ylabel('Z Axis')

#Finally, lets make a pupil appropriate to a 64x64 subarray.
sz = 64
for wave, title, fignum in zip([WAVE_SHORT, WAVE_LONG], ['Shortest Wavelength', 'Longest Wavelength'], [3,5]):
    mm_pix_pupil = wave*M1_F/(sz*PIXEL_PITCH)
    p0 = ot.circle(sz, BEAM_DIAM/mm_pix_pupil)
    pup = np.zeros_like(p0)
    for ploc in PUPIL_LOCATIONS:
        pup += nd.interpolation.shift(p0, ploc[::-1]/mm_pix_pupil, order=1)
    pup = pup[::-1] #Now it is the intuitive way up.

    plt.figure(2)
    plt.clf()
    plt.imshow(pup, extent=[-sz//2*mm_pix_pupil, sz//2*mm_pix_pupil, -sz//2*mm_pix_pupil,sz//2*mm_pix_pupil])
    plt.xlabel('x offset (mm)')
    plt.ylabel('y offset (mm)')
    plt.title('Virtual pupil at M1 (1/4 size at M3)')

    im_highsnr = np.fft.fftshift(np.abs(np.fft.fft2(pup))**2)
    plt.figure(fignum)
    plt.imshow(im_highsnr**.5, aspect='equal')
    plt.title(title)
