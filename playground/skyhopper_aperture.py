"""Lets make an interestng SkyHopper Aperture shape. 

Step 1: Make a polygon approximation and find interior points (with matplotlib)
Step 2 (maybe): Fill in edges.
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.ndimage as nd
import matplotlib.path as mpltPath


#Start off with definitions of the right half of the aperture and mask
APERTURE_PTS = np.array([[100,100],[100,0],[55,0],[55,18],[31,18],[31,28]]) #16,31,12 and 12,55,6
MASK_PTS =     np.array([[25,97], [60,100],[90,90],[100,50],[93,10],[75,0],[60,5],[55,18],[41,23],[31,28],[0,32]]) 


#Copy to the left half as well
neg_x = APERTURE_PTS.copy()
neg_x[:,0] *= -1
APERTURE_PTS = np.concatenate((APERTURE_PTS, neg_x[::-1], [APERTURE_PTS[0]]))

neg_x = MASK_PTS.copy()
neg_x[:,0] *= -1
#MASK_PTS = np.concatenate((MASK_PTS, neg_x[::-1], [MASK_PTS[0]]))

MASK_PTS = np.concatenate((neg_x[::-1], MASK_PTS))

def smooth_corner(edge1, edge2, center, power=0.5, dth=0.01):
    """Create a smooth corner
    
    Parameters
    ----------
    power:
        Power law for the smoothing
    """
    return None
    
def spline_smooth(pts, nnew=500):
    tin = np.arange(len(pts))
    xin = pts[:,0]
    yin = pts[:,1]
    tnew =np.linspace(0,max(tin),nnew)
    xnew = interpolate.CubicSpline(tin,xin,extrapolate='periodic')(tnew)
    ynew = interpolate.CubicSpline(tin,yin,bc_type='periodic')(tnew)
    #import pdb; pdb.set_trace()
    return np.array([xnew, ynew]).T

plt.figure(1)
plt.clf()   
plt.plot(APERTURE_PTS[:,0], APERTURE_PTS[:,1])
plt.plot(MASK_PTS[:,0], MASK_PTS[:,1], '.')
plt.xlabel("X position (mm)")
plt.ylabel("Y position (mm)")
interp_pts = spline_smooth(MASK_PTS)
plt.plot(interp_pts[:,0], interp_pts[:,1])

path = mpltPath.Path(interp_pts)
path_box = mpltPath.Path(APERTURE_PTS)

nx=801
ny=401
scale=0.25
sz = 2048
pxscale = np.degrees(0.85e-6/(0.2*1024/nx))*60 #In arcmin

grid_ix = np.arange(nx)
grid_iy = np.arange(ny)
xy = np.meshgrid(scale*(grid_ix-nx//2), scale*grid_iy)
points = zip(xy[0].flatten(), xy[1].flatten())
inside = path.contains_points(points)
inside_box = path_box.contains_points(points)

pupil = np.zeros((sz,sz))
pupil[sz//2-ny//2:sz//2-ny//2+ny,sz//2-nx//2:sz//2-nx//2+nx]=inside.astype(float).reshape((ny,nx))

plt.figure(2)
plt.clf()
plt.imshow(pupil)
im = np.fft.fftshift(np.abs(np.fft.fft2(pupil))**2)
im /= np.sum(im)
smoothed_im = nd.filters.gaussian_filter(im, 4)

plt.figure(3)
plt.clf()
plt.imshow(np.log10(smoothed_im/np.max(smoothed_im)), vmin=-6, extent=[-pxscale*sz/2, pxscale*sz/2, -pxscale*sz/2, pxscale*sz/2])
plt.xlabel(r'$\Delta$ x (arcmin)')
plt.ylabel(r'$\Delta$ y (arcmin)')
plt.title('PSF (4 arcsec blurred FWHM)')
plt.axis([-8,8,-8,8])
plt.colorbar(label=r'log$_{10}$ (Intensity)')

print("Fraction of aperture inside mask: {0:6.3f}".format(np.sum(inside)/np.sum(inside_box)))