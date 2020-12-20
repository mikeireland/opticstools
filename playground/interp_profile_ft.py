#2.08 microns.

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp2d
from scipy.integrate import cumtrapz
import opticstools as ot
import matplotlib.pyplot as plt


mm = np.genfromtxt('2x2_4um_ex.m00', skip_header=4)
x = np.linspace(-8,8,97)
y = np.linspace(0-7.5,25.5-7.5,97)
xy_new = np.linspace(-7.5,7.5,64)
x_ix = np.interp(xy_new, x, np.arange(97))
y_ix = np.interp(xy_new, y, np.arange(97))

ee_func = RectBivariateSpline(y,x, mm)
ee_square_small = ee_func(xy_new, xy_new)
ee_square = np.zeros( (128,128) )
ee_square[32:32+64,32:32+64] = ee_square_small
dx = xy_new[1]-xy_new[0]
wave_in_pix = 4.0/2.4/0.238

#After a Fourier transform, 1 fourier pixel is this many 
ft_pix_scale = wave_in_pix/128

far_field = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ee_square))))**2
xf = ot.azimuthalAverage(far_field, returnradii=True, center=[64,64], binsize=1)
plt.clf()
plt.plot(xf[0]*ft_pix_scale, xf[1]/xf[1][0], label='Azimuthally Averaged Int')
plt.xlim([0,1])
x_for_sum = np.concatenate([[0],xf[0]*ft_pix_scale])
y_for_sum = np.concatenate([[xf[1][0]],xf[1]])
y_for_sum[-1]=0
encircled = cumtrapz(y_for_sum*x_for_sum, x_for_sum)
encircled /= encircled[-1]
plt.plot(x_for_sum[1:], encircled, label='Encircled Energy')
plt.legend()
plt.xlabel('sin(theta)')
plt.ylabel('Intensity or Int Sum')