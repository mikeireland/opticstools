"""
Compute spherical aberration due to a glass plate
as a ray angle, then convert to wavefront error.

Small Angle that we subtract for the offset:
offset = t*theta/n

Full Angle from Snell's law:

offset_full = t*tan(arcsin(sin(x)/n))
sin(x) ~ x - x^3/6 + ...
arcsin(x) ~ x + x^3/6 + ...

So:
arcsin(sin(x)/n) ~ arcsin(x/n - x^3/(6n))
 ~ x/n + (x/n)^3/6 - x^3/(6n) + ...
 = x/n + x^3(1 - n^2)/(6 n^3) + ...
tan(arcsin(sin(x)/n)) ~ x/n + x^3(1 - n^2)/(6 n^3)  + ...

Wavefront slope is this offset divided by focal length f, and 
offset y = x*f, with x the ray angle in air.
Therefore wavefront error as a slope is:
dw/dx = (t/f) * (y/f)^3(1 - n^2)/(6 n^3)

Peak to valley in length unit sis:
w = 4 * t * (Y/f)^4 * (1 - n^2) / (6 n^3)
 = (2/3) * t * NA^4 * (1 - n^2) / n^3
 
RMS wavefront error is this value scaled by sqrt(5/36):
wrms = sqrt(5)/9 * t * NA^4 * (1 - n^2) / n^3
"""
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import sys
if not '..' in sys.path:
    sys.path.insert(0,'..')
import opticstools as ot

def spherical_aberration_offset(t, n, NA):
    """
    Compute spherical aberration due to a glass plate of thickness t
    and refractive index n for rays with numerical aperture NA.
    Returns wavefront error in units of wavelength.
    
    This function is designed to be a numerical check of an
    analytical expression for spherical aberration.
    """
    theta_air = np.arcsin(NA)  # Maximum ray angle
    rays = np.linspace(0, theta_air, 100)
    
    # Compute the angle in the glass using Snell's law
    theta_glass = np.arcsin(np.sin(rays) / n)   
    
    # Compute the small angular formula equivalent of Snell's law
    theta_glass_small = rays / n
    
    # Compute the offset in position
    offset = t * (np.tan(theta_glass_small) - np.tan(theta_glass))

    return offset

sz = 256
cpup = ot.circle(sz, sz//4)
ref_im = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(cpup)))
ref_psf = np.abs(ref_im)**2
t = np.linspace(0,0.2,10)
n = 1.55
NA = 0.5
wave = 1.06e-3 # wavelength in mm
amps = (np.sqrt(5)/9) * t * NA**4 * (n**2-1) / n**3 / (wave/2/np.pi) 
power_scale = np.zeros(len(t))
for i, amp in enumerate(amps):
    sph = ot.zernike(sz, coeffs=[0,0,0, 0,0,0, 0,0,0,0, 0,0,amp], diam=64, rms_norm=True)
    pup = ot.circle(sz, sz//4)*np.exp(1j*sph)
    im = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pup)))
    psf = np.abs(im)**2
    power_scale[i] = np.max(psf)/np.max(ref_psf)

plt.clf()
plt.figure(1)
plt.plot(t, power_scale, marker='o')
plt.xlabel('Glass thickness (mm)')
plt.ylabel('Required relative power')
plt.title('Predicted laser power NA={:.2f}, n={:.2f}'.format(NA, n))

plt.figure(2)
plt.imshow(psf, vmin=0, vmax=np.max(ref_psf)*0.1, cmap='gray')