"""Lets compute the scattering from the SkyHopper aperture.

We'll do this on the basis of the 4as_nonotch file

Name        ROC     Distance    Dimensions
------------------------------------------
Primary     320     126.1       28 x 54
Secondary   103.6   97.1        54 x 28mm (pupil 45 x 23mm)

The Hole has dimensions of 6 x 20mm, and we can assume that all light making it through
the hole originating from an area the size of the telescope pupil on the secondary makes
it to the detector. Actually, light on the hole edges won't make it all the way to the 
detector, so this is conservative. 

Note that diffraction from a curved aperture goes as intensity \propto r^-3
"""
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

PRIM_F = 320./2
SEC_F=-103.6/2
Z0 = 145.0 #Distance from entrance aperture to primary.
Z1 = 126.1
Z2 = 97.1
PUP_DX = 200.0
PUP_DY = 100.0
SEC_DX = 45.0
SEC_DY = 23.0
HOLE_DX = 20.0
HOLE_DY = 6.0
#Roughness nm RMS, as Rq. 1.5 has been achieved years ago on RSA aluminium, and 2.0
#(20 Angstroms) is within the capabilities of e.g. http://www.iividiamondturning.com.
SURFACE_RMS =  1.0 
WAVE = 854. #in nm
PIX_PER_PSF = 4
PIX_PER_FOV = 512*2048.
DIFFRACTION_X_1EM6 = 1.5/60. #Worst case 2.5 based on initial complex aperture sims.
DIFFRACTION_Y_1EM6 = 3.0/60. #Worst case 5.0 based on initial complex aperture sims.
#----------------

#Scattering fraction into 2 pi
scattering_frac = (4*np.pi*SURFACE_RMS/WAVE)**2

#Normalisation per steradian, assuming isotropic scattering
#2\pi \int_0^{\pi/2} \cos (\theta) \sin (\theta) d\theta 
# = 2\pi \int_0^1 \mu d\mu = \pi
scattering_I = scattering_frac / np.pi / PUP_DX / PUP_DY

#The effect of the secondary is to bring the primary mirror image closer to the Hole,
#as well as making it smaller. Surface brightness (specific intensity) is preserved.
primary_image_from_sec = -1./(1./SEC_F - 1/Z1) 
primary_image_dx = primary_image_from_sec / Z1 * PUP_DX
primary_image_dy = primary_image_from_sec / Z1 * PUP_DY
primary_image_from_hole = Z2 + primary_image_from_sec
primary_scattering_through_hole = scattering_I * primary_image_dx * primary_image_dy * \
    HOLE_DX * HOLE_DY / primary_image_from_hole**2

#Fraction that intercepts secondary, as a check.
scattering_intercepted = scattering_frac / np.pi * SEC_DX * SEC_DY / Z1**2

#Now on to the secondary. The flux is magnified by a significant factor...
secondary_F_mag = PUP_DX*PUP_DY/SEC_DX/SEC_DY
secondary_scattering_through_hole = scattering_frac * secondary_F_mag / np.pi * HOLE_DX * HOLE_DY / Z2**2

#Angles for scattering
primary_scattering_x_fov = np.degrees(np.arctan(PUP_DX/Z0))
primary_scattering_y_fov = np.degrees(np.arctan(PUP_DY/Z0))
secondary_scattering_x_fov = np.degrees(np.arctan(SEC_DX/Z1))
secondary_scattering_y_fov = np.degrees(np.arctan(SEC_DY/Z1))

#Now make the plot
plt.clf()
th = np.linspace(0.1,primary_scattering_x_fov, 1000)

thx_ramp = np.maximum(1 - th/primary_scattering_x_fov,0)
thy_ramp = np.maximum(1 - th/primary_scattering_y_fov,0)

primary_scattering_theta_x = primary_scattering_through_hole*PIX_PER_PSF/PIX_PER_FOV*thx_ramp
primary_scattering_theta_y = primary_scattering_through_hole*PIX_PER_PSF/PIX_PER_FOV*thy_ramp

thx_ramp = np.maximum(1 - th/secondary_scattering_x_fov,0)
thy_ramp = np.maximum(1 - th/secondary_scattering_y_fov,0)

secondary_scattering_theta_x = secondary_scattering_through_hole*PIX_PER_PSF/PIX_PER_FOV*thx_ramp
secondary_scattering_theta_y = secondary_scattering_through_hole*PIX_PER_PSF/PIX_PER_FOV*thy_ramp

plt.loglog(th, primary_scattering_theta_x, '--', label='Pri. X axis')
plt.plot(th, primary_scattering_theta_y,'--', label='Pri.Y axis')
plt.plot(th, secondary_scattering_theta_x,'--', label='Sec. X axis')
plt.plot(th, secondary_scattering_theta_y,'--', label='Sec. Y axis')
plt.plot(th, 1e-6*(th/DIFFRACTION_X_1EM6)**(-3), '--', label='X diffraction')
plt.plot(th, 1e-6*(th/DIFFRACTION_Y_1EM6)**(-3), '--', label='Y diffraction')

xtot = 1e-6*(th/DIFFRACTION_X_1EM6)**(-3) + primary_scattering_theta_x + secondary_scattering_theta_x
ytot = 1e-6*(th/DIFFRACTION_Y_1EM6)**(-3) + primary_scattering_theta_y + secondary_scattering_theta_y

plt.plot(th, xtot, label='X total')
plt.plot(th, ytot, label='Y total')

plt.axis( (0.1,primary_scattering_x_fov, 1e-14,1e-8) )
plt.title('Achievable Scattering per PSF')
plt.xlabel('Offset (degrees)')
plt.ylabel('Fractional Intensity')
plt.legend()