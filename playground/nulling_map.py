"""

A good “spare time” mini-project would be to make Declination=Latitude, 
pointing straight up version of Figure A1 from Laugier 2023.

Start with 4 electric fields, with phases p, and amplitudes A 
(set A=1 at the start)

E0 = A0 exp (i [ 2*pi*(u0*alpha + v0*delta)/lambda + p0] )
Alpha = angle in E direction, and delta = angle in N direction.
E = telescope coordinate in E direction in m and N=telescope coordinate in N direction in m.
Same for telescopes 1, 2, 3 and 4.

The two nullers form nulled outputs with: p0=0, p1=pi, p2=0, p3=pi
Make dark null outputs, En1 = E0 - E1, En2 = E2 - E3
Make final outputs A and B: Ea = En1 + i * En2, Eb = En1 - i * En2
Then get the intensities Ia = |Ea|^2, Ib = |Eb|^2, and plot the 2D map Ia-Ib.
"""
import numpy as np
import matplotlib.pyplot as plt

# Define the location of the 4 ESO UT telescopes in meters, 
# Based on the following (P,Q,E,N). We want E and N
# U1    -16.000    -16.000     -9.925    -20.335
# U2     24.000     24.000     14.887     30.502
# U3     64.0013    47.9725    44.915     66.183
# U4    112.000      8.000    103.306     43.999

UT_uv = np.array([[-9.925, -20.335],
                       [14.887, 30.502],
                       [44.915, 66.183],
                       [103.306, 43.999]])

wave = 3.8e-6 # Wavelength in meters (3.8 microns)

# Make a 2D array of angles in radians, up to
# 1 arcsecond = 5e-6 radians.
alpha = np.linspace(-5e-6, 5e-6, 500)
delta = np.linspace(-5e-6, 5e-6, 500)
alpha, delta = np.meshgrid(alpha, delta)

# Calculate the electric fields for each telescope
E = np.zeros((4, alpha.shape[0], alpha.shape[1]), dtype=complex)
for i in range(4):
    u, v = UT_uv[i]
    E[i] = np.exp(1j * (2 * np.pi * (u * alpha + v * delta) / wave))
    
# Make the nulled outputs
En1 = E[0] - E[1]
En2 = E[2] - E[3]

# Make the final outputs
Ea = En1 + 1j * En2
Eb = En1 - 1j * En2

# Calculate the intensities
Ia = np.abs(Ea)**2
Ib = np.abs(Eb)**2

# Plot the 2D map of Ia - Ib, with extent in arcseconds
plt.figure(figsize=(8, 6))
plt.imshow(Ia - Ib, extent=(-1,1,-1,1), origin='lower', cmap='viridis')
plt.colorbar(label='Intensity Difference (Ia - Ib)')
plt.xlabel('Alpha (arcseconds)')
plt.ylabel('Delta (arcseconds)')