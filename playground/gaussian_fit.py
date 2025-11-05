"""
Fit a Gaussian beam profile - parameters w_0 and offset z_0 to
data.

The functional form is:

w = w_0 * np.sqrt(1 + ((z-z_0)/z_r)**2)
z_r = np.pi * w_0**2 / wave
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

z = np.array([15,18,18+55,18+86,18+110]) # mm
w = np.array([17,18,49,65,76])*0.015 #mm
wave = 0.98 # microns

def beam_waist(z_vals, w0, z0):
    # Gaussian beam waist model; wave given in microns, convert to mm
    wave_mm = wave * 1e-3
    z_r = np.pi * (w0 ** 2) / wave_mm
    return w0 * np.sqrt(1.0 + ((z_vals - z0) / z_r) ** 2)

# Initial parameter guesses
w0_0 = float(np.min(w))
z0_0 = float(z[np.argmin(w)])

# Fit the model to the data
popt, pcov = curve_fit(
    beam_waist, z, w,
    p0=[w0_0, z0_0],
    bounds=([0.0, -np.inf], [np.inf, np.inf]),
    maxfev=10000
)
w0_fit, z0_fit = popt
z_r_fit = np.pi * w0_fit**2 / (wave * 1e-3)

print(f"Fitted w0 = {w0_fit:.4f} mm")
print(f"Fitted z0 = {z0_fit:.4f} mm")
print(f"Derived z_R = {z_r_fit:.2f} mm")

# Add the predicted waist at z0
w_at_z0 = beam_waist(0, w0_fit, z0_fit)
print(f"Predicted 1/e^2 diameter at z0: w(z0) = {2*w_at_z0:.4f} mm")

# Plot data and best-fit curve
z_fit = np.linspace(z.min(), z.max(), 600)
w_fit = beam_waist(z_fit, w0_fit, z0_fit)

plt.figure(figsize=(7, 5))
plt.scatter(z, w, s=40, label="Data")
plt.plot(z_fit, w_fit, 'r-', lw=2, label="Best fit")
plt.xlabel("z (mm)")
plt.ylabel("w (mm)")
plt.title("Gaussian Beam Waist Fit")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

