"""
V loss due to polarisation.

Rdiff = 0.004 * (angle/30 degrees)^2
Pdiff = 0.6 * (angle/30 degrees)^2 * (wave/10 microns) (degrees) 

The rotation angle of the beam trains between the two nulling arms is 9.5 degrees

Lets start with a simple calculation at a single wavelength.

Interestingly, it is polarisations at 45 degrees to the baseline that form the 
eigenpolarisations for visibility, with a tiny retardance in this direction 
fixing the null depth. It could even be static.
"""
import numpy as np

aois = np.array([5,29,24]) #Long Baseline
aois = np.array([1,25,24]) #Short baseline
aois = np.array([22.5,22.5,45]) #In-plane

pdiff_10um = np.sum((aois/30)**2) * 0.01

az_rad = np.radians(9.5)

J_retard = np.array([[1,0],[0,np.exp(1j*pdiff_10um)]])
J_rot = np.array([[np.cos(az_rad), np.sin(az_rad)],[-np.sin(az_rad), np.cos(az_rad)]])

J1 = J_rot.dot(J_retard).dot(J_rot.T)
J2 = J_rot.T.dot(J_retard).dot(J_rot)

W,V = np.linalg.eig(J1.dot(J2.conj().T))

#Now we have two fringes, at the following phase angles:
fringe_phase = np.angle(W)
null_depth = 1 - np.cos(fringe_phase[0])
print("Null depth: {:.2e}".format(null_depth))