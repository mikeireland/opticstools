"""
V loss due to polarisation.

Rdiff = 0.004 * (angle/30 degrees)^2
Pdiff = [0.6 * (angle/30 degrees)^2  ] * (10 microns / wave) (degrees) 

Checking this again with OpenFilters. 

Pdiff @ 3 microns = 1.7 degrees at 30 degrees [formula: 2.0]
 = 4.5 degrees at 45 degrees [formula: 4.5 degrees]
 = 9.1 degrees at 60 degrees [formula: 8.0 degrees]

Pdiff @ 4 microns = 1.3 degrees at 30 degrees [formula: 1.5]
 = 3.3 degrees at 45 degrees [formula: 3.4 degrees]
 = 6.8 degrees at 60 degrees [formula: 6.0 degrees]
 
Pdiff @ 8 microns = 0.7 degrees at 30 degrees
 = 1.7 degrees at 45 degrees
 = 3.6 degrees at 60 degrees

Better formula:

Pdiff = [0.6 * (angle/30)**2  + 0.07 * (angle/30)**4] * (8 microns / wave) (degrees) 

Clearly, if the only issue is uneven angles of incidence, where input and output angles
are the same, we have no problem. And rotating a corner reflector by 5 degrees only
compensates for 0.17 degrees or 0.003 radians (barely enough to help!). A compensator 
would need to handle 60 and 30 degree AOIs, i.e. be able to rotate by 15 degreers.

---- 
 
The rotation angle of the beam trains between the two nulling arms is 9.5 degrees

Lets start with a simple calculation at a single wavelength.

Interestingly, in the adiabatic approx, it is polarisations at 45 degrees to the 
baseline that form the eigenpolarisations for visibility, with a tiny retardance in 
this direction fixing the null depth. It could even be static.

NB Overcoat on transfer mirrors to nulling level - A differential overcoat thickness 
give the needed tiny phase shift.

The real part of J1.dot(J2.conj().T) forms the fringes, i.e. the fringes (total intensity 
for an arbitrary input polarisation) are

E^ J1^t J1 E  + E^ J2^t J2 E + 2 Re (E^ J1^t J2 E e^i phi )
= [IV+IH] + 2 Re ( (VE)^ D (VE) e^i phi )

For a non-unitary J1, J2, this is complex! But generally, the Imax and Imin
for linearly polarized source is:

Ev^ J1^t J1 Ev + Ev^ J2^t J2 Ev +/- 2 |Ev^ J1^t J2 Ev| 

For an unpolarised source, we need to find:

|Ev^ J1^t J2 Ev + Eh^ J1^t J2 Eh|, and the Imax and Imin become:

Ev^ J1^t J1 Ev + Ev^ J2^t J2 Ev  + 
Eh^ J1^t J1 Eh + Eh^ J2^t J2 Eh
+/- 2 |Ev^ J1^t J2 Ev + Eh^ J1^t J2 Eh| 

"""
import numpy as np
#from scipy import linalg

#aois = np.array([5,29,24]) #Long Baseline, but unrealistic!
#aois = np.array([1,25,24]) #Short baseline, but unrealistic!
#aois = np.array([22.5,22.5,45]) #In-plane

aois = np.array([5,41,45]) #Long Baseline, new design.
#aois = np.array([1,44,45]) #Long Baseline, new design.

aois = np.array([3,27,30]) #Av Baseline, better AOIs.

pdiff_10um = np.sum((aois/30)**2) * 0.01 / 0.8**2
adiff_10um = np.sum((aois/30)**2) * 0.002 #0.004 in intensity!
#adiff_10um = 0

az_rad = np.radians(9.5)

J_retard = np.array([[1-adiff_10um,0],[0,np.exp(1j*pdiff_10um)]])
J_rot = np.array([[np.cos(az_rad), np.sin(az_rad)],[-np.sin(az_rad), np.cos(az_rad)]])
J_rot45 = np.array([[1,1],[-1,1]])/np.sqrt(2)

J1 = J_rot.dot(J_retard).dot(J_rot.T)
J2 = J_rot.T.dot(J_retard).dot(J_rot)

#Interference term
Iint_matrix = J1.dot(J2.conj().T)
W,V = np.linalg.eig(Iint_matrix)
print(V)
print(np.dot(V[0], V[1]))

#Now we have two fringes, at the following phase angles:
fringe_phase = np.angle(W)
null_depth = 0.5*(1 - np.cos(fringe_phase[0]))
print("Null depth (unitary approx): {:.2e}".format(null_depth))

Iconst = np.abs(np.sum((np.diag(J1.dot(J1.T.conj()))))) + \
	np.abs(np.sum((np.diag(J2.dot(J2.T.conj())))))
Iint = 2*np.abs(np.trace(Iint_matrix))
print("Actual null depth: {:.2e}".format( (Iconst-Iint)/(Iconst+Iint) ))


#If we make a retarder... !!! Have to place this at 45 degrees to beam !!!
J_overcoat1 = np.array([[1,0],[0,1]])
J_overcoat2 = np.array([[1,0],[0,np.exp(1j*(0.017))]]) #0.04 for 'new design'
J_overcoat1 = J_rot45.dot(J_overcoat1).dot(J_rot45.T)
J_overcoat2 = J_rot45.dot(J_overcoat2).dot(J_rot45.T)

J1r = J_overcoat1.dot(J_rot).dot(J_retard).dot(J_rot.T)
J2r = J_overcoat2.dot(J_rot).T.dot(J_retard).dot(J_rot)

#Errors in the calculation of the next bit!
if False:
	Wr,Vr = np.linalg.eig(J1r.dot(J2r.conj().T))
	null_depth = 0.5*(1 - np.cos(np.angle(Wr[0])))
	print("Null depth with overcoat (unitary approx): {:.2e}".format(null_depth))

Iconst = np.abs(np.sum((np.diag(J1r.dot(J1r.T.conj()))))) + \
	np.abs(np.sum((np.diag(J2r.dot(J2r.T.conj())))))
Iint_matrix = J1r.dot(J2r.conj().T)
Iint = 2*np.abs(np.trace(Iint_matrix))
print("Actual null depth with overcoat: {:.2e}".format( (Iconst-Iint)/(Iconst+Iint) ))


#Finally, lets make some plots/images of what this would actually look like on the spacecraft
