"""
Some code for dealing with the 5 aperture nuller.

For leakage calculations, we need the mean square stellar radius and mean 4th power of 
stellar radius for a standard limb-darkening law:

I = I_0 (1 - 0.47(1-cos(mu)) - 0.23(1-cos(mu))**2)


Zodiacal light has an optical depth of 1e-7 spread over the sky. 
A 290K background has a flux of 2.8e14 Jy/Sr, 
2*6.626e-34*nu**3/3e8**2/(np.exp(6.626e-34*nu/1.38e-23/290) - 1)*1e26
... which is 1e7 times higher than the COBE value.
OR
25 MJy/sr
Vega is 40 Jy
Gives 12 magnitudes per square arcsec.

tau Ceti is mag 1.6

For a Bracewell nuller, transmission to a single output, with
\alpha_p = lambda/2B
T = 1-cos(2 pi \alpha B/lambda)
  = 1-cos(pi \alpha / \alpha_p)
  = 0.5 \pi^2 (\alpha / \alpha_p)^2
  
The integral of I times T in terms of \alpha/\alpha_p=x is:

\int T(x) I(r)^2 r dr / \int I(r)^2 r dr

"""
import matplotlib.pyplot as plt
import numpy as np
import sys
plt.ion()

if not '..' in sys.path:
    sys.path.insert(0,'..')
from opticstools import knull
from opticstools import azimuthalAverage

N = knull.make_nuller_mat5()

#Other Parameters
wave = 10e-6
sz = 400
plx = np.linspace(100,800,100)

#Stellar parameters
Lstar = 0.52
Rstar = 0.79
color = 'b' ; label = 'G8V' ; star='tau Ceti'
#Background for 0.5 square arcsec.
ref_plx = 274
zodiacal = 10**(-0.4*(12-1.6))*0.5 * (ref_plx/plx)**2 #Tau Ceti
initial_run=True
#Run as false then true!
if True:
    Lstar = 0.0017
    Rstar = 0.1542
    color = 'r' ; label = 'M5V' ; star = 'proxima Cen'
    #Background for 0.5 square arcsec.
    ref_plx = 768.5
    zodiacal = 10**(-0.4*(12-3.8))*0.5 * (ref_plx/plx)**2 #Proxima Cen
    initial_run=False


#Create a grid of +/- 2pi
dist = 1000/plx
hz_angle = plx * Lstar**.5
mas_rad = np.degrees(1)*3600e3
bl = wave/2*mas_rad/hz_angle

#bl = 5.5#50 #Baseline in m
angle = np.linspace(-3*np.pi,3*np.pi,sz)

#The null scale is lambda/2B. This should be equal to 1/6 of the full grid size. 
#null_scale = 10e-6/bl/2*mas_rad
#print("Null scale: {:.1f} mas".format(null_scale))
extent = [-1.5*wave/bl[0]*mas_rad,1.5*wave/bl[0]*mas_rad,-1.5*wave/bl[0]*mas_rad,1.5*wave/bl[0]*mas_rad]
mas_pix = wave/bl*mas_rad*3/sz

xy = np.meshgrid(angle, angle, indexing='ij')

#Create some x,y grid coordinates, normalised so that the shortest baseline is 1.
x = N[1].real.copy()
#x *= np.sqrt(5)
y = N[1].imag.copy()
#y *= np.sqrt(5)

response = np.zeros((5,sz,sz), dtype='complex')

for i in range(5):
    for k in range(5):
        response[k] += np.exp(1j*(xy[0]*x[i] + xy[1]*y[i]))*N[k,i]

#Turn into intensity
response = np.abs(response)**2
response /= (np.max(response[0])/5)

#Create the kernel nulls.
k1 = response[2]-response[3]
k2 = response[1]-response[4]

plt.figure(1)
plt.clf()
plt.imshow(k1, extent=extent)
plt.xlabel('Offset (mas)')
plt.ylabel('Offset (mas)')
plt.colorbar()
plt.tight_layout()
plt.figure(2)
plt.clf()
plt.imshow(k2, extent=extent)
plt.xlabel('Offset (mas)')
plt.ylabel('Offset (mas)')
plt.tight_layout()
#plt.colorbar()
plt.figure(3)
plt.imshow(response[0])

#plt.figure(4)
#plt.subplot(121)
#plt.imshow(k1)
#plt.subplot(122)
#plt.imshow(k2)
#plt.colorbar()

#Find the null function numerically
r_ix,y2 = azimuthalAverage(response[1], returnradii=True, binsize=0.8)
r_ix,y4 = azimuthalAverage(response[2], returnradii=True, binsize=0.8)

second_order_coeff = np.median(y2[1:16]/r_ix[1:16]**2)/mas_pix**2
fourth_order_coeff = np.median(y4[1:16]/r_ix[1:16]**2)/mas_pix**2

r = np.linspace(0,1,200)
cosmu = np.sqrt(1-r**2)
I = 1-0.47*(1-cosmu)-0.23*(1-cosmu)**2
mn_r2 = (np.trapz(I*r**3, r)/np.trapz(I*r, r))**.5
mn_r4 = (np.trapz(I*r**5, r)/np.trapz(I*r, r))**.25

#tau Ceti calculation
star_mas = Rstar/214*plx
leakage_2nd = second_order_coeff*(mn_r2*star_mas)**2
leakage_4th = fourth_order_coeff*(mn_r2*star_mas)**4

xy = np.meshgrid(r + 0.5*(r[1]-r[0]),r+ 0.5*(r[1]-r[0]))
rr = np.sqrt(xy[0]**2+xy[1]**2)
leakage_bracewell_const = 0.5*np.pi**2*np.sum(np.interp(rr,r,I)*xy[0]**2)/np.sum(np.interp(rr,r,I))
leakage_bracewell = leakage_bracewell_const*(star_mas/hz_angle)**2


print("{:.1e} {:.1e} {:.1e} {:.1e}".format(leakage_2nd[0], leakage_4th[0], leakage_bracewell[0], zodiacal[0]))

plt.figure(4)
plt.clf()
plt.semilogy(dist, leakage_2nd/zodiacal, color + '-', label=label + ' 5T Second Order')
plt.semilogy(dist, leakage_bracewell/zodiacal, color + '--', label=label + ' Bracewell')
plt.xlabel('Distance (pc)')
plt.ylabel('Leakage (fraction of Zodiacal)')

plt.figure(5)
var1 = (leakage_2nd + zodiacal)/(0.5*0.4**2)
var2 = (leakage_4th + zodiacal)/(0.5*0.4**2)
y = np.sqrt(zodiacal[0]*(1/var1 + 1/var2))
ref_dist = 1000./ref_plx
if initial_run:
    plt.clf()
plt.plot(ref_dist, np.interp(ref_dist, dist[::-1], y[::-1]),color+'o')
plt.text(ref_dist+1, np.interp(ref_dist, dist[::-1], y[::-1]), star)
varz = (leakage_bracewell + zodiacal)/0.35**2
plt.plot(dist, y, color + '-', label=label + ' 5T Combined')
plt.plot(dist, np.sqrt(zodiacal[0]/varz), color + '--', label=label + ' Bracewell')
plt.ylabel('Relative SNR (equal total area)')
plt.xlabel('Distance (pc)')
if not initial_run:
    plt.legend()
    plt.tight_layout()
