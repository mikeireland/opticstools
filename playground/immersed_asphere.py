import numpy as np
import matplotlib.pyplot as plt
plt.ion()

#Thickness of glass in the center.
glass_thickness = 2.08
glass_thickness = 1.04

#Refractive index of glass
n_glass = 2.41

#Distance to focus (a finite conjugate system)
d_focus = 50.0

#Numerical aperture in glass
na_glass = 0.45

#Output angles, in glass.
nx = 1000
#----------------------------------------------
dx = na_glass*glass_thickness/nx
xs = np.arange(nx)*dx
th_gs = [0]
th_wgs = [0]
th_exts = [0]
zs = [glass_thickness]

z_focus = d_focus + glass_thickness
#We iterate from the center of the lens outwards.
for i in range(nx-1):
    #Initial (shooting) model for z
    z = zs[i] - np.tan(th_gs[i])*dx
    x = xs[i+1]
    
    #Simple Trigonometry
    th_wg = np.arctan(x/z)
    th_ext = np.arctan(x/(z_focus-z))
    
    #Snell's law:
    # sin(th_ext-th_g) = n*sin(th_wg - th_g)
    # sin(th_ext)cos(th_g) - cos(th_ext)sin(th_g) = n*sin(th_wg)cos(th_g) - n*cos(th_wg)sin(th_g)
    # sin(th_ext) - cos(th_ext)tan(th_g) = n*sin(th_wg) - n*cos(th_wg)tan(th_g)
    # tan(th_g) = (n*sin(th_wg) - sin(th_ext))/(n*cos(th_wg)-cos(th_ext))
    th_g = np.arctan((n_glass*np.sin(th_wg) - np.sin(th_ext))/\
                     (n_glass*np.cos(th_wg) - np.cos(th_ext)))
    #Adjust z to be more accurate
    z = zs[i] - 0.5*(np.tan(th_gs[i]) + np.tan(th_g))*dx
    
    #Save key outputs.
    th_gs.append(th_g)
    zs.append(z)

zs = np.array(zs)
#plt.clf()
plt.plot(xs, zs,'b')
plt.plot(-xs, zs,'b')
plt.axis([-1.5,1.5,0,3])
plt.xlabel('x (mm)')
plt.ylabel('z (mm)')

#Plot the sphere!
roc = xs[10]/np.sin(th_gs[10])
sphere_centre = glass_thickness - roc
plt.plot(xs, sphere_centre + np.sqrt(roc**2 - xs**2),'r')
plt.plot(-xs, sphere_centre + np.sqrt(roc**2 - xs**2),'r')

#Example polynomial fitting
pfit = np.polyfit(np.concatenate([-xs,xs]), np.concatenate([zs,zs]), 8)
pfit[[1,3,5,7]]=0
pfunc = np.poly1d(pfit)
plt.plot(xs,pfunc(xs),'g')
plt.plot(-xs,pfunc(xs),'g')
#Format this better!
print('Even Polynomial terms: ', pfit[[6,4,2,0]])
