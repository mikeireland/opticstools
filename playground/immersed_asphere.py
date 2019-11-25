import numpy as np
import matplotlib.pyplot as plt

#Thickness of glass in the center.
glass_thickness = 3.0

#Refractive index of glass
n_glass = 2.4

#Distance to focus (a finite conjugate system)
d_focus = 100.0

#Numerical aperture in glass
na_glass = 0.4

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
plt.clf()
plt.plot(xs, zs,'b')
plt.plot(-xs, zs,'b')
plt.axis([-1.5,1.5,0,3])
plt.xlabel('x (mm)')
plt.ylabel('z (mm)')