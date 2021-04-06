"""
Find the best focal length of a lens outside the C-RED One dewar
"""
import numpy as np

dstop = 40  #Distance to stop form detector in mm
wstop = 2   #Width of the stop in mm
pdiam = 10  #Input pupil diameter
fcam = 100   #Focal length of the first camera lens
dprism = 100.01 #Physical distance from lens 2 to the prism

#-----
dprism_opt = 1/(1/dprism - 1/fcam)

x1 = np.arange(10,40,.1)
f1 = np.arange(10,40,.1)
x1f1 = np.meshgrid(x1,f1)

#Pupil diameter on lens 1
p1 = pdiam * x1f1[0]/fcam

#Object distance
x2 = 1/(1/x1f1[1] - 1/x1f1[0])

#Pupil diameter on cold stop
pstop = dstop/x2 * p1

#Distance form lens1 to cold stop
d1 = x2 - dstop

#Distance to prism (i.e. lens) image)
dprism_image = 1/(1/x1f1[1] - 1/(x1f1[0] + fcam + dprism_opt))

metric = np.sqrt(100*(pstop-wstop)**2 + (d1-dprism_image)**2)
ix = np.argmin(metric)
print('x1: {:.1f}'.format(x1f1[0].flatten()[ix]))
print('f1: {:.1f}'.format(x1f1[1].flatten()[ix]))
print('x2: {:.1f}'.format(x2.flatten()[ix]))