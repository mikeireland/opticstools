import numpy as np
import matplotlib.pyplot as plt
import sys
if not '..' in sys.path:
    sys.path.insert(0,'..')
import opticstools as ot
plt.ion()

nwave = 100
wave = np.linspace(1.1,1.8,nwave)

#Thicknesses in microns
sio2 = [0,37.43]
su8 = [35,0]

npath = len(su8)
pathlengths = np.empty((npath,nwave))
for i in range(npath):
    pathlengths[i] = sio2[i] * ot.nglass(wave, 'sio2') + \
                    su8[i] * ot.nglass(wave, 'su8')
                    
    
delta = pathlengths - pathlengths[-1]

#Plot the phase change between two parts of the mask in waves and as opd.
plt.figure(1)
plt.clf()
plt.plot(wave, delta[0]/wave)
plt.ylabel('Phase (waves)')

plt.figure(2)
plt.clf()
plt.plot(wave, delta[0])
plt.ylabel('OPD')