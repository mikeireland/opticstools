"""Some diffraction calculations for the RHEA slit feed. Speak to Mike about details..."""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import opticstools

#Firstly, define a Gaussian beam
m_pix = 1e-6
gauss_width = 65e-6    #1.e^2 radius
hex_diameter = 250.0e-6
wave = 0.7e-6 #in mm...

#Case 1
f1 = 40.0e-3
f2 = 4.22e-3 
d1 = 40.0e-3

#Case 2
#f1 = np.inf #If we come to focus.
#f2 = 4.71e-3#3.54e-3   
#d1 = 40.4e-3


sz = 1024
alpha = 0.0325
#-----
x = (np.arange(sz) - (sz/2))*m_pix
xy = np.meshgrid(x,x)
rr = np.sqrt(xy[0]**2 + xy[1]**2)
beam = np.exp(-(rr/gauss_width)**2) * opticstools.utils.hexagon(sz,hex_diameter/m_pix).astype(complex)

beams = []
beams.append(beam)

#First lens.
if (f1 < np.inf):
    beam *= opticstools.curved_wf(sz,m_pix,f_length=f1, wave=wave)

#Save "common pupil" wavefront.    
beam = opticstools.propagate_by_fresnel(beam, m_pix, d1, wave)
beams.append(beam)

#second lens.
beam = opticstools.propagate_by_fresnel(beam, m_pix, f2, wave)
beam *= opticstools.curved_wf(sz,m_pix,f_length=f2, wave=wave)
d2 = 1.0/(1.0/f2 - 1.0/(f2+d1))
beam = opticstools.propagate_by_fresnel(beam, m_pix, d2, wave)
beams.append(beam)

#Far field
beam = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(beam)))
beams.append(beam)
    
beam_widths_raw = []
for beam in beams:
    beam_widths_raw.append(np.sqrt(np.sum(rr**2*np.abs(beam)**2)/np.sum(np.abs(beam)**2)))
beam_widths_raw = np.array(beam_widths_raw)

#Scale beam widths to 1/e^2 units.
beam_widths = beam_widths_raw * gauss_width/beam_widths_raw[0]
print(beam_widths*1e6)

focused_width = (f1 * wave)/(np.pi * gauss_width)
print( "Lens focussed beam width (microns): {0:5.2f}".format(focused_width*1e6) )
print("Asphere focal length: {0:5.2f}".format(focused_width/alpha*1e3))
propagated_width = gauss_width * np.sqrt(1 + (wave * d1 / np.pi / gauss_width**2)**2)
print( "Propagated beam width (microns): {0:5.2f}".format(propagated_width*1e6) )
print("Asphere focal length: {0:5.2f}".format(propagated_width/alpha*1e3))

