"""Some diffraction calculations for Veoce """

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import opticstools

#A set of seeing numbers
seeings = 1.0 + np.arange(21)*0.1
t_diam = 3.89
arcsec_rad = np.degrees(1)*3600
sz = 512
m_pix = 1.0e-6   #Near field calcs
radians_pix = 1e-3  #Far field calcs

#Hexagon x and y offsets (still to be scaled)
yoffset0 = np.array([-2, 2, -2, -1, -1, 0,
                     -1, -1, 0, 0, 0, 1, 1,
                     0, 1, 1, 2, -2, 2])
xoffset0 = np.array([-2, 0, 2, -3, 3, -4,
                     -1, 1, -2, 0, 2, -1, 1,
                     4, -3, 3, -2, 0, 2])

# "Option 1" : feed 19 fibers of 0.44" flat to flat dimension
# ISSUE: The slit is too narrow. Increasing to 88.5 microns FWHM nominal
# gives a circular fiber full width of 102.2 microns. In turn, this means 
# that 3 fibers have to be removed from the slit.

#"Option 2 big" increases the spot size to 102.2 microns (for 88.5 microns nominal FWHM).
#With the 8% FRD allowance, the individual spot sizes at the output MLA become
#215 microns. 

plt.clf()
labels = ['Option 1', 'Option 1 big','Option 2','Option 2 vbig','Option 2 opt']
diams  = [0.44,0.493,0.47,0.5687, 0.47*93.2/(96.4*np.sqrt(3)/2)]
losses = [1,1,0.99**2,0.99**2,0.99**2]
for run_label, arcsec_lens, loss in zip(labels,diams,losses): #Was 0.44
    #Flat to flat for an octagon.
    fiber_core = 45.0e-6       
    #Focal ratio assuming lenslet flat to flat is injected in to the fiber 
    #diameter
    flat_to_flat_fn = fiber_core/(t_diam*arcsec_lens/arcsec_rad)

    #The loss here is an octagon inside a hexagon, then an octagon inside a circle.
    hex_input = opticstools.utils.hexagon(sz,fiber_core/m_pix)
    oct_fiber = opticstools.utils.octagon(sz,fiber_core/m_pix)
    circ_fiber = opticstools.utils.circle(sz,fiber_core/m_pix)
    injection_loss = np.sum(oct_fiber*hex_input)/np.sum(hex_input)
    fiber_interface_loss = np.sum(circ_fiber*oct_fiber)/np.sum(oct_fiber)

    one_fiber_onsky = oct_fiber*hex_input

    #Convert x and y offsets to pixels. We do this in "fiber core" units, knowing
    #that each fiber core is one arcsec_lens apart.
    yoffset = (yoffset0 * fiber_core/m_pix*np.sqrt(3)/2).astype(int)
    xoffset = (xoffset0 * fiber_core/m_pix*0.5).astype(int)

    mask = np.zeros( (sz,sz) )
    for i in range(len(yoffset)):
        mask += np.roll(np.roll(one_fiber_onsky,xoffset[i],axis=0),yoffset[i],axis=1)
    mask = np.minimum(mask,1)

    ix = np.arange(-sz//2,sz//2) / (fiber_core/m_pix) * arcsec_lens
    xy = np.meshgrid(ix,ix)
    rr = np.sqrt(xy[0]**2 + xy[1]**2)

    throughputs = []
    for seeing in seeings:
        seeing_prof = opticstools.moffat(rr, seeing/2, beta=4.0)
        throughputs.append(np.sum(seeing_prof*mask)*(arcsec_lens/fiber_core*m_pix)**2)
    throughputs_option1 = np.array(throughputs)
    throughputs_option1 *= fiber_interface_loss*loss

    plt.plot(seeings, throughputs_option1, label=run_label)
    plt.xlabel('Seeing (")')
    plt.ylabel('Slit/Geometry Loss')

    
plt.legend()

#Jon's numbers: Option 2 had 96.4 micron spot size based on edge to edge rather than FWHM. 
#The difference is sqrt(3)/2. 
slit_width_2vbig = 96.4 * np.sqrt(3)/2 * (245.0/202.5)
