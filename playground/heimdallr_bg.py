import numpy as np
import matplotlib.pyplot as plt
import sys
if not '..' in sys.path:
    sys.path.insert(0,'..')
import opticstools as ot
import scipy.ndimage as nd

f_ratio = 18    #Focal ratio
px_um = 24      #Pixel width in microns.
qe = 0.6
x = np.arange(300)*0.024
h = 38
y = 150
s = 2

def planck_lambda(wave, T=293):
    """
    Input Wavelength in microns, and flux per unit micron. Area also in 
    microns^2
    
    Sanity check:
    Convert per unit frequency to wavelength by a c/lambda^2 factor
    Convert from per mode to per unit area by dividing by lambda^2
    """
    return 1e-6**3*2*3e8/(wave*1e-6)**4/(np.exp(6.626e-34*3e8/(wave*1e-6)/1.38e-23/T)-1)

def expanded_3pupil(wave=1.5, undersamp=0.625):
    """
    For a 3 telescope non-redundant array, the minimum pupil expansion to fit everything
    in a square is a factor of 7. This however critically samples in the wavelength 
    direction, so a higher factor makes sense. A natural factor to optimally fit in the 
    f/20 pupil is 8.7, so we'll choose 8 for simplicity.
    
    An f/20 beam gives 1.25 pixels sampling at 1.5 microns and 1.96 pixel sampling.
    
    i.e. sampling = wave*20/24
    psize = 24/20/wave * sz
    """
    sz = 256
    psz = 20/24/wave * sz
    subpup = ot.circle(sz, psz*0.625*8/7, interp_edge=True)
    subpup_subarr = ot.rebin(subpup,(sz,sz//8))/8
    subpup[:]=0
    subpup[:,sz//2-sz//16:sz//2+sz//16] = subpup_subarr
    pup3 = nd.shift(subpup,(0,-3*psz*0.625/7)) + \
        nd.shift(subpup,(0,3*psz*0.625/7)) + \
        nd.shift(subpup,(0,-1*psz*0.625/7))
    pup_outer = ot.circle(sz, psz, interp_edge=True)
    return pup3, pup_outer
    
wave = np.linspace(1.5,2.35,100)
#For a pixel of 24 miccrons and an input of f-ratio, we multiply the 
#planck formula by the solid angle and area.
elec_sec = np.trapz(planck_lambda(wave),wave)*24**2*np.pi*(0.5/f_ratio)**2*qe
elec_sec_mode = np.trapz(planck_lambda(wave),wave)*2**2*qe

waveH = np.linspace(1.5,1.8,100)
elecH_sec = np.trapz(planck_lambda(waveH),waveH)*24**2*np.pi*(0.5/f_ratio)**2*qe

pup3, pup_outer = expanded_3pupil()
pfrac = np.sum(pup3)/np.sum(pup_outer)
elec_sec_pup3 = (pfrac  + (1-pfrac)*0.02)* elec_sec

print("Electrons/s/pix: {:.1f}".format(elec_sec))