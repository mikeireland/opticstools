"""
Plan A:
1) Create a 7x7 response matrix and reconstructor
2) Implement simple servo loop.
... which is best done in aotools or as part of pyxao


Plan B:
- Just correct the first N Zernike modes.

Plan C:
- Create the wavefront sensor by itself, and a sensitivity matrix, which 
is Beta in Guyon terminology. 
- Rather than sin and cos terms, we could simply write for a normalised data
flux d and a particular mode's model intensity difference Imod:


For flux normalised to 1, let Imod be the difference in pupil intensity per unit mode 
amplitude, for a given mode.

Idiff = d + sigma = a * Imod 
This has a least squares solution:
a = \sum (Imod * d / dvar) / \sum(Imod**2 / dvar)

Var(a) = \sum(Imod**2/dvar**2 * Var(d)) / [ \sum(Imod**2 / dvar) ]^2
# Photon limited:
Var(a) = \sum(Imod**2/Iflux) / [ \sum(Imod**2 / Iflux) ]^2
    = 1/\sum(Imod**2 / Iflux)

Sanity check: If we double the number of pixels, Imod and Iflux both halve per pixel, 
but there are double the number of pixels to average over so this is OK - the variance stays the
same. 

The practical magnitude limits are:
np.log10(9.7e9*.3*.2*.5/400/1000*np.pi*(1.8/2)**2)*2.5
np.log10(9.7e9*.3*.2*.5/400/1000*np.pi*(8/2)**2)*2.5
... H=11 on UTs or H=8 on ATs, at 1 kHz.
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
plt.ion()
if not '..' in sys.path:
    sys.path.insert(0,'..')
import opticstools as ot
import scipy.ndimage as nd

#Constants. At kReflectivity=1, we get a minimum 7 radians RMS per 1/sqrt(photons), 
#Compared to 9.5 radians RMS at kReflectivity=0.25
kReflectivity = 1.0 #0.25

#Preliminaries
kSz = 128
kDiam = 24
kObst = 3
kPsf_diam = kSz/kDiam

kWidth= kPsf_diam * 1.6#10# 8.5

#calculations based on these constants.
pup_outer = ot.utils.circle(kSz,kDiam)
rpup = pup_outer - ot.utils.circle(kSz,kObst)
lyot_outer = ot.utils.circle(kSz,kDiam*2)

#Reference image/pupil calculations
rim_E = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(rpup)))
iflux_norm = np.sum(np.abs(rim_E)**2)
pflux_norm = np.sum(np.abs(rpup)**2)

rim_E = rim_E*(1-ot.circle(kSz,kWidth, interp_edge=True))+ 1j*rim_E*ot.circle(kSz,kWidth, interp_edge=True)*np.sqrt(kReflectivity)
rpup_lyot = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(rim_E)))
#Intensities
rpup_lyot_I = np.abs(rpup_lyot)**2/pflux_norm
rim_outer = np.fft.fftshift(np.abs(np.fft.fft2(np.fft.fftshift(rpup_lyot*(1-pup_outer)*lyot_outer)))**2)/iflux_norm

def wfs_Idiff(zernikes):
    """
    Simulate the zernike-like wavefront sensor response to an aberration 
    
    Parameters
    ----------
    zernikes:
        Array of zernike coefficients, normalised so that RMS is 1 over a circular aperture.
        
    Returns
    -------
    pupil flux, pupil flux difference, image flux, image flux difference
    """
    #Add aberration
    pup = rpup*ot.zernike_wf(kSz, coeffs=zernikes, diam=kDiam, rms_norm=True)

    #Create the image electric field.
    im_E = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pup)))
    im_E = im_E*(1-ot.circle(kSz, kWidth, interp_edge=True))+ 1j*im_E*ot.circle(kSz,kWidth, interp_edge=True)*np.sqrt(kReflectivity)

    #Now for the "Lyot" stop
    pup_lyot = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(im_E)))
    pup_lyot_I = np.abs(pup_lyot)**2/pflux_norm
    pup_lyot_diff = pup_lyot_I - rpup_lyot_I

    #An image based on the outer pupil.
    im_outer = np.fft.fftshift(np.abs(np.fft.fft2(np.fft.fftshift(pup_lyot*(1-pup_outer)*lyot_outer)))**2)/iflux_norm
    im_outer_diff = im_outer - rim_outer
    
    #import pdb; pdb.set_trace()
    #Resample and cut out on return
    return ot.rebin(nd.shift(pup_lyot_I,(-0.5,-0.5)),(kSz//2,kSz//2))[kSz//4-6:kSz//4+6,kSz//4-6:kSz//4+6],\
           ot.rebin(nd.shift(pup_lyot_diff,(-0.5,-0.5)), (kSz//2,kSz//2))[kSz//4-6:kSz//4+6,kSz//4-6:kSz//4+6],\
        im_outer[kSz//2-7:kSz//2+7,kSz//2-7:kSz//2+7], im_outer_diff[kSz//2-7:kSz//2+7,kSz//2-7:kSz//2+7]
    
if __name__=="__main__":
    nz = 2
    test_amp = 0.5
    threshold = 1e-4
    sigs = np.zeros(nz)
    sigs_tt = np.zeros(nz)
    for i in range(1,nz):
        zernikes = np.zeros(nz)
        zernikes[i]=test_amp
        pup_lyot_Ip, pup_lyot_diffp, im_outerp, im_outer_diffp = wfs_Idiff(zernikes)
        zernikes[i]=-test_amp
        pup_lyot_Im, pup_lyot_diffm, im_outerm, im_outer_diffm = wfs_Idiff(zernikes)
        Imod_pup = (pup_lyot_diffp - pup_lyot_diffm)/(2*test_amp)
        Imod_im = (im_outer_diffp - im_outer_diffm)/(2*test_amp)
        Iflux_pup = (pup_lyot_Ip + pup_lyot_Im)/2
        Iflux_im = (im_outerp + im_outerm)/2
        sigs[i] =  1/np.sqrt(np.sum(Imod_pup**2/(Iflux_pup + threshold)))
        sigs_tt[i] =  1/np.sqrt(np.sum(Imod_im**2/(Iflux_im + threshold)))
        
    print("Lyot flux: {:.3f}".format(np.sum(Iflux_pup)))
    print("TT im flux: {:.3f}".format(np.sum(Iflux_im)))
    print("Beta parameters for additional TT beam (RMS radians wavefront error * sqrt(total incoming photons)")
    print(np.minimum(sigs_tt, 100))
    print("Beta parameters for ZWFS (RMS radians wavefront error * sqrt(total incoming photons)")
    print(sigs)
    print("Practical Flux Limit: {:.1f}".format(1/np.mean(Iflux_pup)))
    #Display outputs
    plt.figure(1)
    plt.clf()
    plt.imshow(pup_lyot_Ip)
    plt.title('Pupil Lyot Intensity')
    plt.figure(2)
    plt.clf()
    plt.imshow(pup_lyot_diffp)
    plt.title('Pupil Lyot Difference')
    plt.figure(3)
    plt.clf()
    plt.imshow(im_outerp)
    plt.title('Image from outer pupil')
    plt.figure(4)
    plt.clf()
    plt.imshow(im_outer_diffp)
    plt.title('Outer pup image difference')
    