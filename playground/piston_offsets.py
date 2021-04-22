"""Assume atmospheric profile which comes from Sarazin and Tokovinin (2002):
- 0.86" seeing
- tau_0=3.9ms, which is 9.4m/s wind. But lets double this to cover more actual seeing
conditions.

We will just move through the  wavefront with nd.interpolation.shift, with a wrap.

"""
import astropy.constants as const
import astropy.units as units
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd
import pdb
import os
import glob
import opticstools as ot
plt.ion()

mm_pix = 100
r_0_500 = 0.98*.5e-6/np.radians(0.86/3600)*1000 #In mm
v_wind = 9.4*1000   #9.4 for median seeing. Actually vbar
angle_wind = 0.1

#Fraction of wavefront that is corrected. In practice, this is spatial-frequency
#dependent, with a larger fraction corrected in e.g. tip/tilt modes. However, 
#there is additional tip/tilt noise due to vibrations, so likely this isn't 
#too far out. Lag multiplies a -5/3 power spectrum by an exponent of 2
#
#Note that for stability, the lag is generally higher by a factor of e.g. 1.5, due 
#to the AO gain being decreased.
ao_correction_lag = 1.5*0.004*v_wind/mm_pix #in pix. 
t_int = 0.005 
nt = 4096

rnoise = 0.35
wl = 2.2e-3

sz = 512

#Comment out one of the blocks below.
#AT
subap_diam = 450.
tel_diam = 1800.
ao_correction_frac = 0.8 

#UT
subap_diam = 1000.
tel_diam = 7900.
ao_correction_frac = 0.9

#Servo parameters
Ki = 0.6
tau = 0.003
Tint = 0.003

#For plotting...
g = np.exp(-np.arange(-15,15)**2/2./5**2)
g /= np.sum(g)
#-----------------------


def evolve(delay, time, v_wind, angle_wind, m_px):
    """Evolve atmospheric delays according the atmosphere and angle"""
    yshift_in_pix = v_wind*time/m_px*np.sin(angle_wind)
    xshift_in_pix = v_wind*time/m_px*np.cos(angle_wind)
    if len(delay.shape)==2:
        return nd.interpolation.shift(delay,(yshift_in_pix,xshift_in_pix),order=1,mode='wrap')
    else:
        new_delays = np.empty_like(delay)
        for newd, d in zip(new_delays, delay):
            newd[:]=nd.interpolation.shift(d,(yshift_in_pix,xshift_in_pix),order=1,mode='wrap')
        return new_delays 


#------------------
if __name__=="__main__":
    #Now create some wavefronts.
    delay = np.zeros( (sz, sz) )
    
    #Create a wavefront in mm. 
    delay_unfiltered = ot.kmf(sz, r_0_pix=r_0_500/mm_pix)*.5e-3/2/np.pi
    
    #Pupil
    subap_pup = np.fft.fftshift(ot.utils.circle(sz, subap_diam/mm_pix, interp_edge=True))
    subap_pup_ft = np.fft.rfft2(subap_pup/np.sum(subap_pup))
    pup = np.fft.fftshift(ot.utils.circle(sz, tel_diam/mm_pix, interp_edge=True))
    pup_ft = np.fft.rfft2(pup/np.sum(pup))
    
    #Gaussian mode in pupil 
    x = ((np.arange(sz) + sz//2) % sz) - sz//2
    xy = np.meshgrid(x,x)
    #FWHM equal to telescope diameter. Not sure if this is correct!
    gbeam = 0.5**( (xy[0]**2 + xy[1]**2)/0.5/(tel_diam/mm_pix)**2 )
    gbeam *= pup
    gbeam_ft = np.fft.rfft2(gbeam/np.sum(gbeam))
    
    #Now simulate the effect of the AO system. We should be left with 
    #delay * pup + (1 - delay * subap)
    delay[:] = delay_unfiltered
    correction = np.fft.irfft2(np.fft.rfft2(delay_unfiltered)*subap_pup_ft)
    correction = nd.interpolation.shift(correction, (ao_correction_lag,0),mode='wrap')
    delay[:] -= ao_correction_frac * correction
    
    #Subtract the convolution of the pupil with the delay
    delay_piston_free = delay_unfiltered - np.fft.irfft2(np.fft.rfft2(delay_unfiltered)*pup_ft)
    delay_filtered_piston_free = delay_unfiltered - np.fft.irfft2(np.fft.rfft2(delay_unfiltered)*gbeam_ft)
    #plt.figure(2)
    #plt.imshow(delay_piston_free)
    #plt.colorbar()
    #plt.pause(.01)
        
    #fiber_delays is the delay as measured in a fiber mode (e.g. Gravity)
    fiber_delays = []
    #gbeam delays is a Gaussian-weighted delay
    gbeam_delays = []
    strehls = []
    window = np.ones(nt)
    window[:16] *= (np.arange(16)+1)/16
    window[-16:] *= ((np.arange(16)+1)/16)[::-1]
    
    #Brute force loop...
    for t_ix, t in enumerate(np.arange(nt)*t_int):
            if (t_ix % 100 == 0):
                print("Done {:d} of {:d} frames".format(t_ix,nt))
            #Create the wavefront as a delay within a telescope
            new_delay = evolve(delay,t,v_wind, angle_wind, mm_pix)
            new_delay -= new_delay[0,0]
            new_delay_piston_free = evolve(delay_piston_free,t,v_wind, angle_wind, mm_pix)
            
            #Compute key parameters
            fiber_delays += [np.angle(np.sum(np.exp(2j*np.pi*new_delay/wl)*gbeam)/np.sum(gbeam))/2/np.pi*wl - np.sum(new_delay*gbeam)/np.sum(gbeam)]
            gbeam_delays += [np.sum(new_delay_piston_free*gbeam)/np.sum(gbeam)]
            strehls += [np.exp(-(np.std(new_delay[pup != 0])*1e6/2200*2*np.pi)**2)]
            
    fiber_delays = np.array(fiber_delays)
    gbeam_delays = np.array(gbeam_delays)
    strehls = np.array(strehls)
    print("Mean Strehl: {:5.3f}".format(np.mean(strehls)))
    
    #Lets look at these in the Fourier domain.
    ps_gbeam_delays = 2*np.convolve(np.abs(np.fft.rfft(gbeam_delays*window)**2),g, mode='same')
    
    #We know that summing the above and dividing by len(gbeam_delays)^2 gives the mean square.
    #Also, integrating should give the mean square.
    #Naieve integral: Trapz computes sum multplied by df=1/(t_int * nt)
    ps_gbeam_delays *= t_int*1e6 #Convert to microns
    ps_gbeam_delays /= nt
    ps_gbeam_delays = ps_gbeam_delays[1:]
    
    #Do the same for the fiber delays.
    ps_fiber_delays = 2*np.convolve(np.abs(np.fft.rfft(fiber_delays*window)**2),g, mode='same')
    ps_fiber_delays *= t_int*1e6
    ps_fiber_delays /= nt
    ps_fiber_delays = ps_fiber_delays[1:]
    
    fs = (np.arange(nt//2)+1)/nt/t_int
    print("RMS gbeam delay (um): {:5.2f}".format(np.sqrt(np.trapz(ps_gbeam_delays, fs))))
    print("RMS gbeam delay (um, direct calc): {:5.2f}".format(np.std(gbeam_delays)*1e3))
    
    #Simulate a simple PID servo loop
    s = 2j*np.pi*fs
    G = Ki*np.exp(-tau*s)*(1-np.exp(-Tint*s))/Tint**2/s**2
    G[fs > .5/Tint] = 0
    error = np.abs(1/(1+G))
    
    print("RMS corrected gbeam delay (um): {:5.3f}".format(np.sqrt(np.trapz(ps_gbeam_delays*error**2, fs))))
    print("RMS fiber delay (um): {:5.3f}".format(np.std(fiber_delays)*1e3))
    
    plt.figure(1)
    plt.clf()
    plt.loglog(fs, ps_gbeam_delays, 'b-', label='Fiber Piston Definition')
    plt.loglog(fs, ps_gbeam_delays*error**2, 'b:', label='Corrected Fiber Piston Definition')
    plt.loglog(fs, ps_fiber_delays, 'g-', label='Third order phase (K fringe tracker)')
    
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'Power ($\mu$m$^2$/Hz)')
    plt.axis([1e-2,1e2, 1e-11,1e3])
