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
#---
    if False:
        np.sum(delay_piston_free*gbeam)/np.sum(gbeam)*1e6
        fiber_delay = np.angle(np.sum(np.exp(2j*np.pi*delay/wl)*pup)/np.sum(pup))/2/np.pi*wl
        pup_delay = np.sum(delay*pup)/np.sum(pup)
        gbeam_delay = np.sum(delay*gbeam)/np.sum(pup)
    
        pistons = np.zeros( (4,sz*oversamp,sz*oversamp) )
        delays_for_strehl = np.zeros( (4,sz*oversamp,sz*oversamp) )

        #For convolving
    
        at_pup = np.fft.fftshift(ot.utils.circle(sz*oversamp, AT_DIAM/mm_pix_tel, interp_edge=True))
        highpass_pup = np.fft.fftshift(ot.utils.circle(sz*oversamp, highpass_diam/mm_pix_tel, interp_edge=True))
    
        at_pup_ft = np.fft.rfft2(at_pup/np.sum(at_pup))
        highpass_pup_ft = np.fft.rfft2(highpass_pup/np.sum(highpass_pup))

        #For thermal background
        pup_longwl, dummy = assemble_pupil(sz, np.max(wl_mns), mm_pix_pupil=mm_pix_lab)
        pup_frac = np.sum(np.abs(pup_longwl))/sz/sz

        #Create delays
        for delay, piston, delay_for_strehl in zip(delays,pistons, delays_for_strehl):
            #Create a wavefront in mm. 
            delay_unfiltered = ot.kmf(sz*oversamp, r_0_pix=r_0_500/mm_pix_tel)*.5e-3/2/np.pi 
    
        
            piston[:] = np.fft.irfft2(np.fft.rfft2(delay_unfiltered)*at_pup_ft)
            if highpass_filter_pistons:
                piston[:] = piston - np.fft.irfft2(np.fft.rfft2(piston)*highpass_pup_ft)
            delay_for_strehl[:] = delay - np.fft.irfft2(np.fft.rfft2(delay)*at_pup_ft)

        ix = np.arange(analyzed_sz)-analyzed_sz//2 + 0.25 
        xy = np.meshgrid(ix,ix)
        ix_full = np.arange(sz*oversamp)-sz*oversamp//2  
        xy_full = np.meshgrid(ix_full,ix_full)

        #AO Phase RMS
        delay_rms = 1e6*np.std(delays_for_strehl)
        print("Approximate AO delay RMS (nm): {0:6.1f}".format(delay_rms))
        print("Approximate Strehl: {0:6.3f}".format(np.exp(-(delay_rms*2*np.pi/1650)**2)))

        #Brute-force approach to finding noise constants
        nphot_test = 10000
        vs_noisy_dark = []
        vs_noisy_bright = []
        nsim = 25
        for i in range(nsim):
            ims = make_ims(waves, lenses, sz, oversamp, mm_pix_lab)
            noisy_ims = add_noise(ims, rnoise=rnoise, nphot=0, sznew=analyzed_sz, waves=waves, \
                skyback_filters=skyback_filters, t_int=t_int)
            vs_noisy_dark.append(compute_vis(noisy_ims, lab_uv, xy, wl_mns))
            noisy_ims = add_noise(ims, rnoise=rnoise, nphot=nphot_test, sznew=analyzed_sz, \
                waves=waves, skyback_filters=skyback_filters, t_int=t_int)
            vs_noisy_bright.append(compute_vis(noisy_ims, lab_uv, xy, wl_mns))
        vs_noisy_dark = np.array(vs_noisy_dark)
        vs_noisy_bright = np.array(vs_noisy_bright)
        #Lets find the relationship between imaginary visibility variance and flux
        var_dark = np.var(vs_noisy_dark.imag, axis=0)
        var_bright = np.var(vs_noisy_bright.imag, axis=0)
        var_scale = (var_bright - var_dark)/nphot_test
        print("Done computing phase error model")

        #Create the images and display... (this should look bad due to no AO!)
        delta_piston = .08e-3
        piston_vect = np.arange(-5e-3,5e-3,delta_piston) #In mm! (like all lengths here)
        tel_pistons = np.meshgrid(piston_vect,piston_vect,piston_vect, indexing='ij')
        #FIXME: This assumes 4 telescopes...
        zero_pistons = np.zeros_like(tel_pistons[0]).reshape((1,tel_pistons[0].shape[0],tel_pistons[0].shape[1],tel_pistons[0].shape[2])) 
        tel_pistons = np.concatenate((zero_pistons, tel_pistons))
        conv_per_step = (v_wind*t_int/r_0_500)**(5/6.)*np.sqrt(6.88)*.5e-3/2/np.pi/delta_piston

        plt.figure(2)
        plt.clf()
        plt.xlabel('Time (s)')
        plt.ylabel(r'Piston ($\mu$m)')
        plt.title('Delay estimate at {:d} photons/frame/bandpass'.format(nphot))
        piston_pdf = np.ones_like(tel_pistons[0])
        piston_diffs = []
        piston_correction = np.zeros(3)
        tilt_correction = np.zeros( (4,2) )
        t_ix_min = 0
    
    
    
        for t_ix, t in enumerate(np.arange(0, total_time, t_int)):
            #Create the wavefront as a delay within a telescope
            new_delay = evolve(delay,t,v_wind, angle_wind, mm_pix_tel)
            new_delay -= new_delay[sz//2, sz//2]

            #Create the image
        
        
            ims = make_ims(waves, lenses, sz, oversamp, mm_pix_lab, delays=new_delays, pistons=new_pistons)
            noisy_ims = add_noise(ims, rnoise=rnoise, nphot=nphot, sznew=analyzed_sz, waves=waves, \
                skyback_filters=skyback_filters, t_int=t_int)
    
            #A tilt of 1 pixel is a tilt of PIXEL_PITCH over a distance of
            #M1_F. This is a delay of PIXEL_PITCH/M1_F*mm_pix_lab
            dummy = compute_splodge_tilts(noisy_ims, lab_uv, xy, wl_mns, window_frac=0.3, ntel=4, normalise=False)
            tel_tilts = tel_tilts_from_splodge_tilts(dummy.T)
            print("RMS tilt: {:5.1f}".format(np.sqrt(np.mean(tel_tilts**2))))
            tilt_correction += tel_tilts*(PIXEL_PITCH/M1_F*mm_pix_lab)
    
            #Now compute the (complex) visibilities and noise.
            vs = compute_vis(noisy_ims, lab_uv, xy, wl_mns)
            phases = np.angle(vs)
            phase_errs = np.sqrt(var_scale*nphot + var_dark)/np.abs(vs)
            phase_errs = np.maximum(phase_errs, 0.4)
    
            #Every phase measurement produces a PDF in piston space. A phase PDF is an integral
            #over amplitude. 
            this_bl=0
            for i in range(0,NTEL-1):
                for j in range(i+1,NTEL):
                    for k in range(n_bandpass):
                        phase_array = 2*np.pi/wl_mns[k]*(tel_pistons[j]-tel_pistons[i])
                        piston_pdf *= phase_pdf(phase_array, phases[k,this_bl], phase_errs[k,this_bl])
                    this_bl += 1
    
    
            #Make figure 1 (Saphira image)
            plt.figure(1)
            plt.clf()
            saphira_im = np.zeros((64,320))
            saphira_im[:,8:analyzed_sz+8] = noisy_ims[2]
            saphira_im[:,8+80:analyzed_sz+8+80] = noisy_ims[0]
            saphira_im[:,8+80*2:analyzed_sz+8+80*2] = noisy_ims[1]
            saphira_im[:,8+80*3:analyzed_sz+8+80*3] = noisy_ims[3]
            plt.imshow(saphira_im)
            plt.tight_layout()
            plt.savefig("movie/movie{:03d}.png".format(t_ix))
            plt.pause(.001)
    
            #Make figure 2 (pistons)
            plt.figure(2)
            piston_est = -piston_vect[list(np.unravel_index(np.argmax(piston_pdf), piston_pdf.shape))]
            piston_correction += piston_est
            plt.plot(t, 1e3*piston_est[0], 'rx')
            plt.plot(t, 1e3*piston_est[1], 'gx')
            plt.plot(t, 1e3*piston_est[2], 'bx')
            plt.plot(t, 1e3*(new_pistons[1] - new_pistons[0]), 'ro')
            plt.plot(t, 1e3*(new_pistons[2] - new_pistons[0]), 'go')
            plt.plot(t, 1e3*(new_pistons[3] - new_pistons[0]), 'bo')
            plt.pause(.001)
    
            ix2 = np.unravel_index(np.argmax(piston_pdf), piston_pdf.shape)[2]
            plt.figure(3)
            plt.title('')
            plt.title('')
            plt.xlabel('Delay 1')
            plt.ylabel('Delay 2')
            plt.imshow(piston_pdf[:,:,ix2]**.05, extent=[-5,5,-5,5])
            plt.tight_layout()
            plt.pause(0.001)
            plt.savefig("movie/pdf{:03d}.png".format(t_ix))
        
    
            #Record the piston differences...
            piston_diffs.append([new_pistons[i+1] - new_pistons[0]-piston_est[i] for i in range(3)])
    
            #Now degrade the piston pdf, and normalise
            if correct_pistons:
                #piston_pdf = np.ones_like(tel_pistons[0]) 
                piston_pdf = nd.filters.gaussian_filter(piston_pdf, 2*conv_per_step)
            else:
                piston_pdf = nd.filters.gaussian_filter(piston_pdf, conv_per_step)
            piston_pdf /= np.mean(piston_pdf)
        
            #Step change
            if (t > time_servo_on) and not correct_pistons:
                correct_pistons=True   
                correct_tilts=True
                piston_pdf = np.ones_like(tel_pistons[0])    
    
            if False:
                #This demonstrates the correct sign.
                print(np.angle(vs))
                print(new_pistons/wl_mns[-1]*2*np.pi)
                print((new_pistons[0]-new_pistons[1:])/wl_mns[-1]*2*np.pi)
                print((new_pistons[1]-new_pistons[2:])/wl_mns[-1]*2*np.pi)
                print((new_pistons[2]-new_pistons[3:])/wl_mns[-1]*2*np.pi)
                pdb.set_trace()
   
        piston_diffs = np.array(piston_diffs)
        print("Piston RMS (nm): {:5.2f}".format(np.std(piston_diffs[t_ix_min:])*1e6))
        plt.tight_layout()
        plt.figure(3)
        plt.clf()
        plt.imshow(ims[0][sz//2-16:sz//2+16,sz//2-16:sz//2+16])
        plt.title('Open Loop H1 Image')
        plt.tight_layout()
        plt.figure(4)
        plt.clf()
        plt.imshow(noisy_ims[0][analyzed_sz//2-16:analyzed_sz//2+16,analyzed_sz//2-16:analyzed_sz//2+16])
        plt.title('Open Loop H1 Noisy Data')
        plt.tight_layout()

