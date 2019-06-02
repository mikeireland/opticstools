"""Lets simulate some actual HEIMDALLR images.

Note that a back-of-envelope computation occurs in optimal_ft_time.py, within the
PFI simulation.

Assume atmospheric profile which comes from Sarazin and Tokovinin (2002):
- 0.86" seeing
- tau_0=3.9ms, which is 9.4m/s wind. But lets double this to cover more actual seeing
conditions.

Rather than taking piston away and adding it back in, we will just move through the 
wavefront with nd.interpolation.shift, with a wrap.

Options for multi-wavelength...

Option 1: Fresnel propagate to the image plane (one extra FT)
Option 2: Convolve the final image and re-sample (two extra FTs)

We went with option 1.

For open-loop fringe acquisition, we have 6 x 4 phasors, which produce a 
maximum likelihood PDF in 3D for the delays. A brute-force approach would 
sample these PDFs with better than 1 radian at the shortest wavelength
(e.g. 0.2 microns) over the maximum plausible acquisition range, i.e.
+/- 12.8 microns. This is a 2 million element likelihood function, which can 
be convolved each time (representing seeing) and have the latest measurement
added on.

NB 250 photons/frame/bandpass is about magnitude 10.6 in H, based on 12% throughput
overall.

np.log10(9.7e9*.15*np.pi*.9**2*.12/250e2)*2.5

Results: 150 photons/frame/bandpass is fine... 

ffmpeg -r 15 -i movie%03d.png -f mp4 -q:v 0 -vcodec mpeg4 -r 15 loop_close.mp4
ffmpeg -r 15 -i pdf%03d.png -f mp4 -q:v 0 -vcodec mpeg4 -r 15 pdf.mp4
"""

from heimdallr_layout import *
import astropy.constants as const
import astropy.units as units
import pdb
import os
import glob
plt.ion()

for f in glob.glob('movie/*png'):
    os.remove(f)

#----- Define servo parameters and fixed constants here ---
#Number of photons per bandpass per exposure. 
nphot=65 #160

r_0_500 = 0.98*.5e-6/np.radians(0.86/3600)*1000 #In mm
r_0_500 = 0.98*.5e-6/np.radians(1.0/3600)*1000 #In mm
#r_0_500 = 0.98*.5e-6/np.radians(0.05/3600)*1000 #In mm
v_wind = 9.4*1000*2.0 #9.4 for median seeing. Actually vbar
angle_wind = 0.1

wl_los=np.array([1.48,1.63,1.95,2.16])*1e-3
wl_his=np.array([1.63,1.81,2.16,2.40])*1e-3
n_bandpass = len(wl_los)
nwl_bandpass=4
mm_pix_lab = 1.5 #Lab pupil...
mm_pix_tel = mm_pix_lab*AT_DIAM/BEAM_DIAM
sz = 128
analyzed_sz = 64
oversamp=2
subap_diam = 450. #In mm
H_C_ON_KB = (const.h*const.c/const.k_B/units.K/units.mm).si.value

#Fraction of wavefront that is corrected. In practice, this is spatial-frequency
#dependent, with a larger fraction corrected in e.g. tip/tilt modes. However, 
#there is additional tip/tilt noise due to vibrations, so likely this isn't 
#too far out. Lag multiplies a -5/3 power spectrum by an exponent of 2
ao_correction_frac = 0.8 #!!! Should be 0.8
ao_correction_lag = 0.004*v_wind/mm_pix_tel #in pix. 
t_int = 0.004 #For faint stars (reference case), should be 0.009
rnoise = 0.35
highpass_diam = 10e3
highpass_filter_pistons = True

#Lets figure out the H to K1 mean background, and use this as a general background.
skyback = np.loadtxt('cp_skybg_zm_43_10_ph.dat.txt')
throughput = 0.12
skyback_filters = []
for wl_lo,wl_hi in zip(wl_los, wl_his):
    ww=np.where((skyback[:,0] > wl_lo*1e6) * (skyback[:,0] < wl_hi*1e6))[0]
    #Sky background is in photons/s/nm/arcsec^2/m^2
    #A diffraction-limited beam has an area of ~(lamdbda/D)^2 steradians, or 
    #(lambda/D*arcsec_per_radians)^2 square arecsec. 
    skyback_nm = np.mean(skyback[ww,1]*(skyback[ww,0]/1e9*np.degrees(3600))**2) * throughput
    skyback_filters.append(skyback_nm*(wl_hi - wl_lo)*1e6)
skyback_filters = np.array(skyback_filters)

correct_pistons=False
correct_tilts=False
#correct_pistons=True
#correct_tilts=True
time_servo_on = 0.05e3
total_time = 0.15
servo_gain = 0.9
#-----------------------

def make_ims_var_pupscale(wl_los, wl_his, wfs=None, nwl_bandpass=4, sz=64):
    """A test case to make images with variable pupil scale.
    """
    n_bandpass = len(wl_los)
    ims = np.zeros( (n_bandpass, sz, sz) )
    for im, wl_lo,wl_hi in zip(ims, wl_los, wl_his):
        waves = np.linspace(wl_lo, wl_hi,nwl_bandpass+1)
        waves = 0.5*(waves[1:] + waves[:-1])
        for wave in waves:
            pup, mm_pix = assemble_pupil(sz, wave, wfs)        
            im += np.fft.fftshift(np.abs(np.fft.fft2(pup))**2)
        #Normalise...
        im /= np.sum(im)
    return ims
    
def make_lenses(wl_los, wl_his, mm_pix, nwl_bandpass=4, sz=64, oversamp=2):
    """Make images where the same pupil scale is used for all wavelengths.
    
    We use Fresnel propagation instead of an FFT.
    """
    n_bandpass = len(wl_los)
    lenses = []
    waves_all=[]
    for wl_lo,wl_hi in zip(wl_los, wl_his):
        waves = np.linspace(wl_lo, wl_hi,nwl_bandpass+1)
        waves = 0.5*(waves[1:] + waves[:-1])  
        waves_all.append(waves)      
        sub_lenses = []
        for wave in waves:
            sub_lenses.append(ot.FocusingLens(sz*oversamp,mm_pix, PIXEL_PITCH/oversamp, M1_F, wave))
        lenses.append(sub_lenses)
    return np.array(waves_all), np.array(lenses)

def make_intermediate_plane(waves, sz, mm_pix_pupil, intermediate_dist, f_length=2500): 
    """Find the intensity at an intermediate plane.

    A key result from this is that even at 400mm from the focal plane, there is a 40%
    increase in background for a mask made large enough to capture 96% of the flux.
    
    If long K matters, then this means that we do want an intermediate focal plane and 
    a cold stop in the pupil. If we really have 170mm of path length for the dichroics,
    then there is 340mm of path length to the lens, which could be warm, or indeed could
    be the dewar window.
    """
    reimaged_dist = 1./(1./intermediate_dist - 1./f_length)
    n_bandpass = waves.shape[0]
    nwl_bandpass = waves.shape[1]
    ims = np.zeros((n_bandpass,sz,sz))
    for im, sub_lenses, sub_waves in zip(ims, lenses, waves):
        for wave in sub_waves:
            pup, dummy = assemble_pupil(sz, wave, mm_pix_pupil=mm_pix_pupil)
            #lens = ot.curved_wf(sz,mm_pix_pupil,f_length=f_length,wave=wave)
            intermediate_pup = ot.propagate_by_fresnel(pup, mm_pix_pupil, reimaged_dist, wave)
            im += np.abs(intermediate_pup)**2
        #Keep normalised
        im /= nwl_bandpass
    return ims, pup


def make_ims(waves, lenses, sz, oversamp, mm_pix_pupil, delays=None, pistons=None,
    test_defocus=False):
    """Make images via Frensel propagation.
    
    
    Parameters
    ----------
    lenses: (n_bandpass, nwl_bandpsss) FocusingLens array
        Fresnel propagators ready to go.
    
    test_defocus:
        Try to manually add some defocus, to see what happens to the peak intensity.
    """
    n_bandpass = waves.shape[0]
    nwl_bandpass = waves.shape[1]
    ims = np.zeros((n_bandpass,sz,sz))
    for im, sub_lenses, sub_waves in zip(ims, lenses, waves):
        for lens, wave in zip(sub_lenses, sub_waves):
            #We have to assemble the pupil for every wavelength, due to the independent 
            #wavefronts. 
            #FIXME: This can be sped-up by saving the shifted pupils and the wavefronts
            #separately. 
            pup, dummy = assemble_pupil(sz*oversamp, wave, delays=delays, \
                pistons=pistons, mm_pix_pupil=mm_pix_pupil)
            if test_defocus:
                defoc_one = ot.curved_wf(sz*oversamp,mm_pix_pupil,f_length=210833.,wave=wave)
                acirc = ot.utils.circle(sz*oversamp, 20/mm_pix_pupil)
                defoc_one = acirc*defoc_one #+ (1-acirc)
                pup *= defoc_one
            im += ot.rebin(lens.focus(pup),(sz,sz))
        #Keep normalised
        im /= nwl_bandpass
    return ims

def add_noise(ims, rnoise=0.5, nphot=400, sznew=64, waves=None, tlab=288.0, \
    pup_frac=0.027, t_int=0.009, skyback_filters=None, print_background=False, eta_c=0.7):
    """Add noise to the image
    
    Parameters
    ----------
    pup_frac: float
        Fraction of the image fourier transform that is warm. 
        FIXME: should be an array. But only the longest wavelength matters, so it is
        close enough.
    """
    if waves is not None:
        dws = np.mean(waves[:,1:]-waves[:,:-1], axis=1)
        bg_fluxes = []
        for sub_waves, dw in zip(waves, dws):
            sub_bg_flux = eta_c*t_int*pup_frac*2/(np.exp(H_C_ON_KB/sub_waves/tlab)-1)*3e8/1e-3*dw/sub_waves**2
            bg_fluxes.append(np.sum(sub_bg_flux))
        bg_fluxes = np.array(bg_fluxes)
        if skyback_filters is not None:
            bg_fluxes += skyback_filters*t_int*pup_frac
        if print_background:
            print(bg_fluxes)
    oldsz = ims.shape[1]
    noisy_ims = ims[:,oldsz//2-sznew//2:oldsz//2+sznew//2,oldsz//2-sznew//2:oldsz//2+sznew//2].copy()
    for noisy_im, bg_flux in zip(noisy_ims, bg_fluxes):
        noisy_im[:] = np.random.poisson(nphot*noisy_im + bg_flux) + np.random.normal(scale=rnoise, size=noisy_im.shape)
    return noisy_ims

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

def compute_vis(noisy_ims, lab_uv, xy, wl_mns, window_frac=0.3, ntel=4, normalise=False):
    """Compute visibilities on all baselines, for all bandpasses. A discrete Fourier 
    transform is used. """
    nbl = lab_uv.shape[1]
    nbp = lab_uv.shape[0]
    vs = np.empty( (nbp, nbl), dtype=complex)
    for i in range(nbp):
        windowed_data = noisy_ims[i]*np.exp(-((xy[0]**2 + xy[1]**2)/(wl_mns[i]/wl_mns[-1]*analyzed_sz*window_frac)**2)**2)
        for j in range(nbl):
            phasors = np.exp(2j*np.pi*(xy[0]*lab_uv[i,j,0] + xy[1]*lab_uv[i,j,1]))
            if normalise:
                vs[i,j] = ntel*np.sum(phasors*windowed_data)/np.sum(windowed_data)
            else:
                vs[i,j] = ntel*np.sum(phasors*windowed_data)
    return vs

def compute_eigenphase(noisy_ims, lab_uv, xy, wl_mns, window_frac=0.3, ntel=4):
    """Compute the Fourier transform at 7 discrete positions per pupil, producing
    19 positions per splodge. """
    return None

def compute_splodge_tilts(noisy_ims, lab_uv, xy, wl_mns, window_frac=0.25, ntel=4, \
    normalise=False, plotit=False):
    """Compute tilts of each splodge, in order to back out tilts on each sub-aperture. 
    Here we window *after* inverse Fourier transorming.
    """
    nbl = lab_uv.shape[1]
    nbp = lab_uv.shape[0]
    sz = noisy_ims.shape[1]
    vs = np.empty( (nbp, nbl), dtype=complex)
    splodge_tilts = np.zeros( (nbp, 2, 7) )
    for i in range(nbp):
        window = np.exp(-((xy[0]**2 + xy[1]**2)/(wl_mns[i]/wl_mns[-1]*analyzed_sz*window_frac)**2)**2)
        wbg = np.where(window < 0.01)
        bg = np.mean(noisy_ims[i][wbg[0], wbg[1]])
        windowed_data = (noisy_ims[i] - bg)*window
        splodge_tilts[i, 0,0] = np.sum(xy[0]*windowed_data)/np.sum(windowed_data)
        splodge_tilts[i, 0,1] = np.sum(xy[1]*windowed_data)/np.sum(windowed_data)
        data_ft = np.fft.fftshift(np.fft.fft2(noisy_ims[i]-bg))
        mask = ot.utils.circle(sz, 1.9*pupil_uv_diam(wl_mns[i])*sz)
        mask = np.fft.fftshift(mask)
        pts = ( (0.5 + lab_uv[i,:])*sz ).astype(int)
        for j in range(nbl):
            data_ft_shifted = np.roll(np.roll(data_ft, -pts[j,0], axis=1), -pts[j,1], axis=0)
            demod_image = np.abs(np.fft.ifft2(data_ft_shifted*mask))**2
            splodge_tilts[i, 0, j+1] = np.sum(xy[0]*demod_image*window)/np.sum(demod_image*window)
            splodge_tilts[i, 1, j+1] = np.sum(xy[1]*demod_image*window)/np.sum(demod_image*window)
            if plotit:
                plt.imshow(demod_image)
                if(np.max(demod_image) < 10):
                    pdb.set_trace()
                plt.pause(.1)
    #plt.clf()
    #plt.plot(splodge_tilts[:,0].T)
    #plt.draw()
    #pdb.set_trace()
    #FIXME: Weighted mean needed.
    return np.mean(splodge_tilts, axis=0)

def tel_tilts_from_splodge_tilts(splodge_tilts, splodge_amps=[4,1,1,1,1,1,1]):
    """Compute optimal telescope tilts from demodulated splodge tilts
    
    If amplitudes are high (e.g. theoretical), then the reconstructor vector
    for telescpe 0 is:
    [8,6,6,6,-5,-5,-5]/11 for telescope 0. This is different to the naieve:
    [4,0,0,0,-1,-1,-1]
    
    This has an error 1.24 times a single splodge error, which in turn has an 
    error 2 times a single telescope tilt error. Overall... a 2.4 times higher
    error.
    """
    w_mat = np.diag(np.array(splodge_amps)**2)
    x_mat = np.array([[.25,.25,.25,.25],[.5, .5, 0,0], [.5, 0, .5, 0], \
        [.5, 0, 0, .5], [0, .5, .5,0], [0, .5, 0, .5], [0, 0, .5, .5]])
    xtw = np.dot(x_mat.T, w_mat) 
    reconstruct_matrix = np.dot(np.linalg.inv(np.dot(xtw, x_mat)), xtw)
    return np.dot(reconstruct_matrix, splodge_tilts)


def phase_pdf(phase_array, phase_mn, phase_err):
    """Compute an approximate phase probability distribution function, taking into account 
    the additional chance of being out by np.pi in an approximate way.
    
    Should work for SNR=2 or more. A better functional form is required for low SNR... 
    e.g. by inerpolating between a Gaussian and a truncated power law."""
    return np.exp(-((phase_array-phase_mn + np.pi) % (2*np.pi) - np.pi)**2/2/phase_err**2) \
        + np.exp(-1/2/phase_err**2)

def phase_pdf(phase_array, phase_mn, phase_err, scale=0.05):
    """Compute an approximate phase probability distribution function, taking into account 
    the additional chance of being out by np.pi in an approximate way.
    
    Should work for SNR=2 or more. A better functional form is required for low SNR... 
    e.g. by inerpolating between a Gaussian and a truncated power law."""
    return np.exp(-((phase_array-phase_mn + np.pi) % (2*np.pi) - np.pi)**2/2/phase_err**2) \
        + scale*np.exp(-1/2/phase_err**2)

def phase_pdf_2d(phase_array, phase_err):
    """Test the PDF"""
    x = np.linspace(-5,5,512)
    xy = np.meshgrid(x,x)
    gg = np.exp(-((xy[0]-1)**2 + xy[1]**2)/2/phase_err**2)
    th = np.arctan2(xy[1], xy[0])
    pdf_out = np.zeros_like(phase_array)[:-1]
    for th_ix in range(len(phase_array) - 1):
        pdf_out[th_ix] = np.sum(gg[np.where((th >= phase_array[th_ix]) & (th < phase_array[th_ix+1]))])
    return pdf_out /np.sum(pdf_out)

#------------------
if __name__=="__main__":
    #Preliminaries
    wl_mns = 0.5*(wl_los + wl_his)
    lab_uv = [lab_uv_coords(w) for w in wl_mns]
    lab_uv = np.array(lab_uv)

    #Old way...
    #ims = make_ims_var_pupscale(wl_los, wl_his)

    #New way of doing things...
    waves, lenses=make_lenses(wl_los, wl_his, mm_pix_lab, nwl_bandpass=4, sz=sz)
    ims_perfect = make_ims(waves, lenses, sz, oversamp, mm_pix_lab, test_defocus=True)

    #Some test lines for testing defocus...
    #print(np.max(np.max(ims_perfect, axis=2), axis=1))
    #pdb.set_trace()

    #Now create some wavefronts.
    delays = np.zeros( (4,sz*oversamp,sz*oversamp) )
    pistons = np.zeros( (4,sz*oversamp,sz*oversamp) )
    delays_for_strehl = np.zeros( (4,sz*oversamp,sz*oversamp) )

    #For convolving
    subap_pup = np.fft.fftshift(ot.utils.circle(sz*oversamp, subap_diam/mm_pix_tel, interp_edge=True))
    at_pup = np.fft.fftshift(ot.utils.circle(sz*oversamp, AT_DIAM/mm_pix_tel, interp_edge=True))
    highpass_pup = np.fft.fftshift(ot.utils.circle(sz*oversamp, highpass_diam/mm_pix_tel, interp_edge=True))
    subap_pup_ft = np.fft.rfft2(subap_pup/np.sum(subap_pup))
    at_pup_ft = np.fft.rfft2(at_pup/np.sum(at_pup))
    highpass_pup_ft = np.fft.rfft2(highpass_pup/np.sum(highpass_pup))

    #For thermal background
    pup_longwl, dummy = assemble_pupil(sz, np.max(wl_mns), mm_pix_pupil=mm_pix_lab)
    pup_frac = np.sum(np.abs(pup_longwl))/sz/sz

    #Create delays
    for delay, piston, delay_for_strehl in zip(delays,pistons, delays_for_strehl):
        #Create a wavefront in mm. 
        delay_unfiltered = ot.kmf(sz*oversamp, r_0_pix=r_0_500/mm_pix_tel)*.5e-3/2/np.pi 
    
        #Now simulate the effect of the AO system. We should be left with 
        #delay * pup + (1 - delay * subap)
        delay[:] = delay_unfiltered
        correction = np.fft.irfft2(np.fft.rfft2(delay_unfiltered)*subap_pup_ft)
        correction = nd.interpolation.shift(correction, (ao_correction_lag,0),mode='wrap')
        delay[:] -= ao_correction_frac * correction
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
        #Do we turn the servos on?
        if (t > time_servo_on) and not correct_pistons:
            t_ix_min = t_ix+2
            piston_correction = np.zeros(3)
            tilt_correction = np.zeros( (4,2) )
    
        #Create the wavefront as a delay within a telescope
        new_delays = evolve(delays,t,v_wind, angle_wind, mm_pix_tel)
        for dd in new_delays:
            dd -= dd[sz*oversamp//2, sz*oversamp//2]
        
        #Create the telescope pistons separately to the delay within a telescope...
        new_pistons = evolve(pistons,t,v_wind, angle_wind, mm_pix_tel)[:,sz*oversamp//2, sz*oversamp//2]
        
        #Correct for the previous measurements of piston and tilt.
        if correct_pistons:
            new_pistons[1:] -= servo_gain*piston_correction
            
        if correct_tilts:
            for i in range(0,NTEL):
                new_delays[i] += servo_gain*tilt_correction[i,0]*xy_full[0]
                new_delays[i] += servo_gain*tilt_correction[i,1]*xy_full[1]
                
        #Create the images
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

