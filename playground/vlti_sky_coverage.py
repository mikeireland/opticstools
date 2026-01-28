"""Determine sky coverage for the VLTI (e.g. Gravity+)
"""

import numpy as np
import pickle
import pdb
import matplotlib.pyplot as plt
import scipy.signal as sig
plt.ion()

#This file has 200 random positions within a 5x5 degree box.
infile = 'random_tt_05_0200.pkl' 
ns = 200

#For more rapid testing, use this, with 50 random positions in a 10x10 degree box.
#infile = 'random_tt_10_0050.pkl'
#ns = 50

#Fringe tracking limit. Note that the 2MASS with faint source extension is really only good
#to 15.3, which means that this is somewhat incomplete beyond m100=13.5
m100 = 14.0
WAVE = 2200
RLIM = 99#17 #Hard limit for tip/tilt
JLIM = 99#17 #Hard limit for tip/tilt, with AO
tracking_lim2 = 300**2 #Fringe tracking limit.
eta_lim1 = 0.5
eta_lim2 = 1/np.sqrt(10)

#----------
def phase(delta, wave=WAVE):
    return 2*np.pi*delta/wave

def phase2(delta2, wave=WAVE):
    "Helper function - return the phase squared"
    return (2*np.pi/wave)**2*delta2

def c2d(a,b):
    return np.fft.irfft2(np.fft.rfft2(a)*np.fft.rfft2(b))

n_too_faint =0
starlist_ix = 0
laser_on_tts = []
eta_singles = []
eta_duals = []
eta_single_on_tgts = []
tt_seps = []
tt_mags = []

with open(infile, 'rb') as f:
    ras, decs, starlists = pickle.load(f)
    radec = np.meshgrid(ras, decs, indexing='ij')
    ddec = (decs[1]-decs[0])
    omegas = np.cos(np.radians(radec[1] + ddec/2))
    for ra in ras:
        print(ra)
        for dec in decs:
            starlist = starlists[starlist_ix]
            laser_on_tt = np.zeros(ns, dtype=np.bool)
            eta_single = np.zeros(ns)
            eta_dual = np.zeros(ns)
            eta_single_on_tgt = np.zeros(ns)
            tt_sep = np.zeros(ns)
            tt_mag = np.zeros(ns)
            for ss in starlist:
                th = ss['_r']
                #This is the target position index. There can be several potential 
                #guide stars per index.
                ix = ss['_q']-1
                
                #Enforce hard tip/tilt limit. This actually *never* happens.
                if ss['Jmag']>JLIM:
                    continue
                
                #Start with Dual laser
                d2aniso_piston = (17.8 * th)**2
                d2ao_science = (9.9 * th)**2
                d2tracking = (100 * 10**(4/15*(ss['Kmag']-m100)))**2
                this_eta = np.exp(-phase2(d2tracking + d2aniso_piston + d2ao_science))
                if d2tracking > tracking_lim2:
                    this_eta=0
                if this_eta > eta_dual[ix]:
                    eta_dual[ix] = this_eta
                
                #Now laser on tip/tilt
                d2ao_science = (44 * th**(5/6))**2
                this_eta = np.exp(-phase2(d2tracking + d2aniso_piston + d2ao_science))
                if d2tracking > tracking_lim2:
                    this_eta=0
                if this_eta > eta_single[ix]:
                    eta_single[ix] = this_eta
                    laser_on_tt[ix] = True
                
                #Now laser on science target
                d2ao_science = (9.9 * th)**2
                d2ao_tt = (40 * th**(5/6))**2
                #Reduce the ability to fringe track due to lower tip/tilt star Strehl.
                #e.g. for an additional 1 radian AO RMS, the photon-noise component
                #of the RMS fringe tracking error increases by 40%. 
                d2tracking *= np.exp(2/3*phase2(d2ao_tt))
                this_eta = np.exp(-phase2(d2tracking + d2aniso_piston + d2ao_science))
                if d2tracking > tracking_lim2:
                    this_eta=0
                    
                #Enforce limit on R magnitude for tip/tilt.
                if (2*ss['Jmag']-ss['Kmag'])>RLIM:
                    n_too_faint += 1
                    continue
                if this_eta > eta_single_on_tgt[ix]:
                    eta_single_on_tgt[ix] = this_eta
                    tt_sep[ix] = th
                    tt_mag[ix] = ss['Kmag']
                if this_eta > eta_single[ix]:
                    eta_single[ix] = this_eta
                    laser_on_tt[ix] = False
            eta_singles += [eta_single]
            eta_duals += [eta_dual]
            eta_single_on_tgts += [eta_single_on_tgt]
            laser_on_tts += [laser_on_tt]
            tt_seps += [tt_sep]
            tt_mags += [tt_mag]
            starlist_ix += 1
laser_on_tts = np.array(laser_on_tts).reshape((len(ras), len(decs), ns))
eta_singles = np.array(eta_singles).reshape((len(ras), len(decs), ns))
eta_duals = np.array(eta_duals).reshape((len(ras), len(decs), ns))
eta_single_on_tgts = np.array(eta_single_on_tgts).reshape((len(ras), len(decs), ns))
tt_mags = np.array(tt_mags).reshape((len(ras), len(decs), ns))
tt_seps = np.array(tt_seps).reshape((len(ras), len(decs), ns))
print('Mean eta (single science laser): {:5.2f}'.format(np.mean(eta_single_on_tgts)))
print('Mean eta (single laser): {:5.2f}'.format(np.mean(eta_singles)))
print('Mean eta (dual laser): {:5.2f}'.format(np.mean(eta_duals)))

xy = np.meshgrid(np.fft.fftfreq(len(ras))*len(ras),  np.fft.fftfreq(len(decs))*len(decs))
gg = np.exp(-(xy[0]**2+xy[1]**2)/2/0.8**2)
gg /= np.sum(gg)

for ix, eta_lim in enumerate([eta_lim1, eta_lim2]):
    skyfrac_single = np.mean(eta_singles>eta_lim, axis=2)
    skyfrac_dual = np.mean(eta_duals>eta_lim, axis=2)
    skyfrac_laser_on_tt = np.mean(eta_single_on_tgts>eta_lim, axis=2)
    print('Mean skyfrac (single science laser): {:5.2f}'.format(np.sum(skyfrac_laser_on_tt*omegas)/np.sum(omegas)))
    print('Mean skyfrac (single laser): {:5.2f}'.format(np.sum(skyfrac_single*omegas)/np.sum(omegas)))
    print('Mean skyfrac (dual laser): {:5.2f}'.format(np.sum(skyfrac_dual*omegas)/np.sum(omegas)))
    print('eta high avg (single science laser): {:5.3f}'.format(np.mean(eta_single_on_tgts[eta_single_on_tgts > eta_lim])))
    print('eta high avg (single laser): {:5.3f}'.format(np.mean(eta_singles[eta_single_on_tgts > eta_lim])))
    print('eta high avg (dual laser): {:5.3f}'.format(np.mean(eta_duals[eta_single_on_tgts > eta_lim])))

    plt.figure(1 + 2*ix)     
    plt.clf()       
    plt.imshow(c2d(skyfrac_single.T[::-1],gg), extent=[0,360,-90,90], vmin=0, vmax=1)
    plt.title(r'Single Laser ($\eta$={:5.2f})'.format(eta_lim))
    plt.xlabel('RA (deg)')
    plt.ylabel('Dec (deg)')
    plt.colorbar()
    plt.tight_layout()

    plt.figure(2 + 2*ix)
    plt.clf()
    plt.imshow(c2d(skyfrac_dual.T[::-1],gg), extent=[0,360,-90,90], vmin=0, vmax=1)
    plt.title(r'Dual Laser ($\eta$={:5.2f})'.format(eta_lim))
    plt.xlabel('RA (deg)')
    plt.ylabel('Dec (deg)')
    plt.colorbar()
    plt.tight_layout()

