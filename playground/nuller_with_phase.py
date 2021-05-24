"""
The CSV input files came from WebPlotDigitizer and Harry's plots.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

imbalance = np.genfromtxt('harry_imbalance.csv', delimiter=',')
phase_deg = np.genfromtxt('harry_phase.csv', delimiter=',')

#Wavelength range
wave = np.linspace(3.7,4.3,51)

def worst_null(p, wave, imbalance, phase_deg, return_nulls=False):
    """
    Return the worst null depth in dB, with negative numbers a deeper Null
    """
    wmn = 0.5*(wave[0]+wave[-1])
    mod_phase = p[0] + p[1]*(wave - wmn)
    I1 = 0.5*(1+np.interp(wave, imbalance[:,0], imbalance[:,1]))
    I2 = 1- I1
    pdiff = np.interp(wave, phase_deg[:,0], phase_deg[:,1]) - mod_phase
    nulls = np.abs(np.sqrt(I1) - np.sqrt(I2)*np.exp(1j*np.radians(pdiff)))**2
    if return_nulls:
        return nulls
    else:
        return 10*np.log10(np.max(nulls))
        
best_p = op.minimize(worst_null, [90, 0], args=(wave, imbalance, phase_deg), method='Nelder-Mead')
nulls = worst_null(best_p.x, wave, imbalance, phase_deg, return_nulls=True)

#Now make a plot of the null depth with the sign convention of Harry's plots.
plt.clf()
plt.plot(wave, -10*np.log10(nulls))
plt.xlabel('Wavelength (microns)')
plt.ylabel('Extinction (dB)')