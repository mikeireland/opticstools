"""
Assume a data model that is:

f(x) = a_0 * PSF(x-a_3) + a_1 * PSF(x-a_4)

Then we linearise about our best solution:

f(x) = f_0(x) + J_0(x) . da

where da=[da_0, da_1, da_2, da_3]
J_0 is the Jacobian at our best solution:

[df/da_0, df/da_1, df/da_2, df/da_3]

We then use the standard formulae for weighted multivariate linear regression in order
to compute the photon-limited uncertainties.
"""
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import opticstools as ot
import pdb

plt.clf()
sz = 256
psz = 64
rnoise = 1e4
nshifts = 16
pup = ot.circle(2*sz, 2*psz)
im = np.fft.fftshift(np.abs(np.fft.fft2(pup))**2)
im = ot.rebin(im, (sz,sz))
im /= np.sum(im)

#Run for a couple of different photon rates.
for nphot, label in zip([6e9, 3e8], ['Idealized', 'Non-saturated']):
    im_var = im * nphot + rnoise

    contrast_sig = []
    js = np.arange(1, nshifts+1)

    for j in js:
        im_shifted = np.roll(im, j, axis=1)
        #FIXME
        #NB If we really cared, this next line would be a sub-pixel interpolation.
        im_deriv0 = 0.5*(np.roll(im, 1, axis=1)-np.roll(im, -1, axis=1))
        im_deriv1 = 0.5*(np.roll(im, j+1, axis=1)-np.roll(im, j-1, axis=1))
        #The Jacobian.
        X = np.array([im.flatten(), im_shifted.flatten(), im_deriv0.flatten(), im_deriv1.flatten()])
        #The weights vector, which is the inverse covariance.
        W = 1.0/im_var.flatten()
        #Row-by-row multiplication of X by W. 
        WX = X.copy()
        for WXrow in WX: WXrow *= W
        #
        cov = np.linalg.inv(np.dot(X, WX.T))
        contrast_sig.append(np.sqrt(cov[1,1])/nphot)
    
    contrast_sig = np.array(contrast_sig)
    plt.semilogy(js*psz/sz, contrast_sig*5, label=label)
    plt.xlabel(r'Separation ($\lambda$/D)')
    plt.ylabel(r'5-$\sigma$ Contrast')
plt.legend()