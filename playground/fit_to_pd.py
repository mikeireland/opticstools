"""
Fit to a set of phase diversity images.

For the laboratory setup done by Hancheng, the focal length of the custom lens was ~0.8mm
in air (2mm in glass), meaning that the image was at ~0.813mm from the lens (thin lens
approx for focusing at 50mm or 0.800mm for focusing at infinity. This means that when 
defocusing, the beam actually diverged. This is apparently not an issue, as can be 
tested by using the "fresnel_focal_length" option of 50000 microns, where an initial
diameter of ~155 pixels is needed (~930 microns).

Alternatively, in a Fraunhofer scenario
"""
from PIL import Image
import numpy as np
import sys
if not '..' in sys.path:
    sys.path.insert(0,'..')
import opticstools as ot 
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import time

ddir = '/Users/mireland/data/midir/Lens2ImageStack/C330TMD/'

im_fns = [ddir + 'Lens2_C330TMD_BestFocus_26mm185um_38_03.tiff', \
    ddir + 'Lens2_C330TMD_Away10um_26mm175um_444_83.tiff', \
    ddir + 'Lens2_C330TMD_Close10um_26mm195um_69_32.tiff']
stage_pos = [0,-10,10]
im_fns = [ddir + 'Lens2_C330TMD_BestFocus_26mm185um_38_03.tiff', \
    ddir + 'Lens2_C330TMD_Away05um_26mm180um_131_91.tiff', \
    ddir + 'Lens2_C330TMD_Close10um_26mm195um_69_32.tiff']
stage_pos = [0,-5,5]
NA=0.58
obstruction_frac=0

ddir = '/Users/mireland/data/midir/Lens2ImageStack/Pike100x/'

im_fns = [ddir + 'Lens2_Pike100x_BestFocus_24mm195um_38_03.tiff', \
    ddir + 'Lens2_Pike100x_Away05um_24mm190um_69_32.tiff', \
    ddir + 'Lens2_Pike100x_Close05um_24mm200um_288_37.tiff']
stage_pos = [0,-5,5]


NA=0.8
obstruction_frac=0.6

#mod_ims = ot.pd_images(outer_diam=110, \
#    phase_zernikes=[0,0, 0,7,0, 0,0,0,0, 0,0,10,0,0], stage_pos=stage_pos, xt_offsets=[5,-5], yt_offsets=[-3,0])
#mod_ims = ot.pd_images(outer_diam=110, \
#    phase_zernikes=[0,0, 0,7,0, 0,-8,8,0, 0,0,9,0,0], fresnel_focal_length=50000)

#phase_zernikes = [-2.441,-0.709,   1.04,-3.21,-0.491,   0.072,2.47,-1.737,-0.481, -0.292,-0.216,9.981,-0.338,0.432, 0,0,0,0,0,0]
phase_zernikes = [-2.533,-1.144,  1.103,-3.29,-0.322,     0.003,1.984,-3.039,-0.668, -0.319,-0.189,10.959,-0.325,0.475, \
                0.101,-0.126,0.067,0.239,-0.138,-0.114]
amp_zernikes = [0.139,  -0.12,0.143,  -0.067,-0.757,0.029,   0.255,0.073,0.238,0.504, 0,0,0,0,0, 0,0,0,0,0,0]
mod_ims = ot.pd_images(outer_diam=120.4, inner_diam=120.4*obstruction_frac, \
    phase_zernikes=phase_zernikes, stage_pos=stage_pos,\
    xt_offsets=[-0.49,-0.07], yt_offsets=[-2.6,-2.3], foc_offsets=[1.33,3.81], \
    amp_zernikes=amp_zernikes)


#def pd_images(foc_offsets=[0,0], xt_offsets = [0,0], yt_offsets = [0,0], 
#    phase_zernikes=[0,0,0,0], amp_zernikes = [0], outer_diam=200, inner_diam=0, \
#    stage_pos=[0,-10,10], radians_per_um=None, NA=0.58, wavelength=0.633, sz=512, \
#    fresnel_focal_length=None, um_per_pix=6.0):

def fit_func(params, ims, NA, obstruction_frac, stage_pos, return_im=False):
    n_ims = len(ims)
    foc_offsets = params[:n_ims-1]
    xt_offsets = params[n_ims-1:2*(n_ims-1)]
    yt_offsets = params[2*(n_ims-1):3*(n_ims-1)]
    outer_diam = params[3*(n_ims-1)]
    amp_zernikes = params[3*(n_ims-1) + 1:3*(n_ims-1) + 11 + 11] #Second set of 11 for full 5th order.
    phase_zernikes = params[3*(n_ims-1) + 11 + 11:]
#    phase_zernikes = params[3*(n_ims-1) + 1:]
#    amp_zernikes = [0,0,0,0,0,0]
    mod_ims = ot.pd_images(foc_offsets=foc_offsets, xt_offsets=xt_offsets, yt_offsets=yt_offsets,\
        phase_zernikes=phase_zernikes, amp_zernikes=amp_zernikes, outer_diam=outer_diam, \
        inner_diam=outer_diam*obstruction_frac, stage_pos=stage_pos)
    if return_im:
        return mod_ims
    resid_ims = ims - mod_ims
    residuals=[]
    for rim in resid_ims:
        residuals.append(rim[256-128:256+128,256-128:256+128])
    residuals = np.array(residuals).flatten()
    return residuals

sz = 512
n_ims = len(im_fns)
ims = []
for fn in im_fns:
    im = np.array(Image.open(fn))
    im = im - np.median(im)
    g = np.exp(-(np.arange(200)-100)**2/2/40**2)
    xmax = np.argmax(np.convolve(np.sum(im, axis=0), g, mode='same'))
    ymax = np.argmax(np.convolve(np.sum(im, axis=1), g, mode='same'))
    im = im[ymax-sz//2:ymax+sz//2, xmax-sz//2:xmax+sz//2]
    im /= np.sum(im)
    ims.append(im)
ims = np.array(ims)    
fig = plt.figure(1)
plt.clf()
for i in range(n_ims):
    a = fig.add_subplot(2, n_ims, i+1)
    plt.imshow(np.maximum(ims[i],0)**.5)
    a.set_title('Data (offset={:.1f})'.format(stage_pos[i]))
for i in range(n_ims):
    a = fig.add_subplot(2, n_ims, i+1+n_ims)
    plt.imshow(mod_ims[i]**.5)
    a.set_title('Model (offset={:.1f})'.format(stage_pos[i]))
plt.tight_layout()

#Try a fit...
#focus, x, y offsets, and diameter, :7
#amplitude parameters, 7:13
#phase parameters, 13:
params = [0.811,   7.788,  -0.288,  -0.21 , -2.083, -3.143,120.992,]

params = np.concatenate([params, amp_zernikes, phase_zernikes])
                  
print("Computing Residual...")
then = time.time()
init_resid = fit_func(params, ims, NA, obstruction_frac, stage_pos)
init_chi2 = np.sum(init_resid**2)
print("Initial Chi-squared: {:.2e}".format(init_chi2))
print(time.time()-then)

if True:
    fitted_params = least_squares(fit_func, params, args=(ims, NA, obstruction_frac, stage_pos), xtol=1e-4, ftol=1e-4, max_nfev=1000)

    #np.set_printoptions(precision=3)
    #np.set_printoptions(suppress=True)

    final_resid = fit_func(fitted_params.x, ims, NA, obstruction_frac, stage_pos)
    final_chi2 = np.sum(final_resid**2)
    print("Final Chi-squared: {:.2e}".format(final_chi2))
    mod_ims = fit_func(fitted_params.x, ims,  NA, obstruction_frac, stage_pos, return_im=True)