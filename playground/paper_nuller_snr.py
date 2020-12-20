from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pdb
import sys
plt.ion()
np.set_printoptions(precision=5)
if not '..' in sys.path:
    sys.path.insert(0,'..')
from opticstools import knull
from matplotlib import rc
plt.ion()
rc('text', usetex=True)
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Palatino']})

mat = knull.other_nuller_mat4(phasor=np.pi/2)

#cov_elts is a list of either phases we want to be linearly independent of, or
#pairs of phases we want to be independent of to 2nd order.
#We can ignore one of the telescope phases here as a "reference" phase. 
cov_elts = [[1,1],[2,2],[3,3],[1,2],[1,3],[2,3]]


#For numerical computation, the delta-phase we use.
dph = 0.01

#The A_mat is an extension of the matrix "A" from the kernel-phase papers, where
#we both include first and second-order phase differences in the pupil plane 
#phase vector. They are arranged according to cov_elts.
#Output - Perfect_ouput = A_mat DOT (dph[0], dph[0]**2, dph[2], dph[1]*dph[3], etc...)
A_mat = np.empty( (len(cov_elts), mat.shape[0]) )

#We approximate in 2D the intensity as:
#f(x,y) = a_x x + a_y y + b_xx x^2 + b_yy*y^2 + b_xy * xy
for ix, elt in enumerate(cov_elts):
    p1 = np.zeros(mat.shape[1], dtype=complex)
    p2 = np.zeros(mat.shape[1])
    if len(elt)==1:
        p1[elt[0]] = dph
        A_mat[ix] = 0.5*(knull.odiff(mat, p1) - knull.odiff(mat, -p1))/dph
    elif elt[0]==elt[1]:
        p1[elt[0]] = dph
        A_mat[ix] = 0.5*(knull.odiff(mat, p1) + knull.odiff(mat, -p1))/dph**2
    else:
        p1[elt[0]] = dph
        p1[elt[1]] = dph
        p2[elt[0]] = dph
        p2[elt[1]] = -dph        
        A_mat[ix] = 0.25*( knull.odiff(mat, p1) + knull.odiff(mat, -p1) \
            - knull.odiff(mat, p2) - knull.odiff(mat, -p2) )/dph**2
        
U, W, V = np.linalg.svd(A_mat.T)
#Kernel-output matrix...
K = np.transpose(U[:,np.sum(W>1e-10):])
K = np.array([[1,-1,0,0,0,0],[0,0,1,-1,0,0],[0,0,0,0,1,-1]])


output_record, kernel_record = knull.response_random_pistons(mat, K)

plt.clf()
plt.plot(K.T)
plt.plot(K.T,'o')
print("Shape of K: " + str(K.shape))

rms_phase = 0.01
flux  = 1e6
bflux = 1e4

output_highphase, kernel_highphase = knull.response_random_pistons(mat, K, rms_piston=10000.)
output_record, kernel_record = knull.response_random_pistons(mat, K, \
    rms_piston=4000./2/np.pi*rms_phase, cwavel=4e-6, flux=flux, bflux=bflux)
knull.response_random_pistons(mat, K, rms_piston=10000.)


print("Kernel Stds (recalc): ")
print(np.std(kernel_record, axis=0))

print("Scaling Factor: ")
stdk = np.std(kernel_record)
print(stdk/rms_phase**3)

print("Flux Scaling Factor: ")
#remove the phase RMS in quadrature
stdk_flux = np.sqrt(stdk**2 - rms_phase**6)
print(stdk_flux * np.sqrt(flux) / rms_phase)

print("Background Scaling Factor: ")
#remove the target flux RMS in quadrature
stdk_bg = np.sqrt(stdk_flux**2 - rms_phase**2/flux)
print(stdk_bg * flux/np.sqrt(bflux))

#Now lets make a plot based on these formulas we've tested.
filter = 'L'
eta_c=0.4
eta_w=0.25
nu_ft = 200.
deltaT = 3600.

lines=[':','-','--']
mags = np.array([3,6,9])
colours=['r','g','b']
nulling = False
if filter=='K':
    diam = 1.8
    label_str = r'$m_{{K}}$={:d}'
    f_tgts = 8.7e10*eta_c*eta_w*10**(-0.4*mags)*(diam/8.0)**2
    f_bgd = 7.8e3*eta_c*(1-eta_w)
    wave_nm = 2200.
    sig_Is=[0,0.05,0.25]
    delay_rms = 10**np.log10(np.linspace(10,100,100))
elif filter == 'L':
    diam = 8.0
    label_str = r'$m_{{L^\prime}}$={:d}'
    f_tgts = 3.5e10*eta_c*eta_w*10**(-0.4*mags)*(diam/8.0)**2
    f_bgd = 5.4e7*eta_c*(1-eta_w)
    sig_Is=[0,0.02,0.1]
    wave_nm = 3750.
    delay_rms = 10**np.log10(np.linspace(30,300,100))
else:
    raise UserWarning("Unknown Filter")

dph = delay_rms/wave_nm*2*np.pi
plt.clf()
for f_tgt, mag, colour in zip(f_tgts,mags,colours):
    for sig_I, line in zip(sig_Is, lines):
        if line=='-':
            label=label_str.format(mag)
        else:
            label=None
        #!!! Where does the 0.5 in front of sig_I come from???
        if nulling:
            null_term = dph**2/f_tgt/deltaT
        else:
            null_term = 1.0/f_tgt/deltaT
        plt.semilogy(delay_rms, 0.8*np.sqrt(dph**6/nu_ft/deltaT + \
            null_term + f_bgd/f_tgt**2/deltaT + 0.5*sig_I**2*dph**2/nu_ft/deltaT), \
            label=label, linestyle=line, color=colour)
    #plt.semilogy(delay_rms, np.sqrt(dph**2/f_tgt/deltaT), '--')
plt.legend()
plt.title(r'Contrast Uncertainty: $\Delta$ T=1hr, D={:5.1f}m'.format(diam))
plt.xlabel('Fringe Tracking RMS (nm)')
plt.ylabel(r'$\sigma_C$')
