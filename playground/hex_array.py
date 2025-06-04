"""
conclusions: a loop or even right angle of fiber requires ~5cm, which is enough for
substantial long-wavelength losses:
https://irflex.com/products/irf-se-series/
AsSe fibers are mature, but can not be used beyond 10 microns at the moment.

On the other hand, several optical materials are highly transparent to 20 microns. 
Unfortunately, these tend to be crystalline, and not glassy.

At long wavelengths, this structure behaves as if it has an average refractive index of
1.35 for the cladding, i.e. a numerical aperture of ~0.65. Probably more like 0.5

this would give:
V = 1.46, U=1.295, W=0.674
Delta = 0.065
... which givs a bend loss of 2.7% for a 90 degree bend with radius of 1mm.
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
import scipy.ndimage as nd
plt.ion()
if not '..' in sys.path:
    sys.path.insert(0,'..')
import opticstools as ot
from matplotlib.colors import LogNorm

#Add this!
from philsol.classy import phil_class

sz = 138 #Apparently has to be even!
microns_pix = 0.8
diam = 8
cell_edge = 16

x_pcf = (np.arange(sz)-sz//2) * microns_pix
xy = np.meshgrid(x_pcf,x_pcf)
rr2 = xy[0]**2 + xy[1]**2
rr = np.sqrt(rr2)

hole = ot.circle(sz,diam/microns_pix, interp_edge=True)

holes = np.zeros_like(hole)
x0 = -1.5*cell_edge
y0 = -np.sqrt(3)/2*cell_edge * 3
for dx in np.arange(4)*cell_edge:
    holes += nd.shift(hole,((x0 + dx)/microns_pix, y0/microns_pix), order=1)
    
x0 = -2*cell_edge
y0 = -np.sqrt(3)/2*cell_edge * 2
for dx in np.arange(5)*cell_edge:
    holes += nd.shift(hole,((x0 + dx)/microns_pix, y0/microns_pix), order=1)
    
x0 = -2.5*cell_edge
y0 = -np.sqrt(3)/2*cell_edge * 1
for dx in np.arange(6)*cell_edge:
    holes += nd.shift(hole,((x0 + dx)/microns_pix, y0/microns_pix), order=1)

x0 = -3*cell_edge
y0 = 0
for dx in np.arange(3)*cell_edge:
    holes += nd.shift(hole,((x0 + dx)/microns_pix, y0/microns_pix), order=1)

x0 = 1*cell_edge
y0 = 0
for dx in np.arange(3)*cell_edge:
    holes += nd.shift(hole,((x0 + dx)/microns_pix, y0/microns_pix), order=1)

x0 = -2.5*cell_edge
y0 = np.sqrt(3)/2*cell_edge * 1
for dx in np.arange(6)*cell_edge:
    holes += nd.shift(hole,((x0 + dx)/microns_pix, y0/microns_pix), order=1)

x0 = -2*cell_edge
y0 = np.sqrt(3)/2*cell_edge * 2
for dx in np.arange(5)*cell_edge:
    holes += nd.shift(hole,((x0 + dx)/microns_pix, y0/microns_pix), order=1)

x0 = -1.5*cell_edge
y0 = np.sqrt(3)/2*cell_edge * 3
for dx in np.arange(4)*cell_edge:
    holes += nd.shift(hole,((x0 + dx)/microns_pix, y0/microns_pix), order=1)

    
plt.clf()
plt.imshow(holes, extent=[-sz/2*microns_pix,sz/2*microns_pix,-sz/2*microns_pix,sz/2*microns_pix])
plt.xlabel('Microns')
plt.ylabel('Microns')

#Let's try the scalar approx
#if True:
#	n_wg = 1.5-0.5*holes

if True:
	#---
	wavelength_in_mm = np.linspace(0.008,0.0185,3)
	#---
	
	n_wg = 1.5-0.5*holes
	#n_wg[rr>3.4*cell_edge]=1
	n_wg = np.repeat(n_wg[:,:,np.newaxis],3, axis = 2) 
	
	f_rat = 1.3
	apod_scale = 1.25
	psize = sz/(wavelength_in_mm*f_rat/(microns_pix/1000))
	asize = sz/(wavelength_in_mm[-1]/(microns_pix/1000)) * (wavelength_in_mm/wavelength_in_mm[-1])**(-0.4)
	
	couplings_pcf = []
	acouplings_pcf = []
	couplings_pcf_scalar = []
	acouplings_pcf_scalar = []
	pups_pcf = []
	Is = []
	Es = []
	
	oldap = ot.circle(sz,sz/asize[-1]*2.7)
	
	for ix,wave in enumerate(wavelength_in_mm):
		print("Run: {:d}".format(ix))
		k = 2*np.pi/(wave*1000)
		nmodes = 100
		neff_guess = 1.5
		beta_guess = k*neff_guess
  
  		# A scalar approximation for the modes.
		delta = microns_pix*2*np.pi/(wave*1000)
		print(f'delta: {delta:.2f}')
		P = ot.scalar_eigen_build(n_wg[:,:,0], delta)
		neff, E = ot.solve_scalar_eigen(P, neff_guess, neigs=nmodes)
		E = E.reshape((sz, sz, nmodes))
		Es += [E[:,:,4]]
  
		# A vectorised computation of the modes.
		pcf = phil_class(n_wg, k, dx = microns_pix, dy=microns_pix)  
		
		pcf.build_stuff()
		pcf.solve_stuff(nmodes, beta_guess)
		shape = np.insert(np.flip(np.shape(pcf.n)[:2]),0,nmodes)
		Ex = pcf.Ex.reshape(shape)
		Ey = pcf.Ey.reshape(shape)
		ixm = np.argsort(-np.abs(Ex[:,sz//2,sz//2])**2 - np.abs(Ey[:,sz//2,sz//2])**2)
		Is += [np.abs(Ex[ixm[0]])**2 + np.abs(Ex[ixm[0]])**2 + np.abs(Ey[ixm[0]])**2 + np.abs(Ey[ixm[0]])**2]
  
		#Now compute the electric field for the overlap
		pup = ot.circle(sz,psize[ix], interp_edge=True)
		pup_apod = ot.circle(sz,asize[ix], interp_edge=True)*np.exp(-rr2/(asize[ix]/2)**2)
		im_e = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pup)))
		im_e_apod = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pup_apod)))
		
		#!!! There was a "coldap*" on theline below, for im_e_apod
		couplings_pcf += [np.abs(np.sum(im_e*Ex[ixm[0]].conj()))**2/np.sum(np.abs(im_e)**2)/np.sum(np.abs(Ex[ixm[0]])**2)]
		acouplings_pcf += [np.abs(np.sum(im_e_apod*Ex[ixm[0]].conj()))**2/np.sum(np.abs(im_e_apod)**2)/np.sum(np.abs(Ex[ixm[0]])**2)]
		print("Wave: {:.4f} Coupling: {:.3f} Apod: {:.3f}".format(wave, couplings_pcf[-1], acouplings_pcf[-1]))
		couplings_pcf_scalar += [np.abs(np.sum(im_e*E[:,:,4].conj()))**2/np.sum(np.abs(im_e)**2)/np.sum(np.abs(E[:,:,4])**2)]
		acouplings_pcf_scalar += [np.abs(np.sum(im_e_apod*E[:,:,4].conj()))**2/np.sum(np.abs(im_e_apod)**2)/np.sum(np.abs(E[:,:,4])**2)]
		print("Wave: {:.4f} Coupling (scalar): {:.3f} Apod (scalar): {:.3f}".format(wave, couplings_pcf_scalar[-1], acouplings_pcf_scalar[-1]))
		
	fig, axs = plt.subplots(1, 5, width_ratios=[1,1,1,1,.2])
	axs[0].imshow(holes, extent=[-sz/2*microns_pix,sz/2*microns_pix,-sz/2*microns_pix,sz/2*microns_pix])
	axs[0].set_ylabel(r'Offset ($\mu$m)')
	axs[0].set_xlabel(r'Offset ($\mu$m)')
	Is = np.array(Is)
	for i in range(3):
		Is[i] /= np.max(Is[i])
		im=axs[i+1].imshow(Is[i], norm=LogNorm(.001,1), extent=[-sz/2*microns_pix,sz/2*microns_pix,-sz/2*microns_pix,sz/2*microns_pix])
		axs[i+1].set_xlabel(r'Offset ($\mu$m)')
		
	#ax_cb = fig.add_axes([.9,.1,.05,.8])
	#ax_cb.axis('off')
	axs[-1].axis('off')
	plt.colorbar(im, ax=axs[-1])
	fig.tight_layout()
	
	azav = ot.azimuthalAverage(np.abs(Ex[0])**2 + np.abs(Ex[1])**2, returnradii=True, binsize=1)

#------ simple apodized below here -------
if False:
	R_bs = [1,2,5,10,20,50,100,200,500]
	R_bs = [1,2,5,10,20,50,100]
	
	core_radius = 0.015
	na = 0.2
	f_rat = 2.9
	mm_pix = 0.002
	
	core_radius = 0.0064
	na = 0.47
	f_rat = 1.2
	mm_pix = 0.0008
	
	n_co = 2.4
	n_cl = np.sqrt(n_co**2-na**2)
	Delta = 0.5*(n_co**2-n_cl**2)/n_co**2
	
	
	sz = 512
	psize = sz/(wavelength_in_mm*f_rat/mm_pix)
	couplings = []
	acouplings = []
	acouplings_chrom = []
	xy = np.meshgrid(np.arange(sz)-sz//2, np.arange(sz)-sz//2)
	rr2 = xy[1]**2 + xy[0]**2
	asize_chrom = sz/(wavelength_in_mm[0]*(f_rat/apod_scale)/mm_pix) * (wavelength_in_mm/wavelength_in_mm[0])**(-1.7)

	#Now go through 1 wavelength at a time
	for ix, wave in enumerate(wavelength_in_mm):
		pup = ot.circle(sz,psize[ix], interp_edge=True)
		pup_apod = ot.circle(sz,psize[ix]*apod_scale, interp_edge=True)*np.exp(-rr2/(psize[ix]*apod_scale/2)**2)
		pup_apod_chrom = ot.circle(sz,asize_chrom[ix], interp_edge=True)*np.exp(-rr2/(asize_chrom[ix]/2)**2)
		im_e = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pup)))
		im_e_apod = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pup_apod)))
		im_e_apod_chrom =  np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pup_apod_chrom)))
		V = ot.compute_v_number(wave, core_radius, na)
		U, n_per_j = ot.neff(V)
		U = U[0]
		n_eff = np.sqrt(n_co**2 - U**2 * wave**2/core_radius**2/4/np.pi**2 )
		W = np.sqrt(V**2-U**2)
		print('n_eff: {:.3f}'.format(n_eff))
		#From Selena's stack exchange post:
		#https://physics.stackexchange.com/questions/286840/how-can-i-estimate-optical-bending-loss
		for R_b in R_bs:
			#bend_loss = np.sqrt(np.pi/2*core_radius/R_b) * (n_co**2-n_cl**2)/(n_eff**2-n_cl**2)*(2*np.pi/wave)*np.sqrt(n_co**2-n_eff**2)\
			#	*np.exp(-8/3*2*np.pi/wave*n_co**2*np.sqrt(n_co**2-n_cl**2)*R_b)
			bend_loss = np.sqrt(np.pi/4*core_radius/R_b)*V**2*np.sqrt(W)/U**2*np.exp(-4/3*R_b/core_radius*W**3*Delta/V**2)
			print("bend loss {:.3f} for radius {:.2f}. total: {:.2f}".format(bend_loss, R_b, 2*np.pi*R_b*bend_loss))
		mode = ot.mode_2d(V, core_radius, sampling=mm_pix, sz=sz)
		if ix==0:
			mode0 = mode.copy()
		couplings += [np.abs(np.sum(im_e*mode.conj()))**2/np.sum(np.abs(im_e)**2)/np.sum(np.abs(mode)**2)]
		acouplings += [np.abs(np.sum(im_e_apod*mode.conj()))**2/np.sum(np.abs(im_e_apod)**2)/np.sum(np.abs(mode)**2)]
		acouplings_chrom += [np.abs(np.sum(im_e_apod_chrom*mode.conj()))**2/np.sum(np.abs(im_e_apod_chrom)**2)/np.sum(np.abs(mode)**2)]
		print("Wave: {:.4f} V: {:.2f} Coupling: {:.3f} Apod: {:.3f} Chrom: {:.3f}".format(\
			wave, V, couplings[-1], acouplings[-1], acouplings_chrom[-1]))

	plt.figure(1)
	plt.clf()
	plt.plot(wavelength_in_mm*1e3, couplings, label='Airy Disk')
	plt.plot(wavelength_in_mm*1e3, acouplings, label=r'1/e$^2$ PIAA')
	plt.plot(wavelength_in_mm*1e3, acouplings_chrom, label=r'$\lambda$-dep 1/e$^2$ PIAA')
	plt.plot(wavelength_in_mm*1e3, acouplings_pcf, label=r'$\lambda$-dep 1/e$^2$ PCF')
	plt.xlabel(r'Wavelength ($\mu$m)')
	plt.ylabel('Coupling')
	plt.tight_layout()
	plt.legend()
	plt.show()

	x = (np.arange(sz)-sz//2)*mm_pix*1e3
	plt.figure(2)
	plt.clf()
	plt.plot(x, mode[256]/np.max(mode[256]))
	plt.plot([-1e3*core_radius,-1e3*core_radius],[0,1],'r')
	plt.plot([1e3*core_radius,1e3*core_radius],[0,1],'r')
	plt.xlabel(r'Offset ($\mu$m)')
	plt.ylabel('Field Amplitude')
	plt.tight_layout()
	
	
