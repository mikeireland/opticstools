"""
	- Effective focal lengths? ROCs are 1363.1 and 908.74, with off-axis angles of 7.5 degrees and 17 degrees.
	- DM to OAP2 distance: 2.88 + 3/n_glass + 308.12
	- OAP2 to spherical mirror: 850 + 1300
	- Spherical ROC is 4793mm
	- Distance narcissis mirror to focus is ??

"""
import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
plt.ion()
np.set_printoptions(precision=5)
if not '..' in sys.path:
    sys.path.insert(0,'..')
import opticstools as ot

#The effective focal length of an off-axis section is:
#2 f_p/(1 + cos(theta)) = R_p / (1 + cos(theta))
f1 = 1363.1/(1 + np.cos(np.radians(7.5)))
f2 = 908.74/(1 + np.cos(np.radians(17)))

#Pupil diameter on the spherical mirror
pdiam = 18 * f2/f1

#Input pupil is the DM. To find where the pupil is in the converging beam from the
#spherical mirror, this is all we need to know.
d_dm_oap2 = 2.88 + 3/1.4 + 308.12
d_oap2_sph = 850 + 1300
fsph = 4793/2
f_extra_sph = 200

d_narcissis_implane = 150
d_cstop_implane = 39
f_rat_coldstop = 20

dx = 0.05
sz = 2048
newsz = 512
wave = 2.4e-3 #in mm!

beam_sep_deg = 0.96
#-----
target_frat = 2/np.radians(beam_sep_deg)
#Thin lens formula for virtual position of pupil with respect to OAP2
d_vpup_oap2 = 1/(1/f2 - 1/d_dm_oap2)

#Same, but for the spherical mirror
d_vpup_sph = 1/(1/fsph - 1/(d_oap2_sph - d_vpup_oap2))
print("Pupil Position in front of spherical mirror: {:.1f}".format(d_vpup_sph))

fresnel_f = d_vpup_sph - fsph
fresnel_pin_diam = fresnel_f/fsph * pdiam
fresnel_pup = ot.circle(sz, fresnel_pin_diam/dx, interp_edge=True) - \
    ot.circle(sz, fresnel_pin_diam/dx*0.19, interp_edge=True)

fresnel_pup = fresnel_pup*ot.curved_wf(sz, dx, f_length=fresnel_f, wave=wave)

pup_coldstop = ot.propagate_by_fresnel(fresnel_pup, dx, fresnel_f+ d_cstop_implane, wave)
pup_narcissis = ot.propagate_by_fresnel(fresnel_pup, dx, fresnel_f + d_narcissis_implane , wave)

#Firstly, a cold stop plot
plt.figure(1)
plt.clf()
plt.imshow(np.abs(pup_coldstop[sz//2-20:sz//2+20, sz//2-20:sz//2+20])**2, extent = np.array([-20,20,-20,20])*dx)
plt.xlabel('Dist (mm)')
plt.ylabel('Dist (mm)')
plt.title('Central Beam at Cold Stop')
plt.tight_layout()

#Now lets overplot the notional stop and beams at the cold stop location
r1 = d_cstop_implane/f_rat_coldstop/2
th = np.linspace(0,2*np.pi,100)
plt.plot(r1*np.sin(th), r1*np.cos(th), 'r')
r2 = np.radians(beam_sep_deg)/4*d_cstop_implane
plt.plot(r2*np.sin(th), r2*np.cos(th), 'g')
plt.plot(r2*np.sin(th), r2*np.cos(th) + 4*r2, 'g')
plt.plot(r2*np.sin(th) + 2*np.sqrt(3)*r2, r2*np.cos(th) - 2*r2, 'g')
plt.plot(r2*np.sin(th) - 2*np.sqrt(3)*r2, r2*np.cos(th) - 2*r2, 'g')
plt.savefig('fig1.png')

#Next, the narcissis mirror plot
plt.figure(2)
plt.clf()
plt.imshow(np.abs(pup_narcissis[sz//2-50:sz//2+50, sz//2-50:sz//2+50])**2, extent = np.array([-50,50,-50,50])*dx)
plt.xlabel('Dist (mm)')
plt.ylabel('Dist (mm)')
plt.title('Central Beam at Narcissis Stop')
r3 = np.radians(beam_sep_deg)/4*d_narcissis_implane 
plt.plot(r3*np.sin(th), r3*np.cos(th), 'g')
plt.tight_layout()
plt.savefig('fig2.png')

#OK - what we really need to do is re-image with a second mirror 
mag_extra_sph = target_frat/(fsph/pdiam)
#1/x + 1/(mag * x) = 1/f
#x = f*(1 + 1/mag)
x_post_focus = f_extra_sph*(1 + 1/mag_extra_sph)
extra_sph_to_focus = mag_extra_sph * x_post_focus
print("Extra Sph mirror (or lens) to focus: {:.1f}".format(extra_sph_to_focus))
print("Has to be larger than Narcissis to focus: {:.1f}".format(d_narcissis_implane))

newpup_dist_from_extra_sph = 1/(1/f_extra_sph - 1/(x_post_focus - fresnel_f))
newpup_dist_to_focus = x_post_focus*mag_extra_sph - newpup_dist_from_extra_sph
print("Newpup dist to focus {:.1f}".format(newpup_dist_to_focus))
newpup_diam = newpup_dist_to_focus/target_frat
newpup = ot.circle(newsz, newpup_diam/dx, interp_edge=True) - \
    ot.circle(newsz, newpup_diam/dx*0.19, interp_edge=True)
newpup = newpup * ot.curved_wf(newsz, dx, f_length=newpup_dist_to_focus, wave=wave) 
newpup_coldstop  = ot.propagate_by_fresnel(newpup, dx, newpup_dist_to_focus - d_cstop_implane, wave)
newpup_narcissis = ot.propagate_by_fresnel(newpup, dx, newpup_dist_to_focus - d_narcissis_implane , wave)

#Repeat the coldstop  plot
plt.figure(3)
plt.clf()
plt.imshow(np.abs(newpup_coldstop[newsz//2-20:newsz//2+20, newsz//2-20:newsz//2+20])**2, \
    extent = np.array([-20,20,-20,20])*dx)
plt.xlabel('Dist (mm)')
plt.ylabel('Dist (mm)')
plt.title('Central Beam at Cold Stop (reimaged)')
plt.plot(r1*np.sin(th), r1*np.cos(th), 'r')
plt.plot(r2*np.sin(th), r2*np.cos(th), 'g')
plt.plot(r2*np.sin(th), r2*np.cos(th) + 4*r2, 'g')
plt.plot(r2*np.sin(th) + 2*np.sqrt(3)*r2, r2*np.cos(th) - 2*r2, 'g')
plt.plot(r2*np.sin(th) - 2*np.sqrt(3)*r2, r2*np.cos(th) - 2*r2, 'g')
plt.tight_layout()
plt.savefig('fig3.png')

#Repeat the narcissis mirror plot
plt.figure(4)
plt.clf()
plt.imshow(np.abs(newpup_narcissis[newsz//2-50:newsz//2+50, newsz//2-50:newsz//2+50])**2, \
    extent = np.array([-50,50,-50,50])*dx)
plt.xlabel('Dist (mm)')
plt.ylabel('Dist (mm)')
plt.title('Central Beam at Narcissis Stop (reimaged)')
plt.plot(r3*np.sin(th), r3*np.cos(th), 'g')
plt.tight_layout()
plt.savefig('fig4.png')

ftot = np.sum(np.abs(newpup_narcissis)**2)
stop_12 = ot.circle(newsz,d_narcissis_implane/newpup_dist_to_focus*newpup_diam/dx*1.2, interp_edge=True)
print("Fractional Flux in 1.2 times stop: {:.2f}".format(\
    np.sum(np.abs(newpup_narcissis*stop_12)**2)/ftot))
