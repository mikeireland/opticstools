"""A selection of useful functions for optics, especially Fourier optics. The
documentation is designed to be used with sphinx (still lots to do)

Authors:
Dr Michael Ireland
Adam Rains
"""

from __future__ import print_function, division
import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy import optimize
from .utils import *
try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(1.0)
    nthreads=6 
except:
    nthreads=0

def azimuthalAverage(image, center=None, stddev=False, returnradii=False, return_nr=False, 
        binsize=0.5, weights=None, steps=False, interpnan=False, left=None, right=None, return_max=False):
    """
    Calculate the azimuthally averaged radial profile.
    NB: This was found online and should be properly credited! Modified by MJI

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fractional pixels).
    stddev - if specified, return the azimuthal standard deviation instead of the average
    returnradii - if specified, return (radii_array,radial_profile)
    return_nr   - if specified, return number of pixels per radius *and* radius
    binsize - size of the averaging bin.  Can lead to strange results if
        non-binsize factors are used to specify the center and the binsize is
        too large
    weights - can do a weighted average instead of a simple average if this keyword parameter
        is set.  weights.shape must = image.shape.  weighted stddev is undefined, so don't
        set weights and stddev.
    steps - if specified, will return a double-length bin array and radial
        profile so you can plot a step-form radial profile (which more accurately
        represents what's going on)
    interpnan - Interpolate over NAN values, i.e. bins where there is no data?
        left,right - passed to interpnan; they set the extrapolated values
    return_max - (MJI) Return the maximum index.

    If a bin contains NO DATA, it will have a NAN value because of the
    divide-by-sum-of-weights component.  I think this is a useful way to denote
    lack of data, but users let me know if an alternative is prefered...
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    if weights is None:
        weights = np.ones(image.shape)
    elif stddev:
        raise ValueError("Weighted standard deviation is not defined.")

    # the 'bins' as initially defined are lower/upper bounds for each bin
    # so that values will be in [lower,upper)  
    nbins = int(np.round(r.max() / binsize)+1)
    maxbin = nbins * binsize
    bins = np.linspace(0,maxbin,nbins+1)
    # but we're probably more interested in the bin centers than their left or right sides...
    bin_centers = (bins[1:]+bins[:-1])/2.0

    # Find out which radial bin each point in the map belongs to
    whichbin = np.digitize(r.flat,bins)

    # how many per bin (i.e., histogram)?
    # there are never any in bin 0, because the lowest index returned by digitize is 1
    nr = np.bincount(whichbin)[1:]

    # recall that bins are from 1 to nbins (which is expressed in array terms by arange(nbins)+1 or xrange(1,nbins+1) )
    # radial_prof.shape = bin_centers.shape

    if stddev:
        radial_prof = np.array([image.flat[whichbin==b].std() for b in xrange(1,nbins+1)])
    elif return_max:
        radial_prof = np.array([np.append((image*weights).flat[whichbin==b],-np.inf).max() for b in xrange(1,nbins+1)])
    else:
        radial_prof = np.array([(image*weights).flat[whichbin==b].sum() / weights.flat[whichbin==b].sum() for b in xrange(1,nbins+1)])

    #import pdb; pdb.set_trace()

    if interpnan:
        radial_prof = np.interp(bin_centers,bin_centers[radial_prof==radial_prof],radial_prof[radial_prof==radial_prof],left=left,right=right)

    if steps:
        xarr = np.array(zip(bins[:-1],bins[1:])).ravel() 
        yarr = np.array(zip(radial_prof,radial_prof)).ravel() 
        return xarr,yarr
    elif returnradii: 
        return bin_centers,radial_prof
    elif return_nr:
        return nr,bin_centers,radial_prof
    else:
        return radial_prof

def propagate_by_fresnel(wf, m_per_pix, d, wave):
    """Propagate a wave by Fresnel diffraction
    
    Parameters
    ----------
    wf: float array
        Wavefront, i.e. a complex electric field in the scalar approximation.
    m_per_pix: float
        Scale of the pixels in the input wavefront in metres.
    d: float
        Distance to propagate the wavefront.
    wave: float
        Wavelength in metres.
        
    Returns
    -------
    wf_new: float array
        Wavefront after propagating.
    """
    #Notation on Mike's board
    sz = wf.shape[0]
    if (wf.shape[0] != wf.shape[1]):
        print("ERROR: Input wavefront must be square")
        raise UserWarning
    
    #The code below came from the board, i.e. via Huygen's principle.
    #We got all mixed up when converting to Fourier transform co-ordinates.
    #Co-ordinate axis of the wavefront. Not that 0 must be in the corner.
    #x = (((np.arange(sz)+sz/2) % sz) - sz/2)*m_per_pix
    #xy = np.meshgrid(x,x)
    #rr =np.sqrt(xy[0]**2 + xy[1]**2)
    #h_func = np.exp(1j*np.pi*rr**2/wave/d)
    #h_ft = np.fft.fft2(h_func)
    
    #Co-ordinate axis of the wavefront Fourier transform. Not that 0 must be in the corner.
    #x is in cycles per wavefront dimension.
    x = (((np.arange(sz)+sz/2) % sz) - sz/2)/m_per_pix/sz
    xy = np.meshgrid(x,x)
    uu =np.sqrt(xy[0]**2 + xy[1]**2)
    h_ft = np.exp(1j*np.pi*uu**2*wave*d)
    
    g_ft = np.fft.fft2(np.fft.fftshift(wf))*h_ft
    wf_new = np.fft.ifft2(g_ft)
    return np.fft.fftshift(wf_new)

def airy(x, obstruction_sz=0):
    """Return an Airy disk as an electric field as a function of angle in units of 
    lambda/D, with the possibility of a circular obstruction that is a fraction of the
    pupil size.
    
    The total intensity is proportional to the area, so the peak intensity is 
    proportional to the square of the area, and the peak electric field proportional
    to the area."""
    ix = np.where(x>0)[0]
    y1 = np.ones(len(x))
    y1[ix] = 2*special.jn(1,np.pi*x[ix])/(np.pi*x[ix])
    if obstruction_sz>0:
        y2 = np.ones(len(x))
        y2[ix] = 2*special.jn(1,np.pi*x[ix]*obstruction_sz)/(np.pi*x[ix]*obstruction_sz)
        y1 -= obstruction_sz**2 * y2
        y1 /= (1 - obstruction_sz**2)
    
    return y1

def curved_wf(sz,m_per_pix,f_length=np.infty,wave=633e-9, tilt=[0.0,0.0], power=None,diam=None,defocus=None):
    """A curved wavefront centered on the *middle*
    of the python array.
    
    Parameters
    ----------
    sz: int
        Size of the wavefront in pixels
    m_per_pix: float
        Meters per pixel
    tilt: float (optional)
        Tilt of the wavefront in radians in the x and y directions. 
    wave: float
        Wavelength in m       
    """
    x = np.arange(sz) - sz//2
    xy = np.meshgrid(x,x)
    rr =np.sqrt(xy[0]**2 + xy[1]**2)
    if not power:
        power = 1.0/f_length
    if not diam:
        diam=sz*m_per_pix
    #The following line computes phase in *wavelengths*
    if defocus:
        phase = defocus*(rr*m_per_pix/diam*2)**2
    else:
        phase = 0.5*m_per_pix**2/wave*power*rr**2 
    phase += tilt[0]*xy[0]*diam/sz/wave
    phase += tilt[1]*xy[1]*diam/sz/wave

    return np.exp(2j*np.pi*phase)

def fourier_wf(sz,xcyc_aperture,ycyc_aperture,amp,phase):
    """This function creates a phase aberration, centered on the
    middle of a python array
    
    Parameters
    ----------
    dim: int
        Size of the 2D array
    xcyc_aperture: float
        cycles per aperture in the x direction.
    ycyc_aperture: float
        cycles per aperture in the y direction.
    amp: float
        amplitude of the aberration in radians
    phase: float
        phase of the aberration in radians.
        
    Returns
    -------
    pupil: float array (sz,sz)
        2D array circular pupil mask
    """
    x = np.arange(sz) - sz//2
    xy = np.meshgrid(x,x)
    xx = xy[0]
    yy = xy[1]
    zz = 2*np.pi*(xx*xcyc_aperture/sz + yy*ycyc_aperture/sz)
    aberration = np.exp( 1j * amp * (np.cos(phase)*np.cos(zz) + np.sin(phase)*np.sin(zz)))
    return aberration


def gmt(dim,widths=None,pistons=[0,0,0,0,0,0],m_pix=None):
    """This function creates a GMT pupil.
    http://www.gmto.org/Resources/GMT-ID-01467-Chapter_6_Optics.pdf
    
    Parameters
    ----------
    dim: int
        Size of the 2D array
    width: int
        diameter of the primary mirror (scaled to 25.448m)
        
    Returns
    -------
    pupil: float array (sz,sz)
        2D array circular pupil mask
    """
    #The geometry is complex... with eliptical segments due to their tilt.
    #We'll just approximate by segments of approximately the right size.
    pupils=[]
    if m_pix:
        widths = 25.448/m_pix
    elif not widths:
        print("ERROR: Must set widths or m_pix")
        raise UserWarning
    try:
        awidth = widths[0]
    except:
        widths = [widths]
    for width in widths:
        segment_dim = width*8.27/25.448
        segment_sep = width*(8.27 + 0.3)/25.448
        obstruct = width*3.2/25.448
        rollx = int(np.round(np.sqrt(3)/2.0*segment_sep))
        one_seg = circle(dim, segment_dim)
        pupil = one_seg - circle(dim, obstruct) + 0j
        pupil += np.exp(1j*pistons[0])*np.roll(np.roll(one_seg,  int(np.round(0.5*segment_sep)), axis=0),rollx, axis=1)
        pupil += np.exp(1j*pistons[1])*np.roll(np.roll(one_seg, -int(np.round(0.5*segment_sep)), axis=0),rollx, axis=1)
        pupil += np.exp(1j*pistons[4])*np.roll(np.roll(one_seg,  int(np.round(0.5*segment_sep)), axis=0),-rollx, axis=1)
        pupil += np.exp(1j*pistons[3])*np.roll(np.roll(one_seg, -int(np.round(0.5*segment_sep)), axis=0),-rollx, axis=1)
        pupil += np.exp(1j*pistons[5])*np.roll(one_seg, int(segment_sep), axis=0)
        pupil += np.exp(1j*pistons[2])*np.roll(one_seg, -int(segment_sep), axis=0)
        pupils.append(pupil)
    return np.array(pupils)

#--- Start masks ---

def mask2s(dim):
    """ Returns 4 pupil mask that split the pupil into halves.
    """
    masks = np.zeros( (4,dim,dim) )
    masks[0,0:dim/2,:]=1
    masks[1,dim/2:,:]=1
    masks[2,:,0:dim/2]=1
    masks[3,:,dim/2:]=1
    return masks

def mask6s(dim):
    """ Returns 4 pupil mask that split the pupil into halves, with a
    six-way symmetry
    """
    masks = np.zeros( (4,dim,dim) )
    x = np.arange(dim) - dim//2
    xy = np.meshgrid(x,x)
    theta = np.arctan2(xy[0],xy[1])
    twelfths = ( (theta + np.pi)/2/np.pi*12).astype(int)
    masks[0,:,:]=(twelfths//2) % 2
    masks[1,:,:]=(twelfths//2 + 1) % 2
    masks[2,:,:]=((twelfths+1)//2) % 2
    masks[3,:,:]=((twelfths+1)//2 + 1) % 2
    
    return masks

def angel_mask(sz,m_per_pix,diam=25.5):
    """Create a mask like Roger Angel et al's original GMT tilt and piston sensor.
    """
    diam_in_pix = diam/m_per_pix
    inner_circ = circle(sz,int(round(diam_in_pix/3)))
    outer_an = circle(sz,int(round(diam_in_pix))) - inner_circ
    mask6s = mask6s(sz)
    masks = np.array([inner_circ + outer_an*mask6s[2,:,:],inner_circ + outer_an*mask6s[3,:,:],outer_an])
    return masks

def angel_mask_mod(sz,wave,diam=25.5):
    """Create a mask like Roger Angel et al's original GMT tilt and piston sensor, except 
    we 50/50 split the inner segment.
    """
    diam_in_pix = diam/m_per_pix
    inner_circ = circle(sz,int(round(diam_in_pix/3)))
    outer_an = circle(sz,int(round(diam_in_pix))) - inner_circ
    mask6s = mask6s(sz)
    masks = np.array([0.5*inner_circ + outer_an*mask6s[2,:,:],0.5*inner_circ + outer_an*mask6s[3,:,:]])
    return masks

def diversity_mask(sz,m_per_pix,defocus=2.0):
    """Create a traditional phase diversity mask.
    """
    wf1 = curved_wf(sz,m_per_pix,defocus=defocus)
    wf2 = curved_wf(sz,m_per_pix,defocus=-defocus)
    masks = np.array([wf1,wf2])
    return masks
    
#--- End Masks ---

def kmf(sz):
    """This function creates a periodic wavefront produced by Kolmogorov turbulence. 
    It SHOULD normalised so that the variance at a distance of 1 pixel is 1 radian^2.
    To scale this to an r_0 of r_0_pix, multiply by sqrt(6.88*r_0_pix**(-5/3))
    
    Parameters
    ----------
    sz: int
        Size of the 2D array
    
    Returns
    -------
    wavefront: float array (sz,sz)
        2D array wavefront.
    """
    xy = np.meshgrid(np.arange(sz/2 + 1)/float(sz), (((np.arange(sz) + sz/2) % sz)-sz/2)/float(sz))
    dist2 = np.maximum( xy[1]**2 + xy[0]**2, 1e-12)
    ft_wf = np.exp(2j * np.pi * np.random.random((sz,sz/2+1)))*dist2**(-11.0/12.0)*sz/15.81
    ft_wf[0,0]=0
    return np.fft.irfft2(ft_wf)
    
def test_kmf(sz,ntests):
    vars = np.zeros(ntests)
    for i in range(ntests):
        wf = kmf(sz)
        vars[i] = 0.5* ( np.mean((wf[1:,:] - wf[:-1,:])**2) + \
                      np.mean((wf[:,1:] - wf[:,:-1])**2) )
    print("Mean var: {0:7.3e} Sdev var: {1:7.3e}".format(np.mean(vars),np.std(vars)))
    
def moffat(theta, hw, beta=4.0):
    """This creates a moffatt function for simulating seeing.
    The output is an array with the same dimensions as theta.
    Total Flux" is set to 1 - this only applies if sampling
    of thetat is 1 per unit area (e.g. arange(100)).
    
    From Racine (1996), beta=4 is a good approximation for seeing
    
    Parameters
    ----------
    theta: float or float array
        Angle at which to calculate the moffat profile (same units as hw)
    hw: float
        Half-width of the profile
    beta: float
        beta parameters
    
    """
    denom = (1 + (2**(1.0/beta) - 1)*(theta/hw)**2)**beta
    return (2.0**(1.0/beta)-1)*(beta-1)/np.pi/hw**2/denom
    
def moffat2d(sz,hw, beta=4.0):
    """A 2D version of a moffat function
    """
    x = np.arange(sz) - sz/2.0
    xy = np.meshgrid(x,x)
    r = np.sqrt(xy[0]**2 + xy[1]**2)
    return moffat(r, hw, beta=beta)
       
def snell(u, f, n_i, n_f):
    """Snell's law at an interface between two dielectrics
    
    Parameters
    ----------
    u: float array(3)
        Input unit vector
    f: float array(3)
        surface normal  unit vector
    n_i: float
        initial refractive index
    n_f: float
        final refractive index.
    """
    u_p = u - np.sum(u*f)*f
    u_p /= np.sqrt(np.sum(u_p**2))
    theta_i = np.arccos(np.sum(u*f))
    theta_f = np.arcsin(n_i*np.sin(theta_i)/n_f)
    v = u_p*np.sin(theta_f) + f*np.cos(theta_f)
    return v

def grating_sim(u, l, s, ml_d, refract=False):
    """This function computes an output unit vector based on an input unit
    vector and grating properties.

    Math: v \cdot l = u \cdot l (reflection)
          v \cdot s = u \cdot s + ml_d
    The blaze wavelength is when m \lambda = 2 d sin(theta)
     i.e. ml_d = 2 sin(theta)

    x : to the right
    y : out of page
    z : down the page
    
    Parameters
    ----------
    u: float array(3)
        initial unit vector
    l: float array(3)
        unit vector along grating lines
    s: float array(3)
        unit vector along grating surface, perpendicular to lines
    ml_d: float
        order * \lambda/d
    refract: bool
        Is the grating a refractive grating? 
    """
    if (np.abs(np.sum(l*s)) > 1e-3):    
        print('Error: input l and s must be orthogonal!')
        raise UserWarning
    n = np.cross(s,l)
    if refract:
        n *= -1
    v_l = np.sum(u*l)
    v_s = np.sum(u*s) + ml_d
    v_n = np.sqrt(1-v_l**2 - v_s**2)
    v = v_l*l + v_s*s + v_n*n
    
    return v
      
def nglass(l, glass='sio2'):
    """Refractive index of fused silica and other glasses. Note that C is
    in microns^{-2}
    
    Parameters
    ----------
    l: wavelength 
    """
    try:
        nl = len(l)
    except:
        l = [l]
        nl=1
    l = np.array(l)
    if (glass == 'sio2'):
        B = np.array([0.696166300, 0.407942600, 0.897479400])
        C = np.array([4.67914826e-3,1.35120631e-2,97.9340025])
    elif (glass == 'bk7'):
        B = np.array([1.03961212,0.231792344,1.01046945])
        C = np.array([6.00069867e-3,2.00179144e-2,1.03560653e2])
    elif (glass == 'nf2'):
        B = np.array( [1.39757037,1.59201403e-1,1.26865430])
        C = np.array( [9.95906143e-3,5.46931752e-2,1.19248346e2])
    else:
        print("ERROR: Unknown glass {0:s}".format(glass))
        raise UserWarning
    n = np.ones(nl)
    for i in range(len(B)):
            n += B[i]*l**2/(l**2 - C[i])
    return np.sqrt(n)

def join_bessel(U,V,j):
    """In order to solve the Laplace equation in cylindrical co-ordinates, both the
    electric field and its derivative must be continuous at the edge of the fiber...
    i.e. the Bessel J and Bessel K have to be joined together. 
    
    The solution of this equation is the n_eff value that satisfies this continuity
    relationship"""
    W = np.sqrt(V**2 - U**2)
    return U*special.jn(j+1,U)*special.kn(j,W) - W*special.kn(j+1,W)*special.jn(j,U)
    
def neff(V, accurate_roots=True):
 """Find the effective indices of all modes for a given value of 
 the fiber V number. """
 delu = 0.04
 U = np.arange(delu/2,V,delu)
 W = np.sqrt(V**2 - U**2)
 all_roots=np.array([])
 n_per_j=np.array([],dtype=int)
 n_modes=0
 for j in range(int(V+1)):
   f = U*special.jn(j+1,U)*special.kn(j,W) - W*special.kn(j+1,W)*special.jn(j,U)
   crossings = np.where(f[0:-1]*f[1:] < 0)[0]
   roots = U[crossings] - f[crossings]*( U[crossings+1] - U[crossings] )/( f[crossings+1] - f[crossings] )
   if accurate_roots:
     for i,root in enumerate(roots):
         roots[i] = optimize.newton(join_bessel, root, args=(V,j))
   #import pdb; pdb.set_trace()
   if (j == 0): 
     n_modes = n_modes + len(roots)
     n_per_j = np.append(n_per_j, len(roots))
   else:
     n_modes = n_modes + 2*len(roots)
     n_per_j = np.append(n_per_j, len(roots)) #could be 2*length(roots) to account for sin and cos.
   all_roots = np.append(all_roots,roots)
 return all_roots, n_per_j
 
def mode_2d(V, r, j=0, n=0, sampling=0.3,  sz=1024):
    """Create a 2D mode profile. 
    
    Parameters
    ----------
    V: Fiber V number
    
    r: core radius in microns
    
    sampling: microns per pixel
    
    n: radial order of the mode (0 is fundumental)
    
    j: azimuthal order of the mode (0 is pure radial modes)
    TODO: Nonradial modes."""
    #First, find the neff values...
    u_all,n_per_j = neff(V)
    ix = np.sum(n_per_j[0:j]) + n
    U0 = u_all[ix]
    W0 = np.sqrt(V**2 - U0**2)
    x = (np.arange(sz)-sz/2)*sampling/r
    xy = np.meshgrid(x,x)
    r = np.sqrt(xy[0]**2 + xy[1]**2)
    win = np.where(r < 1)
    wout = np.where(r >= 1)
    the_mode = np.zeros( (sz,sz) )
    the_mode[win] = special.jn(j,r[win]*U0)
    scale = special.jn(j,U0)/special.kn(j,W0)
    the_mode[wout] = scale * special.kn(j,r[wout]*W0)
    return the_mode/np.sqrt(np.sum(the_mode**2))

def compute_v_number(wavelength_in_mm, core_radius, numerical_aperture):
    """Computes the V number (can be interpreted as a kind of normalized optical frequency) for an optical fibre
    
    Parameters
    ----------
    wavelength_in_mm: float
        The wavelength of light in mm
    core_radius: float
        The core radius of the fibre in mm
    numerical_aperture: float
        The numerical aperture of the optical fibre, defined be refractive indices of the core and cladding
        
    Returns
    -------
    v: float
        The v number of the fibre
        
    """
    v = 2 * np.pi / wavelength_in_mm * core_radius * numerical_aperture
    return v
    
def shift_and_ft(im):
    """Sub-pixel shift an image to the origin and Fourier-transform it

    Parameters
    ----------
    im: (ny,nx) float array
    ftpix: optional ( (nphi) array, (nphi) array) of Fourier sampling points. 
    If included, the mean square Fourier phase will be minimised.

    Returns
    ----------
    ftim: (ny,nx/2+1)  complex array
    """
    ny = im.shape[0]
    nx = im.shape[1]
    im = regrid_fft(im,(3*ny,3*nx))
    shifts = np.unravel_index(im.argmax(), im.shape)
    im = np.roll(np.roll(im,-shifts[0]+1,axis=0),-shifts[1]+1,axis=1)
    im = rebin(im,(ny,nx))
    ftim = np.fft.rfft2(im)
    return ftim

def rebin(a, shape):
    """Re-bins an image to a new (smaller) image with summing	

    Originally from:
    http://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array

    Parameters
    ----------
    a: array
        Input image
    shape: (xshape,yshape)
        New shape
    """
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).sum(-1).sum(1)

def correct_tip_tilt(turbulent_wf, pupil, size):
    """Given a turbulent wavefront, calculate the tip/tilt (horizontal and vertical slope)
    
    TODO: Only compute turbulence over square immediately surrounding the pupil to save on unnecessary computation
    
    Parameters
    ----------
    turbulent_wf: np.array([[...]...])
        2D square of numbers representing a turbulent patch of atmosphere
    pupil: np.array([[...]...])
        The pupil of the telescope receiving the light pasing through the turbulence.
    size: int
        Size of input_wf per side, preferentially a power of two (npix=2**n)
        
    Return
    ------
    corrected_wf: np.array([[...]...])
        Tip/Tilt corrected turbulent_wf
    """
    x = np.arange(size) - size/2
    xy = np.meshgrid(x, x)
    xtilt_func = xy[0]*pupil
    ytilt_func = xy[1]*pupil
    
    xtilt = np.sum(xtilt_func * turbulent_wf)/np.sum(xtilt_func**2)
    ytilt = np.sum(ytilt_func * turbulent_wf)/np.sum(ytilt_func**2)
    
    corrected_wf = turbulent_wf*pupil - ytilt_func*ytilt - xtilt_func*xtilt
    
    return corrected_wf   

def apply_and_scale_turbulent_ef(turbulence, npix, wavelength, dx, seeing):
    """ Applies an atmosphere in the form of Kolmogorov turbulence to an initial wavefront and scales
    
    Parameters
    ----------
    npix: integer
        The size of the square of Kolmogorov turbulence generated
    wavelength: float
        The wavelength in mm. Amount of atmospheric distortion depends on the wavelength. 
    dx: float
        Resolution in mm/pixel
    seeing: float
        Seeing in arcseconds before magnification
    Returns
    -------
    turbulent_ef or 1.0: np.array([[...]...]) or 1.0
        Return an array of phase shifts in imperfect seeing, otherwise return 1.0, indicating no change to the incident wave.
    """
    if seeing > 0.0:
        # Convert seeing to radians
        seeing_in_radians = np.radians(seeing/3600.)
        
        # Generate the Kolmogorov turbulence
        #turbulence = optics_tools.kmf(npix)
        
        # Calculate r0 (Fried's parameter), which is a measure of the strength of seeing distortions
        r0 = 0.98 * wavelength / seeing_in_radians 
        
        # Apply the atmosphere and scale
        wf_in_radians = turbulence * np.sqrt(6.88*(dx/r0)**(5.0/3.0))
            
        # Convert the wavefront to an electric field
        turbulent_ef = np.exp(1.0j * wf_in_radians)
        
        return turbulent_ef
    else:
        # Do not apply phase distortions --> multiply by unity
        return 1.0

def calculate_fibre_mode(wavelength_in_mm, fibre_core_radius, numerical_aperture, npix, dx):
    """Computes the mode of the optical fibre.
    Parameters
    ----------
    wavelength_in_mm: float
        The wavelength in mm
    fibre_core_radius: float
        The radius of the fibre core in mm
    numerical_aperture: float
        The numerical aperture of the fibre
    npix: int
        Size of input_wf per side, preferentially a power of two (npix=2**n)
    dx: float
        Resolution of the wave in mm/pixel    
    Returns
    -------
    fibre_mode: np.array([[...]...])
        The mode of the optical fibre
    """
    # Calculate the V number for the model
    v = compute_v_number(wavelength_in_mm, fibre_core_radius, numerical_aperture)
    
    # Use the V number to calculate the mode
    fibre_mode = mode_2d(v, fibre_core_radius, sampling=dx, sz=npix)

    return fibre_mode


def compute_coupling(npix, dx, electric_field, lens_width, fibre_mode, x_offset, y_offset):
    """Computes the coupling between the electric field and the optical fibre using an overlap integral.
    
    Parameters
    ----------
    npix: int
        Size of input_wf per side, preferentially a power of two (npix=2**n)
    dx: float
        Resolution of the wave in mm/pixel      
    electric_field: np.array([[...]...])
        The electric field at the fibre plane
    lens_width: float
        The width of the a single microlens (used for minimising the unnecessary calculations)
    fibre_mode: np.array([[...]...])
        The mode of the optical fibre   
    x_offset: int
        x offset of the focal point at the fibre plane relative to the centre of the microlens.
    y_offset: int
        y offset of the focal point at the fibre plane relative to the centre of the microlens.           
   
    Returns
    -------
    coupling: float
        The coupling between the fibre mode and the electric_field (Max 1)
    """
    # Crop the electric field to the central 1/4
    low = npix/2 - int(lens_width / dx / 2) #* 3/8
    upper = npix/2 + int(lens_width / dx / 2) #* 5/8
    
    # Compute the fibre mode and shift (if required)
    fibre_mode = fibre_mode[(low + x_offset):(upper + x_offset), (low + y_offset):(upper + y_offset)]
    
    # Compute overlap integral - denominator first
    den = np.sum(np.abs(fibre_mode)**2) * np.sum(np.abs(electric_field)**2)
    
    #Crop the electric field and compute the numerator
    #electric_field = electric_field[low:upper,low:upper]
    num = np.abs(np.sum(fibre_mode*np.conj(electric_field)))**2

    coupling = num / den
    
    return coupling

def nglass(l, glass='sio2'):
    """Refractive index of fused silica and other glasses. Note that C is
    in microns^{-2}
    
    Parameters
    ----------
    l: wavelength 
    """
    try:
        nl = len(l)
    except:
        l = [l]
        nl=1
    l = np.array(l)
    if (glass == 'sio2'):
        B = np.array([0.696166300, 0.407942600, 0.897479400])
        C = np.array([4.67914826e-3,1.35120631e-2,97.9340025])
    elif (glass == 'bk7'):
        B = np.array([1.03961212,0.231792344,1.01046945])
        C = np.array([6.00069867e-3,2.00179144e-2,1.03560653e2])
    elif (glass == 'nf2'):
        B = np.array( [1.39757037,1.59201403e-1,1.26865430])
        C = np.array( [9.95906143e-3,5.46931752e-2,1.19248346e2])
    elif (glass == 'nsf11'):
        B = np.array([1.73759695E+00,   3.13747346E-01, 1.89878101E+00])
        C = np.array([1.31887070E-02,   6.23068142E-02, 1.55236290E+02])
    elif (glass == 'ncaf2'):
        B = np.array([0.5675888, 0.4710914, 3.8484723])
        C = np.array([0.050263605,  0.1003909,  34.649040])
    else:
        print("ERROR: Unknown glass {0:s}".format(glass))
        raise UserWarning
    n = np.ones(nl)
    for i in range(len(B)):
            n += B[i]*l**2/(l**2 - C[i])
    return np.sqrt(n)

class FresnelPropagator():
    """Propagate a wave by Fresnel diffraction"""
    def __init__(self,sz,m_per_pix, d, wave,nthreads=nthreads):
        """Initiate this fresnel_propagator for a particular wavelength, 
        distance etc.
    
        Parameters
        ----------
        wf: float array
            
        m_per_pix: float
            Scale of the pixels in the input wavefront in metres.
        d: float
            Distance to propagate the wavefront.
        wave: float
            Wavelength in metres.
        nthreads: int
            Number of threads. 
        """
        self.sz = sz
        self.nthreads=nthreads
        #Co-ordinate axis of the wavefront Fourier transform. Not that 0 must be in the corner.
        #x is in cycles per wavefront dimension.
        x = (((np.arange(sz)+sz/2) % sz) - sz/2)/m_per_pix/sz
        xy = np.meshgrid(x,x)
        uu =np.sqrt(xy[0]**2 + xy[1]**2)
        self.h_ft = np.exp(1j*np.pi*uu**2*wave*d)
    
    def propagate(self,wf):
        """Propagate a wavefront, according to the parameters established on the
        __init__. No error checking for speed.
        
        Parameters
        ----------
        wf: complex array
            Wavefront, i.e. a complex electric field in the scalar approximation.
        
        Returns
        -------
        wf_new: float array
            Wavefront after propagating.
        """

        if (wf.shape[0] != self.sz | wf.shape[1] != self.sz):
            print("ERROR: Input wavefront must match the size!")
            raise UserWarning
        if (self.nthreads>0):
            g_ft = pyfftw.interfaces.numpy_fft.fft2(wf,threads=self.nthreads)*self.h_ft
            wf_new = pyfftw.interfaces.numpy_fft.ifft2(g_ft,threads=self.nthreads)
        else:
            g_ft = np.fft.fft2(wf)*self.h_ft
            wf_new = np.fft.ifft2(g_ft)
        return wf_new
