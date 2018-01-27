"""
To add: simulation of modulation and servo loop

With synchronous demod at frequencies 2,3,5,7,8,9 with 4 samples/frequency, 
a frame for processing is then 9*4=28 or if really needed 9*4*4=144 raw frames.
This is 1.8 (or maybe 7.2 s) at 20Hz frame rate. 
Sounds achievable and compatible with Aladin detector 1/f noise.
"""
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import pdb

PHI0 = np.exp(2j*np.pi/3) # phasor for tri-coupler
#PHI0 = np.exp(1j)#*np.pi/2) # could be anything?
# ==================================================================
def make_nuller_mat3():
    """Make a nuller electric field matrix for a 3 telescope combiner"""
    #First, we have a unitary matrix for a nulling combiner with 1 bright and 
    #2 nulled outputs. How to manufacture this? Likely possible for a tri-coupler
    #but would be wavelength dependent.
    initial_mat = np.array([[1, 1, 1],
                            [1, PHI0**1, PHI0**2],
                            [1, PHI0**2, PHI0**1]])

    initial_mat = initial_mat / np.sqrt(3)
    
    #3x3 is hard, because 2 channels have to be split into 4 outputs 
    #to get two intensities and two visibility components. An obvious 
    #way to do this is to split each channel 1/3,2/3 in flux, reducing
    #the fringe visibility to 0.94. With C and S the cosine and sine
    #fluxes, we'd have F1 = 2*C-S, F2 = 2*S-C
    
    final_mat = np.ones( (5,3), dtype=complex)
    final_mat[1] = (initial_mat[1] + initial_mat[2]/np.sqrt(2))/np.sqrt(3)
    final_mat[2] = (initial_mat[1] - initial_mat[2]/np.sqrt(2))/np.sqrt(3)
    final_mat[3] = (initial_mat[1]/np.sqrt(2) + 1j*initial_mat[2])/np.sqrt(3)
    final_mat[4] = (initial_mat[1]/np.sqrt(2) - 1j*initial_mat[2])/np.sqrt(3)
    return final_mat

# ==================================================================
def make_nuller_mat4(bracewell_design=True, ker_only=False,
    phase_shifters=np.array([1,1,1], dtype=complex)):
    """Make a nuller electric field matrix for a 3 telescope combiner
    
    Parameters
    ----------
    bracewell_design: bool
        Do we use a Bracewell-like design? If not, then we use an 
        obvious symmetrical matrix. Harry can build a bracewell-like
        design already.
                
    ker_only: bool
        Do we use kernel outputs only or all 9?
        
    phase_shifters: array-like
        Three phasors to represent the on-chip phase modulation. For no
        phase shift, these should be 1, and for 180 degrees they should 
        be -1 (i.e. interferometric chopping)
    """
    #4x4 is nice and symmetric because you can get flux and visibilities
    #from 3 tri-couplers in the nulled channels.
    if bracewell_design:
        sq2 = np.sqrt(2)
        MM0 = 0.5 * np.array([[1,     1,   1,   1],
                              [1,     1,  -1,  -1],
                              [sq2, sq2,   0,   0],
                              [0,     0, sq2, sq2]], dtype=complex)
    else:
        MM0 = 0.5 * np.array([[1, 1, 1, 1],
                              [1, 1,-1,-1],
                              [1,-1, 1,-1],
                              [1,-1,-1, 1]], dtype=complex)

    #Add in the phase shifters
    for ix,phasor in enumerate(phase_shifters):
        MM0[ix+1] *= phasor

    if ker_only:
        #Now lets take the three nulled outputs, and put each of these into a 2x2-coupler.
        MM1 = np.ones( (7,4), dtype=complex)
        MM1[1] = (MM0[1] + 1j*MM0[2]) / 2
        MM1[2] = (1j*MM0[1] + MM0[2]) / 2
        
        MM1[3] = (MM0[1] + 1j*MM0[3]) / 2
        MM1[4] = (1j*MM0[1] + MM0[3]) / 2
        
        MM1[5] = (MM0[2] + 1j*MM0[3]) / 2
        MM1[6] = (1j*MM0[2] + MM0[3]) / 2
    else:
        #Now lets take the three nulled outputs, and put each of these into a tri-coupler.
        MM1 = np.ones( (10,4), dtype=complex)
        MM1[1] = (MM0[1] + MM0[2] * PHI0**0) / np.sqrt(6)
        MM1[2] = (MM0[1] + MM0[2] * PHI0**1) / np.sqrt(6)
        MM1[3] = (MM0[1] + MM0[2] * PHI0**2) / np.sqrt(6)
    
        MM1[4] = (MM0[1] + MM0[3] * PHI0**0) / np.sqrt(6)
        MM1[5] = (MM0[1] + MM0[3] * PHI0**1) / np.sqrt(6)
        MM1[6] = (MM0[1] + MM0[3] * PHI0**2) / np.sqrt(6)
    
        MM1[7] = (MM0[2] + MM0[3] * PHI0**0) / np.sqrt(6)
        MM1[8] = (MM0[2] + MM0[3] * PHI0**1) / np.sqrt(6)
        MM1[9] = (MM0[2] + MM0[3] * PHI0**2) / np.sqrt(6)
    return MM1

# ==================================================================
def other_nuller_mat4(phasor=False):
    if phasor is False:
        phi1 = np.exp(2j*np.pi/3) # phasor for tri-coupler
    else:
        phi1 = np.exp(j*phasor) # for a custom phase shift
        
    phi2 = np.conj(phi1)

    # this one is an explicit form of the matrix
    MM1 = (1.0 / (2 * np.sqrt(6))) * np.array(
        [[2,       0,       0,      -2     ],
         [1+phi1,  1-phi1, -1+phi1, -1-phi1],
         [1+phi2,  1-phi2, -1+phi2, -1-phi2],
         [2,      -2,       0,      0      ],
         [1+phi1, -1-phi1,  1-phi1, -1+phi1],
         [1+phi2, -1-phi2,  1-phi2, -1+phi2],
         [2,       0,      -2,       0     ],
         [1+phi1, -1+phi1, -1-phi1,  1-phi1],
         [1+phi2, -1+phi2, -1-phi2,  1-phi2]])

    # experiment: ditch out the rows with no phase shift
    MM1 = (1.0 / (2 * np.sqrt(2))) * np.array(
        [[1+phi1,  1-phi1, -1+phi1, -1-phi1],
         [1+phi2,  1-phi2, -1+phi2, -1-phi2],
         [1+phi1, -1-phi1,  1-phi1, -1+phi1],
         [1+phi2, -1-phi2,  1-phi2, -1+phi2],
         [1+phi1, -1+phi1, -1-phi1,  1-phi1],
         [1+phi2, -1+phi2, -1-phi2,  1-phi2]])

    '''
    # this one is identical to the result of Mike's
    # initial function
    MM1 = (1.0 / (2 * np.sqrt(6))) * np.array(
        [[2,       0,       0,      -2     ],
         [1+phi1,  1-phi1, -1+phi1, -1-phi1],
         [1+phi2,  1-phi2, -1+phi2, -1-phi2],
         [2,       0,      -2,       0     ],
         [1+phi1,  1-phi1, -1-phi1, -1+phi1],
         [1+phi2,  1-phi2, -1-phi2, -1+phi2],
         [2,      -2,       0,       0     ],
         [1+phi1, -1-phi1,  1-phi1, -1+phi1],
         [1+phi2, -1-phi2,  1-phi2, -1+phi2]])
    '''
    return MM1

# ==================================================================
def odiff(mat, phases): # output differences
    """Return the difference in output intensity between an input modified by 
    electric field phase and a perfect input. Note that inputting 
    imaginary phases tests for amplitude changes. """

    i0  = np.abs(mat.dot(np.ones(len(phases))))**2 # ref intensities (zeros!)
    ef1 = mat.dot(np.exp(1j*np.array(phases)))     # output E-field

    # Note that i0 = 0 (!), if we are dealing with the nulled output only.
    # This approach can be generalised to a combiner with some nulled and some 
    # non-nulled outputs. 
    return np.abs(ef1)**2 - i0

# ==================================================================
def mas2rad(x):
    ''' convert milliarcsec to radians '''
    return(x * 4.8481368110953599e-09) # = x*np.pi/(180*3600*1000)

# ==================================================================
def incoherent_sum(mat, E_on, E_off, con):
    '''Computes the output intensity vector by a combiner "mat", looking at a
    binary star with primary on-axis for which the two E-field vectors are
    pre-computed. The contrast "con" (< 1.0) is applied to the off-axis source. '''
        
    temp = (np.abs(mat.dot(E_on))**2 + con * np.abs(mat.dot(E_off))**2) / (1.0 + con)
    return(temp)

# ==================================================================
def make_nuller_mat_n(n=6):
    """Make a nuller matrix with N inputs. 
    
    For odd n, one could start with 
    the van der mont matrix, then put each pair into a tri-coupler. This of 
    course gives more outputs than necessary for N>=5, but should be sufficient
    """
    return None
