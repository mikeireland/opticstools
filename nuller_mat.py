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

TRICOUPLER_PHASOR = np.exp(2j*np.pi/3)

def output_int_diff(mat, phases):
    """Return the difference in output intensity between an input modified by 
    electric field phase and a perfect input. Note that inputting 
    imaginary phases tests for amplitude changes. """
    i0 = np.dot(mat, np.ones(len(phases)))
    i0 = np.abs(i0)**2
    phasors = np.exp(1j*np.array(phases))
    efields = np.dot(mat, phasors)
    return np.abs(efields)**2 - i0

def make_nuller_mat3():
    """Make a nuller electric field matrix for a 3 telescope combiner"""
    #First, we have a unitary matrix for a nulling combiner with 1 bright and 
    #2 nulled outputs. How to manufacture this? Likely possible for a tri-coupler
    #but would be wavelength dependent.
    initial_mat = np.array([[1,1,1],[1,TRICOUPLER_PHASOR,np.conj(TRICOUPLER_PHASOR)],[1,np.conj(TRICOUPLER_PHASOR), TRICOUPLER_PHASOR]])
    
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

def make_nuller_mat4(bracewell_design=True, ker_only=True, phase_shifters=[1,1,1]):
    """Make a nuller electric field matrix for a 3 telescope combiner
    
    Parameters
    ----------
    bracewell_design: bool
        Do we use a Bracewell-like design? If not, then we use an 
        obvious symmetrical matrix. Harry can build a bracewell-like
        design.  
        
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
        initial_mat = np.array([[1,1,1,1],[1,1,-1,-1],[np.sqrt(2),-np.sqrt(2),0,0],[0,0,np.sqrt(2),-np.sqrt(2)]], dtype=complex)
    else:
        initial_mat = np.array([[1,1,1,1],[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]], dtype=complex)

    #Add in the phase shifters
    for ix,phasor in enumerate(phase_shifters):
        initial_mat[1+ix] *= phasor

    if ker_only:
        #Now lets take the three nulled outputs, and put each of these into a 2x2-coupler.
        final_mat = np.ones( (7,4), dtype=complex)
        final_mat[1] = (initial_mat[1] + 1j*initial_mat[2])/2
        final_mat[2] = (1j*initial_mat[1] + initial_mat[2])/2
        
        final_mat[3] = (initial_mat[1] + 1j*initial_mat[3])/2
        final_mat[4] = (1j*initial_mat[1] + initial_mat[3])/2
        
        final_mat[5] = (initial_mat[2] + 1j*initial_mat[3])/2
        final_mat[6] = (1j*initial_mat[2] + initial_mat[3])/2
        
    else:
        #Now lets take the three nulled outputs, and put each of these into a tri-coupler.
        final_mat = np.ones( (10,4), dtype=complex)
        final_mat[1] = (initial_mat[1] + initial_mat[2])/np.sqrt(6)
        final_mat[2] = (initial_mat[1] + initial_mat[2]*TRICOUPLER_PHASOR)/np.sqrt(6)
        final_mat[3] = (initial_mat[1] + initial_mat[2]*np.conj(TRICOUPLER_PHASOR))/np.sqrt(6)
    
        final_mat[4] = (initial_mat[1] + initial_mat[3])/np.sqrt(6)
        final_mat[5] = (initial_mat[1] + initial_mat[3]*TRICOUPLER_PHASOR)/np.sqrt(6)
        final_mat[6] = (initial_mat[1] + initial_mat[3]*np.conj(TRICOUPLER_PHASOR))/np.sqrt(6)
    
        final_mat[7] = (initial_mat[2] + initial_mat[3])/np.sqrt(6)
        final_mat[8] = (initial_mat[2] + initial_mat[3]*TRICOUPLER_PHASOR)/np.sqrt(6)
        final_mat[9] = (initial_mat[2] + initial_mat[3]*np.conj(TRICOUPLER_PHASOR))/np.sqrt(6)
    return final_mat
    
def make_nuller_mat_n(n=6):
    """Make a nuller matrix with N inputs. 
    
    For odd n, one could start with 
    the van der mont matrix, then put each pair into a tri-coupler. This of 
    course gives more outputs than necessary for N>=5, but should be sufficient
    """
    return None

if __name__=="__main__":
    #Start with a matrix that is all but the bright output of the 4 input coupler
    mat = make_nuller_mat4(bracewell_design=False)[1:]
    
    #cov_elts is a list of either phases we want to be linearly independent of, or
    #pairs of phases we want to be independent of to 2nd order.
    #We can ignore one of the telescope phases here as a "reference" phase. 
    cov_elts = [[1,1],[2,2],[3,3],[1,2],[1,3],[2,3]]
    
    #For 3 telescopes...
    #mat = make_nuller_mat4()[1:] 
    #cov_elts = [[1,1],[2,2],[1,2]]
    
    #For a hypothetical phase matrix with e.g. 12 outputs:
    #Independence of amplitude would require a modification of the code
    #below (which has been checked for the simple 4 telescope case)
    #cov_elts = [[1],[2],[3],[1,1],[2,2],[3,3],[1,2],[1,3],[2,3]]

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
            A_mat[ix] = 0.5*(output_int_diff(mat, p1) - output_int_diff(mat, -p1))/dph
        elif elt[0]==elt[1]:
            p1[elt[0]] = dph
            A_mat[ix] = 0.5*(output_int_diff(mat, p1) + output_int_diff(mat, -p1))/dph**2
        else:
            p1[elt[0]] = dph
            p1[elt[1]] = dph
            p2[elt[0]] = dph
            p2[elt[1]] = -dph        
            A_mat[ix] = 0.25*( output_int_diff(mat, p1) + output_int_diff(mat, -p1) \
                - output_int_diff(mat, p2) - output_int_diff(mat, -p2) )/dph**2
            
    U, W, V = np.linalg.svd(A_mat.T)
    #Kernel-output matrix...
    K = np.transpose(U[:,sum(W>1e-10):])
    plt.clf()
    plt.plot(K.T)
    plt.plot(K.T,'o')
    print("Shape of K: " + str(K.shape))

    #FIXME
    #Next step: Figure out now these kernel outputs relate to visibilities
    #and on-sky coordinates for simple telescope arrangements.
