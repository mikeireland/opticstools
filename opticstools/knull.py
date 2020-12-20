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
#PHI0 = np.exp(1j)#*np.pi/2) # could be anything? No - coupler matrices must have 
                             # orthogonal columns.

#Some telescope arrays
UTS = np.array([[-9.925,  14.887, 44.915, 103.306],  # VLTI
                [-20.335, 30.502,  66.183,  44.999]]) # VLTI                     
        
def guyon_nuller5():
    """Create the Null stage for the Guyon Beamsplitter Nuller
    
    After the first stage, to obtain the even split, the next
    stage is includes 0.5*initial_mat[3] then complex reflections
    for the other components. 
    
    Symmetrical beamsplitters give phase independent of wavelength... I think.s
    """
    initial_mat = np.zeros( (5,5) )
    initial_mat[0,:2] = np.array([1,-1])*np.sqrt(1/2)
    initial_mat[1,:3] = np.array([1,1,-2])*np.sqrt(2/3)/2
    initial_mat[2,:4] = np.array([1,1,1,-3])*np.sqrt(3/4)/3
    initial_mat[3,:] = np.array([1,1,1,1,-4])*np.sqrt(4/5)/4
    initial_mat[4,:] = np.ones(5)*np.sqrt(1/5)
    return initial_mat
            
def make_nuller_mat5():
    """Create a 5x5 Nuller matrix"""
    initial_mat = np.zeros( (5,5) )
    for i in range(5):
        initial_mat[i] = np.arange(5) * i
    initial_mat = np.exp(2j*np.pi/5*initial_mat)
    return initial_mat
        
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
    phase_shifters=np.array([1,1,1], dtype=complex), ker_coupler_angle=np.pi/2):
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
        
    ker_coupler_angle: float
        In the case of kernel outputs only, the final 50/50 couplers are defined by the 
        one phase. See e.g. https://en.wikipedia.org/wiki/Unitary_matrix, setting theta
        to pi/2 (50/50), and phase angle varphi_1 to 0. 
    """
    #4x4 is nice and symmetric because you can get flux and visibilities
    #from 3 tri-couplers in the nulled channels.
    if bracewell_design:
        sq2 = np.sqrt(2)
        #For a "Bracewell" design, each element has a 
        #matrix 1/sq2 * np.array([[1,1],[1,-1]])
        MM0 = 0.5 * np.array([[1,      1,   1,    1],
                              [1,      1,  -1,   -1],
                              [sq2, -sq2,   0,    0],
                              [0,      0, sq2, -sq2]], dtype=complex)
    else:
        MM0 = 0.5 * np.array([[1, 1, 1, 1],
                              [1, 1,-1,-1],
                              [1,-1, 1,-1],
                              [1,-1,-1, 1]], dtype=complex)

    #Add in the phase shifters
    for ix,phasor in enumerate(phase_shifters):
        MM0[ix+1] *= phasor

    if ker_only:
        MM1 = np.zeros( (7,4), dtype=complex)
        #Start with the bright output.
        MM1[0] = MM0[0]
        
        #Now lets take the three nulled outputs, and put each of these into a 2x2-coupler.
        PHI0 = np.exp(1j*ker_coupler_angle)
        PHI1 = np.conj(PHI0)
        
        MM1[1] = (MM0[1] + PHI0*MM0[2]) / 2
        MM1[2] = (-PHI1*MM0[1] + MM0[2]) / 2
        
        MM1[3] = (MM0[1] + PHI0*MM0[3]) / 2
        MM1[4] = (-PHI1*MM0[1] + MM0[3]) / 2
        
        MM1[5] = (MM0[2] + PHI0*MM0[3]) / 2
        MM1[6] = (-PHI1*MM0[2] + MM0[3]) / 2
    else:
        #Now lets take the three nulled outputs, and put each of these into a tri-coupler.
        PHI0 = np.exp(2j*np.pi/3)
        MM1 = np.zeros( (10,4), dtype=complex)
        MM1[0] = MM0[0]
        
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
        phi1 = np.exp(1j*phasor) # for a custom phase shift
        
    phi2 = -np.conj(phi1) #!!!Warning - the minus sign here was added by Mike

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
#    MM1 = (1.0 / (2 * np.sqrt(2))) * np.array(
#        [[1+phi1,  1-phi1, -1+phi1, -1-phi1],
#         [1+phi2,  1-phi2, -1+phi2, -1-phi2],
#         [1+phi1, -1-phi1,  1-phi1, -1+phi1],
#         [1+phi2, -1-phi2,  1-phi2, -1+phi2],
#         [1+phi1, -1+phi1, -1-phi1,  1-phi1],
#         [1+phi2, -1+phi2, -1-phi2,  1-phi2]])
    #!!!Mike - actually do this physically... with an on-paper 
    #tedious matrix multiplication.
    MM1 = (1.0 / 4.0) * np.array(
        [[1+phi1,  1-phi1, -1+phi1, -1-phi1],
         [1+phi2, -1+phi2,  1-phi2, -1-phi2],
         [1+phi1,  1-phi1, -1-phi1, -1+phi1],
         [1+phi2, -1+phi2, -1-phi2,  1-phi2],
         [1+phi1, -1-phi1,  1-phi1, -1+phi1],
         [1+phi2, -1-phi2, -1+phi2,  1-phi2]])

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
    
    For n>=4, one could start with 
    the van der mont matrix, then put each pair into a tri-coupler. This of 
    course gives more outputs than necessary for N>=5, but should be sufficient
    """
    return None

# ==================================================================
def make_lacour_mat():
    """Make the matrix from Lacour et al (2014)
    
    There are 8 non-nulled outputs followed by 12 nulled outputs.
    
    There are many "kernel" quantities that can be derived from this.
    However, some are meaningless, like A+C-B-D in each ABCD combiner.
    """
    tri_nuller = np.array([[PHI0**1, 1, PHI0**2],
                          [-PHI0**2,-1, -PHI0**1]])/np.sqrt(3)
    tri_nuller = tri_nuller.T
    y_splitter = np.array([1,1])/np.sqrt(2)
    abcd = np.array([[1,   1],
                     [1,  1j],
                     [1,  -1],
                     [1, -1j]])/2.0
    
    #Start with the splitters
    MM0 = np.zeros((8,4), dtype=complex)
    MM0[0:2,0] = y_splitter
    MM0[2:4,1] = y_splitter
    MM0[4:6,2] = y_splitter
    MM0[6:8,3] = y_splitter
    
    #Next the nullers
    MM1 = np.zeros((11,8), dtype=complex)
    MM1[0,0] = 1
    MM1[1:4,1:3] = tri_nuller
    MM1[4:7,3:5] = tri_nuller
    MM1[7:10,5:7] = tri_nuller
    MM1[-1,-1] = 1
    
    #Now the re-routing and splitting for ABCD
    MM2 = np.zeros((14,11), dtype=complex)
    #Flux 
    MM2[0,0]=1
    MM2[1,10]=1
    #Fringe tracking 
    MM2[2,1]=1
    MM2[3,3]=1
    MM2[4,4]=1            
    MM2[5,6]=1
    MM2[6,7]=1            
    MM2[7,9]=1
    #Nulled outputs... need splitting
    #First, C(12-23)
    MM2[8,2]=y_splitter[0]
    MM2[9,5]=y_splitter[1]
    #Next, C(12-34)
    MM2[10,2]=y_splitter[0]
    MM2[11,8]=y_splitter[1]
    #Finally, C(23-34)
    MM2[12,5]=y_splitter[0]
    MM2[13,8]=y_splitter[1]
    
    #Finally, the ABCD
    MM3 = np.zeros((20,14), dtype=complex)
    ix = np.arange(8, dtype=int)
    MM3[ix,ix] = 1
    MM3[8:12,  8:10] = abcd
    MM3[12:16,10:12] = abcd
    MM3[16:20,12:14] = abcd

    MM = MM3.dot(MM2.dot(MM1.dot(MM0)))

    return MM
    

def response_random_pistons(mat, K, ntest=100000, rms_piston=50.0, \
    con=0.0, off_axis=None, cwavel=3.6e-6, flux=np.inf, \
    bflux=0, bflux_scale=0.5, rms_amp=0.01):
    """Record the response of/6 the system to random pistons
    
    bflux_scale: float
        The scaling of the flux per telescope to flux per output for the
        background
        Get this by e.g. np.mean(output_highphase)
        
    rms_piston: float
        the RMS piston error
        
    rms_amp: float
        The RMS amplitude error. 10% RMS for H-band is 2% RMS at 3.7 microns, or 
        1% in electric field.
    """

    if off_axis is None:
        off_axis=np.zeros(mat.shape[1])

    ntest = 10000

    piston_record = [] # record of random pistons generated
    output_record = [] # record of the output of the nuller+sensor matrix
    kernel_record = [] # record of the kernel-output

    for i in range(ntest):
        pistons = np.random.randn(4) *  rms_piston     # atmospheric pistons in nanometers
        piston_record.append(pistons)
        amps = 1 + np.random.normal(size=4) * rms_amp

        E_on  = amps*np.exp(-1j*2*np.pi/cwavel * pistons * 1e-9)
        E_off = amps*np.exp(-1j*2*np.pi/cwavel * (pistons * 1e-9 + off_axis))

        output = incoherent_sum(mat, E_on, E_off, con)

        if flux != np.inf:
            output = np.random.poisson(flux * output + bflux_scale*bflux)/float(flux)

        output_record.append(output)
        kernel_record.append(K.dot(output))

    output_record = np.array(output_record)
    kernel_record = np.array(kernel_record)
    return output_record, kernel_record

def lacour_response_random_pistons(mat, ntest=400, nsubsample=100, rms_piston=50.0, \
    con=0.0, off_axis=None, cwavel=3.6e-6):
    """Record the closure-phase response of the system to random pistons.
    
    Assume an incoherent sum of outputs for nsubsample integration times. """

    if off_axis is None:
        off_axis=np.zeros(mat.shape[1])

    ntest = 10000

    piston_record = [] # record of random pistons generated
    output_record = [] # record of the output of the nuller+sensor matrix
    closure_record = [] # record of the kernel-output

    for i in range(ntest):
        output=np.zeros(mat.shape[0])
        for j in range(nsubsample):
            pistons = np.random.randn(4) *  rms_piston     # atmospheric pistons in nanometers
            piston_record.append(pistons)

            E_on  = np.exp(-1j*2*np.pi/cwavel * pistons * 1e-9)
            E_off = np.exp(-1j*2*np.pi/cwavel * (pistons * 1e-9 + off_axis))

            output += incoherent_sum(mat, E_on, E_off, con)
        #Now compute the closure-phase
        vis1 = (output[0]-output[2]) + 1j*(output[1]-output[3])
        vis2 = (output[4]-output[6]) - 1j*(output[5]-output[7])
        vis3 = (output[8]-output[10]) + 1j*(output[9]-output[11])
        
        output_record.append(output)    
        closure_record.append(np.angle(vis1*vis2*vis3))

    output_record = np.array(output_record)
    closure_record = np.array(closure_record)
    return output_record, closure_record
    