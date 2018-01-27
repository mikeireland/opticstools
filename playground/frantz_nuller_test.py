from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pdb
import sys
plt.ion()
if not '..' in sys.path:
    sys.path.insert(0,'..')
from opticstools import knull

#Start with a matrix that is all but the bright output of the 4 input coupler
#mat = make_nuller_mat4(bracewell_design=False)[1:]
mat = knull.other_nuller_mat4() # test with reformatted matrix

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

ncov = len(cov_elts) # number of covariance elements
nout = mat.shape[0]  # number of outputs
ntel = mat.shape[1]  # number of sub-apertures
dph = 0.01           # delta-phase (for numerical computation)

#AA is an extension of the matrix "A" from the kernel-phase papers, where
#we both include first and second-order phase differences in the pupil plane 
#phase vector. They are arranged according to cov_elts.
#Output - Perfect_ouput = AA DOT (dph[0], dph[0]**2, dph[2], dph[1]*dph[3], etc...)
AA = np.empty((ncov, nout))

#We approximate in 2D the intensity as:
#f(x,y) = a_x x + a_y y + b_xx x^2 + b_yy*y^2 + b_xy * xy

for ix, elt in enumerate(cov_elts):
    p1 = np.zeros(ntel, dtype=complex)
    p2 = np.zeros(ntel)
    
    p1[elt[0]] = dph
    
    if (len(elt) == 1):
        AA[ix] = 0.5 * (knull.odiff(mat, p1) - knull.odiff(mat, -p1)) / dph
        
    elif (elt[0] == elt[1]):
        AA[ix] = 0.5 * (knull.odiff(mat, p1) + knull.odiff(mat, -p1)) / dph**2

    else:
        p1[elt[1]] = dph
        p2[elt[0]] = dph
        p2[elt[1]] = -dph        
        AA[ix] = 0.25 * (knull.odiff(mat, p1) + knull.odiff(mat, -p1) -
                         knull.odiff(mat, p2) - knull.odiff(mat, -p2) ) / dph**2

U, W, V = np.linalg.svd(AA.T)
#Kernel-output matrix...
# K = np.transpose(U[:,len(W):]) # Mike's version
K = np.transpose(U[:,np.sum(W < 1e-3):]) #
'''plt.figure(1)
plt.clf()
plt.plot(K.T)
plt.plot(K.T,'o')'''


print("Shape of AA.T: " + str(AA.T.shape))
print(np.round(AA.T, 3))

# NOTE that 8 * AA.T (or AA) is filled with integer values!

print("Shape of K: " + str(K.shape))
print(np.round(K, 2))

#pdb.set_trace()






#FIXME
#Next step: Figure out now these kernel outputs relate to visibilities
#and on-sky coordinates for simple telescope arrangements.















# ---------------------------------------
# definition of the interferometric array
# ---------------------------------------
apc = np.array([[0.0, 10.0, 40.0, 60.0],  # start with an in-line interferometer
                [0.0,  0.0,  0.0,  0.0]]) # x-y coordinates of the apertures in meters
apc = np.array([[-9.925,  14.887, 44.915, 103.306],  # VLTI
                [-20.335, 30.502,  66.183,  44.999]]) # VLTI

cwavel       = 3.6e-6                    # wavelength of observation (in meters)
alpha, delta = 5.0, 0.0                  # coordinates of the off-axis companion in mas
con          = 1.0e-2                    # 100:1 companion contrat
off_axis     = knull.mas2rad(alpha * apc[0] + delta * apc[1])
    
'''    plt.figure(2)
plt.clf()
plt.plot(apc[0], apc[1], 'ro')

plt.xlim([-10,50])
plt.ylim([-10,50])
'''

MM0 = 0.5 * np.array([[1, 1, 1, 1],  # the nuller matrix, alone
                      [1, 1,-1,-1],
                      [1,-1, 1,-1],
                      [1,-1,-1, 1]])

# ---------------------------------------------------
# record the response of the system to random pistons
# ---------------------------------------------------
ntest = 10000

piston_record = [] # record of random pistons generated
nuller_record = [] # record of the output of the nuller matrix alone
output_record = [] # record of the output of the nuller+sensor matrix
kernel_record = [] # record of the kernel-output

for i in range(ntest):
    pistons = np.random.randn(4) *  50.0     # atmospheric pistons in nanometers
    pistons[0] = 0.0                         # piston measured relative to first aperture
    piston_record.append(pistons)

    E_on  = np.exp(-1j*2*np.pi/cwavel * pistons * 1e-9)
    E_off = np.exp(-1j*2*np.pi/cwavel * (pistons * 1e-9 + off_axis))

    output = knull.incoherent_sum(mat, E_on, E_off, con)

    output_record.append(output)
    kernel_record.append(K.dot(output))

    temp = knull.incoherent_sum(MM0, E_on, E_off, con)
    nuller_record.append(temp)

# --------------------------------------------
# record what the perfect output should be
# --------------------------------------------
pistons[:] = 0.0
E_on  = np.exp(-1j*2*np.pi/cwavel * pistons * 1e-9)
E_off = np.exp(-1j*2*np.pi/cwavel * (pistons * 1e-9 + off_axis))

temp = knull.incoherent_sum(MM0, E_on, E_off, con)
perfect_null_data = temp.copy()

temp = knull.incoherent_sum(mat, E_on, E_off, con)
perfect_kernel_data = temp.copy()

# --------------------------------------------

output_record = np.array(output_record)
kernel_record = np.array(kernel_record)
nuller_record = np.array(nuller_record)

'''
f1 = plt.figure(3, figsize=(15,6))
f1.clf()
ax1 = f1.add_subplot(121)
ax2 = f1.add_subplot(122)
for i in range(nout):
    try:
        ax1.plot(output_record[:100,i])
    except:
        pass
    
for i in range(3):
    try:
        ax2.plot(kernel_record[:100,i])
    except:
        pass

rmax = 1.5e-2
ax1.set_ylim([0, rmax])
ax1.set_xlabel("observation index", fontsize=14)
ax1.set_ylabel("raw nuller output", fontsize=14)

ax2.set_ylim([-rmax/2, rmax/2])
ax2.set_xlabel("observation index", fontsize=14)
ax2.set_ylabel("kernel nuller output", fontsize=14)

f1.set_tight_layout(True)
#ax1.set_yscale('log')
'''
print("------------------------------------------------------------")
print("raw output variability: ", output_record.std(axis=0).round(4))
print("kernel variability:     ", kernel_record.std(axis=0).round(4))
print("------------------------------------------------------------")


# ===============================================================
# now looking at distribution of nulled outputs vs kernel-outputs
# ===============================================================
f0 = plt.figure(10, figsize=(18,5))
plt.clf()

ax0 = f0.add_subplot(131)
for ii in range(3):
    plt.hist(nuller_record[:,ii+1], bins=100, range=[0,0.05], label="output #%d" % (ii+1),
             histtype="step")
plt.ylabel("# of occurences")
plt.xlabel("Null-depth")
plt.xlim([0.0, 0.05])
plt.legend(fontsize=11)
plt.title("Classical nuller outputs distribution")

ax1 = f0.add_subplot(132)
for ii in range(nout):
    plt.hist(output_record[:,ii], bins=100, range=[0.0, 0.05], label="output #%d" % (ii+1,), histtype="step")
plt.ylabel("# of occurences")
plt.xlabel("Null-depth")
plt.xlim([0.0, 0.05])
plt.legend(fontsize=11)
plt.title("Modified nuller outputs distribution")

ax2 = f0.add_subplot(133)
for ii in range(3):
    plt.hist(kernel_record[:,ii], bins=100, label="output #%d" % (ii+1,), histtype="step")
plt.ylabel("# of occurences")
plt.xlabel("Kernel-Null")
plt.xlim([-0.025, 0.025])
plt.legend(fontsize=11)
plt.title("Kernel-outputs distribution")

f0.set_tight_layout(True)
# ---------------------------------------------------
# record the response of the system to variable pos
# ---------------------------------------------------
ntest = 100
output_record2 = []
kernel_record2 = []

pos = 20 * (np.arange(ntest)/float(ntest)-0.5) # position from -10 to 10 mas
direction = "E-W"

for i in range(ntest):
    pistons = np.random.randn(4) *  0.0     # atmospheric pistons in nanometers
    pistons[0] = 0.0                         # piston measured relative to first aperture

    if direction == "E-W":
        off_axis = knull.mas2rad(pos[i] * apc[0] + delta * apc[1]) # along east-west axis
    else:
        off_axis = knull.mas2rad(0.0 * apc[0] + pos[i] * apc[1]) # along north-south axis
    
    E_on  = np.exp(-1j*2*np.pi/cwavel * pistons * 1e-9)
    E_off = np.exp(-1j*2*np.pi/cwavel * (pistons * 1e-9 + off_axis))
    
    output = knull.incoherent_sum(mat, E_on, E_off, con)
    output_record2.append(output)
    kernel_record2.append(K.dot(output))

output_record2 = np.array(output_record2)
kernel_record2 = np.array(kernel_record2)



f2 = plt.figure(4, figsize=(15,6))
f2.clf()
ax1 = f2.add_subplot(121)
ax2 = f2.add_subplot(122)
for i in range(nout):
    try:
        ax1.plot(pos, output_record2[:,i])
    except:
        pass

for i in range(3):
    try:
        ax2.plot(pos, kernel_record2[:,i])
    except:
        pass

rmax = 1.5e-2
ax1.set_ylim([0, rmax])
ax1.set_xlabel("%s off-axis position (mas)" % (direction,), fontsize=14)
ax1.set_ylabel("VLTI-UTs raw nuller output", fontsize=14)

ax2.set_ylim([-rmax/2, rmax/2])
ax2.set_xlabel("%s off-axis position (mas)" % (direction,), fontsize=14)
ax2.set_ylabel("VLTI-UTs kernel nuller output", fontsize=14)

f2.set_tight_layout(True)

# =========================================
# 1D-plot of the response of the nuller alone
# =========================================
'''
output_record3 = []

pos = 20 * (np.arange(ntest)/float(ntest)-0.5) # position from -10 to +10 mas
for i in range(ntest):
    pistons = np.random.randn(4) *  0.0     # atmospheric pistons in nanometers
    pistons[0] = 0.0                         # piston measured relative to first apert

    if direction == "E-W":
        off_axis = mas2rad(pos[i] * apc[0] + delta * apc[1]) # along east-west axis
    else:
        off_axis = mas2rad(0.0 * apc[0] + pos[i] * apc[1]) # along north-south axis
    
    E_on   = np.exp(-1j*2*np.pi/cwavel * pistons * 1e-9)
    E_off  = np.exp(-1j*2*np.pi/cwavel * (pistons * 1e-9 + off_axis))
    output = incoherent_sum(MM0, E_on, E_off, con)
    
    output_record3.append(output)

output_record3 = np.array(output_record3)

f3 = plt.figure(5, figsize=(8,6))
f3.clf()
ax1 = f3.add_subplot(111)
for ii in range(1,4):
    ax1.plot(pos, output_record3[:,ii])
'''




# =========================================================================
#                             2D Kernel-output maps
# =========================================================================

pos = 20 * (np.arange(ntest)/float(ntest)-0.5) # position from -10 to +10 mas
xx, yy = np.meshgrid(pos,pos)
for ii in range(nout):
    exec("omap%d = np.zeros((ntest, ntest))" % (ii+1,)) # maps of the SxNx outputs
    
for ii in range(3):
    exec("kmap%d = np.zeros((ntest, ntest))" % (ii+1,)) # maps of the KxSxNx outputs

for ii in range(3):
    exec("nmap%d = np.zeros((ntest, ntest))" % (ii+1,)) # maps of the Nx outputs

pistons[:] = 0.0                         # no piston errors
E_on  = np.exp(-1j*2*np.pi/cwavel * pistons * 1e-9)
for ii in range(ntest):
    for jj in range(ntest):
        off_axis = knull.mas2rad(xx[ii,jj] * apc[0] + yy[ii,jj] * apc[1])
        E_off    = np.exp(-1j*2*np.pi/cwavel * (pistons * 1e-9 + off_axis))
    
        output = knull.incoherent_sum(mat, E_on, E_off, con)
        kernel = K.dot(output)

        for kk in range(3):
            exec("kmap%d[ii,jj] = kernel[%d]" % (kk+1, kk))
            
        for kk in range(nout):
            exec("omap%d[ii,jj] = output[%d]" % (kk+1, kk))

        temp = knull.incoherent_sum(MM0, E_on, E_off, con)
        for kk in range(3):
            exec("nmap%d[ii,jj] = temp[%d]" % (kk+1, kk+1))

# ===============================================================
#                         plot the 2D maps
# ===============================================================
f4 = plt.figure(6, figsize=(15,5))
mmax = np.round(pos.max())
mycmap = cm.magma

for kk in range(3):
    exec('ax%d = f%d.add_subplot(1,3,%d)' % (kk+1, 4, kk+1))
    exec('ax%d.imshow(kmap%d, extent=(-mmax, mmax, -mmax, mmax), cmap=mycmap)' % (kk+1, kk+1))
    exec('ax%d.set_xlabel("R.A. Kernel-output map #%d (mas)")' % (kk+1, kk+1))
    exec('ax%d.set_ylabel("Dec. Kernel-output map #%d (mas)")' % (kk+1, kk+1))

f4.set_tight_layout(True)

# =======
f5 = plt.figure(7, figsize=(15,15/(nout/3)))

for kk in range(nout):
    exec('ax%d = f%d.add_subplot(%d,3,%d)' % (kk+1, 5, nout/3, kk+1))
    exec('ax%d.imshow(omap%d, extent=(-mmax, mmax, -mmax, mmax), cmap=mycmap)' % (kk+1, kk+1))
    exec('ax%d.set_xlabel("R.A. Modified output map #%d (mas)")' % (kk+1, kk+1))
    exec('ax%d.set_ylabel("Dec. Modified output map #%d (mas)")' % (kk+1, kk+1))

f5.set_tight_layout(True)

# =======
f6 = plt.figure(8, figsize=(15,5))

for kk in range(3):
    exec('ax%d = f%d.add_subplot(1,3,%d)' % (kk+1, 6, kk+1))
    exec('ax%d.imshow(nmap%d, extent=(-mmax, mmax, -mmax, mmax), cmap=mycmap)' % (kk+1, kk+1))
    exec('ax%d.set_xlabel("R.A. Nuller output map #%d (mas)")' % (kk+1, kk+1))
    exec('ax%d.set_ylabel("Dec. Nuller output map #%d (mas)")' % (kk+1, kk+1))

f6.set_tight_layout(True)

