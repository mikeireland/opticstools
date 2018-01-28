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

diagonalise=True #Find the best kernel outputs...
#Start with a matrix that is all but the bright output of the 4 input coupler
lacour=False
mat = knull.make_nuller_mat4(bracewell_design=False, ker_only=True)[1:]

#mat = knull.make_lacour_mat()
mat = knull.make_lacour_mat()[8:] #Nulled outputs only.
lacour=True

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

if lacour:
    old_K=K.copy()
    #Try to remove the obvious problems with K.
    garbage_kernels = np.zeros((3,12))
    garbage_kernels[0, 0:4] = np.array([1,-1,1,-1])/2.
    garbage_kernels[1, 4:8] = np.array([1,-1,1,-1])/2.
    garbage_kernels[2,8:12] = np.array([1,-1,1,-1])/2.
    for g in garbage_kernels:
        for krow in K:
            krow -= np.sum(g*krow)*g
    #Reconstruct K
    UK, wk, VK = np.linalg.svd(K)
    nnew = np.sum(wk>1e-10)
    WK = np.zeros( (nnew,K.shape[1]) )
    WK[:nnew,:nnew] = np.diag(wk[:nnew])
    K = WK.dot(VK)
    #Compute the closure phases
    output_record, closure_record = knull.lacour_response_random_pistons(mat, rms_piston=50.0)
    output_record, closure_highphase = knull.lacour_response_random_pistons(mat, rms_piston=10000.0, nsubsample=1)
    print("CP Std (per frame):")
    print(np.std(closure_record, axis=0)*10)
    
    print("Dodgy CP:")
    dodgy_normalisation = np.std(closure_highphase, axis=0)
    print(dodgy_normalisation)

    print("Normalised CP Stds: ")
    print(np.std(closure_record, axis=0)/dodgy_normalisation*10.0)

output_record, kernel_record = knull.response_random_pistons(mat, K)

if diagonalise:
    #Diagonalise the covariance matrix
    kernel_cov = kernel_record.T.dot(kernel_record)/kernel_record.shape[0]
    W, V = np.linalg.eigh(kernel_cov)
    print("Kernel Stds: ")
    print(np.sqrt(W))
    K= V.T.dot(K)

plt.clf()
plt.plot(K.T)
plt.plot(K.T,'o')
print("Shape of K: " + str(K.shape))

if lacour:
    #Interestingly, all imaginary parts of the Lacour ABCD combiners are kernel null
    #outputs. And the closure-phase is not the best linear combination of them!
    print(K[:,[1,3,5,7,9,11]])

output_highphase, kernel_highphase = knull.response_random_pistons(mat, K, rms_piston=10000.)
output_record, kernel_record = knull.response_random_pistons(mat, K)
knull.response_random_pistons(mat, K, rms_piston=10000.)


print("Kernel Stds (recalc): ")
print(np.std(kernel_record, axis=0))

#Rather than normalising by looking at the response as a function of angle for a particular
#telescope configuration, look at the response averaged over large piston offsets to 
#approximate the response to off-axis sources.
print("Dodgy Normalisation:")
dodgy_normalisation = np.std(kernel_highphase, axis=0)
print(dodgy_normalisation)

print("Normalised Stds: ")
print(np.std(kernel_record, axis=0)/dodgy_normalisation)


#FIXME
#Next step: Figure out now these kernel outputs relate to visibilities
#and on-sky coordinates for simple telescope arrangements.