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
mat = knull.make_nuller_mat4(bracewell_design=False, ker_only=True)[1:]

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
K = np.transpose(U[:,sum(W>1e-10):])
plt.clf()
plt.plot(K.T)
plt.plot(K.T,'o')
print("Shape of K: " + str(K.shape))

#FIXME
#Next step: Figure out now these kernel outputs relate to visibilities
#and on-sky coordinates for simple telescope arrangements.
