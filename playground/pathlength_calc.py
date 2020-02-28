"""
Lets numerically evaluate the pathlength in Hansen and Ireland
"""
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

#Distance chief to deputy
l = 10
#Separation B to D and A to C
h = 0.3
#Offset of deputy 1 in the vertical direction
delta = np.linspace(-.01,.01,100)

#Lets compute the first term:
d_A = np.sqrt((h/2 + delta)**2 + l**2)
d_C = np.sqrt((h/2 - delta)**2 + l**2)
delta_est = (d_A - d_C) * (d_A + d_C)/2/h
plt.figure(1)
plt.clf()
plt.plot(delta, delta_est-delta)
plt.title('First Term')
plt.xlabel('Delta')
plt.ylabel('Error')

#Now lets compute the difference between separation vector magnitudes
l2 = l #Deputy 2 at same distance as deputy 1
d_B = np.sqrt((h/2)**2 + l2**2)
d_D = np.sqrt((h/2)**2 + l2**2)
Lambda_expr = 0.25*( (d_B-d_A) + (d_D-d_C) + (d_B-d_C) + (d_D-d_A))
r1_minus_r2 = l - l2
plt.figure(2)
plt.clf()
plt.title('3rd term')
plt.xlabel('Delta')
plt.ylabel('Error')
plt.plot(delta, Lambda_expr - r1_minus_r2)