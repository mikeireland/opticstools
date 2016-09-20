from __future__ import division, print_function
import opticstools as ot
import numpy as np
import matplotlib.pyplot as plt

the_V = 10
the_j = 0
the_n = 0

plt.clf()
ax = plt.gca()
m33 = ot.mode_2d(the_V,10,j=the_j,n=the_n,sz=100)
plt.imshow(m33.real, extent=[-50*0.3,50*0.3,-50*0.3,50*0.3])
circle = plt.Circle((0, 0), 10, color='k', fill=False)
ax.add_artist(circle)
plt.xlabel('X pos (microns)')
plt.ylabel('Y pos (microns)')
plt.title("n={0:d} j={1:d} V={2:5.1f}".format(the_n, the_j, the_V))