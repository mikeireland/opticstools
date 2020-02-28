import numpy as np
import matplotlib.pyplot as plt

def raised_sine(x0,y0,x1,y1,N=100):
    x_norm = np.linspace(0,1,N)
    xs = x0 + (x1-x0)*x_norm
    y_norm = (x_norm*2*np.pi - np.sin(x_norm*2*np.pi))/2/np.pi
    ys = y0 + y_norm*(y1-y0)
    return xs, ys
    
def raised_sine_vert(x0,y0,x1,y1,N=100):
    ys,xs = raised_sine(y0,x0,y1,x1,N)
    return xs, ys
    
def quarter(x0,y0,x1,y1,N=100):
    theta = np.linspace(0,np.pi/2,N)
    xs = x0 + (x1-x0)*np.sin(theta)
    ys = y0 + (y1-y0)*(1-np.cos(theta))
    return xs, ys

plt.clf()
  
vert_sep = 1.0  
output_ys = 24 - vert_sep*np.arange(4)
output_x = 25
straight_length = 0.5
straight_frac = 1.4
coupler_length = 1.0
coupler_long = 4.0
coupler_dsep = 0.05
bend_rad = 6.0
pd_sep = 6.0
nr_pos = 12.5 + np.array([-4,0,1,3])*0.5

x0 = output_x - straight_length
x1 = x0 - coupler_length
for ix, o in enumerate(output_ys):
    plt.plot([output_x, output_x-straight_length], [o,o],'r') 
    plt.plot(*raised_sine(x0,o,x1,o+coupler_dsep), 'r')
    plt.plot([x1,x1-straight_frac*bend_rad],[o+coupler_dsep,o+coupler_dsep],'r')
    plt.plot(*raised_sine(x0,o,x1,o-coupler_dsep), 'r')
    x2 = x1-vert_sep*(3-ix)
    plt.plot([x1,x2],[o-coupler_dsep,o-coupler_dsep],'r')
    x3 = x2-bend_rad
    y3 = o-coupler_dsep-bend_rad
    plt.plot(*quarter(x2,o-coupler_dsep, x3, y3),'r')
    x4 = x3-coupler_dsep
    y4 = y3-coupler_length
    plt.plot(*raised_sine_vert(x3,y3, x4, y4),'r')
    plt.plot(*raised_sine_vert(x3,y3, x3+coupler_dsep, y3-coupler_length),'r')
    plt.plot(*quarter(x4-bend_rad,y4-bend_rad,x4,y4),'r')
    y_pd = 1+(3-ix)*pd_sep
    plt.plot(*raised_sine(x4-bend_rad,y4-bend_rad,0,y_pd),'r')
    
    #Now, lets follow the path lengths to the microlens array.
    x4 = x3+coupler_dsep
    plt.plot([x4,x4],[y4,y4-straight_frac*bend_rad],'r')
    plt.plot(*raised_sine_vert(x4,y4-straight_frac*bend_rad,nr_pos[ix],0),'r')
    
    #Finally, lets add in the couplers heading to the laser diodes.
    x2 = x1-straight_frac*bend_rad
    y2 = o+coupler_dsep
    y3 = y2-vert_sep/2+vert_sep*(ix % 2)
    plt.plot(*raised_sine(x2,y2,x2-coupler_long,y3),'r')
    y4 = np.mean(output_ys)
    if (ix % 2):
        x4 = x2-2*coupler_long
        plt.plot(*raised_sine(x2-coupler_long,y3,x4,y4),'r')

#Finally, the common waveguides.
plt.plot(*raised_sine(x4,y4,x4-coupler_length, y4+coupler_dsep),'r')
plt.plot(*raised_sine(x4,y4,x4-coupler_length, y4-coupler_dsep),'r')
x5 = x4-coupler_length-2
y5 = y4-coupler_dsep
plt.plot([x4-coupler_length,x5], [y5,y5],'r')
plt.plot(*raised_sine(x5,y5,0,23),'r')
x5 = x4-coupler_length
y5 = y4+coupler_dsep
plt.plot(*raised_sine(x5,y5,0,24),'r')

plt.plot([0,0],[0,25], 'b:')
plt.plot([0,25],[25,25], 'b:')
plt.plot([25,25],[25,0], 'b:')
plt.plot([25,0],[0,0], 'b:')
plt.xlabel('Position (mm)')
plt.ylabel('Position (mm)')
plt.text(1,21,'A')
plt.text(1,5,'B')
plt.text(23,18,'C')
plt.text(8,1,'D')