"""Some example calculations for a CHARA beam combiner, to be fabricated in 
Chalcogenide by Harry."""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import opticstools as ot
plt.ion()

L=1585
D=1500
BendRad=180

L=800
D=700
BendRad=200

#----------------


#int variables
Position2 = [-39.42, -31.58, -7.91, 11.86, 27.65, 39.43];
 
StartOffset = -33.2;
ModeSize = 1.66*2;
Position0 = [StartOffset,StartOffset+ModeSize*2,StartOffset+ModeSize*8,StartOffset+ModeSize*13,StartOffset+ModeSize*17,StartOffset+ModeSize*20]            

WGLength = 30;
LengthCavity = 156.4285971;

#calculating the angles.
def TheAngle(x):
    angle=np.arctan(-1*(StartOffset+ModeSize*x)/LengthCavity)
    return angle

Angle1 = [TheAngle(0),TheAngle(2),TheAngle(8),TheAngle(13),TheAngle(17),TheAngle(20)]            


def WG(z):
    WG = WGLength*np.cos(np.deg2rad(Angle1[z]))
    return WG

#The position of the start of the bends
t=-400 #An extra offset if need be
offset = 30
Position1 = [t,offset+t,2*offset+t,3*offset+t,4*offset+t,5*offset+t]

plt.clf()
initp = None
for j in range(len(Position1)):
    p=ot.pathlength.solve_pathlength_bendrad(edge_y=[Position1[j],Position0[j]], edge_dydx=[0,Angle1[j]], D=D, L=L, BendRad=BendRad, init_par=initp);
    print(j)
    initp = p.c
    x=np.linspace(0.0,D,1000.0);
    plt.plot(x,p(x));

plt.show()


"""



from technologies import silicon_photonics
import sys
from picazzo3.traces.wire_wg.trace import WireWaveguideTemplate
from ipkiss.aspects.layout.visualization.matplotlib_figure import *

from ipcore.properties.predefined import *
import ipkiss3.all as i3
import ipkiss.all as i2


class SplineWG(i3.PCell):

    _name_prefix = "MyMMI" # a prefix added to the unique identifier

    class Layout(i3.LayoutView):

        edge_y = i3.Coord2Property(default = (0.0,0.0))
        edge_dydx = i3.Coord2Property(default = (0.0,0.0))
        D = i3.PositiveNumberProperty(default = 1.0)
        L = i3.PositiveNumberProperty(default = 1.2)
        
        def validate_properties(self):
            # ....
            return True

        # This method generates the layout by adding elements to "elems"
        def _generate_elements(self, elems):
             # Here, we add the layout elements based shapes

             # 1.Build the waveguides that go into the cavity.
            StartOffset = -33.2;
            ModeSize = 1.66*2;
            Position0 = [StartOffset,StartOffset+ModeSize*2,StartOffset+ModeSize*8,StartOffset+ModeSize*13,StartOffset+ModeSize*17,StartOffset+ModeSize*20]            
            #This is the inital position and angle of the waveguides
            Position2 = [-39.42, -31.58, -7.91, 11.86, 27.65, 39.43];
            
            StartOffset = -33.2;
            ModeSize = 1.66*2;
            WGLength = 30;
            LengthCavity = 156.4285971;
            
            def TheAngle(x):
                angle=(np.arctan(-1*(StartOffset+ModeSize*x)/LengthCavity))
                return angle
            Angle1 = [TheAngle(0),TheAngle(2),TheAngle(8),TheAngle(13),TheAngle(17),TheAngle(20)]            
            print(np.rad2deg(Angle1))
 

            def WG(z):
                WG = WGLength*np.cos(Angle1[z])
                return WG
            
            #Intital WG that go into the cavity
            Shape1=i2.Shape([(Position0[0],0),(Position2[0],-WG(0))])
            Shape2=i2.Shape([(Position0[1],0),(Position2[1],-WG(1))])
            Shape3=i2.Shape([(Position0[2],0),(Position2[2],-WG(2))])
            Shape4=i2.Shape([(Position0[3],0),(Position2[3],-WG(3))])
            Shape5=i2.Shape([(Position0[4],0),(Position2[4],-WG(4))])                
            Shape6=i2.Shape([(Position0[5],0),(Position2[5],-WG(5))])
            
            elems += i3.Path(layer=i3.PPLayer(i3.TECH.PROCESS.WG, i3.TECH.PURPOSE.LF.LINE),shape = Shape1, line_width=1.3)
            elems += i3.Path(layer=i3.PPLayer(i3.TECH.PROCESS.WG, i3.TECH.PURPOSE.LF.LINE),shape = Shape2, line_width=1.3)
            elems += i3.Path(layer=i3.PPLayer(i3.TECH.PROCESS.WG, i3.TECH.PURPOSE.LF.LINE),shape = Shape3, line_width=1.3)
            elems += i3.Path(layer=i3.PPLayer(i3.TECH.PROCESS.WG, i3.TECH.PURPOSE.LF.LINE),shape = Shape4, line_width=1.3)
            elems += i3.Path(layer=i3.PPLayer(i3.TECH.PROCESS.WG, i3.TECH.PURPOSE.LF.LINE),shape = Shape5, line_width=1.3)
            elems += i3.Path(layer=i3.PPLayer(i3.TECH.PROCESS.WG, i3.TECH.PURPOSE.LF.LINE),shape = Shape6, line_width=1.3)            
            #
            
            #The position of the start of the bends
            t=-400
            OffDis = 30
            Position1 = [t,t+OffDis,t+OffDis*2,t+OffDis*3,t+OffDis*4,t+OffDis*5];
            
            #Length between the two ends
            i=1500
            
            j = len(Position1)-1

            while j > -1:
                p=solve_pathlength(edge_y=[Position1[j],Position0[j]], edge_dydx=[0,Angle1[j]], D=i, BendRad=180, numb=-j+len(Position1)-1);
                print('next')
                print(j)
                x=np.linspace(0.0,i,5000.0);
               # plt.plot(x,p(x));
                j-=1
                m=0
                d=p(x)


                a = [() for k in range(len(d))]
                while m < len(p(x)):
                    a[m] = (d[m],x[m])
                    m+=1
                
                shapeSpline = i2.Shape(a,closed=False)

             
             # 2. We add the shapes to elems.

                elems += i3.Path(layer=i3.PPLayer(i3.TECH.PROCESS.WG, i3.TECH.PURPOSE.LF.LINE),shape = shapeSpline, line_width=1.3, transformation = i3.Translation((0,-i)))
                elems += i3.Path(layer=i3.PPLayer(i3.TECH.PROCESS.WG, i3.TECH.PURPOSE.LF.LINE),shape = i2.Shape(points=[(Position1[j],0),(Position1[j],-10)]), line_width=1.3, transformation = i3.Translation((0,-i)))
            
            EndWidth = 2*2*2/(3.2/2*(2.63))*LengthCavity
            Cavity = i2.Shape([(2*StartOffset,0),(-EndWidth,LengthCavity),(EndWidth,LengthCavity),(-2*StartOffset,0)],closed = True)
            elems += i3.Boundary(layer=i3.PPLayer(i3.TECH.PROCESS.WG, i3.TECH.PURPOSE.LF.LINE),shape = Cavity)
            return elems









my_WG = SplineWG(name="my_unique_spline_name")
#my_WG = i3.RoundedWaveguide(trace_template=i3.TECH.PCELLS.WG.DEFAULT)

#my_spline_rounding = i2.SplineRoundingAlgorithm(adiabatic_angles = (20.0, 20.0)),,,,,,,,,,,my_spline_rounding = SplineRoundingAlgorithm(adiabatic_angles = (20.0, 20.0))



# instantiate the ring layout view.

my_WG_layout = my_WG.Layout()

# 2. Visualizing the layout
my_WG_layout.visualize()

# 3. Export to GDSII
my_WG_layout.write_gdsii("SplineWaveguides.gds")  # fast writing to GDSII

"""
