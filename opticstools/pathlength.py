from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.optimize as op
import pdb

def polynomial_pathlength(x,p_d):
    """Integrand for  a path y(x) defined by a polynomial
    
    The line element ds is given by:
    
    ds      = sqrt(dx**2 + dy**2)
    ds/dx   = sqrt(1 + dy/dx**2) 
    
    Parameters
    ----------
    x: float
        distance to integrate
    p_d: np.poly1d
        Derivative of a np.poly1d object, e.g. np.polyder(np.poly1d([params]))
    """
    return np.sqrt(1 + p_d(x)**2)

def bend_radius(x, p_d, p_d_d):
    """Find the bend radius of a curve 
    
    Parameters
    ----------
    x: float
        distance to integrate
    p_d: np.poly1d
        Derivative of a np.poly1d object, e.g. np.polyder(np.poly1d([params]))
    p_d_d: np.poly1d
        Second derivative of a np.poly1d object, e.g. np.polyder(np.poly1d([params]))
    """
    return (1+p_d(x)**2.0)**(3.0/2.0)/np.abs(p_d_d(x))

def solve_pathlength_func(p,edge_y,edge_dydx,D,L0):
    """A helper function for solve_pathlength
    
    Parameters
    ----------
    p: array-like
        Polynomial coefficients.
    edge_y: array(2)
        Edge y coordinates at start and finish.
    edge_dydx: array(2)
        Edge gradient at start and finish.
    D: float
        Length in x coordinate.
    L0: float
        Waveguide length.
        
    Returns
    -------
    Residuals in y co-ord, dydx and length.
    """
    pathlength_poly = np.poly1d(p)
    p_d = np.polyder(pathlength_poly)
    L = integrate.quad(polynomial_pathlength,0,D,args=(p_d))
    return [pathlength_poly(0)-edge_y[0],pathlength_poly(D)-edge_y[1],p_d(0)-edge_dydx[1],p_d(D)-edge_dydx[1],L[0]-L0]


def solve_pathlength_func_bendrad(p,edge_y,edge_dydx,D,L0,BendRad,n_grid=100):
    """A helper function for solve_pathlength where a fixed minimum bend radius 
    is desired.
    
    Parameters
    ----------
    BendRad: float
        Enforced minimum bend radius.
    """
    pathlength_poly = np.poly1d(p)
    p_d = np.polyder(pathlength_poly)
    p_d_d = np.polyder(p_d)
    L = integrate.quad(polynomial_pathlength,0,D,args=(p_d))
    
    #Find the minimum radius of curvature along the curve.
    #Given the the curve is non-differentiable, try brute-force with n_grid points 
    #along the curve.
    x_vect = np.meshgrid(0,D,n_grid)
    
    a = bend_radius(x_vect, p_d, p_d_d)
    
    #Find the minimum radius of curvature.
    SmallCurve = np.min(a)
    
    retpar = [pathlength_poly(0)-edge_y[0], pathlength_poly(D)-edge_y[1],p_d(0)-edge_dydx[0],p_d(D)-edge_dydx[1], L[0]-L0,SmallCurve-BendRad]
    #print(retpar); pdb.set_trace()
    return retpar
    
def solve_pathlength_bendrad(edge_y=[0,0], edge_dydx=[0,0], D=1.0, L=1.2, BendRad=100.0, init_par=None):
    """Solve for polynomial coefficients for a spline of fixed
    pathlength between two points.
    
    Parameters
    ----------
    edge_y: [float,float]
        Beginning and end waveguide co-ordinate.
        
    edge_dydx: [float,float]
        Beginning and end waveguide derivatives (angles).
        
    D: float
        Length of the array in the x co-ordinate direction
        
    L: float
        Pathlength of the waveguide
        To be calculated
        
    BendRad: float
        Minimum BendRaius Allowed by the waveguide
        
    Notes
    -----
    Multiple solutions are possible.
    """
        
    #From https://en.wikipedia.org/wiki/Cubic_Hermite_spline. This gives us an
    #initial spline that fits.
    params = np.array([2,-3,0,1])*edge_y[0] + np.array([1,-2,1,0]) *D*edge_dydx[0] + \
             np.array([-2,3,0,0])*edge_y[1] + np.array([1,-1,0,0])*D*edge_dydx[1]    
    params = params.astype(float) #In case edge_y etc were integers.

    params /= [D**3.0,D**2.0,D**1.0,1.0]
    #Initialize the spline direction to one side.
    init_params = np.append([-0.01/D**5/1e3,0.01/D**4],params)
    if init_par != None:
        init_params = init_par
    final_params,infodict,ier,mesg = op.fsolve(solve_pathlength_func_bendrad, init_params,args=(edge_y,edge_dydx,D,L,BendRad),full_output=True)
    if ier != 1:
        print(mesg)
        raise UserWarning
        
    #print(init_params)
    #print(final_params)
    return np.poly1d(final_params)    
    
def solve_pathlength(edge_y=[0,0], edge_dydx=[0,0], D=1.0, L=1.2):
    """Solve for polynomial coefficients for a spline of fixed
    pathlength between two points.
    
    Parameters
    ----------
    edge_y: [float,float]
        Beginning and end waveguide co-ordinate.
        
    edge_dydx: [float,float]
        Beginning and end waveguide derivatives.
        
    D: float
        Length of the array in the x co-ordinate direction
        
    L: float
        Pathlength of the waveguide"""
    
    #From https://en.wikipedia.org/wiki/Cubic_Hermite_spline. This gives us an
    #initial spline that fits.
    params = np.array([2,-3,0,1])*edge_y[0] + np.array([1,-2,1,0]) *D*edge_dydx[0] + \
             np.array([-2,3,0,0])*edge_y[1] + np.array([1,-1,0,0])*D*edge_dydx[1]
    params = params.astype(float) #In case edge_y etc were integers.

    params /= [D**3,D**2,D**1,1]
    #Initialize the spline direction to one side.
    init_params = np.append(1.0/D**3/1e3,params)
    print(init_params)
    final_params,infodict,ier,mesg = op.fsolve(solve_pathlength_func, init_params,args=(edge_y,edge_dydx,D,L),full_output=True)
    if ier != 1:
        print(mesg)
        raise UserWarning
    return np.poly1d(final_params)
    