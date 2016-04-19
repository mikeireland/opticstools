from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.optimize as op

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
    
def solve_pathlength_func(p,edge_y,edge_dydx,D,L0):
    """A helper function for solve_pathlength"""
    pathlength_poly = np.poly1d(p)
    p_d = np.polyder(pathlength_poly)
    L = integrate.quad(polynomial_pathlength,0,D,args=(p_d))
    return [pathlength_poly(0)-edge_y[0],pathlength_poly(D)-edge_y[1],p_d(0)-edge_dydx[1],p_d(D)-edge_dydx[1],L[0]-L0]
    
def solve_pathlength(edge_y=[0,0], edge_dydx=[0,0], D=1.0, L=1.2):
    """Solve for polynomial coefficients for a spline of fixed
    pathlength between two points."""
    
    #From https://en.wikipedia.org/wiki/Cubic_Hermite_spline. This gives us an
    #initial spline that fits.
    params = np.array([2,-3,0,1])*edge_y[0] + np.array([1,-2,1,0]) *D*edge_dydx[0] + \
             np.array([-2,3,0,0])*edge_y[1] + np.array([1,-1,0,0])*D*edge_dydx[1]
    params /= [D**3,D**2,D**1,1]
    #Initialize the spline direction to one side.
    init_params = np.append(1.0/D**3/1e3,params)
    print(init_params)
    final_params,infodict,ier,mesg = op.fsolve(solve_pathlength_func, init_params,args=(edge_y,edge_dydx,D,L),full_output=True)
    if ier != 1:
        print(mesg)
        raise UserWarning
    return np.poly1d(final_params)
    