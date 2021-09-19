#!/usr/bin/python
"""
cosmodist_subs.py -- functions for computing comoving and 
                     angular diameter distances

The main function is cosmodist, which in turn calls Simpson's rule
and trapzd integration routines.  
The function hratio, which is the integrand for cosmodist, may
also be of use on its own.
"""

import numpy as np	
nstepmax=2e8    	# maximum number of allowed integration steps
tolerance=1.e-6		# require convergence to this fractional error

# using upper case variable names for constants
SPEED_OF_LIGHT=2.997925e5	# km/s units
OMEGA_GAMMA_HSQR=1.710e-5	# photon radiation density (Omega_\gamma h^2) 
OMEGA_NU_HSQR=2.473e-5		# neutrino energy density (Omega_\nu h^2) 
				      # (assumes 3 massless neutrinos)

def cosmodist(redshift,h0,omega_m,omega_k=0.0,w=-1.0,omega_r_h2=1.710e-5,omega_nu_h2=2.473e-5):
    """
    cosmodist(redshift,h0,omega_m,omega_k=0.0,w=-1.0)
    redshift = redshift at which distances are to be evaluated
    h0 = present day Hubble constant in km/s/Mpc
    omega_m = matter density parameter at z=0
    omega_k = curvature parameter, default is 0.0 (flat universe)
    w = equation-of-state parameter, default is -1.0 (cosmological constant)

   Returns [d_c(z), D_C(z), D_M(z)]
      d_C(z) = \int_0^z H_0/H(z') dz' 
      D_C(z) = (c/H_0) d_c(z)
      D_M(z) = D_C(z) for Omega_k=0
               (c/H_0)*sin(sqrt(-Omega_k)*d_c)/sqrt(-Omega_k)  otherwise
	            [using sin(ix) = i sinh(x)]
      d_c is dimensionless, D_C and D_M are in physical Mpc for specified H0
      Multiplying D_M by (1+z) gives the luminosity distance D_L.
    """

    # compute radiation energy density parameter
    h=h0/100.
    # omega_r=(OMEGA_GAMMA_HSQR+OMEGA_NU_HSQR)/(h**2)	
    omega_r=(omega_r_h2 + omega_nu_h2)/(h**2)	

    # Evaluate integral for d_C(z)
    params=[omega_m,omega_r,omega_k,w]
    [dc, nstep]=simpson_driver(hratio,params,0.0,redshift,tolerance,nstepmax)
 
    if (omega_k==0.0):
        dm = dc
    else:
        if (omega_k>0.0):
            dm=np.sinh(np.sqrt(omega_k)*dc)/np.sqrt(omega_k)
        else:
            dm=np.sin(np.sqrt(-omega_k)*dc)/np.sqrt(-omega_k)

    dcphys=dc*SPEED_OF_LIGHT/h0
    dmphys=dm*SPEED_OF_LIGHT/h0
    return([dc,dcphys,dmphys])

def hratio(z,params):
    """
    Returns the ratio H0/H(z).
    The params list contains constants that are independent of z:
      params = [omega_m, omega_r, omega_k, w]
          omega_m = matter density parameter
	  omega_r = radiation energy density parameter (including neutrinos)
	  omega_k = curvature parameter
	  w = equation-of-state parameter
    """

    [omega_m,omega_r,omega_k,w]=params
    omega_de=1.-omega_m-omega_r-omega_k		# DE density parameter
    zp1=1.+z

    # ratio of DE density at redshift z to value at z=0
    if (w != -1.0):
        de_ratio=zp1**(3.*(1.+w))
    else:
        de_ratio=1.0

    eratio=omega_m*zp1**3 + omega_r*zp1**4 + omega_k*zp1**2 + omega_de*de_ratio
    return(1./np.sqrt(eratio))

def simpson_driver(func,params,a,b,tolerance,nstepmax):
    """
    Integrate a function func() using Simpson's rule implemented
      via successive applications of trapezoidal rule
    params = list of parameters to function
    a = lower limit of integration
    b = upper limit of integration
    tolerance = fractional convergence required for integral
    nstepmax = maximum number of steps allowed

    Number of steps starts at 4 and doubles until convergence or nstep>nstepmax
    """

    nstep=4

    # compute integral via trapezoidal rule for initial value of nstep
    int_trap=trapzd(func,params,a,b,nstep)

    # Note that int_trap and oldint_trap refer to trapezoidal rule evaluations,
    # integral and oldint to Simpson rule evaluations
    oldint=0.0    
    oldint_trap=int_trap
    integral=int_trap

    while ((np.fabs(oldint/integral-1.0) > tolerance) and (2*nstep<nstepmax)):
        oldint=integral
        nstep*=2
        int_trap=trapzd(func,params,a,b,nstep)
	# Here is the Numerical Recipes trick for using two successive trapzd
	# evaluations to achieve a Simpson's rule evaluation
        integral=(4*int_trap-oldint_trap)/3.
        oldint_trap=int_trap

    if (np.fabs(oldint/integral-1.0) > tolerance):
        print("SimpsonDriver Warning, fractional convergence is only ", \
	  np.fabs(oldint/integral-1.0))
    return [integral, nstep]

def trapzd(func,params,a,b,nstep):
    """ 
    Evaluate [\int_a^b func(x) dx] using trapezoidal rule with nstep steps
    The parameter list params is passed to the function
    """
    x=np.linspace(a,b,nstep+1)		# note nstep+1 !
    hstep=(b-a)/nstep
    y=func(x,params)*hstep
    # subtract half of first and last steps to get to trapezoidal rule
    return (np.sum(y)-0.5*hstep*(func(b,params)+func(a,params)))
