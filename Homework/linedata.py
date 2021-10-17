#!/usr/bin/python
# linedata -- generate points on a line with uncorrelated Gaussian errors
# linedata xmin xmax npoints slope intercept sigma seed filename
#   xmin,xmax = range of x values
#   npoints = number of points, evenly distributed in xmin,xmax
#   slope, intercept = slope and intercept of line
#   sigma = scatter about line (constant from point to point)
#   seed = random number seed
#   filename = name of output file

import numpy as np
import sys

xmin=float(sys.argv[1])
xmax=float(sys.argv[2])
npoints=int(sys.argv[3])
a=float(sys.argv[4])
b=float(sys.argv[5])
sigma=float(sys.argv[6])
seed=int(sys.argv[7])
filename=sys.argv[8]

x=np.linspace(xmin,xmax,num=npoints)
y=a*x+b

# generate errors with diagonal covariance matrix, add to y values, save
errors=sigma*np.ones(npoints)
mu=np.zeros(npoints)
cov=np.diag(errors)
dy=np.random.multivariate_normal(mu,cov)    # one realization of errors
y=y+dy

np.savetxt(filename,np.transpose([x,y,errors]),'%9.5f')
