import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import scipy.integrate
import scipy.interpolate
import scipy.special
from tqdm import tqdm
import re, sys
from matplotlib.animation import FuncAnimation
import time
import numba
from numba import jit
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

from decomp import integrate, integrate_rk45, decomp11, decomp22, decomp33, decomp44, decomp45



startstate = [0,0.5,1,0]

ts, states = integrate(startstate=startstate, dt = 0.1, tmax=10, lamb=1, decomp=decomp11)
#ts, states = integrate_rk45(startstate=startstate, dt = 0.1, tmax=1, lamb=0)


plt.plot(states[0,:], states[1,:])
plt.axis("equal")
plt.show()



