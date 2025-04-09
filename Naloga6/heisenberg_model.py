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
from numba import jit, njit
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))


import model_2d

N = 25
iters = 4000
J = 1
h = 0
beta = 10
T = 1/beta

states, energies = model_2d.ising(N,iters,J,h,T)

plt.plot(energies)
plt.show()

plt.imshow(states[:,:,-1], cmap=cm.get_cmap("binary"))

plt.show()