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
iters = 10000
J = -1
q = 10
beta = 0.1
T = 1/beta

states, energies = model_2d.potts(N,iters,J,T,q)

plt.plot(energies)
plt.show()

cmap = plt.cm.gray  # define the colormap
# extract all colors from the .jet map
cmaplist = [cmap(i/q) for i in range(q)]
"""# force the first color entry to be grey
cmaplist[0] = (.5, .5, .5, 1.0)"""

# create the new map
cmap = colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, q)

plt.imshow(states[:,:,-1], cmap=cmap)
plt.colorbar()

plt.show()