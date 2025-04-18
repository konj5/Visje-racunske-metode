import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import scipy.integrate
import scipy.interpolate
import scipy.special
from tqdm import tqdm, trange
import re, sys
from matplotlib.animation import FuncAnimation
import time
import numba
from numba import jit, njit
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))


import model_2d

"""N = 25
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
# force the first color entry to be grey
#cmaplist[0] = (.5, .5, .5, 1.0)

# create the new map
cmap = colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, q)

plt.imshow(states[:,:,-1], cmap=cmap)
plt.colorbar()

plt.show()
"""


"""#### Solutions for different temps
N = 100
iters = 100000
J = 1

Ts = [0.1, 1, 2, 2.269, 3, 5]
datas = np.zeros((len(Ts), N, N), dtype=np.byte)
fig, axs = plt.subplots(6,len(Ts))


for i in trange(0, len(Ts)):
    T = Ts[i]

    q = 2
    cmap = plt.cm.gray  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i/q) for i in range(q)]
    cmap = colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, q)
    states, energies = model_2d.potts(N,iters,J,T,q)
    axs[0,i].imshow(states[:,:,-1], cmap=cmap) 
    axs[0,i].set_xticks([])
    axs[0,i].set_yticks([])

    q = 3
    cmap = plt.cm.gray  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i/q) for i in range(q)]
    cmap = colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, q)
    states, energies = model_2d.potts(N,iters,J,T,q)
    axs[1,i].imshow(states[:,:,-1], cmap=cmap) 
    axs[1,i].set_xticks([])
    axs[1,i].set_yticks([])

    q = 4
    cmap = plt.cm.gray  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i/q) for i in range(q)]
    cmap = colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, q)
    states, energies = model_2d.potts(N,iters,J,T,q)
    axs[2,i].imshow(states[:,:,-1], cmap=cmap) 
    axs[2,i].set_xticks([])
    axs[2,i].set_yticks([])

    q = 5
    cmap = plt.cm.gray  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i/q) for i in range(q)]
    cmap = colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, q)
    states, energies = model_2d.potts(N,iters,J,T,q)
    axs[3,i].imshow(states[:,:,-1], cmap=cmap) 
    axs[3,i].set_xticks([])
    axs[3,i].set_yticks([])

    q = 6
    cmap = plt.cm.gray  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i/q) for i in range(q)]
    cmap = colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, q)
    states, energies = model_2d.potts(N,iters,J,T,q)
    axs[4,i].imshow(states[:,:,-1], cmap=cmap) 
    axs[4,i].set_xticks([])
    axs[4,i].set_yticks([])

    q = 7
    cmap = plt.cm.gray  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i/q) for i in range(q)]
    cmap = colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, q)
    states, energies = model_2d.potts(N,iters,J,T,q)
    axs[5,i].imshow(states[:,:,-1], cmap=cmap) 
    axs[5,i].set_xticks([])
    axs[5,i].set_yticks([])

    axs[0,i].set_title(f"$T = {T}$")

axs[0,0].set_ylabel("$q=2$")
axs[1,0].set_ylabel("$q=3$")
axs[2,0].set_ylabel("$q=4$")
axs[3,0].set_ylabel("$q=5$")
axs[4,0].set_ylabel("$q=6$")
axs[5,0].set_ylabel("$q=7$")

plt.show()"""


#Susceptibilnost

betas = np.linspace(1/5, 1/0.5, 500)

N = 50
iters = 10000
J = 1
h = 0
fig, axs = plt.subplots(1,3)

for q in trange(2,6):

    magss = np.zeros_like(betas)
    susceps = np.zeros_like(betas)
    sptopls = np.zeros_like(betas)

    startstate = None

    for i in trange(0, len(betas)):
        beta = betas[i]
        T = 1/beta
        states, energies = model_2d.potts(N,iters,J,T,q,startstate)

        #Try if this works or not!
        startstate = states[:,:,-1]
        #print(startstate)

        mags = np.abs(model_2d.Potts_M(states[:,:,-1],q))
        sus = np.abs(model_2d.Potts_sus(states[:,:,-1], beta, q))
        sph = np.abs(model_2d.Potts_heat(states[:,:,-1], beta, J))

        magss[i] = mags
        susceps[i] = sus
        sptopls[i] = sph


    axs[0].plot(1/betas, np.abs(magss), marker = "+", linestyle='None')
    axs[1].plot(1/betas, susceps, marker = "+", linestyle='None')
    axs[2].plot(1/betas, sptopls, marker = "+", linestyle='None', label = f"$q = {q}$")


axs[0].set_ylabel("Absolutna Magnetizacija")
axs[0].set_xlabel("T")

axs[1].set_ylabel("Susceptibilnost")
axs[1].set_xlabel("T")

axs[2].set_ylabel("Specifična toplota")
axs[2].set_xlabel("T")

plt.tight_layout()
plt.legend()
plt.show()