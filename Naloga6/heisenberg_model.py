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
iters = 4000
J = 1
h = 0
beta = 10
T = 1/beta

states, energies = model_2d.ising(N,iters,J,h,T)

plt.plot(energies)
plt.show()

plt.imshow(states[:,:,-1], cmap=plt.get_cmap("binary"))

plt.show()"""


"""#### Solutions for different temps
N = 25
iters = 10000

h = 0

Ts = [0.1, 1, 2, 2.269, 3, 5]
datas = np.zeros((len(Ts), N, N), dtype=np.byte)
fig, axs = plt.subplots(2,len(Ts))

for i in trange(0, len(Ts)):
    T = Ts[i]

    J = 1
    states, energies = model_2d.ising(N,iters,J,h,T)
    if np.sum(states[:,:,-1]*2-1) < 0: states[:,:,-1] = -states[:,:,-1]
    axs[0,i].imshow(states[:,:,-1], cmap=plt.get_cmap("binary"))
    axs[0,i].set_xticks([])
    axs[0,i].set_yticks([])

    J = -1
    states, energies = model_2d.ising(N,iters,J,h,T)
    axs[1,i].imshow(states[:,:,-1], cmap=plt.get_cmap("binary"))
    axs[1,i].set_xticks([])
    axs[1,i].set_yticks([])

plt.show()"""



#Susceptibilnost

betas = np.linspace(0.01, 10, 10)

N = 25
iters = 10000
J = 1
h = 0

magss = np.zeros_like(betas)
susceps = np.zeros_like(betas)
sptopls = np.zeros_like(betas)

startstate = None

for i in trange(0, len(betas)):
    beta = betas[-i-1]
    T = 1/beta
    states, energies = model_2d.ising(N,iters,J,h,T,startstate)

    #Try if this works or not!
    startstate = states[:,:,-1]

    mags = model_2d.Ising_M(states)
    sus = model_2d.Ising_sus(mags, beta)
    sph = model_2d.Ising_heat(energies, beta)

    print(mags[-1])

    magss[-i-1] = mags[-1]
    susceps[-i-1] = sus
    sptopls[-i-1] = sph
    
fig, axs = plt.subplots(1,3)

axs[0].plot(betas, mags)
axs[1].plot(betas, sus)
axs[2].plot(betas, sph)

plt.show()


