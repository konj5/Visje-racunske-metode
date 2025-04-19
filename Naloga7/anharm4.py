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
from numba import jit, njit, prange
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))


#Bring unto me Death, for Death is sweeter than this ******* ****!

def single(beta, lamb, ratio, iters):
    M = int(beta * ratio)
    epsilon = 0.3

    state = np.random.normal(0,1,M)

    for i in range(iters):
        while(True):
            j = np.random.randint(0,M)
            dq = np.random.normal(0,epsilon)
            
            E0 = -ratio/2 * ((state[(j-1)%M]-state[j])**2 + (state[(j+1)%M]-state[j])**2) - (state[j]**2/2 + lamb * state[j]**4)/ratio
            E1 = -ratio/2 * ((state[(j-1)%M]-state[j]-dq)**2 + (state[(j+1)%M]-state[j]-dq)**2) - ((state[j]+dq)**2/2 + lamb * (state[j]+dq)**4)/ratio
            dE = E1 - E0

            if dE < 0 or np.random.rand() < np.exp(dE):
                state[j] += dq
                break

    return state

def expected_V(state, lamb):
    M = len(state)

    V = 0
    for j in range(M):
        V += state[j]**2/2 + lamb * state[j]**4

    return V / (M)

def expected_E(state, lamb, beta):
    M = len(state)

    E = M/(2*beta)
    for j in range(M):
        E += M/(2*beta**2) * np.sum((state[(j+1)%M] - state[j])**2)
        E += 1/(M) * np.sum((state[j])**2)/2
        E += 1/(M) * lamb *  np.sum((state[j])**4)

    return E

lamb = 0
ratio = 100
iters = 1000

betas = np.linspace(0.1, 100, 100)
E, V = np.zeros(betas.shape), np.zeros(betas.shape)
for i in trange(len(betas)):
    beta = betas[i]
    state = single(beta, lamb, ratio, iters)
    E[i] = expected_E(state, lamb, beta)
    V[i] = expected_V(state, lamb)
    
plt.plot(betas, E, label = "E")
plt.plot(betas, V, label = "V")
plt.plot(betas, E-V, label = "T")

plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.show()



