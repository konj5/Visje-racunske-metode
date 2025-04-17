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

"""def energy(state, lamb, beta, M):
    E = 0
    for j in range(0,M):
        E += M/(2*beta)*(state[(j+1)%M]-state[(j-1)%M])**2

        E += beta/M * (1/2 * state[j]**2 + 0 if lamb == 0 else lamb * state[j]**4)
    return E"""

def Pij(n1,n2, lamb, beta, M):
    E = 0
    E += M/(2*beta)*(n1-n2)**2

    E += beta/M * (1/2 * n1.dot(n1) + 0 if lamb == 0 else lamb * n1.dot(n1)**2)
    return np.sum(E)

def move(state, lamb, beta, M, N):
    sig = 0.1
    while(True):
        i = np.random.choice(M)
        diff = np.random.normal(0,sig, N)
        newstate = state.copy()
        newstate[:,i] += diff

        prob = Pij(state[(i-1)%M], newstate[i], lamb, beta, M) * Pij(newstate[i], state[(i+1)%M], lamb, beta, M) / (Pij(state[(i-1)%M], state[i], lamb, beta, M) * Pij(state[i], state[(i+1)%M], lamb, beta, M))

        if prob >= 1 or np.random.rand() < prob:
            return newstate


def anharm(iters, lamb, beta, M, N):
    state = np.ones((N,M), dtype=np.complex128)
    for i in trange(iters):
        state = move(state, lamb, beta, M, N)

    return state

def expected_V(state, lamb):
    V = state**2 / 2 + lamb * state**4 / 2
    return np.average(V)

def expected_E(state, lamb, beta, M, N):
    E = M/(2*beta)

    for j in range(M):
        E += M/(2*beta**2*N) * np.sum((state[:,(j+1)%M] - state[:,j])**2)
        E += 1/(M*N) * np.sum((state[:,j])**2)/2
        E += 1/(M*N) * lamb *  np.sum((state[:,j])**4)

    return E



state = anharm(1000, 0, 1000,40,40)
print(state)
print(expected_V(state, 0))
print(expected_E(state, 0, 1000, 40, 40))
