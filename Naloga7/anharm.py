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

"""def energy(state, lamb, beta, M):
    E = 0
    for j in range(0,M):
        E += M/(2*beta)*(state[(j+1)%M]-state[(j-1)%M])**2

        E += beta/M * (1/2 * state[j]**2 + 0 if lamb == 0 else lamb * state[j]**4)
    return E"""

@njit(nopython = True)
def Pij(n1,n2, lamb, beta, M):
    E = 0
    E = E + M/(2*beta)*(n1-n2)**2

    E = E + beta/M * (1/2 * np.dot(n1,n1) + 0 if lamb == 0 else lamb * np.dot(n1,n1)**2)
    E = np.real(np.sum(E))
    return E
    print(E)
    print(np.exp(-E))
    print("==============")
    return np.exp(-E)


@njit(nopython = True)
def move(state, lamb, beta, M, N):
    sig = 0.1
    while(True):
        i = np.random.choice(M)
        diff = np.random.normal(0,sig, N)
        newstate = state.copy()
        newstate[:,i] += diff

        prob = Pij(state[:,(i-1)%M], newstate[:,i], lamb, beta, M) * Pij(newstate[:,i], state[:,(i+1)%M], lamb, beta, M) / (Pij(state[:,(i-1)%M], state[:,i], lamb, beta, M) * Pij(state[:,i], state[:,(i+1)%M], lamb, beta, M))

        if prob >= 1 or np.random.rand() < prob:
            return newstate

@njit(nopython = True)
def anharm(iters, lamb, beta, M, N):
    #state = np.ones((N,M), dtype=np.complex128)
    state = np.random.random((N,M))*10
    for i in range(iters):
        state = move(state, lamb, beta, M, N)

    return state

@njit(nopython = True)
def expected_V(state, lamb, beta, M, N):
    """print(state)
    print(beta)
    V = state**2 / 2 + lamb * state**4
    print(np.sum(V))
    print(np.size(V))
    print(np.average(V))
    print(1/0)
    return np.average(V)"""

    V = 0
    for i in range(N):
        for j in range(M):
            V += state[i,j]**2/2 + lamb * state[i,j]**4

    return V / (M*N)

@njit(nopython = True)
def expected_E(state, lamb, beta, M, N):
    E = M/(2*beta)

    for j in range(M):
        E += M/(2*beta**2*N) * np.sum((state[:,(j+1)%M] - state[:,j])**2)
        E += 1/(M*N) * np.sum((state[:,j])**2)/2
        E += 1/(M*N) * lamb *  np.sum((state[:,j])**4)

    return E

@njit(nopython = True)
def process(state, lamb, beta, M, N):
    E = expected_E(state, lamb, beta, M, N)
    V = expected_V(state, lamb, beta, M, N)
    # E, V, T
    """    print(E)
    print(V)
    print(E-V)
    print("------------")"""
    return E, V, E-V


#######
@njit(nopython = True)
def move3(state, lamb, beta, ratio, M, epsilon):
    while(True):
        i = np.random.randint(0,M)
        dq = np.random.normal(0,epsilon)
        
        dE = 0
        dE += ratio/2 * ((state[(i+1)%M]-state[i]-dq)**2 - (state[(i+1)%M]-state[i])**2)
        dE += ratio/2 * ((state[(i-1)%M]-state[i]-dq)**2 - (state[(i-1)%M]-state[i])**2)
        dE += 1/ratio * ((state[i]+dq)**2-(state[i])**2)/2
        dE += 1/ratio * lamb * ((state[i]+dq)**4-(state[i])**4)

        if dE < 0 or np.exp(-beta*dE) > np.random.random():
            state[i] += dq
            return state

@njit(nopython = True)
def attempt3(iters, lamb, beta, ratio):
    M=int(beta*ratio)
    if M<=2:
        raise ValueError(f"Bad ratio it seems! M = {M}, beta = {beta}, ratio = {ratio}")
    state = np.random.normal(0,10,M)

    epsilon = 0.1
    counter = 0
    for i in range(iters):
        state = move3(state, lamb, beta, ratio, M, epsilon)
        counter += 1
        #epsilon = epsilon - epsilon *  (counter/(i+1)-0.5)/(M if beta >1 else M*100)  #TOLE NARAŠČA!!!!
        #print(epsilon)

    #Energy
    E = 2/ratio
    for j in range(M):
        E -= M/(2*beta**2) * (state[(j+1)%M] - state[j])**2
        E += 1/M / 2 * (state[j])**2
        E += 1/M * lamb *  (state[j])**4

    #Potential
    V = np.average(state**2 / 2 + lamb * state**4)

    #print(beta)
    #print(state)

    return E, V, E-V



"""state = anharm(1000, 0, 1000,40,40)
print(state)
print(expected_V(state, 0))
print(expected_E(state, 0, 1000, 40, 40))

input()"""


@njit(parallel=True)
def par():
    iters = 10**3
    lamb = 0
    N = 10
    #M = 40

    betas = np.linspace(0.1,1000, 100)

    #np.random.shuffle(betas)
    data = np.zeros((3,len(betas)), dtype=np.complex128)
    done = np.zeros(len(betas), dtype=np.bool)

    for i in prange(len(betas)):
        beta = betas[i]
        M = N + 2*int(beta)
        state = anharm(iters, lamb, beta, M, N)

        data[:,i] = process(state, lamb, beta, M, N)
        done[i] = True

        #print(np.sum(done), end="")
        #print("/", end="")
        #print(len(done), end="")
        #print(",  ", end="")
        #print(np.sum(done)/len(done), end="")
        #print(",  i=", end="")
        """print(np.sum(done))
        print(len(done))
        print(i)
        print("---------")"""

    return betas, data
    
betas, data = par()

plt.plot(betas, data[0,:], label = "E")
plt.plot(betas, data[1,:], label = "V")
plt.plot(betas, data[2,:], label = "T")

plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.show()


"""@njit(parallel=True)
def par():
    iters = 10**7
    lamb = 0
    ratio = 10

    #M = 40

    betas = np.linspace(10,1000, 100)


    data = np.zeros((3,len(betas)), dtype=np.float32)
    done = np.zeros(len(betas), dtype=np.bool)

    for i in prange(len(betas)):
        beta = betas[i]

        data[:,i] = attempt3(iters, lamb, beta, ratio)
        done[i] = True

        print(np.sum(done))
        print(len(done))
        print(i)
        print("---------")

    return betas, data
    
betas, data = par()

plt.plot(betas, data[0,:], label = "E")
plt.plot(betas, data[1,:], label = "V")
plt.plot(betas, data[2,:], label = "T")

plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.show()"""
