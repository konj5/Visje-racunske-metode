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

@njit(nopython = True)
def expected_V(state, lamb):
    M = len(state)

    V = 0
    for j in range(M):
        V += state[j]**2/2 + lamb * state[j]**4

    return V / M

@njit(nopython = True)
def expected_E(state, lamb, beta):
    M = len(state)

    E = M**2/(2*beta)
    for j in range(M):
        E -= M**2/(2*beta**2) * (state[(j+1)%M] - state[j])**2
        E += state[j]**2/2 + lamb * state[j]**4

    return E/M

@njit(nopython = True)
def single(beta, lamb, ratio, iters, split):
    M = int(beta * ratio)
    epsilon = 0.3

    state = np.random.normal(0,1,M)
    H = 0
    #E = expected_E(state, lamb, beta)
    n = 0

    for i in range(iters):
        while(True):
            j = np.random.randint(0,M)
            dq = np.random.normal(0,epsilon)
            
            E0 = ratio/2 * ((state[(j-1)%M]-state[j])**2 + (state[(j+1)%M]-state[j])**2) + (state[j]**2/2 + lamb * state[j]**4)/ratio
            E1 = ratio/2 * ((state[(j-1)%M]-state[j]-dq)**2 + (state[(j+1)%M]-state[j]-dq)**2) + ((state[j]+dq)**2/2 + lamb * (state[j]+dq)**4)/ratio
            dE = E1 - E0
            #print(dE)

            if dE < 0 or np.random.rand() < np.exp(-dE):
                state[j] += dq
                #E += dE                                                                 ############# dE is likely somehow wrong?????????????????
                break
        
        if i > iters * split and i % 10 == 0:
            #H += E
            H += expected_E(state, lamb, beta)
            n += 1

    return state, H/n


"""lamb = 0
ratio = 10
iters = 10**6
N = 100
split = 0.9

@njit(nopython = True, parallel = True)
def run():
    
    #betas = np.linspace(0.1, 100, 100)
    betas = 10**np.linspace(np.log10(0.1), np.log10(100), 200)

    doing = np.zeros(len(betas), dtype=np.bool)
    done = np.zeros(len(betas), dtype=np.bool)
    E, V = np.zeros(betas.shape), np.zeros(betas.shape)
    for i in prange(len(betas)):
        doing[i] = True
        for j in range(N):
            beta = betas[i]
            state, H = single(beta, lamb, ratio, iters, split)

            #state = state[-1:]

            E[i] += H/N
            V[i] += expected_V(state, lamb)/N
        
        done[i] = True

        s_done = np.sum(done)
        s_doing = np.sum(doing)

        print(s_done)
        print(len(betas))
        print(s_doing-s_done)
        print("==============")

    return betas, E, V
betas, E, V = run()"""

"""a = 4.5
fig, axs = plt.subplots(1,2, figsize = (2.3*a,1*a)); ax1,ax2 = axs


ax1.plot(betas, E, label = "E", marker = "+", linestyle = "none")
ax1.plot(betas, V, label = "V", marker = "+", linestyle = "none")
ax1.plot(betas, E-V, label = "T", marker = "+", linestyle = "none")

ax1.set_yscale("log")
ax1.set_xscale("log")
#ax1.legend()
ax1.set_xlabel("$\\beta$")



ax2.plot(betas, E, label = "E", marker = "+", linestyle = "none")
ax2.plot(betas, V, label = "V", marker = "+", linestyle = "none")
ax2.plot(betas, E-V, label = "T", marker = "+", linestyle = "none")

ax2.set_xscale("log")
ax2.legend()
ax2.set_xlabel("$\\beta$")

fig.suptitle(f"$\\lambda = {lamb}$, " + "$\\frac{M}{\\beta}=$"+f"${ratio}$, Število iteracij $= 10^{int(np.round(np.log10(iters), decimals=0))}, N = {N}$")

plt.tight_layout()
plt.show()"""


"""lamb = 0
ratio = 10
iters = 10**7
N = 5
split = 0.9

betas = np.linspace(0.1, 100, 100)
E, V = np.zeros(betas.shape), np.zeros(betas.shape)
for i in trange(len(betas)):
    for j in range(N):
        beta = betas[i]
        state, H = single(beta, lamb, ratio, iters, split)

        #state = state[-1:]

        E[i] += H/N
        V[i] += expected_V(state, lamb)/N
    
plt.plot(betas, E, label = "E")
plt.plot(betas, V, label = "V")
plt.plot(betas, E-V, label = "T")

plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.show()"""

def min_of(A,B,C):
    return min(np.min(A), np.min(B), np.min(C))

def max_of(A,B,C):
    return max(np.max(A), np.max(B), np.max(C))

from solver import mat_v_bazi2

def spectre(lamb, Nmax):
    vals, vects = mat_v_bazi2(Nmax, lamb)
    return vals

def analyE(lamb, betas):
    E = np.zeros(len(betas))
    Es = spectre(lamb, 30)

    for i in range(len(betas)):
        E[i] = np.sum(Es * np.exp(-betas[i]*Es)) / np.sum(np.exp(-betas[i]*Es))

    return E


"""lamb = 1
ratio = 10
iters = 10**6
N = 100
split = 0.9

@njit(nopython = True, parallel = True)
def run():
    
    #betas = np.linspace(0.1, 100, 100)
    betas = 10**np.linspace(np.log10(0.1), np.log10(100), 200)

    doing = np.zeros(len(betas), dtype=np.bool)
    done = np.zeros(len(betas), dtype=np.bool)
    E, V = np.zeros(betas.shape), np.zeros(betas.shape)
    for i in prange(len(betas)):
        doing[i] = True
        for j in range(N):
            beta = betas[i]
            state, H = single(beta, lamb, ratio, iters, split)

            #state = state[-1:]

            E[i] += H/N
            V[i] += expected_V(state, lamb)/N
        
        done[i] = True

        s_done = np.sum(done)
        s_doing = np.sum(doing)

        print(s_done)
        print(len(betas))
        print(s_doing-s_done)
        print("==============")

    return betas, E, V
betas, E, V = run()


plt.plot(betas, E, label = "E", marker = "+", linestyle = "none")
plt.plot(betas, V, label = "V", marker = "+", linestyle = "none")
plt.plot(betas, E-V, label = "T", marker = "+", linestyle = "none")

#ax1.legend()
plt.xlabel("$\\beta$")

#plt.axhline(y=1/2, xmin=min_of(E,V,E-V), xmax=max_of(E,V,E-V), linestyle = "dotted", label = "Osnovno stanje", color = "black")
betas2 = np.linspace(np.min(betas), np.max(betas), 400)
Es = analyE(lamb, betas2)
plt.plot(betas2, Es, linestyle = "dashed", label = "Z diagonalizacijo", color = "black" )

plt.title(f"$\\lambda = {lamb}$, " + "$\\frac{M}{\\beta}=$"+f"${ratio}$, Število iteracij $= 10^{int(np.round(np.log10(iters), decimals=0))}, N = {N}$")

plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.tight_layout()
plt.show()"""



ratio = 10
iters = 10**6
N = 100
split = 0.9

@njit(nopython = True, parallel = True)
def run(lamb):
    
    #betas = np.linspace(0.1, 100, 100)
    betas = 10**np.linspace(np.log10(0.1), np.log10(100), 200)

    doing = np.zeros(len(betas), dtype=np.bool)
    done = np.zeros(len(betas), dtype=np.bool)
    E, V = np.zeros(betas.shape), np.zeros(betas.shape)
    for i in prange(len(betas)):
        doing[i] = True
        for j in range(N):
            beta = betas[i]
            state, H = single(beta, lamb, ratio, iters, split)

            #state = state[-1:]

            E[i] += H/N
            V[i] += expected_V(state, lamb)/N
        
        done[i] = True

        s_done = np.sum(done)
        s_doing = np.sum(doing)

        print(s_done)
        print(len(betas))
        print(s_doing-s_done)
        print("==============")

    return betas, E, V

lamb = 0
betas, E, V = run(lamb)
plt.plot(betas, E, label = "$E$, $\\lambda = 0$", marker = "+", linestyle = "none")
betas2 = np.linspace(np.min(betas), np.max(betas), 400)
Es = analyE(lamb, betas2)
plt.plot(betas2, Es, linestyle = "dashed", label = "Z diagonalizacijo", color = "black")


lamb = 0.1
betas, E, V = run(lamb)
plt.plot(betas, E, label = "$E$, $\\lambda = 0.1$", marker = "+", linestyle = "none")
betas2 = np.linspace(np.min(betas), np.max(betas), 400)
Es = analyE(lamb, betas2)
plt.plot(betas2, Es, linestyle = "dashed", color = "black")

lamb = 1
betas, E, V = run(lamb)
plt.plot(betas, E, label = "$E$, $\\lambda = 1$", marker = "+", linestyle = "none")
betas2 = np.linspace(np.min(betas), np.max(betas), 400)
Es = analyE(lamb, betas2)
plt.plot(betas2, Es, linestyle = "dashed", color = "black")

lamb = 10
betas, E, V = run(lamb)
plt.plot(betas, E, label = "$E$, $\\lambda = 10$", marker = "+", linestyle = "none")
betas2 = np.linspace(np.min(betas), np.max(betas), 400)
Es = analyE(lamb, betas2)
plt.plot(betas2, Es, linestyle = "dashed", color = "black")


#ax1.legend()
plt.xlabel("$\\beta$")

#plt.axhline(y=1/2, xmin=min_of(E,V,E-V), xmax=max_of(E,V,E-V), linestyle = "dotted", label = "Osnovno stanje", color = "black")

plt.title("$\\frac{M}{\\beta}=$"+f"${ratio}$, Število iteracij $= 10^{int(np.round(np.log10(iters), decimals=0))}, N = {N}$")

plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.tight_layout()
plt.show()



