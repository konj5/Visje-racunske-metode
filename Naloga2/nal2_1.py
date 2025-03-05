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
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))


import solverOld as solver
from solver import mat_v_bazi2 as basic_herm


def V(x,lamb):
    return 1/2 * x**2 + lamb * x**4

def eigenstate(N, x):
    return 1/(np.pi**(1/4) * np.sqrt(2**N * scipy.special.factorial(N))) * scipy.special.hermite(N)(x) * np.exp(-x**2/2)

def evolve(startstate, ts, lamb):
    N = 200
    energs, states = basic_herm(N,lamb)


    psi = np.zeros((len(states[:,0]),len(ts)), dtype=complex)
    for i in tqdm(range(len(energs))):
        for j in range(len(ts)):
            psi[:,j] += states[:,i].dot(startstate) * states[:,i] * np.exp(-1j * energs[i]*ts[j])

    print(psi)

    return psi

L = 10
dx = 0.1
dt = 0.01

tmax = 10
nmax = int(tmax/dt)

xs = np.arange(-L,L, dx)
Vmn = np.zeros((len(xs), nmax))

spacewise2 = [1/2, -1, 1/2]
spacewise4 = [-1/12, 4/3, -5/3, 4/3, -1/12]


lambs = np.arange(0,4,1)
Ns = np.arange(0,4,1)

sols = np.zeros((len(lambs), len(Ns), len(xs), nmax), dtype=np.complex128)

for i in tqdm(range(len(lambs))):
    lamb = lambs[i]

    V0 = V(xs,lamb)
    for k in range(nmax):
        Vmn[:,k] = V0

    for j in tqdm(range(2),leave=False):
        N = Ns[j]
        solution = solver.implicinator(
            startstate=eigenstate(N,xs),
            V=Vmn,
            tau=dt,
            h=dx,
            nmax=nmax,
            spacewise=spacewise4
        )
        sols[i,j,:,:] = solution

    for j in tqdm(range(2),leave=False):
        N = Ns[j]

        startstate = np.zeros(200)
        startstate[N] = 1

        states =  evolve(startstate, np.arange(0,tmax,dt), lamb)

        #convert to space representation:

        

        sols[i,j+2,:,:]  =



fig, axs = plt.subplots(len(Ns), len(lambs), figsize = (len(Ns)*2, len(lambs)*2))

cmap = plt.get_cmap("hot")
norm = colors.Normalize(0,1)
for i in tqdm(range(len(lambs))):
    lamb = lambs[i]
    for j in tqdm(range(2),leave=False):
        N = Ns[j]
        axs[i,j].imshow(np.abs(sols[i,j,:,:].T)**2, cmap=cmap, norm=norm, aspect="auto", extent=(-L,L,tmax,0))

        axs[i,j].set_title(f"$N = {N}$, $\\lambda = {lamb}, implicitna$")

        axs[i,j].set_xlabel("x")
        axs[i,j].set_ylabel("t")
    
    for j in tqdm(range(2),leave=False):
        N = Ns[j]
        axs[i,j+2].imshow(np.abs(sols[i,j+2,:,:].T)**2, cmap=cmap, norm=norm, aspect="auto", extent=(-L,L,tmax,0))

        axs[i,j+2].set_title(f"$N = {N}$, $\\lambda = {lamb}, spekter$")

        axs[i,j+2].set_xlabel("x")
        axs[i,j+2].set_ylabel("t")



fig.tight_layout()
plt.show()