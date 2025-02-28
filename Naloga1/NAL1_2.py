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


import solver

def V(x,lamb):
    return 1/2 * x**2 + lamb * x**4

def eigenstate(N, x):
    return 1/(np.pi**(1/4) * np.sqrt(2**N * scipy.special.factorial(N))) * scipy.special.hermite(N)(x) * np.exp(-x**2/2)

def coherent_state(d,x):
    return eigenstate(0,x-d)

L = 15
dx = 0.03
dt = 0.001

tmax = 10
nmax = int(tmax/dt)

xs = np.arange(-L,L, dx)
Vmn = np.zeros((len(xs), nmax))

spacewise2 = [1/2, -1, 1/2]
spacewise4 = [-1/12, 4/3, -5/3, 4/3, -1/12]


"""lambs = np.arange(0,4,1)
Ns = np.arange(0,4,1)

sols = np.zeros((len(lambs), len(Ns), len(xs), nmax), dtype=np.complex128)

for i in tqdm(range(len(lambs))):
    lamb = lambs[i]

    V0 = V(xs,lamb)
    for k in range(nmax):
        Vmn[:,k] = V0

    for j in tqdm(range(len(Ns)),leave=False):
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

fig, axs = plt.subplots(len(Ns), len(lambs), figsize = (len(Ns)*2, len(lambs)*2))

cmap = plt.get_cmap("hot")
norm = colors.Normalize(0,1)
for i in tqdm(range(len(lambs))):
    lamb = lambs[i]
    for j in tqdm(range(len(Ns)),leave=False):
        N = Ns[j]
        axs[i,j].imshow(np.abs(sols[i,j,:,:].T)**2, cmap=cmap, norm=norm, aspect="auto", extent=(-L,L,tmax,0))

        axs[i,j].set_title(f"$N = {N}$, $\\lambda = {lamb}$")

        axs[i,j].set_xlabel("x")
        axs[i,j].set_ylabel("t")

fig.tight_layout()
plt.show()"""


"""lambs = [0,0.1]
As = [1,10]

sols = np.zeros((len(lambs), len(As), len(xs), nmax), dtype=np.complex128)

for i in tqdm(range(len(lambs))):
    lamb = lambs[i]

    V0 = V(xs,lamb)
    for k in range(nmax):
        Vmn[:,k] = V0

    for j in tqdm(range(len(As)),leave=False):
        a = As[j]
        solution = solver.implicinator(
            startstate=coherent_state(a,xs),
            V=Vmn,
            tau=dt,
            h=dx,
            nmax=nmax,
            spacewise=spacewise4
        )
        sols[i,j,:,:] = solution

fig, axs = plt.subplots(len(lambs), len(As), figsize = (len(lambs)*2, len(As)*2))

cmap = plt.get_cmap("hot")
norm = colors.Normalize(0,1)
for i in tqdm(range(len(lambs))):
    lamb = lambs[i]
    for j in tqdm(range(len(As)),leave=False):
        N = As[j]
        axs[i,j].imshow(np.abs(sols[i,j,:,:].T)**2, cmap=cmap, norm=norm, aspect="auto", extent=(-L,L,tmax,0))

        axs[i,j].set_title(f"$a = {N}$, $\\lambda = {lamb}$")

        axs[i,j].set_xlabel("x")
        axs[i,j].set_ylabel("t")

fig.tight_layout()
plt.savefig("backup.png")
plt.show()"""


lambs = [0,0.1]
As = [1,10]

sols = np.zeros((len(lambs), len(As), len(xs), nmax), dtype=np.complex128)

for i in tqdm(range(len(lambs))):
    lamb = lambs[i]

    
    for k in range(nmax):
        V0 = V(xs,lamb* k/nmax)
        Vmn[:,k] = V0

    for j in tqdm(range(len(As)),leave=False):
        a = As[j]
        solution = solver.implicinator(
            startstate=coherent_state(a,xs),
            V=Vmn,
            tau=dt,
            h=dx,
            nmax=nmax,
            spacewise=spacewise4
        )
        sols[i,j,:,:] = solution

fig, axs = plt.subplots(len(lambs), len(As), figsize = (len(lambs)*2, len(As)*2))

cmap = plt.get_cmap("hot")
norm = colors.Normalize(0,1)
for i in tqdm(range(len(lambs))):
    lamb = lambs[i]
    for j in tqdm(range(len(As)),leave=False):
        N = As[j]
        axs[i,j].imshow(np.abs(sols[i,j,:,:].T)**2, cmap=cmap, norm=norm, aspect="auto", extent=(-L,L,tmax,0))

        axs[i,j].set_title(f"$a = {N}$, $\\lambda = {lamb}$")

        axs[i,j].set_xlabel("x")
        axs[i,j].set_ylabel("t")

fig.tight_layout()
plt.savefig("backup.png")
plt.show()












