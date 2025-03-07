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

def evolve(startstate, ts, lamb, takenN):
    N = 200
    energs, states = basic_herm(N,lamb)

    energs = energs[:takenN]
    states = states[:takenN,:]


    psi = np.zeros((len(states[:,0]),len(ts)), dtype=complex)
    for i in tqdm(range(len(energs))):
        for j in range(len(ts)):
            psi[:,j] += states[:,i].dot(startstate) * states[:,i] * np.exp(-1j * energs[i]*ts[j])


    return psi

L = 10
dx = 0.05
dt = 0.002

dx = 0.1
dt = 0.01

tmax = 30
nmax = int(tmax/dt)

xs = np.arange(-L,L, dx)
Vmn = np.zeros((len(xs), nmax))

spacewise2 = [1/2, -1, 1/2]
spacewise4 = [-1/12, 4/3, -5/3, 4/3, -1/12]


lambs = np.arange(0,4,1)
Ns = np.arange(0,4,1)

sols = np.zeros((len(lambs), len(Ns), len(xs), nmax), dtype=np.complex128)

basis = np.zeros((200, len(xs)))
for i in range(len(basis[:,0])):
    basis[i,:] = eigenstate(i, xs)

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

        takenN = 100

        startstate = np.zeros(takenN)
        startstate[N] = 1

        states =  evolve(startstate, np.arange(0,tmax,dt), lamb, takenN)

        basisN = basis[:takenN,:]

        #convert to space representation:

        #print(basisN.shape)
        #print(states.shape)

        sols[i,j+2,:,:]  = np.einsum("ji,jk->ik", basisN, states)

        


fig, axs = plt.subplots(len(Ns), len(lambs), figsize = (len(Ns)*2+1, len(lambs)*2))

cmap = plt.get_cmap("hot")
norm = colors.Normalize(0,np.max(np.abs(sols)**2))
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



cbar_ax = fig.add_axes([0.92, 0.15, 0.025, 0.8])
fig.colorbar(cm.ScalarMappable(norm,cmap), cax=cbar_ax)

fig.tight_layout(rect=(0,0,0.9,1))
#fig.subplots_adjust(wspace=1, hspace=1)
plt.savefig("evolves.png")
plt.show()



duration = 30

fig, axs = plt.subplots(1,2); ax1, ax2 = axs

i = 0; j = 0

lamb = lambs[i]
N = Ns[j]


line1, = ax1.plot(xs, np.abs(sols[i,j,:,0])**2)
line2, = ax2.plot(xs, np.abs(sols[i,j+2,:,0])**2)

ax1.set_title("Implicitna")
ax2.set_title("Spektralna")

fig.suptitle(f"$\\lambda = {lamb}$, $N = {N}$")

def animate(k):
    

    print(f"animating: {k}/{30*duration}")

    k = int(k * nmax/(30*duration))

    line1.set_ydata(np.abs(sols[i,j,:,k])**2)
    line2.set_ydata(np.abs(sols[i,j+2,:,k])**2)

    ax1.set_ylim(0,max(np.max(np.abs(sols[i,j,:,:])**2), np.max(np.abs(sols[i,j+2,:,:])**2)))
    ax2.set_ylim(0,max(np.max(np.abs(sols[i,j,:,:])**2), np.max(np.abs(sols[i,j+2,:,:])**2)))
    


ani = FuncAnimation(fig, animate, frames=30*duration, )
ani.save(f'animacija_{i}_{j}.mp4',  
          writer = 'ffmpeg', fps = 30) 
print("done")
plt.cla()

duration = 30

fig, axs = plt.subplots(1,2); ax1, ax2 = axs

i = 1; j = 0

lamb = lambs[i]
N = Ns[j]


line1, = ax1.plot(xs, np.abs(sols[i,j,:,0])**2)
line2, = ax2.plot(xs, np.abs(sols[i,j+2,:,0])**2)

ax1.set_title("Implicitna")
ax2.set_title("Spektralna")

fig.suptitle(f"$\\lambda = {lamb}$, $N = {N}$")

def animate(k):
    

    print(f"animating: {k}/{30*duration}")

    k = int(k * nmax/(30*duration))

    line1.set_ydata(np.abs(sols[i,j,:,k])**2)
    line2.set_ydata(np.abs(sols[i,j+2,:,k])**2)

    ax1.set_ylim(0,max(np.max(np.abs(sols[i,j,:,:])**2), np.max(np.abs(sols[i,j+2,:,:])**2)))
    ax2.set_ylim(0,max(np.max(np.abs(sols[i,j,:,:])**2), np.max(np.abs(sols[i,j+2,:,:])**2)))
    

ani = FuncAnimation(fig, animate, frames=30*duration, )
ani.save(f'animacija_{i}_{j}.mp4',  
          writer = 'ffmpeg', fps = 30) 
print("done")
plt.cla()

duration = 30

fig, axs = plt.subplots(1,2); ax1, ax2 = axs

i = 1; j = 1

lamb = lambs[i]
N = Ns[j]


line1, = ax1.plot(xs, np.abs(sols[i,j,:,0])**2)
line2, = ax2.plot(xs, np.abs(sols[i,j+2,:,0])**2)

ax1.set_title("Implicitna")
ax2.set_title("Spektralna")

fig.suptitle(f"$\\lambda = {lamb}$, $N = {N}$")

def animate(k):
    

    print(f"animating: {k}/{30*duration}")

    k = int(k * nmax/(30*duration))

    line1.set_ydata(np.abs(sols[i,j,:,k])**2)
    line2.set_ydata(np.abs(sols[i,j+2,:,k])**2)

    ax1.set_ylim(0,max(np.max(np.abs(sols[i,j,:,:])**2), np.max(np.abs(sols[i,j+2,:,:])**2)))
    ax2.set_ylim(0,max(np.max(np.abs(sols[i,j,:,:])**2), np.max(np.abs(sols[i,j+2,:,:])**2)))
    


ani = FuncAnimation(fig, animate, frames=30*duration, )
ani.save(f'animacija_{i}_{j}.mp4',  
          writer = 'ffmpeg', fps = 30) 
print("done")
plt.cla()

fig, axs = plt.subplots(1,2); ax1, ax2 = axs
i = 2; j = 0

lamb = lambs[i]
N = Ns[j]


line1, = ax1.plot(xs, np.abs(sols[i,j,:,0])**2)
line2, = ax2.plot(xs, np.abs(sols[i,j+2,:,0])**2)

ax1.set_title("Implicitna")
ax2.set_title("Spektralna")

fig.suptitle(f"$\\lambda = {lamb}$, $N = {N}$")

def animate(k):
    

    print(f"animating: {k}/{30*duration}")

    k = int(k * nmax/(30*duration))

    line1.set_ydata(np.abs(sols[i,j,:,k])**2)
    line2.set_ydata(np.abs(sols[i,j+2,:,k])**2)

    ax1.set_ylim(0,max(np.max(np.abs(sols[i,j,:,:])**2), np.max(np.abs(sols[i,j+2,:,:])**2)))
    ax2.set_ylim(0,max(np.max(np.abs(sols[i,j,:,:])**2), np.max(np.abs(sols[i,j+2,:,:])**2)))
    

ani = FuncAnimation(fig, animate, frames=30*duration, )
ani.save(f'animacija_{i}_{j}.mp4',  
          writer = 'ffmpeg', fps = 30) 
print("done")
plt.cla()

duration = 30

fig, axs = plt.subplots(1,2); ax1, ax2 = axs

i = 2; j = 1

lamb = lambs[i]
N = Ns[j]


line1, = ax1.plot(xs, np.abs(sols[i,j,:,0])**2)
line2, = ax2.plot(xs, np.abs(sols[i,j+2,:,0])**2)

ax1.set_title("Implicitna")
ax2.set_title("Spektralna")

fig.suptitle(f"$\\lambda = {lamb}$, $N = {N}$")

def animate(k):
    

    print(f"animating: {k}/{30*duration}")

    k = int(k * nmax/(30*duration))

    line1.set_ydata(np.abs(sols[i,j,:,k])**2)
    line2.set_ydata(np.abs(sols[i,j+2,:,k])**2)

    ax1.set_ylim(0,max(np.max(np.abs(sols[i,j,:,:])**2), np.max(np.abs(sols[i,j+2,:,:])**2)))
    ax2.set_ylim(0,max(np.max(np.abs(sols[i,j,:,:])**2), np.max(np.abs(sols[i,j+2,:,:])**2)))


ani = FuncAnimation(fig, animate, frames=30*duration, )
ani.save(f'animacija_{i}_{j}.mp4',  
          writer = 'ffmpeg', fps = 30) 
print("done")
plt.show()