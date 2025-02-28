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

def ground_state_2D(x,y):
    return eigenstate(0,x) * eigenstate(0,y)
    

def coherent_state2D(xs,dx,dy):
    state = np.zeros((len(xs),len(xs)), dtype=np.complex128)

    for i in tqdm(range(len(xs))):
        for j in range(len(xs)):
            state[i,j] = ground_state_2D(xs[i]-dx, xs[j]-dy)
    return state

L = 2.5
dx = 0.1
dt = 0.001

xs = np.arange(-L,L, dx)
Vm = np.zeros((len(xs),len(xs)))

lamb = 0

for i in range(len(xs)):
    for j in range(len(xs)):
        Vm[i,j] = 1/2 * (xs[i]**2 + xs[j]**2) + lamb * xs[i]**2 * xs[j]**2

spacewise2 = [1/2, -1, 1/2]
spacewise4 = [-1/12, 4/3, -5/3, 4/3, -1/12]



tskip = 0.5
tmax = 1
fps = 2
frames = int(tmax/tskip)

state = coherent_state2D(xs,1,0)
cmap = plt.get_cmap("hot")
norm = colors.Normalize(0,1)

state[0,:] = state[-1,:] = state[:,0] = state[:,-1] = 0


plt.imshow(np.abs(state[:,:].T)**2, cmap=cmap, norm=norm, aspect="equal", extent=(-L,L,L,-L))

def animate(i):
    plt.cla()

    plt.title(f"Metoda: Končni propagator, $h={dx}$, $\\tau={dt}$, $t = {i/(frames)*tmax:0.3f}$")

    #i = int(i * nmax/(30*duration))
    #print(f"animating: {int(i * nmax/(30*duration))}/{nmax}")

    print(f"animating: {i}/{frames}")

    state = solver.finite_propagatonator2D_single_step(coherent_state2D(xs,1,0), Vm, dt, dx, tskip, spacewise=spacewise2)

    print(state)
    

    #norm = colors.Normalize(0,np.max(np.abs(state[:,:].T)**2))


    plt.imshow(np.abs(state[:,:].T)**2, cmap=cmap, norm=norm, aspect="equal", extent=(-L,L,L,-L))

    #plt.xlabel("Rdeča - Re$\\psi^2$, Modra - Im$\\psi^2$,  Črna - $|\\psi|^2$")
    


fig = plt.figure()

ani = FuncAnimation(fig, animate, frames=frames, )
ani.save('animacija.mp4',  
          writer = 'ffmpeg', fps = fps) 
print("done")
#plt.show()












