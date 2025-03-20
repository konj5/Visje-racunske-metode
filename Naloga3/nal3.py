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
import time
import numba
from numba import jit
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

from decomp import integrate, integrate_rk45, decomp11, decomp22, decomp33, decomp44, decomp45



#startstate = [0,0.5,1,0]
startstate = [0,1,1,0]

"""ts, states = integrate(startstate=startstate, dt = 0.1, tmax=100, lamb=0, decomp=decomp11)
#ts, states = integrate_rk45(startstate=startstate, dt = 0.1, tmax=1, lamb=0)


plt.plot(states[0,:], states[1,:])
plt.axis("equal")
plt.show()
"""

fig, axs = plt.subplots(4,4, figsize = (4*2, 4*2))
lambs = [0, 0.1, 0.5, 1,
         2, 3, 4, 5,
         7, 10, 12, 15,
         20, 30, 40, 50
         ]

data = [[], [], [], []]
dt = 0.01

for i in tqdm(range(4)):
    for j in tqdm(range(4), leave=False):
        lamb = lambs[4*i+j]
        ts, states = integrate(startstate=startstate, dt = dt, tmax=40, lamb=lamb, decomp=decomp44)
        
        data[i].append(states)

        axs[i,j].plot(states[0,:], states[1,:], )
        axs[i,j].set_title(f"$\\lambda = {lamb}$")

plt.tight_layout()
plt.show()


fig, axs = plt.subplots(4,4, figsize = (4*2, 4*2))
duration = 30
nmax = len(ts)

lines = [[], [], [], []]

for i in tqdm(range(4)):
        for j in tqdm(range(4), leave=False):
            lamb = lambs[4*i+j]
            axs[i,j].set_title(f"$\\lambda = {lamb}$")

            axs[i,j].set_xlim(np.min(states[0,:]),np.max(states[0,:]))
            axs[i,j].set_ylim(np.min(states[1,:]),np.max(states[1,:]))


            line, = axs[i,j].plot(states[0,:], states[1,:])

            

            plt.tight_layout()

            lines[i].append(line)

def animate(k):
    

    print(f"animating: {k}/{30*duration}")

    k = int(k * nmax/(30*duration))

    for i in range(4):
        for j in range(4):
            states = data[i][j]

            axs[i,j].set_xlim(np.min(states[0,:]),np.max(states[0,:]))
            axs[i,j].set_ylim(np.min(states[1,:]),np.max(states[1,:]))

            states = states[:,:k]

            lines[i][j].set_ydata(states[1,:])
            lines[i][j].set_xdata(states[0,:])

            
            
    


ani = FuncAnimation(fig, animate, frames=30*duration, )
ani.save('animacija_adsadasdasda.mp4',  
          writer = 'ffmpeg', fps = 30) 
print("done")
plt.show()

        

