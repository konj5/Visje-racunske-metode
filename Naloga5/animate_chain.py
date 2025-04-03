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
from numba import jit, njit
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))


import deterministic, maxwell
N = 500
M = 1
lamb = 1
tau = 1
TL=1
TR=0
tmax = 1000

print("calculating")
ts, ys = deterministic.run(N,M,lamb,tau,TL,TR,tmax)
print("recalculating")

"""plt.plot(ts, 1/tau * (np.sum(ys[0:2*M:2]**2, axis = 0)-M*TL), label = "p levi rob")
plt.plot(ts, ys[1,:]**2-TR, label = "p desni rob")
plt.legend()
plt.show()
plt.plot(ts, ys[-2,:], label = "levi rob")
plt.plot(ts, ys[-1,:], label = "desni rob")
plt.legend()
plt.show()"""

#q
#ys = ys[0:-2:2,:]

#p
#ys = ys[1:-2:2,:]

#energy_smoothed
a = 30
for i in range(len(ys[0,:])):
    ys[1:-2:2,i] = np.convolve(ys[1:-2:2,i]**2, np.ones(a)/a, mode="same")
ys = ys[1:-2:2,:]



spline = scipy.interpolate.CubicSpline(ts, ys, axis=1)
ts = np.linspace(0,tmax,300)
ys = spline(ts)

#print(ys.shape)

print("animating")
fig, ax = plt.subplots(figsize=(8, 8))
ns = np.array([i for i in range(len(ys[:,0]))])
#print(len(ns))
ax.set_ylim(np.min(ys),np.max(ys))
sca = ax.scatter(ns, ys[:,0])
def animate(k):
    #fig.clear()
    print(f"{k}/{len(ts)}", end="\r")
    x = ns
    y = ys[:,k]
    data = np.c_[x,y]
    #print(data)
    sca.set_offsets(data)
    #plt.scatter(ns, ys[:,k], s=10)
    
    #print(ns)
    #print(ys[:,k])
    return sca

ani = FuncAnimation(fig, animate, frames = len(ts))

ani.save(f'test.mp4',  
          writer = 'ffmpeg', fps = 30) 
print("\ndone")
plt.cla()