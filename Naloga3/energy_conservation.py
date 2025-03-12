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

#@jit
def H(states, lamb):
    E = np.zeros(len(states[0,:]))
    for i in range(len(E)):
        q1, q2, p1, p2 = states[:,i]

        E[i] = (q1**2 + q2**2 + p1**2 + p2**2)/2 + lamb * q1**2 * q2**2

    return E


"""startstate = [0,0.5,1,0]

lambs = [0,0.1,1]

dt = 0.01
tmax = 100

ts = np.arange(0,tmax,dt)

Hs = np.zeros((len(lambs), len(ts)))

for i in range(len(lambs)):
    lamb = lambs[i]
    ts, states = integrate(startstate, dt, tmax, lamb=lamb, decomp=decomp11)
    Hs[i,:] = H(states, lamb)

    plt.plot(ts, np.abs(Hs[i,:]-Hs[i,0]), label = f"razcep11 $\\lambda = {lamb}$")

    ts, states = integrate_rk45(startstate, dt, tmax, lamb=lamb)

    plt.plot(ts, np.abs(H(states,lamb)-H(states,lamb)[0]), label = f"RK45 $\\lambda = {lamb}$")

plt.legend()
plt.xlabel("t")
plt.ylabel("$|E(t)-E(0)|$")
plt.yscale("log")
plt.show()"""











