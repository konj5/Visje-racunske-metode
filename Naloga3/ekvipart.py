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

startstate = [0,0.5,1,0]
lambs = np.linspace(0,10,100)
tmax = 100
dt = 0.1

ekv1 = []
ekv2 = []
for i in tqdm(range(len(lambs))):
    lamb = lambs[i]

    ts, sol = integrate(startstate, dt, tmax, lamb, decomp44)

    ekv1.append(np.sum(sol[2,:]**2 * dt)/tmax)
    ekv2.append(np.sum(sol[3,:]**2 * dt)/tmax)

plt.plot(lambs, ekv1, label = "$\\langle p_1^2 \\rangle$")
plt.plot(lambs, ekv2, label = "$\\langle p_2^2 \\rangle$")

plt.legend()
plt.xlabel("$\\lambda$")

plt.show()
