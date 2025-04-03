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
Nreps = 1
N = 20
M = 1
lamb = 0
tau = 1
TL=2
TR=1
tmax = 1000

print("kalkulieren")
ts, ys = maxwell.run(N,M,lamb,tau,TL,TR,tmax)

print("wiederkalkunieren schön wieder")
ps2s = ys[1::2, :]**2

ps2s = np.trapezoid(ps2s[:,len(ts)//10:],ts[len(ts)//10:])/tmax

plt.plot([i for i in range(len(ps2s))], ps2s, linestyle = "dashed", marker='o')
plt.show()



"""tmaxs = 10**np.array([1,2,3,4])
for tmax in tmaxs:
    print("kalkulieren")
    ts, ys = deterministic.run(N,M,lamb,tau,TL,TR,tmax)

    print("wiederkalkunieren schön wieder")
    ps2s = ys[1:-2:2, :]**2

    ps2s = np.trapz(ps2s[:,len(ts)//10:],ts[len(ts)//10:])/tmax

    plt.plot([i for i in range(len(ps2s))], ps2s, linestyle = "dashed", marker='o', label = "$t_{max} = $" + f"${tmax}$")

plt.legend()
plt.xlabel("Zaporedno število delca")
plt.ylabel("$\\langle \\text{T}\\rangle$")
plt.title(f"$N={N}$, $M = {M}$, $\\lambda = {lamb}$, $T_L = {TL}$, $T_R = {TR}$")
plt.show()"""



"""tmaxs = 10**np.array([1,2,3,4])
for j in tqdm(range(len(tmaxs))):
    tmax = tmaxs[-j-1]
    pcssummed = np.zeros(N)
    for i in tqdm(range(Nreps), leave=False):
        #print("kalkulieren")
        ts, ys = deterministic.run(N,M,lamb,tau,TL,TR,tmax)

        #print("wiederkalkunieren schön wieder")
        ps2s = ys[1:-2:2, :]**2

        ps2s = np.trapz(ps2s[:,len(ts)//10:],ts[len(ts)//10:])/tmax
        pcssummed += ps2s
    pcssummed = pcssummed / Nreps

    plt.plot([i for i in range(len(pcssummed))], pcssummed, linestyle = "dashed", marker='o', label = "$t_{max} = $" + f"${tmax}$")

plt.legend()
plt.xlabel("Zaporedno število delca")
plt.ylabel("$\\langle text{T}\\rangle$")
plt.title(f"$N={N}$, $M = {M}$, $\\lambda = {lamb}$, $T_L = {TL}$, $T_R = {TR}$, $N_r = {Nreps}$")
plt.show()"""


"""from tqdm import trange

lambs = np.round(10**np.linspace(np.log10(0.01), 0, 3), decimals=3)
lambs = np.insert(lambs, 0, 0)
#lambs = np.round(np.linspace(0,1,4), decimals=3)

vals = []
for i in trange(0, len(lambs)):
    lamb = lambs[i]


    ts, ys = maxwell.run(N,M,lamb,tau,TL,TR,tmax)

    ps2s = ys[1::2, :]**2

    ps2s = np.trapezoid(ps2s[:,len(ts)//10:],ts[len(ts)//10:])/tmax

    plt.plot([i for i in range(len(ps2s))], ps2s, linestyle = "dashed", marker='o', label = "$\\lambda = $" + f"${lamb}$")
    #vals.append(np.average(J))

plt.legend()
plt.xlabel("$\\tau$")
plt.ylabel("$\\langle J \\rangle$")
plt.title(f"$N={N}$, $M = {M}$, $\\lambda = {lamb}$, $T_L = {TL}$, $T_R = {TR}$"+", $t_{max}=$"+f"${tmax}$")
plt.show()"""