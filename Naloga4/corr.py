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

import solver


t = 10
dt = 0.05
dsites = [0,1,5]
n = 12
Jx, Jy, Jz = [-1,-1,-1]
N_psi = 20

for dsite in dsites:
    corr = solver.correlator_zz(0,dsite, N_psi, n, dt, t, Jx, Jy, Jz, solver.decomp22)
    ts = np.arange(0,t,dt)
    plt.plot(ts, corr, label = f"$d = {dsite}$")

plt.legend()
plt.title(f"$J_x={Jx}, J_y={Jy}, J_z={Jz}, dt = {dt}, N_\\psi = {N_psi}$, $n = {n}$")
plt.xlabel("t")
plt.ylabel("$C_{zz}$")
plt.savefig("0.png")
plt.cla()

t = 10
dt = 0.01
ns = [2,4,6,8,10,12]
Jx, Jy, Jz = [1,1,1]
N_psi = 10


for n in ns[::]:
    corr = solver.correlator_zz(0,0,N_psi, n, dt, t, Jx, Jy, Jz, solver.decomp22)
    ts = np.arange(0,t,dt)
    plt.plot(ts, corr, label = f"$n = {n}$")

plt.legend()
plt.xlabel("t")
plt.ylabel("$C_{zz}$")
plt.title(f"$J_x={Jx}, J_y={Jy}, J_z={Jz}, dt = {dt}, N_\\psi = {N_psi}$")
plt.savefig("1.png")



plt.cla()
t = 10
dt = 0.01
ns = [2,4,6,8,10,12]
Jx, Jy, Jz = [1,1,1]
N_psi = 10

ints = []

for n in ns[::]:
    corr = solver.correlator_JJ(N_psi, n, dt, t, Jx, Jy, Jz, solver.decomp22)
    ts = np.arange(0,t,dt)
    plt.plot(ts, corr, label = f"$n = {n}$")
    ints.append(np.trapz(corr, ts))

plt.legend()
plt.xlabel("t")
plt.ylabel("$C_{JJ}$")
plt.title(f"$J_x={Jx}, J_y={Jy}, J_z={Jz}, dt = {dt}, N_\\psi = {N_psi}$")
plt.savefig("2.png")

plt.cla()
plt.plot(ns, ints)
plt.xlabel("n")
plt.ylabel("$C_{JJ}$")
plt.title(f"$J_x={Jx}, J_y={Jy}, J_z={Jz}, dt = {dt}, N_\\psi = {N_psi}$")
plt.savefig("3.png")
