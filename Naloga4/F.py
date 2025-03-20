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


"""b = 10
dbs = [1,0.1,0.01,0.001]
n = 4
Jx, Jy, Jz = [1,1,1]
N_psi = 50

for db in dbs[::-1]:
    Fs = solver.F_zs(N_psi, n, db, b, Jx, Jy, Jz, solver.decomp44)
    bs = np.linspace(0, b, len(Fs))
    plt.plot(bs, Fs, label = f"$d\\beta={db}$")

plt.xlabel("$\\beta$")
plt.ylabel("$F(\\beta)$")
plt.xscale("log")

plt.title(f"$J_x={Jx}, J_y={Jy}, J_z={Jz}, n = {n}, N_\\psi = {N_psi}$")
plt.legend()
plt.show()"""


"""b = 10
db = 0.01
n = 4
Jx, Jy, Jz = [1,1,1]
N_psis = [1,10,50,100]

for N_psi in N_psis[::-1]:
    Fs = solver.F_zs(N_psi, n, db, b, Jx, Jy, Jz, solver.decomp44)
    bs = np.linspace(0, b, len(Fs))
    plt.plot(bs, Fs, label = f"$N_\\psi={N_psi}$")

plt.xlabel("$\\beta$")
plt.ylabel("$F(\\beta)$")
plt.xscale("log")

plt.title(f"$J_x={Jx}, J_y={Jy}, J_z={Jz}, d\\beta = {db}, n = {n}$")
plt.legend()
plt.show()"""


"""b = 10
db = 0.01
n = 4
Jx, Jy, Jz = [1,1,1]
N_psi = 50

decompnames = ["razcep22", "razcep33", "razcep44", "razcep45"]

for i, decomp in enumerate([solver.decomp22, solver.decomp33, solver.decomp44, solver.decomp45]):
    Fs = solver.F_zs(N_psi, n, db, b, Jx, Jy, Jz, decomp)
    bs = np.linspace(0, b, len(Fs))
    plt.plot(bs, Fs, label = f"{decompnames[i]}")

plt.xlabel("$\\beta$")
plt.ylabel("$F(\\beta)$")
plt.xscale("log")

plt.title(f"$J_x={Jx}, J_y={Jy}, J_z={Jz}, d\\beta = {db}, n = {n}, N_\\psi = {N_psi}$")
plt.legend()
plt.show()"""


b = 10
db = 0.01
ns = [2,4]
Jx, Jy, Jz = [1,1,1]
N_psi = 40

for n in ns[::-1]:
    Fs = solver.F_zs(N_psi, n, db, b, Jx, Jy, Jz, solver.decomp22)
    bs = np.linspace(0, b, len(Fs))
    plt.plot(bs, Fs, label = f"$n={n}$")

plt.xlabel("$\\beta$")
plt.ylabel("$F(\\beta)$")
plt.xscale("log")

plt.title(f"$J_x={Jx}, J_y={Jy}, J_z={Jz}, d\\beta = {db}, N_\\psi = {N_psi}$")
plt.legend()
plt.show()


"""b = 10
db = 0.01
ns = [2,4]
Jx, Jy, Jz = [1,1,1]
N_psi = 40

for n in ns[::-1]:
    Bs = solver.Z_zs(N_psi, n, db, b, Jx, Jy, Jz, solver.decomp22)
    bs = np.linspace(0, b, len(Bs))
    Gs = np.log(Bs)
    Hs = 1/(bs[1]-Bs[0]) * (Gs[1:]-Gs[:len(Gs)-1])
    bs = (bs[1:] + bs[:len(bs)-1])/2


    #Fs = solver.E_zs(N_psi, n, db, b, Jx, Jy, Jz, solver.decomp22)
    #bs = np.linspace(0, b, len(Fs))
    plt.plot(bs, Hs, label = f"$n={n}$")
    #plt.plot(bs, Fs, label = f"$n={n}$")

plt.xlabel("$\\beta$")
plt.ylabel("$\\langle E\\rangle(\\beta)$")
plt.xscale("log")

plt.title(f"$J_x={Jx}, J_y={Jy}, J_z={Jz}, d\\beta = {db}, N_\\psi = {N_psi}$")
plt.legend()
plt.show()"""