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


"""startstate = [0,0.5,1,0]

dts = [1,0.1,0.05,0.01]




tmax = 40
lamb = 10

decomps = [decomp11, decomp22, decomp44, decomp33, decomp45]
decompnames = ["razcep11", "razcep22", "razcep44", "razcep33", "razcep45"]

fig, axs = plt.subplots(2,2, figsize = (4*2,4*2))

for i in tqdm(range(len(dts))):
    dt = dts[i]
    for j in tqdm(range(len(decomps)), leave=False):
        decomp = decomps[j]
        dname = decompnames[j]
        ts, states = integrate(startstate, dt, tmax, lamb, decomp)
        Hs = H(states, lamb)
        axs[i%2,i//2].plot(ts, np.abs(Hs-Hs[0]), label = f"{dname}")
        axs[i%2,i//2].set_xlabel("t")
        axs[i%2,i//2].set_ylabel("$|E(t)-E(0)|$")
        axs[i%2,i//2].set_yscale("log")
        axs[i%2,i//2].set_title(f"$dt = {dt}$")
        axs[i%2,i//2].legend()

plt.tight_layout()
plt.savefig("econserv closeup lamb = 10.png")
plt.show()"""


"""startstate = [0,0.5,1,0]

dts = 10**np.linspace(-3,0,100)


tmax = 40
lamb = 0

decomps = [decomp11, decomp22, decomp44, decomp33, decomp45]
decompnames = ["razcep11", "razcep22", "razcep44", "razcep33", "razcep45"]


for i in tqdm(range(len(decomps))):
    decomp = decomps[i]
    dname = decompnames[i]
    dHs = []
    for j in tqdm(range(len(dts)), leave=False):
        dt = dts[j]
        
        ts, states = integrate(startstate, dt, tmax, lamb, decomp)
        Hs = H(states, lamb)
        dHs.append(np.average(np.abs(Hs-Hs[0])[len(Hs)//2:]))

    plt.plot(dts, dHs, label = f"{dname}")        

plt.xlabel("$dt$")
plt.ylabel("povprečje $|E(t)-E(0)|$")
plt.yscale("log")
plt.xscale("log")
plt.tight_layout()
plt.legend()
plt.show()"""



"""startstate = [0,0.5,1,0]

dts = 10**np.linspace(-3,0,100)


tmax = 40
lamb = 0

decomps = [decomp11, decomp22, decomp44, decomp33, decomp45]
decompnames = ["razcep11", "razcep22", "razcep44", "razcep33", "razcep45"]


for i in tqdm(range(len(decomps))):
    decomp = decomps[i]
    dname = decompnames[i]
    dHs = []
    for j in tqdm(range(len(dts)), leave=False):
        dt = dts[j]
        
        stime = time.time()
        ts, states = integrate(startstate, dt, tmax, lamb, decomp)
        dHs.append(time.time()-stime)

    plt.plot(dts, dHs, label = f"{dname}")        

plt.xlabel("$dt$")
plt.ylabel("čas računanja [s]")
plt.yscale("log")
plt.xscale("log")
plt.tight_layout()
plt.legend()
plt.show()"""
    

startstate = [0,0.5,1,0]

dts = 10**np.linspace(-3,0,50)


tmax = 40
lamb = 10

decomps = [decomp11, decomp22, decomp44, decomp33, decomp45]
decompnames = ["razcep11", "razcep22", "razcep44", "razcep33", "razcep45"]


for i in tqdm(range(len(decomps))):
    decomp = decomps[i]
    dname = decompnames[i]
    dHs = []
    dHsRK = []
    for j in tqdm(range(len(dts)), leave=False):
        dt = dts[j]
        
        ts, states = integrate(startstate, dt, tmax, lamb, decomp)
        Hs = H(states, lamb)
        dHs.append(np.average(np.abs(Hs-Hs[0])[len(Hs)//2:]))

        if i == 0:
            ts, states = integrate_rk45(startstate, dt, tmax, lamb)
            Hs = H(states, lamb)
            dHsRK.append(np.average(np.abs(Hs-Hs[0])[len(Hs)//2:]))


    plt.plot(dts, dHs, label = f"{dname}")
    if i == 0:
        plt.plot(dts, dHsRK, label = f"RK45")        

plt.xlabel("$dt$")
plt.ylabel("povprečje $|E(t)-E(0)|$")
plt.yscale("log")
plt.xscale("log")
plt.tight_layout()
plt.legend()
plt.show()



startstate = [0,0.5,1,0]

dts = 10**np.linspace(-3,0,50)


tmax = 40
lamb = 10

decomps = [decomp11, decomp22, decomp44, decomp33, decomp45]
decompnames = ["razcep11", "razcep22", "razcep44", "razcep33", "razcep45"]


for i in tqdm(range(len(decomps))):
    decomp = decomps[i]
    dname = decompnames[i]
    dHs = []
    dHsRK = []
    for j in tqdm(range(len(dts)), leave=False):
        dt = dts[j]
        
        stime = time.time()
        ts, states = integrate(startstate, dt, tmax, lamb, decomp)
        dHs.append(time.time()-stime)

        if i == 0:
            stime = time.time()
            ts, states = integrate_rk45(startstate, dt, tmax, lamb)
            dHsRK.append(time.time()-stime)


    plt.plot(dts, dHs, label = f"{dname}")
    if i == 0:
        plt.plot(dts, dHsRK, label = f"RK45")         

plt.xlabel("$dt$")
plt.ylabel("čas računanja [s]")
plt.yscale("log")
plt.xscale("log")
plt.tight_layout()
plt.legend()
plt.show()


"""startstate = [0,0.5,1,0]

lambs = np.linspace(0,10,20)

dt = 0.05

tmax = 40


decomps = [decomp11, decomp22, decomp44, decomp33, decomp45]
decompnames = ["razcep11", "razcep22", "razcep44", "razcep33", "razcep45"]


for i in tqdm(range(len(decomps))):
    decomp = decomps[i]
    dname = decompnames[i]
    dHs = []
    dHsRK = []
    for j in tqdm(range(len(lambs)), leave=False):
        lamb = lambs[j]
        
        ts, states = integrate(startstate, dt, tmax, lamb, decomp)
        Hs = H(states, lamb)
        dHs.append(np.average(np.abs(Hs-Hs[0])[len(Hs)//2:]))

        if i == 0:
            ts, states = integrate_rk45(startstate, dt, tmax, lamb)
            Hs = H(states, lamb)
            dHsRK.append(np.average(np.abs(Hs-Hs[0])[len(Hs)//2:]))


    plt.plot(lambs, dHs, label = f"{dname}")
    if i == 0:
        plt.plot(lambs, dHsRK, label = f"RK45")        

plt.xlabel("$\\lambda$")
plt.ylabel("povprečje $|E(t)-E(0)|$")
plt.yscale("log")
plt.tight_layout()
plt.legend()
plt.show()"""














