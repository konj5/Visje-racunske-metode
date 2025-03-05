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


from solver import mat_v_bazi22 as basic, mat_v_bazi2 as basic_herm, Lanczos_method as lanczos, eigenstate





"""Nmaxs = np.int32(10**np.linspace(0,2.1,10))[::-1] + 4
t_basic, t_basic_h, t_lancz_0, t_lancz_r = [], [], [], []

for i in tqdm(range(len(Nmaxs))):
    Nmax = Nmaxs[i]

    stime = time.time()
    basic(Nmax,1)
    t_basic.append(time.time()-stime)

    stime = time.time()
    basic_herm(Nmax,1)
    t_basic_h.append(time.time()-stime)


    psi0 = np.zeros(Nmax)
    psi0[0] = 1
    stime = time.time()
    lanczos(psi0,Nmax,1)
    t_lancz_0.append(time.time()-stime)

    psi0 = np.random.random(Nmax)
    psi0 = psi0/np.linalg.norm(psi0)
    stime = time.time()
    lanczos(psi0,Nmax,1)
    t_lancz_r.append(time.time()-stime)


plt.plot(Nmaxs, t_basic, label = "scipy.linalg.eig")
plt.plot(Nmaxs, t_basic_h, label = "scipy.linalg.eigh")
plt.plot(Nmaxs, t_lancz_0, label = "Lanzcos 0")
plt.plot(Nmaxs, t_lancz_r, label = "Lanzcos random")

plt.legend()
plt.xlabel("$N$")
plt.ylabel("Čas računanja")
plt.xscale("log")
plt.yscale("log")
plt.show()"""


"""Nmax = 100
lamb = 0.01

es, fis = basic(Nmax,lamb)
plt.scatter([i for i in range(len(es))], es[::-1], label = "scipy.linalg.eig")
es, fis = basic_herm(Nmax,lamb)
plt.scatter([i for i in range(len(es))], es, label = "scipy.linalg.eigh")



psi0 = np.zeros(Nmax)
psi0[0] = 1
es, fis = lanczos(psi0,Nmax,lamb)
plt.scatter([i for i in range(len(es))], es, label = "Lanzcos 0")

psi0 = np.random.random(Nmax)
psi0 = psi0/np.linalg.norm(psi0)
es, fis = lanczos(psi0,Nmax,lamb)
plt.scatter([i for i in range(len(es))], es, label = "Lanzcos random")

plt.legend()
plt.xlabel("Zaporedno št. stanja")
plt.ylabel("Energija")
plt.yscale("log")
plt.show()
"""


"""Nmaxs = np.arange(10,200,20)

lamb = 1

for i in tqdm(range(len(Nmaxs))):
    Nmax = Nmaxs[i]
    es, fis = basic_herm(Nmax,lamb)
    plt.scatter([i for i in range(len(es))], es, label = f"$N = {Nmax}$", s = 3)

plt.legend()
plt.xlabel("Zaporedno št. stanja")
plt.ylabel("Energija")
plt.yscale("log")
plt.title(f"$\\lambda = {lamb}$")
plt.show()"""


Nmax = 200

lambs = [0,0.1,0.5,1,2]

for i in tqdm(range(len(lambs))):
    lamb = lambs[i]
    es, fis = basic_herm(Nmax,lamb)
    #plt.scatter([i for i in range(len(es))], es, label = f"$\\lambda = {lamb}$", s = 3)
    plt.scatter([i for i in range(len(es))], es, label = f"$\\lambda = {lamb}$", s = 6)

plt.legend()
plt.xlabel("Zaporedno št. stanja")
plt.ylabel("Energija")
plt.yscale("log")

plt.show()




"""def is_it_in(lamb,E):
    xmax = np.sqrt(np.sqrt(1/(16*lamb**2)+E/lamb)-1/(4*lamb))
    xs = np.linspace(0,xmax,100)
    ps = np.sqrt(2*E - xs**2 - 2*lamb*xs**4)

    rs2 = xs**2 + ps**2

    truths = rs2 > 2*E

    if int(np.sum(truths)) != 0:
        return False
    return True

def max_fittable_E(lamb):
    Emin = 0
    Emax = 10**6

    while(Emax - Emin > 0.1):
        Emid = Emin/2 + Emax/2
        val = is_it_in(lamb,Emid)
        if val:
            Emin = Emid
        else:
            Emax = Emid

    return Emid

def mu_surface(lamb, E):
    xmax = np.sqrt(np.sqrt(1/(16*lamb**2)+E/lamb)-1/(4*lamb))
    xs = np.linspace(0,xmax,500)

    return 4 * np.trapz(np.sqrt(2*E - xs**2 - 2*lamb*xs**4), xs)
"""


