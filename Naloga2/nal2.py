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





"""Nmaxs = np.int32(10**np.linspace(0,2.3,100))[::-1] + 4
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
plt.show()
"""

"""Nmax = 200
lamb = 1

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
plt.show()"""



"""Nmaxs = np.arange(10,250,20)

lamb = 0.1

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


"""Nmax = 200

lambs = [0,0.1,0.5,1.2]

for i in tqdm(range(len(lambs))):
    lamb = lambs[i]
    es, fis = basic_herm(Nmax,lamb)
    plt.scatter([i for i in range(len(es))], es, label = f"$\\lambda = {lamb}$", s = 3)

    #c = np.polyfit([i for i in range(len(es[0:25]))],es[0:25],deg=2)
    #xs = np.linspace(0,25,100)
    #plt.plot(xs,c[0]*xs**2 + c[1]*xs+c[0], linestyle = "dashed", c ="gray")

    #xs = np.linspace(0,25,100)
    #plt.plot(xs,xs + 1/2 + lamb /2**4 * (7*xs**2 + 7*xs+3), c ="gray")

    #plt.scatter([i for i in range(len(es[0:25]))], es[0:25], label = f"$\\lambda = {lamb}$", s = 6)




plt.legend()
plt.xlabel("Zaporedno št. stanja")
plt.ylabel("Energija")
plt.yscale("log")

#plt.xlim(-5,26)

plt.show()"""



"""Nmax = 200

lambs = [0,0.5,1,2]

N = 

xs = np.linspace(-5,5,1000)

basis = np.zeros((Nmax, len(xs)), dtype=complex)
for i in range(Nmax):
    basis[i,:] = eigenstate(i,xs)

for i in tqdm(range(len(lambs))):
    lamb = lambs[i]
    es, fis = basic_herm(Nmax,lamb)

    ys = np.zeros_like(xs, dtype=complex)
    for j in range(Nmax):
        ys += fis[j,N] * basis[j,:]

    plt.plot(xs,np.abs(ys)**2, label = f"$\\lambda = {lamb}$")

plt.title("Osnovno stanje" if N == 0 else f"{N:0.0f}. vzbujeno stanje")

plt.legend()
plt.xlabel("x")
plt.ylabel("$|\\psi|^2$")

plt.show()"""


"""import matplotlib.pyplot
from numpy import arange
from numpy import meshgrid

fig, ax = plt.subplots(1,1)
delta = 0.025
xrange = arange(-5, 5, delta)
yrange = arange(-5, 5, delta)
X, Y = meshgrid(xrange,yrange)

# F is one side of the equation, G is the other
F = (X*X + Y*Y)/2
G = 2*np.pi*1
cntr1 = matplotlib.pyplot.contour(X, Y, (F - G), [0], label = "$\\lambda = 0$")

F = (X*X + Y*Y)/2 + 0.1 * X*X*X*X
G = 2*np.pi*1
cntr2 = matplotlib.pyplot.contour(X, Y, (F - G), [0], label = "$\\lambda = 0.1$", colors = "red")

F = (X*X + Y*Y)/2 + 0.5 * X*X*X*X
G = 2*np.pi*1
cntr3 = matplotlib.pyplot.contour(X, Y, (F - G), [0], label = "$\\lambda = 0.5$", colors = "blue")

F = (X*X + Y*Y)/2 + 1 * X*X*X*X
G = 2*np.pi*1
cntr4 = matplotlib.pyplot.contour(X, Y, (F - G), [0], label = "$\\lambda = 1$", colors = "green")

h1,_ = cntr1.legend_elements()
h2,_ = cntr2.legend_elements()
h3,_ = cntr3.legend_elements()
h4,_ = cntr4.legend_elements()
ax.legend([h1[0], h2[0], h3[0], h4[0]], ["$\\lambda = 0$", "$\\lambda = 0.1$", "$\\lambda = 0.5$", "$\\lambda = 1$"])

matplotlib.pyplot.axis("equal")
plt.xlabel("x")
plt.ylabel("p")
matplotlib.pyplot.show()"""




def is_it_in(lamb,E,N):
    En = N + 1/2

    xmax = np.sqrt(np.sqrt(1/(16*lamb**2)+E/lamb)-1/(4*lamb))
    xs = np.linspace(0,xmax,100)
    ps = np.sqrt(2*E - xs**2 - 2*lamb*xs**4)

    rs2 = xs**2 + ps**2

    truths = rs2 > 2*En

    if int(np.sum(truths)) != 0:
        return False
    return True

#I SOLVED THIS ANALYTICALLY LATER SO USELESS!
def max_fittable_E(lamb,N):
    Emin = 0
    Emax = 10**6

    while(Emax - Emin > 0.1):
        Emid = Emin/2 + Emax/2
        val = is_it_in(lamb,Emid,N)
        if val:
            Emin = Emid
        else:
            Emax = Emid

    return Emid

def mu_surface(lamb, E):
    xmax = np.sqrt(np.sqrt(1/(16*lamb**2)+E/lamb)-1/(4*lamb)) * 0.99


    #xmax = 1/2 * np.sqrt((np.sqrt(16*E*lamb+1)-1)/lamb)

    xs = np.linspace(0,xmax,500)

    return 4 * np.trapz(np.sqrt(2*E - xs**2 - 2*lamb*xs**4), xs) / 0.99


"""lamb = 1
Nmaxs = np.arange(10,200,20)

N_count = []
N_smc = []

r_count = []
r_smc = []


for i in tqdm(range(len(Nmaxs))):
    Nmax = Nmaxs[i]
    es, fis = basic_herm(Nmax,lamb)

    Emax = Nmax + 1/2

    N_count.append(np.sum(es < Emax))
    r_count.append(np.sum(es < Emax) / len(es))
    
    I = mu_surface(lamb,Emax)
    N_smc.append(I / (2*np.pi))
    r_smc.append(I / (2*np.pi*Emax))

print(N_smc)

plt.plot(Nmaxs, N_count, label = "iz spektra")
plt.plot(Nmaxs, N_smc, label = "samiklasično")
    
plt.legend()
plt.xlabel("Število baznih stanj")
plt.ylabel("Število točnih lastnih stanj")
plt.title(f"$\\lambda = {lamb}$")
plt.show()

plt.plot(Nmaxs, r_count, label = "iz spektra")
plt.plot(Nmaxs, r_smc, label = "samiklasično")
    
plt.legend()
plt.xlabel("Število baznih stanj")
plt.ylabel("Delež točnih lastnih stanj")
plt.title(f"$\\lambda = {lamb}$")
plt.show()"""



"""Nmax = 50
lambs = [0.05,0.1,0.5,1,2]

N_count = []
N_smc = []

r_count = []
r_smc = []


for i in tqdm(range(len(lambs))):
    lamb = lambs[i]
    es, fis = basic_herm(Nmax,lamb)

    Emax = Nmax + 1/2

    N_count.append(np.sum(es < Emax))
    r_count.append(np.sum(es < Emax) / len(es))
    
    I = mu_surface(lamb,Emax)
    N_smc.append(I / (2*np.pi))
    r_smc.append(I / (2*np.pi*Emax))

print(N_smc)

plt.plot(lambs, N_count, label = "iz spektra")
plt.plot(lambs, N_smc, label = "samiklasično")
    
plt.legend()
plt.xlabel("$\\lambda$")
plt.ylabel("Število točnih lastnih stanj")
plt.title(f"$N = {Nmax}$")
plt.show()

plt.plot(lambs, r_count, label = "iz spektra")
plt.plot(lambs, r_smc, label = "samiklasično")
    
plt.legend()
plt.xlabel("$\\lambda$")
plt.ylabel("Delež točnih lastnih stanj")
plt.title(f"$N = {Nmax}$")
plt.show()"""
