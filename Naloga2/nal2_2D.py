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
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

def val_from_sequence(seq,n1,n2):
    val = 1
    for op in seq[::-1]:
        if op == "b":
            val *= np.sqrt(n2+1)
            n2 = n2 +1
        elif op == "a":
            val *= np.sqrt(n2)
            n2 = n2 -1
            if n2 < 0: return 0
    if n1 != n2:
        val = 0

    return val

seqs = ["aa", "ab", "ba", "bb"]
def mat_element(nx1,ny1,nx2,ny2):

    valx = 0
    for seq in seqs:
        valx += val_from_sequence(seq,nx1,nx2)

    valx = valx / 2

    if valx == 0: return 0

    valy = 0
    for seq in seqs:
        valy += val_from_sequence(seq,ny1,ny2)

    valy = valy / 2

    return valx * valy

def n_to_nxny(n):
    k = 0
    dk = 0
    while k <= n:
        dk += 1
        k += dk

    sumindex = dk-1
    s = k - dk
    delta = n - s

    ny = delta
    nx = sumindex - delta

    return nx,ny



def generateHamiltonian(Nmax, lamb):
    dim = int((Nmax+1) * (Nmax+2)//2)
    H = np.zeros((dim,dim))

    for n1 in range(dim):
        nx1, ny1 = n_to_nxny(n1)
        for n2 in range(dim):
            nx2, ny2 = n_to_nxny(n2)

            if n1 == n2:
                H[n1,n2] += 1 + nx1 + ny1

            H[n1,n2] += lamb * mat_element(nx1,ny1,nx2,ny2)

    return H


"""H = generateHamiltonian(3,0)
print(H)
energies, vects = np.linalg.eigh(H)
print(energies)"""


"""lambs = [0,0.5,1,2]
for i in tqdm(range(len(lambs))):
    lamb = lambs[i]

    H = generateHamiltonian(40,lamb)
    energies, vects = np.linalg.eigh(H)

    plt.scatter([i for i in range(len(energies))], energies, label = f"$\\lambda = {lamb}$", s = 3)

plt.xlabel("Zaporedno število stanja")
plt.ylabel("Energija")
plt.legend()
plt.yscale("log")
plt.show()"""
"""
lamb = 1
E = 1

def monte_carlo_integrator(f, fmax, box_halfradius):
    Nhit = 0
    N = 0

    V0 = (2*box_halfradius)**4

    for i in tqdm(range(10000), desc="integrating", leave=False):
        point = (2*np.random.random(4)-1) * box_halfradius
        if f(point) <= fmax:
            Nhit += 1
        N += 1

    return V0 * Nhit/N

def f(v):
    return 1/2 * np.linalg.norm(v) + lamb*v[0]**2*v[1]**2

H = generateHamiltonian(40,lamb)
energies, vects = np.linalg.eigh(H)
energies = energies[:100]
Es = np.linspace(0,energies[-1],100)

N1s, N2s = [], []
for i in tqdm(range(len(Es))):
    N1 = np.sum(energies < Es[i])
    N2 = monte_carlo_integrator(f,Es[i],np.sqrt(Es[i]))/(2*np.pi)**2
    N1s.append(N1)
    N2s.append(N2)

plt.plot(Es, N1s, label = "iz spektra")
plt.plot(Es, N2s, label = "semiklasično")
plt.xlabel("Energija")
plt.ylabel("Število stanj")
plt.legend()
plt.show()"""


lamb = 1
H = generateHamiltonian(50,lamb)
energies, vects = np.linalg.eigh(H)

energies = energies[:500]

dEs = np.abs(energies[:len(energies)-1] - energies[1:])

plt.hist(dEs, bins=np.linspace(0,np.max(dEs),30))
plt.show()


