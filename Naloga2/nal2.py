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


def eigenstate(N, x):
    return 1/(np.pi**(1/4) * np.sqrt(2**N * scipy.special.factorial(N))) * scipy.special.hermite(N)(x) * np.exp(-x**2/2)


def mat_element(n1,n2,f):
    xs = np.linspace(-20,20,1000)
    ys = eigenstate(n1,xs) * f(xs) * eigenstate(n2,xs)

    return np.trapz(ys,xs)

def mat_element_fsymmetric(n1,n2,f):
    if np.logical_xor(n1 % 2 == 1, n2 % 2 == 1): return 0
    xs = np.linspace(-20,20,1000)
    ys = eigenstate(n1,xs) * f(xs) * eigenstate(n2,xs)

    return np.trapz(ys,xs)

def mat_element_fantisymmetric(n1,n2,f):
    if (n1 % 2 == 1) == (n2 % 2 == 1): return 0
    xs = np.linspace(-20,20,1000)
    ys = eigenstate(n1,xs) * f(xs) * eigenstate(n2,xs)

    return np.trapz(ys,xs)

def mat_v_bazi(Nmax, lamb):
    Es = np.array([n + 1/2 + lamb * mat_element(n,n,lambda x:x**4) for n in range(Nmax)], dtype=complex)
    H = np.diag(Es)

    for i in range(Nmax):
        for j in range(i):
            print(f"{i}, {j}")
            H[i,j] += lamb * mat_element(i,j,lambda x:x**4)
            H[j,i] = np.conj(H[i,j])

    eigvals, eigvects = np.linalg.eigh(H)

    return eigvals, eigvects

def mat_v_bazi2(Nmax, lamb): #Če je H' simetričen
    Es = np.array([n + 1/2 + lamb * mat_element_fsymmetric(n,n,lambda x:x**4) for n in range(Nmax)], dtype=complex)
    H = np.diag(Es)

    for i in range(Nmax):
        for j in range(i):
            print(f"{i}, {j}")
            H[i,j] += lamb * mat_element_fsymmetric(i,j,lambda x:x**4)
            H[j,i] = np.conj(H[i,j])

    eigvals, eigvects = np.linalg.eigh(H)

    return eigvals, eigvects

def mat_v_bazi3(Nmax, lamb): #Če je H' antisimetričen
    Es = np.array([n + 1/2 + lamb * mat_element_fantisymmetric(n,n,lambda x:x**4) for n in range(Nmax)], dtype=complex)
    H = np.diag(Es)

    for i in range(Nmax):
        for j in range(i):
            print(f"{i}, {j}")
            H[i,j] += lamb * mat_element_fantisymmetric(i,j,lambda x:x**3)
            H[j,i] = np.conj(H[i,j])

    eigvals, eigvects = np.linalg.eigh(H)

    return eigvals, eigvects

