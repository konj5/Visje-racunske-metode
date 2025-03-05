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




from scipy.linalg import eigh, eig

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
            #print(f"{i}, {j}")
            H[i,j] += lamb * mat_element(i,j,lambda x:x**4)
            H[j,i] = np.conj(H[i,j])

    eigvals, eigvects = eigh(H)

    return eigvals, eigvects

def mat_v_bazi22(Nmax, lamb): #Če je H' simetričen
    Es = np.array([n + 1/2 + lamb * mat_element_fsymmetric(n,n,lambda x:x**4) for n in range(Nmax)], dtype=complex)
    H = np.diag(Es)

    for i in range(Nmax):
        for j in range(i):
            #print(f"{i}, {j}")
            H[i,j] += lamb * mat_element_fsymmetric(i,j,lambda x:x**4)
            H[j,i] = np.conj(H[i,j])

    eigvals, eigvects = eig(H)

    return eigvals, eigvects

def mat_v_bazi2(Nmax, lamb): #Če je H' simetričen
    Es = np.array([n + 1/2 + lamb * mat_element_fsymmetric(n,n,lambda x:x**4) for n in range(Nmax)], dtype=complex)
    H = np.diag(Es)

    for i in range(Nmax):
        for j in range(i):
            #print(f"{i}, {j}")
            H[i,j] += lamb * mat_element_fsymmetric(i,j,lambda x:x**4)
            H[j,i] = np.conj(H[i,j])

    eigvals, eigvects = eigh(H)

    return eigvals, eigvects

def mat_v_bazi3(Nmax, lamb): #Če je H' antisimetričen
    Es = np.array([n + 1/2 + lamb * mat_element_fantisymmetric(n,n,lambda x:x**4) for n in range(Nmax)], dtype=complex)
    H = np.diag(Es)

    for i in range(Nmax):
        for j in range(i):
            #print(f"{i}, {j}")
            H[i,j] += lamb * mat_element_fantisymmetric(i,j,lambda x:x**3)
            H[j,i] = np.conj(H[i,j])

    eigvals, eigvects = eigh(H)

    return eigvals, eigvects


def shooting_method_Numerov(Emax, L, npoints, V, lamb):
    xs = np.linspace(-L, L, npoints)
    Vs = xs**2/2 + lamb * xs**4

    def evolvesol(E, psi1):
        ks2 = 2 * (E - Vs)
        dx = xs[1]-xs[0]

        psis = np.zeros_like(xs, dtype=complex)
        psis[1] = psi1

        for i in range(2,len(xs)):
            psis[i] = (2 * (1-5*dx**2/12*ks2[i-2]) * psis[i-2] - (1+ dx**2/12 * ks2[i-1]) * psis[i-1]) / (1 + dx**2/12 * ks2[i])

    # TO-DO (KILL ME I HATE THIS METHOD!)

def getHmatSym(Nmax, lamb):
    Es = np.array([n + 1/2 + lamb * mat_element_fsymmetric(n,n,lambda x:x**4) for n in range(Nmax)], dtype=complex)
    H = np.diag(Es)

    for i in range(Nmax):
        for j in range(i):
            #print(f"{i}, {j}")
            H[i,j] += lamb * mat_element_fsymmetric(i,j,lambda x:x**4)
            H[j,i] = np.conj(H[i,j])
    return H
    
def Lanczos_method(psi0, Nmax, lamb):
    vects = np.zeros((Nmax, Nmax), dtype=complex) # (number of basis vectors, components in OG eigenbasis)
    vects[0,:] = psi0
    
    H = getHmatSym(Nmax, lamb)

    newH = np.zeros_like(H, dtype=complex)

    #Generate basis
    for i in range(1, Nmax):
        a = vects[i-1,:].dot(H.dot(vects[i-1,:]))
        b = vects[i-2,:].dot(H.dot(vects[i-1,:]))

        newH[i,i] = a
        newH[i,i-1] = b
        newH[i-1,i] = np.conj(b)

        vects[i,:] = H.dot(vects[i-1,:]) - a * vects[i-1,:]
        if i != 1:
            vects[i,:] += -b * vects[i-2,:]

        vects[i,:] /= np.linalg.norm(vects[i,:])

    eigvals, eigvects = eigh(newH)

    return eigvals, eigvects

    

    




