import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import scipy.special
from tqdm import tqdm
import re, sys

from scipy.linalg import solve

np.set_printoptions(linewidth=10000, threshold=sys.maxsize)

def basedwise_scalar_product(i, state, coefs):
    mid = int(len(coefs)//2)
    val = 0
    for k in range(len(coefs)):
        try:
            val += state[i-mid+k] * coefs[k]
        except IndexError:
            pass
    
    return val



def finite_differencer(startstate, V, tau, h, nmax, spacewise = [1/2, -1, 1/2]):
    state = np.zeros((len(startstate), nmax), dtype=np.complex128)
    state[:,0] = startstate
    for n in range(1,nmax,1):
        print(n)
        for m in range(len(startstate)):
            if m < len(spacewise) or m > len(startstate)-1-len(spacewise):
                state[m,n]
                continue

            state[m,n] = state[m,n-1] + 1j*tau * (1/h**2 * basedwise_scalar_product(m, state[:,n-1], spacewise) - V[m,n])
            #state[m,n] = state[m,n-1] + 1j*tau * (1/h**2 * (state[m+1,n-1]-2*state[m,n-1]+state[m-1,n-1]) - V[m,n])

    return state


def finite_propagatorer(startstate, V, tau, h, nmax, spacewise = [1/2, -1, 1/2]):
    K = 10

    HmatrixP = np.zeros((len(startstate), len(startstate)), dtype=np.complex128)
    ###momentum part
    for i in range(len(spacewise)):
        mid = int(len(spacewise)//2)

        HmatrixP += -1/2 * 1/h**2 * np.eye(len(startstate), k=i-mid, dtype=float) * spacewise[i]

    state = np.zeros((len(startstate), nmax), dtype=np.complex128)
    state[:,0] = startstate

    for n in range(1,nmax,1):
        print(f"{n}/{nmax}")
        Hk = np.eye(len(startstate))

        #Potential part
        HmatrixV = np.diag(V[:,n])
        Hmatrix = HmatrixP + HmatrixV

        
        for k in range(0,K+1,1):

            state[:,n] += (-1j * tau)**k/scipy.special.factorial(k) * Hk.dot(state[:,n-1])
            Hk = Hk.dot(Hmatrix)

    return state

def implicinator(startstate, V, tau, h, nmax, spacewise = [1/2, -1, 1/2]):
    HmatrixP = np.zeros((len(startstate), len(startstate)), dtype=np.complex128)
    ###momentum part
    for i in range(len(spacewise)):
        mid = int(len(spacewise)//2)
        HmatrixP += -1/2 * 1/h**2 * np.eye(len(startstate), k=i-mid, dtype=float) * spacewise[i]

    state = np.zeros((len(startstate), nmax), dtype=np.complex128)
    state[:,0] = startstate

    print("\n")

    for n in range(1,nmax,1):
        print(f"\r{n}/{nmax}", end="                 ")
        #Potential part
        HmatrixV = np.diag(V[:,n])
        Hmatrix = HmatrixP + HmatrixV


        b = (np.eye(len(startstate)) - 1j*tau/2 * Hmatrix).dot(state[:,n-1])

        A = (np.eye(len(startstate)) + 1j*tau/2 * Hmatrix)

        state[:,n] = solve(A,b,assume_a="banded")

    return state



def finite_propagatonator2D_single_step(startstate, V, tau, h, tstep, spacewise = [1/2, -1, 1/2]):
    K = 1
    state = startstate.copy()

    def H_ij(i,j):
        ret = 0

        if np.abs(i-j) < len(spacewise)/2:
            ret += -1/2 * 1/h**2 * spacewise[i-j]
        
        if i == j:
            ret += V[i,j]

        return ret
    
    def matrix_vector_multiplication(A,b):
        bnew = np.zeros_like(b)
        for i in range(len(b)):
            for j in range(len(b)):
                bnew[i] += A(i,j)*b[j]
        return bnew

    t = 0
    while t < tstep:
        print(f"\r{np.round(t,decimals=2):0.2f}/{tstep}", end="            ")
        t += tau
        Hkpsi = state.copy()
        new_state = np.zeros_like(state)
        for k in range(0,K+1,1):
            new_state += (-1j * tau)**k/scipy.special.factorial(k) * Hkpsi
            Hkpsi = matrix_vector_multiplication(H_ij, Hkpsi)
        state = new_state.copy()


    return state






