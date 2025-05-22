import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import scipy.integrate
import scipy.interpolate
import scipy.special
from tqdm import tqdm, trange
import re, sys
from matplotlib.animation import FuncAnimation
import time
import numba
from numba import jit, njit, prange
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))


def to_binary_state(k, n):
    """state = np.zeros(n,dtype=np.byte)
    dat = np.array(list(bin(k)[2:]))
    state[-len(dat):] = dat
    return state"""
    state = np.zeros(n, dtype=np.byte)
    if k == 0:
        return state

    binary = []
    while k != 0:
        bit = k % 2
        binary.insert(0, bit)
        k = k // 2
    state[-len(binary):] = binary
    return state

def to_decimal_state(binarr):
    val = 0
    for i in range(len(binarr)):
        val += binarr[-(i+1)] * 2**i
    return val


from MPA import MPA, MPA_B_form

def scalar_product(state1, state2, N):
    As1 = MPA(state1, N)
    As2 = MPA(state2, N)


    #actual algorithm
    active_matrix = As1[0]

    #start
    active_matrix = np.einsum("ij,kj->ik", active_matrix, As2[0])


    #middle steps
    for i in range(1,N-1):
        #print(active_matrix.shape)
        #print(As1[i].shape)
        #print("----")

        active_matrix = np.einsum("ij,ikl->jkl", active_matrix, As1[i])

        #print(active_matrix.shape)
        #print(As2[i].shape)
        #print("----")


        active_matrix = np.einsum("ikl,iml->km", active_matrix, As2[i])

        #print(active_matrix.shape)
        #print(As1[-1].shape)
        #print("----")


    #end
    active_matrix = np.einsum("ij,il->jl", active_matrix, As1[-1])
    active_matrix = np.einsum("il,il", active_matrix, As2[-1])


    return active_matrix
    

"""N = 5
s1 = np.random.normal(0,1,2**N); s1 = s1/np.linalg.norm(s1)
s2 = np.random.normal(0,1,2**N); s2 = s2/np.linalg.norm(s2)

print(np.einsum("i,i", s1, s2))
print(scalar_product(s1,s2,N))"""


def local_propagator(z):
    return np.array(
        [
            [np.exp(2*z), 0, 0, 0],
            [0, np.cosh(2*z), np.sinh(2*z), 0],
            [0, np.sinh(2*z), np.cosh(2*z), 0],
            [0,0,0,np.exp(2*z)]
        ]
    ) * np.exp(-z)

def four_index_propagator(z):
    new = np.zeros((2,2,2,2), dtype=np.complex128)
    old = local_propagator(z)
    for n1 in range(2):
        for n2 in range(2):
            for n3 in range(2):
                for n4 in range(2):
                    new[n1,n2,n3,n4] = old[to_decimal_state([n1,n2]), to_decimal_state([n3,n4])]
                    
def applyLocalHamiltonian(B1,B2, lamb, U2):
    return np.einsum("ijkl,abk,bc,cdl->adij", (U2, B1, np.diag(lamb), B2))

def hamiltonian(state, N):
    Bs, lambs = MPA_B_form(state, N)


