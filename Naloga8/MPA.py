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


###############################



THIS IS BROKEN DOES NOT WORK LOOK AT NALOGA9!!!!


#############################

s0 = np.eye(2, dtype=np.complex128)
sx = np.array([[0,1],[1,0]], dtype=np.complex128)
sy = np.array([[0,-1j],[1j,0]], dtype=np.complex128)
sz = np.array([[1,0],[0,-1]], dtype=np.complex128)

@njit
def comb_product(A,B):
    N = np.shape(A)[0]
    M = np.shape(B)[0]

    AB = np.zeros((N*M, N*M), dtype=np.complex128)

    for i in range(N):
        for j in range(N):
            AB[i*M:i*M+M, j*M:j*M+M] = A[i,j] * B

    return AB

@njit
def tensor(ops):
    for k in range(len(ops)-1):
        A = ops[-1-k-1]
        B = ops[-1-k]
        ops[-1-k-1] = comb_product(A,B)

    return ops[0]

@njit
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


@njit
def to_decimal_state(binarr):
    val = 0
    for i in range(len(binarr)):
        val += binarr[-(i+1)] * 2**i
    return val

from numpy.linalg import svd

def from_kindex(n):
    arr = []
    arr.append(n // 2)
    arr.append(n - 2*arr[-1])
    return arr

def to_kindex(state):
    n = state[0]*2
    n += state[1]
    return n




def MPA(state, N):

    A0s = []
    A1s = []
    Ms = []
    
    ####step 1:

    #converting state to matrix form
    active_state = np.copy(state)
    Psi = np.zeros((2,2**(N-1)), dtype=np.complex128)
    for n2 in range(2**(N-1)):
        substate2 = to_binary_state(n2, N-1)
        for n1 in range(2):
            substate1 = to_binary_state(n1,1)

            combstate = np.append(substate1, substate2)
            combn = to_decimal_state(combstate)
            Psi[n1,n2] = active_state[combn]

    #first svd
    U, lamb, Vt = svd(Psi)
    Ms.append(len(lamb))

    #converting indices of U into form of A
    A00 = np.zeros((1,Ms[-1]), dtype=np.complex128)
    A00 += U[0,:]
    A0s.append(A00)

    A10 = np.zeros((1,Ms[-1]), dtype=np.complex128)
    A10 += U[1,:]
    A1s.append(A10)


   

    ####other steps
    
    for j in range(1,N-1):
        #converting indices of Psi into form for SVD
        nextPsi = np.zeros((Ms[-1]*2,2**(N-j-1)), dtype=np.complex128)
        for n2 in range(2**(N-j-1)):
            substate2 = to_binary_state(n2, N-2)
            for n1 in range(Ms[-1]*2):
                substate1 = from_kindex(n1)


                k1 = substate1[0]
                combstate = np.append(np.array([substate1[1]]), substate2)
                combn = to_decimal_state(combstate)        

                nextPsi[n1,n2] = lamb[k1] * Vt[k1, combn]

        Psi_j = nextPsi.copy()

        #other SVD-s
        U, lamb, Vt = svd(Psi_j)

        Ms.append(len(lamb))

        #converting indices of U into form of A
        A0j = np.zeros((Ms[-2],Ms[-1]), dtype=np.complex128)
        A1j = np.zeros((Ms[-2],Ms[-1]), dtype=np.complex128)
        
        for k1 in range(Ms[-2]):
            for k2 in range(Ms[-1]):

                n0 = to_kindex([k1,0])
                A0j[k1,k2] = U[n0, k2]
                A1j[k1,k2] = U[n0+1, k2]

        A0s.append(A0j)
        A1s.append(A1j)

    if N == 2:
        ####last step
        #converting indices of U into form of A 
        A00 = np.zeros((Vt[:,0].shape[0],1), dtype=np.complex128)
        A00 = A00.T
        A00 += Vt[:,0] * lamb
        A0s.append(A00.T)

        A10 = np.zeros((Vt[:,1].shape[0],1), dtype=np.complex128)
        A10 = A10.T
        A10 += Vt[:,1] * lamb
        A1s.append(A10.T)
        
        return A0s, A1s


    ####last step
    #converting indices of U into form of A
    A00 = np.zeros((Psi_j[:,0].shape[0],1), dtype=np.complex128)
    A00 = A00.T
    A00 += Psi_j[:,0]
    A0s.append(A00.T)

    A10 = np.zeros((Psi_j[:,1].shape[0],1), dtype=np.complex128)
    A10 = A10.T
    A10 += Psi_j[:,1]
    A1s.append(A10.T)



    return A0s, A1s # A0s -> A matrike za s_i = 0,        A1s -> A matrike za s_i = 1

"""A0s, A1s = MPA([1/np.sqrt(2),0,1/np.sqrt(2),0,0,0,0,0],3)

for A0 in A0s:
    print(A0)"""


# Naključno stanje N spinov
"""N = 2
state = np.random.normal(0,1,2**N)
state = state/np.linalg.norm(state)
A0s, A1s = MPA(state,N)


print(state)
for A0 in A0s:
    print(np.real(A0))
    print("\n")
for A1 in A1s:
    print(np.real(A1))"""


# Osnovno stanje heisenberg n=2

state = np.array([0,0.707,-0.707,0])
A0s, A1s = MPA(state,2)


print(state)
for A0 in A0s:
    print(np.real(A0))
    print("\n")
for A1 in A1s:
    print(np.real(A1))
    print("\n")




            


        





        
