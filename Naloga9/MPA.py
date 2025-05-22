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

    As = []
    Ms = []

    #create starting Psi

    Psi = np.zeros((2,2**(N-1)), dtype=np.complex128)
    for n2 in range(2**(N-1)):
        for n1 in range(2):

            Psi[n1,n2] = state[2**(N-1) * n1 + n2]

    
    ######################first split
    U, lamb, Vt = svd(Psi)
    Ms.append(len(lamb))

    #Assingn A
    As.append(np.einsum("ij->ji", U))

    #####################General step

    for i in range(1,N-1):
        
        #redefine Psi
        Psi = np.zeros((Ms[-1]*2, 2**(N-1-i)), dtype=np.complex128)
        for n2 in range(2**(N-1-i)):
            for n1 in range(Ms[-1]*2):
                k, s = from_kindex(n1)
                Psi[n1,n2] = lamb[k] * Vt[k, 2**(N-1-i) * s + n2]
        
        #svd
        U, lamb, Vt = svd(Psi)
        Ms.append(len(lamb))
    
        #Assign A
        A = np.zeros((Ms[-2], Ms[-1], 2), dtype=np.complex128)
        for s in range(2):
            for k1 in range(Ms[-2]):
                for k2 in range(Ms[-1]):
                    A[k1,k2,s] = U[to_kindex([k1,s]), k2]
        As.append(A)
    
    ############# Last step
    A = np.zeros((Ms[-1], 2), dtype=np.complex128)
    for s in range(2):
        for k in range(Ms[-1]):
            A[k,s] = lamb[k] * Vt[k, s]
    As.append(A)

    return As


def MPA_B_form(state, N):

    Bs = []
    lambs = []
    Ms = []

    #create starting Psi

    Psi = np.zeros((2,2**(N-1)), dtype=np.complex128)
    for n2 in range(2**(N-1)):
        for n1 in range(2):

            Psi[n1,n2] = state[2**(N-1) * n1 + n2]

    
    ######################first split
    U, lamb, Vt = svd(Psi)
    Ms.append(len(lamb))

    #Assingn A
    A = np.einsum("ij->ji", U)
    B = np.einsum("ij,jk->ik", np.diag(1/lamb), A)
    Bs.append(B)
    lambs.append(lamb)

    #####################General step

    for i in range(1,N-1):
        
        #redefine Psi
        Psi = np.zeros((Ms[-1]*2, 2**(N-1-i)), dtype=np.complex128)
        for n2 in range(2**(N-1-i)):
            for n1 in range(Ms[-1]*2):
                k, s = from_kindex(n1)
                Psi[n1,n2] = lamb[k] * Vt[k, 2**(N-1-i) * s + n2]
        
        #svd
        U, lamb, Vt = svd(Psi)
        Ms.append(len(lamb))
    
        #Assign A
        A = np.zeros((Ms[-2], Ms[-1], 2), dtype=np.complex128)
        for s in range(2):
            for k1 in range(Ms[-2]):
                for k2 in range(Ms[-1]):
                    A[k1,k2,s] = U[to_kindex([k1,s]), k2]
        
        B = np.einsum("ij,jk->ik", np.diag(1/lamb), A)
        Bs.append(B)
        lambs.append(lamb)
    
    ############# Last step
    A = np.zeros((Ms[-1], 2), dtype=np.complex128)
    for s in range(2):
        for k in range(Ms[-1]):
            A[k,s] = lamb[k] * Vt[k, s]
    
    B = np.einsum("ij,jk->ik", np.diag(1/lamb), A)
    Bs.append(B)
    lambs.append(lamb)

    return Bs, lambs



"""N = 4
s1 = np.random.normal(0,1,2**N); s1 = s1/np.linalg.norm(s1)

As = MPA(s1,N)

for i in range(N):
    try:
        print(As[i][:,:,0].shape)
    except:
        print(As[i][:,0].shape)"""


"""v = np.array([0,1,1,0])
M = np.array([[0,1,0,1], [0,0,4,4], [3,2,3,6], [1,0,0,1]])

print(np.einsum("i,ij->ij", v, M))
print(np.einsum("ij,jk->ik", np.diag(v), M))"""




            


        





        
