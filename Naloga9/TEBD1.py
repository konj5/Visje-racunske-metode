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

##################################################################


#                     THIS ONE IS BROKEN!



###################################################################


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

def from_kindex(n):
    arr = []
    arr.append(n // 2)
    arr.append(n - 2*arr[-1])
    return arr

def to_kindex(state):
    n = state[0]*2
    n += state[1]
    return n

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

from numpy.linalg import svd


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
    return new
                    
def applyLocalPropagator(lamb0, B1, lamb1, B2, lamb2, U2):
    #Apply local propagator
    """print(U2.shape)
    print(B1.shape)
    print(lamb1.shape)
    print(B2.shape)
"""
    big_B = np.einsum("abcd,iqc,qw,wjd->ijab", U2, B1, np.diag(lamb1), B2)

    #Reindex
    BB = np.zeros((lamb0.size*2, lamb2.size*2), dtype=np.complex128)
    for n1 in range(lamb0.size*2):
        for n2 in range(lamb2.size*2):
            k1,s1 = from_kindex(n1)
            k2,s2 = from_kindex(n2)
            BB[n1,n2] = lamb0[k1]*big_B[k1,k2,s1,s2]*lamb2[k2]

    #SVD
    U, lamb, Vt = svd(BB)
    
    #New B1, lamb1, B2 matrices
    Mmax = lamb.size
    B1 = np.zeros((lamb0.size, Mmax,2), dtype=np.complex128)
    for k1 in range(lamb0.size):
        for k2 in range(Mmax):
            for s in range(2):
                B1[k1,k2,s] = 1/lamb0[k1] * U[to_kindex([k1,s]),k2]

    lamb1 = lamb[0:Mmax]

    B2 = np.zeros((Mmax, lamb2.size,2), dtype=np.complex128)
    for k1 in range(Mmax):
        for k2 in range(lamb2.size):
            for s in range(2):
                B2[k1,k2,s] = Vt[k1, to_kindex([k2,s])] * 1/lamb2[k2]

    return B1, lamb1, B2


def applyLocalPropagatorGlobally(U2, position1, Bs, lambs):
    if position1 not in [0, len(Bs)-2]:
        B1, lamb, B2 = applyLocalPropagator(lambs[position1-1], Bs[position1], lambs[position1], Bs[position1+1], lambs[position1+1], U2)
        lambs[position1] = lamb
        Bs[position1] = B1
        Bs[position1+1] = B2

    elif position1 == 0:
        B1, lamb, B2 = applyLocalPropagator(np.array([1]), Bs[position1].reshape((1,Bs[position1].shape[-2],2)), lambs[position1], Bs[position1+1], lambs[position1+1], U2)
        lambs[position1] = lamb
        Bs[position1] = B1
        Bs[position1+1] = B2

    elif position1 == len(Bs)-2:
        B1, lamb, B2 = applyLocalPropagator(lambs[position1-1], Bs[position1], lambs[position1], Bs[position1+1].reshape((Bs[position1].shape[0],1,2)), np.array([1]), U2)
        lambs[position1] = lamb
        Bs[position1] = B1
        Bs[position1+1] = B2

    return Bs, lambs


def scalar_product_Bs(Bs1,lambs1,Bs2,lambs2):
    if len(Bs1) == 2:
        active_matrix = Bs1[0]
        active_matrix = np.einsum("ia,ja->ij",active_matrix, Bs2[0])
        active_matrix = np.einsum("ij,ik->kj",active_matrix, np.diag(lambs1[0]))
        active_matrix = np.einsum("kj,jl->kl",active_matrix, np.diag(lambs2[0]))

        active_matrix = np.einsum("kl,ka->la",active_matrix, Bs1[1])
        active_matrix = np.einsum("la,la",active_matrix, Bs2[1])

        return active_matrix

    #start
    

    if len(Bs1[0].shape) == 3: Bs1[0] = np.sum(Bs1[0], axis=0)
    if len(Bs2[0].shape) == 3: Bs2[0] = np.sum(Bs2[0], axis=0)
    if len(Bs1[-1].shape) == 3: Bs1[-1] = np.sum(Bs1[-1], axis=1)
    if len(Bs2[-1].shape) == 3: Bs2[-1] = np.sum(Bs2[-1], axis=1)

    active_matrix = Bs1[0]

    if len(active_matrix.shape) == 3:
        active_matrix = np.einsum("oia,oja->ij",active_matrix, Bs2[0])
        active_matrix = np.einsum("ij,ik->kj",active_matrix, np.diag(lambs1[0]))
        active_matrix = np.einsum("kj,jl->kl",active_matrix, np.diag(lambs2[0]))

        active_matrix = np.einsum("kl,kja->jla",active_matrix, Bs1[1])

    else:
        active_matrix = np.einsum("ia,ja->ij",active_matrix, Bs2[0])
        active_matrix = np.einsum("ij,ik->kj",active_matrix, np.diag(lambs1[0]))
        active_matrix = np.einsum("kj,jl->kl",active_matrix, np.diag(lambs2[0]))

        active_matrix = np.einsum("kl,kja->jla",active_matrix, Bs1[1])

    #general step
    for i in range(1,len(Bs1)-2):
        active_matrix = np.einsum("jla,lia->ji",active_matrix, Bs2[i])
        active_matrix = np.einsum("ji,jk->ki",active_matrix, np.diag(lambs1[i]))
        active_matrix = np.einsum("ki,il->kl",active_matrix, np.diag(lambs2[i]))
        active_matrix = np.einsum("kl,kia->ila",active_matrix, Bs1[i+1])

    #last step
    if len(Bs1[len(Bs1)-1].shape) == 3:
        active_matrix = np.einsum("jla,lia->ji",active_matrix, Bs2[len(Bs1)-2])
        active_matrix = np.einsum("ji,jk->ki",active_matrix, np.diag(lambs1[len(Bs1)-2]))
        active_matrix = np.einsum("ki,il->kl",active_matrix, np.diag(lambs2[len(Bs1)-2]))
        active_matrix = np.einsum("kl,ka->la",active_matrix, np.sum(Bs1[len(Bs1)-1], axis=1))
        active_matrix = np.einsum("la,la",active_matrix, np.sum(Bs2[len(Bs2)-1], axis=1))

    else:
        active_matrix = np.einsum("jla,lia->ji",active_matrix, Bs2[len(Bs1)-2])
        active_matrix = np.einsum("ji,jk->ki",active_matrix, np.diag(lambs1[len(Bs1)-2]))
        active_matrix = np.einsum("ki,il->kl",active_matrix, np.diag(lambs2[len(Bs1)-2]))

        """print(active_matrix.shape)
        print(Bs1[-1].shape)
        print(Bs2[-1].shape)"""
        active_matrix = np.einsum("kl,ka->la",active_matrix, Bs1[len(Bs1)-1])
        active_matrix = np.einsum("la,la",active_matrix, Bs2[len(Bs2)-1])

    return active_matrix

"""Bs, lambs = MPA_B_form([0,1,1,0], 2)
print(Bs)
print(scalar_product_Bs(Bs,lambs,Bs,lambs))"""



def normalize(x):
    return x / np.linalg.norm(x)

def propagate_imag_time(state, N, z):
    Bs, lambs = MPA_B_form(state, N)

    dz = 0.1 * np.sign(z)
    betas = -np.arange(0, z, dz)
    iters = len(betas)
    evals = np.zeros(betas.shape)

    U4i = four_index_propagator(dz)

    initBs = list.copy(Bs)
    initlambs = list.copy(lambs)
    randBs, randlambs = MPA_B_form(normalize(np.random.normal(0,1,len(state))), N)
    initrandscala = scalar_product_Bs(randBs,randlambs,initBs,initlambs)

    for k in trange(0,iters, desc="steps"):
        #Even terms
        for i in range(0, len(Bs)-1, 2):
            Bs, lambs = applyLocalPropagatorGlobally(U4i, i, Bs, lambs)

        #Odd terms
        for i in range(1, len(Bs)-1, 2):
            Bs, lambs = applyLocalPropagatorGlobally(U4i, i, Bs, lambs)

        evals[k] = scalar_product_Bs(randBs,randlambs,Bs,lambs) / initrandscala
    
    return betas, evals

betas, evals = propagate_imag_time([1,0,1,0,0,1,0,1], 3, -1)
print(evals)
plt.plot(betas, -1/betas * np.log(evals))
plt.show()





    





