import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import scipy.integrate
import scipy.interpolate
import scipy.optimize
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

def MPA_B_real(state, N, Mlim):
    Bs, lambs = MPA_B_form(state,N, Mlim)

    Bs[0] = Bs[0].reshape((1,Bs[0].shape[0],Bs[0].shape[1]))
    Bs[-1] = Bs[-1].reshape((Bs[-1].shape[0],1,Bs[-1].shape[1]))
    lambs.insert(0,np.array([1]))
    lambs.append(np.array([1]))

    return Bs, lambs



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


def scalar_product(Bs1,lambs1,Bs2,lambs2):
    if len(Bs1) == 2:
        active_matrix = Bs1[0][0,:,:]
        lambs1 = lambs1[1:-1]
        lambs2 = lambs2[1:-1]
        active_matrix = np.einsum("ia,ja->ij",active_matrix, Bs2[0][0,:,:])
        active_matrix = np.einsum("ij,ik->kj",active_matrix, np.diag(lambs1[0]))
        active_matrix = np.einsum("kj,jl->kl",active_matrix, np.diag(lambs2[0]))


        active_matrix = np.einsum("kl,ka->la",active_matrix, Bs1[-1][:,0,:])
        active_matrix = np.einsum("la,la",active_matrix, Bs2[-1][:,0,:])

        return active_matrix

    #start
    

    active_matrix = np.conj(Bs1[0])

    lambs1 = lambs1[1:-1]
    lambs2 = lambs2[1:-1]

    active_matrix = np.einsum("oia,oja->ij",active_matrix, Bs2[0])
    active_matrix = np.einsum("ij,ik->kj",active_matrix, np.diag(np.conj(lambs1[0])))
    active_matrix = np.einsum("kj,jl->kl",active_matrix, np.diag(lambs2[0]))
    active_matrix = np.einsum("kl,kja->jla",active_matrix, np.conj(Bs1[1]))

    #general step
    for i in range(1,len(Bs1)-2):
        active_matrix = np.einsum("jla,lia->ji",active_matrix, Bs2[i])
        active_matrix = np.einsum("ji,jk->ki",active_matrix, np.diag(np.conj(lambs1[i])))
        active_matrix = np.einsum("ki,il->kl",active_matrix, np.diag(lambs2[i]))
        active_matrix = np.einsum("kl,kia->ila",active_matrix, np.conj(Bs1[i+1]))

    #last step
    active_matrix = np.einsum("jla,lia->ji",active_matrix, Bs2[len(Bs1)-2])
    active_matrix = np.einsum("ji,jk->ki",active_matrix, np.diag(np.conj(lambs1[len(Bs1)-2])))
    active_matrix = np.einsum("ki,il->kl",active_matrix, np.diag(lambs2[len(Bs1)-2]))
    active_matrix = np.einsum("kl,ka->la",active_matrix, np.sum(np.conj(Bs1[len(Bs1)-1]), axis=1))
    active_matrix = np.einsum("la,la",active_matrix, np.sum(Bs2[len(Bs2)-1], axis=1))

    return active_matrix


"""Bs, lambs = MPA_B_real([0,1,1,0,0,0,0,0], 3)
print(Bs)
print(scalar_product(Bs,lambs,Bs,lambs))"""

def applyLocalPropagator(loc, U2, Bs, lambs, Mlim):

    lamb0 = lambs[loc]
    lamb1 = lambs[loc+1]
    lamb2 = lambs[loc+2]
    B1 = Bs[loc]
    B2 = Bs[loc+1]

    #Apply local propagator
    big_B = np.einsum("abcd,iqc,qw,wjd->ijab", U2, B1, np.diag(lamb1), B2)

    #Reindex
    Q = np.zeros((lamb0.size*2, lamb2.size*2), dtype=np.complex128)

    #print(big_B.shape)
    #print(lamb0.size*2)
    #print(lamb2.size*2)
    #print(lamb2)

    for n1 in range(lamb0.size*2):
        for n2 in range(lamb2.size*2):
            k1,s1 = from_kindex(n1)
            k2,s2 = from_kindex(n2)

            #print(k1,k2,s1,s2)
            Q[n1,n2] = lamb0[k1]*big_B[k1,k2,s1,s2]*lamb2[k2]

    #SVD
    U, lamb, Vt = svd(Q, full_matrices=False)
    

    #New B1, lamb1, B2 matrices

    if Mlim == 0:
        if 0 not in lamb:
            Mmax = lamb.size
        else:
            Mmax = min(lamb.size, np.where(lamb == 0)[0][0])
    else:
        if 0 not in lamb:
            Mmax = min(Mlim, lamb.size)
        else:
            Mmax = min(Mlim, lamb.size, np.where(lamb == 0)[0][0])

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

    lambs[loc+1] = lamb1
    Bs[loc] = B1
    Bs[loc+1] = B2 


    return Bs, lambs



def normalize(x):
    return x / np.linalg.norm(x)

def propagate_time(state, N, t, Mlim):
    Bs, lambs = MPA_B_real(state, N, Mlim)

    dz = -0.001j
    ts = np.arange(0, t, np.abs(dz))
    iters = len(ts)
    evals = np.zeros((N, len(ts)), dtype=np.complex128)

    U4i = four_index_propagator(dz)
    U4iHalf = four_index_propagator(dz/2)


    for k in trange(0,iters, desc="steps"):
        #Even terms
        for i in range(0, len(Bs)-1, 2):
            Bs, lambs = applyLocalPropagator(i, U4iHalf, Bs, lambs, Mlim)

        #Odd terms
        for i in range(1, len(Bs)-1, 2):
            Bs, lambs = applyLocalPropagator(i, U4i, Bs, lambs, Mlim)

        #Even terms
        for i in range(0, len(Bs)-1, 2):
            Bs, lambs = applyLocalPropagator(i, U4iHalf, Bs, lambs, Mlim)

        

        for i in range(N):
            BsN = list.copy(Bs)
            lambsN = list.copy(lambs)

            sz = np.array([[1,0],[0,-1]])
            BsN[i] = np.einsum("ija,ab->ijb", Bs[i], sz)

            evals[i,k] = scalar_product(Bs,lambs, BsN, lambsN)

        evals[:,k] = np.real(evals[:,k] / scalar_product(Bs,lambs, Bs,lambs))
    
    return ts, evals

N = 8
state = np.zeros((2**N), dtype=np.complex128)
k = to_decimal_state([1 if i < N/2 else 0 for i in range(N)])
state[k] = 1

ts, evals = propagate_time(state, N, 10, 100)


"""plt.plot([i for i in range(N)], evals[:, -1])
plt.show()"""


from matplotlib import pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation 
 
# initializing a figure in 
# which the graph will be plotted
fig = plt.figure() 

Ns = np.arange(0,N,1)
line, = plt.plot(Ns, evals[:,0], marker ="o", linestyle = "none")
plt.ylim(-2,2)
plt.xlabel("Pozicija spina")
plt.ylabel("Magnetizacija")

def animate(i):
   
    line.set_data(Ns, evals[:,i])
    
    return line,
 
anim = FuncAnimation(fig, animate,
                     frames = np.arange(0,len(evals[0,:]), len(evals[0,:])//1000), blit = True)

 
anim.save(f'N={N}.mp4', 
          writer = 'ffmpeg', fps = 60)


            









