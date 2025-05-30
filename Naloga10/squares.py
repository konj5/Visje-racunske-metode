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


from numpy.linalg import svd as npsvd

def svd(A, Mlim):
    #return npsvd(A)

    U, lamb, Vt = npsvd(A, full_matrices=False)

    if Mlim == 0:
        if 0 in lamb:
            k = np.where(lamb == 0)[0][0]
            lamb = lamb[0:k]
            U = U[:,0:k]
            Vt = Vt[0:k,:]
        else:
            pass
    else:
        if 0 in lamb:
            k = min(Mlim, np.where(lamb == 0)[0][0])
            lamb = lamb[0:k]
            U = U[:,0:k]
            Vt = Vt[0:k,:]
        else:
            k = min(Mlim, len(lamb))
            lamb = lamb[0:k]
            U = U[:,0:k]
            Vt = Vt[0:k,:]
    
    return U, lamb, Vt


def create_A0(q, beta, J):
    A = np.zeros((q,q,q,q), dtype=np.float64)
    for l in range(q):
        for r in range(q):
            for t in range(q):
                for b in range(q):
                    A[l,r,t,b] = np.exp(-beta*J/2 * (
                        int(t==r) + int(t==l) + int(b==r) + int(b==l)
                    ))
    return A

def to_quindex(a,b,q):
    return a*q + b

def from_quindex(n,q):
    return n//q, n%q

def tensor_renormalize(A, q, Mlim):

    #diagonal pair split

    Ad = np.zeros((q*q, q*q), dtype=np.float64)
    for i in range(q*q):
        for j in range(q*q):
            Ad[i,j] = A[i//q, j//q, j%q, i%q]

    U,S,Vt = svd(Ad, Mlim)

    Sroot = np.diag(np.sqrt(S))

    F1_b = np.einsum("ij,jk->ik", U,Sroot)
    F3_b = np.einsum("ij,jk->ik", Vt,Sroot)

    F1 = np.zeros((q,q,len(S)))
    F3 = np.zeros((q,q,len(S)))

    for i in range(q):
        for j in range(q):
            for k in range(len(S)):
                F1[i,j,k] = F1_b[q*i+j, k]
                F3[i,j,k] = F3_b[q*i+j, k]



    #anti-diagonal pair split

    Ad = np.zeros((q*q, q*q), dtype=np.float64)
    for i in range(q*q):
        for j in range(q*q):
            Ad[i,j] = A[j//q, i%q, i//q, j%q]

    U,S,Vt = svd(Ad, Mlim)

    Sroot = np.diag(np.sqrt(S))

    F2_b = np.einsum("ij,jk->ik", U,Sroot)
    F4_b = np.einsum("ij,jk->ik", Sroot,Vt)

    F2 = np.zeros((q,q,len(S)), dtype=np.float64)
    F4 = np.zeros((q,q,len(S)), dtype=np.float64)

    for i in range(q):
        for j in range(q):
            for k in range(len(S)):
                F2[i,j,k] = F2_b[q*i+j, k]
                F4[i,j,k] = F4_b[q*i+j, k]

    Anew = np.einsum("ijc,lia,lkd,kjb->abcd", F1, F2, F3, F4)
    return Anew


def tensor_renormalize2(A, q, Mlim):

    #Diagonal F1,F3

    Ad = np.zeros((q*q, q*q), dtype=np.float64)
    for i in range(q*q):
        for j in range(q*q):
            a,c = from_quindex(i,q)
            b,d = from_quindex(j,q)
            Ad[i,j] = A[a,b,c,d]

    U,S,Vt = svd(Ad, Mlim)

    Sroot = np.diag(np.sqrt(S))

    F1_b = np.einsum("ij,jk->ik", U,Sroot)
    F3_b = np.einsum("ij,jk->ik", Sroot,Vt)

    F1 = np.zeros((q,q,len(S)), dtype=np.float64)
    F3 = np.zeros((q,q,len(S)), dtype=np.float64)


    for i in range(q):
        for j in range(q):
            for k in range(len(S)):
                F1[i,j,k] = F1_b[to_quindex(i,j,q), k]
                F3[i,j,k] = F3_b[k, to_quindex(i,j,q)]

    #Anti-Diagonal F2,F4

    Ad = np.zeros((q*q, q*q), dtype=np.float64)
    for i in range(q*q):
        for j in range(q*q):
            b,c = from_quindex(i,q)
            a,d = from_quindex(j,q)
            Ad[i,j] = A[a,b,c,d]

    U,S,Vt = svd(Ad, Mlim)

    Sroot = np.diag(np.sqrt(S))

    F2_b = np.einsum("ij,jk->ik", U,Sroot)
    F4_b = np.einsum("ij,jk->ik", Sroot,Vt)

    F2 = np.zeros((q,q,len(S)), dtype=np.float64)
    F4 = np.zeros((q,q,len(S)), dtype=np.float64)

    for i in range(q):
        for j in range(q):
            for k in range(len(S)):
                F2[i,j,k] = F2_b[to_quindex(i,j,q), k]
                F4[i,j,k] = F4_b[k, to_quindex(i,j,q)]
    
    Anew = np.einsum("ijc,lia,lkd,kib->abcd", F1, F2, F3, F4)
    return Anew

def traceA(A):
    return np.einsum("aabb", A)


"""q = 2
beta = 10
J = 1
Mlim = 10
A = create_A0(q,beta,J)
norms = []
while(True):
    print(A.shape)
    #print(A)
    print(traceA(A))
    print(np.prod(norms))
    #A = tensor_renormalize(A,q,Mlim)
    A = tensor_renormalize2(A,q,Mlim)
    norm = np.sqrt(np.einsum("abcd,abcd", A,A))
    norms.append(norm)
    A = A / norm
    #input()"""


def Z(beta, J, q, Mlim):

    A = create_A0(q,beta,J)
    Aold = A.copy()

    Zval = traceA(A)
    dz = 100 * Zval
    normfactor = 1
    #while(dz/Zval > 0.001):
    while(A.shape == Aold.shape and 0.01 < np.sqrt(np.einsum("abcd,abcd", Aold,A))/np.sqrt(np.einsum("abcd,abcd", A,A))):
        Aold = A.copy()
        A = tensor_renormalize2(A, q, Mlim)
        norm = np.sqrt(np.einsum("abcd,abcd", A,A))
        A /= norm
        normfactor *= norm

        newZ = traceA(A) * normfactor
        dz = np.abs(newZ - Zval)
        Zval = newZ
    
    return traceA(A) * normfactor

J = 1
q = 2
Mlim = 20

    
betas = 10**np.linspace(-10,1,10)
Zs = []
for i in trange(0,len(betas)):
    beta = betas[i]
    Zs.append(Z(beta, J, q, Mlim))

plt.plot(betas, Zs)
#plt.xscale("log")
plt.show()

plt.plot(betas, np.log(Zs))
#plt.xscale("log")
plt.show()


plt.plot(betas, -np.gradient(np.log(Zs), betas))
#plt.xscale("log")
plt.show()

"""plt.plot(1/betas, -np.gradient(np.log(Zs), betas))
#plt.xscale("log")
plt.show()"""
    
            

