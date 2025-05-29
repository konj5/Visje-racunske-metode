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
    

    active_matrix = Bs1[0]

    lambs1 = lambs1[1:-1]
    lambs2 = lambs2[1:-1]

    active_matrix = np.einsum("oia,oja->ij",active_matrix, Bs2[0])
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
    active_matrix = np.einsum("jla,lia->ji",active_matrix, Bs2[len(Bs1)-2])
    active_matrix = np.einsum("ji,jk->ki",active_matrix, np.diag(lambs1[len(Bs1)-2]))
    active_matrix = np.einsum("ki,il->kl",active_matrix, np.diag(lambs2[len(Bs1)-2]))
    active_matrix = np.einsum("kl,ka->la",active_matrix, np.sum(Bs1[len(Bs1)-1], axis=1))
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

def propagate_imag_time_E0_by_log(state, N, z, Mlim, dz=0.1):
    Bs, lambs = MPA_B_real(state, N, Mlim)

    dz = dz * np.sign(z)
    betas = -np.arange(0, z, dz)
    iters = len(betas)
    evals = np.zeros(betas.shape)

    U4i = four_index_propagator(dz)
    U4iHalf = four_index_propagator(dz/2)

    initBs = list.copy(Bs)
    initlambs = list.copy(lambs)

    #randBs, randlambs = MPA_B_real(normalize(np.random.normal(0,5,len(state))), N)
    randBs = initBs
    randlambs = initlambs


    initrandscala = scalar_product(randBs,randlambs,initBs,initlambs)

    for k in trange(0,iters, desc="steps", leave=False):
        #Even terms
        for i in range(0, len(Bs)-1, 2):
            Bs, lambs = applyLocalPropagator(i, U4iHalf, Bs, lambs, Mlim)

        #Odd terms
        for i in range(1, len(Bs)-1, 2):
            Bs, lambs = applyLocalPropagator(i, U4i, Bs, lambs, Mlim)

        #Even terms
        for i in range(0, len(Bs)-1, 2):
            Bs, lambs = applyLocalPropagator(i, U4iHalf, Bs, lambs, Mlim)

        evals[k] = scalar_product(randBs,randlambs,Bs,lambs) / initrandscala
    
    return betas, evals


#Show shape
"""for N in np.arange(2,8,1):
    betas, evals = propagate_imag_time_E0_by_log(normalize(np.random.normal(0,1,2**N)), N, -30)
    #print(evals)
    plt.plot(betas, -1/betas * np.log(evals), label = f"$N = {N}$")


plt.xlabel("$\\beta$")
plt.ylabel("$\\frac{1}{\\beta} \\log\\langle e^{-\\beta H}\\rangle$")
plt.legend()
plt.show()"""


#Plot energies
"""Ns = np.arange(2,12,1)[::-1]
Mlim = 100
vals = []
for i in trange(0,len(Ns)):
    N = Ns[i]
    betas, evals = propagate_imag_time_E0_by_log(normalize(np.random.normal(0,1,2**N)), N, -30, Mlim)
    #print(evals)
    vals.append(-1/betas[-1] * np.log(evals[-1]))

plt.plot(Ns, vals, marker = "o", linestyle = "dashed")
plt.xlabel("$N$")
plt.ylabel("$E_0$")
plt.show()"""


"""from exactdiag import calculate_ground

with open("data.txt", mode="w") as f:

    Ns = [2,4,6]
    for i in trange(0,len(Ns)):
        print("\n")
        print(i)
        print("\n")
        N = Ns[i]

        E0, trash = calculate_ground(-1, N)

        Ms = [5,20,0]

        for Mlim in Ms:

            dzs = [0.1,0.01]
            for dz in dzs:
                
                tms = [-30,-60]
                for tm in tms:

                    betas, evals = propagate_imag_time_E0_by_log(normalize(np.random.normal(0,1,2**N)), N, tm, Mlim, dz)
                    E = -1/betas[-1] * np.log(evals[-1])
                    DE = np.abs(E-E0)
                    f.write(f"{N} & {Mlim} & {dz} & {-tm} & {np.round(E, int(-np.log10(DE))+1)} & {np.round(np.log10(DE), 2)}\\\\ \n")
                    #f.write("\\hline\n")"""

            





def propagate_imag_time_E0_by_fit(state, N, z, Mlim, dz=0.1):
    Bs, lambs = MPA_B_real(state, N, Mlim)

    dz = dz * np.sign(z)
    betas = -np.arange(0, z, dz)
    iters = len(betas)
    evals = np.zeros(betas.shape)

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

        evals[k] = scalar_product(Bs,lambs,Bs,lambs)
    
    return betas, evals

"""betas, evals = propagate_imag_time_E0_by_log(normalize(np.array([1,0,1,0,0,1,0,1])), 3, -10)
popt, pcov = scipy.optimize.curve_fit(lambda x,A,k:A * np.exp(k*x), betas, evals)
plt.plot(betas, evals)
A, k = popt
plt.plot(betas, A * np.exp(k*betas))
plt.yscale("log")
plt.show()"""


"""for N in np.arange(2,13,2):
    betas, evals = propagate_imag_time_E0_by_fit(normalize(np.random.normal(0,1,2**N)), N, -30,Mlim=10)
    #print(evals)
    plt.plot(betas, np.log(np.sqrt(evals)), label = f"$N = {N}$")


plt.xlabel("$\\beta$")
plt.ylabel("$\\ln\\sqrt{\\langle\\psi_{\\beta}|\\psi_{\\beta} \\rangle}$")
plt.legend()
plt.show()"""



"""from exactdiag import calculate_ground

with open("data.txt", mode="w") as f:

    Ns = [2,4,6]
    for i in trange(0,len(Ns)):
        print("\n")
        print(i)
        print("\n")
        N = Ns[i]

        E0, trash = calculate_ground(-1, N)

        Ms = [5,20,0]

        for Mlim in Ms:

            dzs = [0.1,0.01]
            for dz in dzs:
                
                tms = [-30,-60]
                for tm in tms:

                    betas, evals = propagate_imag_time_E0_by_fit(normalize(np.random.normal(0,1,2**N)), N, tm, Mlim, dz)
                    
                    evals = np.log(np.sqrt(evals))

                    idx = np.isfinite(evals)

                    p = np.polyfit(betas[idx], evals[idx], 1)
                    #print(p)
                    E = -p[0]
                    
                    #print(E)
                    DE = np.abs(E-E0)
                    f.write(f"{N} & {Mlim} & {dz} & {-tm} & {np.round(E, int(-np.log10(DE))+1)} & {np.round(np.log10(DE), 2)}\\\\ \n")
                    #f.write("\\hline\n")"""


def get_ground_state(state, N, z, dz):
    Bs, lambs = MPA_B_real(state, N, Mlim=30)

    assert dz > 0
    assert z > 0

    betas = np.arange(0, z, dz)
    iters = len(betas)


    U4i = four_index_propagator(-dz)

    for k in trange(0,iters, desc="steps"):
        #Even terms
        for i in range(0, len(Bs)-1, 2):
            Bs, lambs = applyLocalPropagator(i, U4i, Bs, lambs,Mlim=30)

        #Odd terms
        for i in range(1, len(Bs)-1, 2):
            Bs, lambs = applyLocalPropagator(i, U4i, Bs, lambs,Mlim=30)
    
    return Bs, lambs


def correlation(N):
    BsG, lambsG = get_ground_state(normalize(np.random.normal(0,1,2**N)),N,10,0.01)

    
    sz = np.array([[1,0],[0,-1]])
    #id = np.eye(2)

    corrs = np.zeros((N,N), dtype=np.complex128)
    #nullops = np.array([id for i in range(N)])

    for i in range(N):
        for j in range(N):
            Bs = list.copy(BsG)
            lambs = list.copy(lambsG)

            #Cool for general use maybe but not sefull right now
            """ops = nullops.copy(); ops[i] = sz; ops[j] = sz

            for k in range(N):
                Bs[k] = np.einsum("ija,ab->ijb", Bs[k]) """    
                   

            Bs[i] = np.einsum("ija,ab->ijb", Bs[i], sz)
            Bs[j] = np.einsum("ija,ab->ijb", Bs[j], sz)

            corrs[i,j] = scalar_product(BsG,lambsG, Bs, lambs)

    return corrs / scalar_product(BsG,lambsG,BsG,lambsG)

"""N = 5
corrs = correlation(N)
ns = np.array([i for i in range(N)])
for i in range(N):
    plt.plot(ns, corrs[i,:], label = f"$i = {i}$", marker = "o")

plt.ylabel("$\\langle \\sigma_i^z\\sigma_j^z\\rangle$")
plt.xlabel("j")
plt.legend()
plt.tight_layout()

plt.show()"""

fig, axs = plt.subplots(4,2)
norm = colors.Normalize(-1,1)
cmap = plt.get_cmap("seismic")
for N in trange(2,10,1):
    ax = axs[N//2-1, N%2]

    corrs = np.real(correlation(N))
    
    
    im = ax.imshow(corrs, cmap, norm, aspect="equal")


    ax.set_ylabel("i")
    ax.set_xlabel("j")

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.show()




#ALSO TRY 2nd ORDER TROTTER SUZUKI









