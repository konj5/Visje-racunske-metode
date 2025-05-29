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

"""A = s0
B = sz

print(A)
print(B)
print(comb_product(A,B))
print(tensor([A,B]))"""

#print(tensor([sx,sx]) + tensor([sy,sy]) + tensor([sz,sz]))
#print(scipy.linalg.eigh(tensor([sx,sx]) + tensor([sy,sy]) + tensor([sz,sz])))


#open boundary
@njit
def full_hamiltonian(J, n):
    H = np.zeros((2**n, 2**n), dtype=np.complex128)
    ids = [s0 for i in range(n)]
    #for i in trange(0,n-1,leave=False,desc="constructing Hamiltonian"):
    for i in range(0,n-1):
        ops = ids.copy()
        ops[i], ops[i+1] = sx, sx
        H += -J * tensor(ops)

        ops = ids.copy()
        ops[i], ops[i+1] = sy, sy
        H += -J * tensor(ops)

        ops = ids.copy()
        ops[i], ops[i+1] = sz, sz
        H += -J * tensor(ops)

    return H

#periodic boundary
@njit
def full_hamiltonian_periodic(J, n):
    H = np.zeros((2**n, 2**n), dtype=np.complex128)
    ids = [s0 for i in range(n)]
    #for i in trange(0,n-1,leave=False,desc="constructing Hamiltonian"):
    for i in range(0,n-1):
        ops = ids.copy()
        ops[i], ops[i+1] = sx, sx
        H += -J * tensor(ops)


        ops = ids.copy()
        ops[i], ops[i+1] = sy, sy
        H += -J * tensor(ops)

        ops = ids.copy()
        ops[i], ops[i+1] = sz, sz
        H += -J * tensor(ops)

    #Periodicity
    ops = ids.copy()
    ops[-1], ops[0] = sx, sx
    H += -J * tensor(ops)

    ops = ids.copy()
    ops[-1], ops[0] = sy, sy
    H += -J * tensor(ops)

    ops = ids.copy()
    ops[-1], ops[0] = sz, sz
    H += -J * tensor(ops)
    

    return H

from numpy.linalg import eigh


@njit
def calculate_ground_state(J,n):
    H = full_hamiltonian(J,n)
    eigvals, eigvects = eigh(H)

    return eigvects[:,0]

def calculate_ground(J,n):
    H = full_hamiltonian(J,n)
    eigvals, eigvects = eigh(H)

    return eigvals[0], eigvects[:,0]

#print(calculate_ground_state(-1,2))

@njit
def calculate_ground_state_per(J,n):
    H = full_hamiltonian_periodic(J,n)
    eigvals, eigvects = eigh(H)

    return eigvects[:,0]


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

@njit
def I_HATE_NUMBA(arr1, arr2, indices):
    for i in range(len(indices)):
        arr1[indices[i]] = arr2[i]
    return arr1

@njit
def Schmidt(state, A, B):

    ### Build up Psi
    Psi = np.zeros((2**len(A),2**len(B)), dtype=np.complex128)
    for a in range(2**len(A)):
        for b in range(2**len(B)):
            bin_a = to_binary_state(a, len(A))
            bin_b = to_binary_state(b, len(B))
            bin = np.zeros(len(bin_a) + len(bin_b))

            #bin[A] = bin_a; bin[B] = bin_b
            bin = I_HATE_NUMBA(bin, bin_a,A)
            bin = I_HATE_NUMBA(bin, bin_b,B)

            dec = to_decimal_state(bin)
            Psi[a,b] = state[int(dec)]
    
    U,d,Vh = svd(Psi)
    return U,d,Vh

@njit
def rhoAB(U,d,Vh):
    return U.dot(np.diag(d**2).dot(U.H)), Vh.H.dot(np.diag(d**2).dot(Vh))

@njit
def EE(d):
    spec = []
    for i in range(len(d)):
        if d[i] != 0:
            spec.append(d[i])
    d = np.array(spec)

    return -np.sum(d**2 * np.log(d**2))

"""n = 8
state = calculate_ground_state(-1,n)
print(np.abs(state)**2)
A = [i for i in range(0,n//2,1)]
B = [i for i in range(n//2,n,1)]
U,d,Vh = Schmidt(state, A, B)
print(A)
print(B)
print(d)
print(EE(d))"""


## EE by length
"""ns = [2,4,6,8,10]
@njit(nopython = True, parallel = True)
def ddadfsggfmnald(ns):
    vals = np.zeros(len(ns), dtype=np.float64)
    done = np.zeros(len(ns), dtype=np.byte)
    for i in prange(len(ns)):
        n = ns[i]
        state = calculate_ground_state(-1,n)
        A = [j for j in range(0,n//2,1)]
        B = [j for j in range(n//2,n,1)]    
        U,d,Vh = Schmidt(state, A, B)
        vals[i] = EE(d)
        done[i] = 1
        print(i)
        print(np.average(done)*100)
        print("#################")
    return vals

vals = ddadfsggfmnald(ns)
plt.plot(ns, vals, marker = "o", linestyle = "dashed")
plt.xlabel("$n$")
plt.ylabel("$EE_{AB}$")
plt.title("odprti robni pogoji")
plt.show()"""


## EE by split position
"""ns = [2,3,4,5,6,7,8]
@njit(nopython = True, parallel = True)
def ddadfsggfmnald(ns):
    vals = np.zeros((len(ns), ns[-1]), dtype=np.float64)
    for i in range(len(ns)):
        n = ns[i]
        state = calculate_ground_state(-1,n)
        done = np.zeros(n-2, dtype=np.byte)
        for k in prange(1,n):
            A = [j for j in range(0,k,1)]
            B = [j for j in range(k,n,1)]    
            U,d,Vh = Schmidt(state, A, B)
            vals[i,k] = EE(d)
            done[k] = 1

            print(i)
            print(np.average(done)*100)
            print("#################")
    return vals

vals = ddadfsggfmnald(ns)

for i in range(len(ns)):
    ratios = np.array([_ for _ in range(1,ns[i])])/ns[i]
    plt.plot(ratios, vals[i,1:ns[i]], marker = "o", linestyle = "dashed", label = f"$n={ns[i]}$")

plt.legend()
plt.xlabel("Razmerje razpolovitve")
plt.ylabel("$EE_{AB}$")
plt.title("odprti robni pogoji")
plt.show()"""


"""## EE by length (per)
ns = [2,4,6,8,10]
@njit(nopython = True, parallel = True)
def ddadfsggfmnald(ns):
    vals = np.zeros(len(ns), dtype=np.float64)
    done = np.zeros(len(ns), dtype=np.byte)
    for i in prange(len(ns)):
        n = ns[i]
        state = calculate_ground_state_per(-1,n)
        A = [j for j in range(0,n//2,1)]
        B = [j for j in range(n//2,n,1)]    
        U,d,Vh = Schmidt(state, A, B)
        vals[i] = EE(d)
        done[i] = 1
        print(i)
        print(np.average(done)*100)
        print("#################")
    return vals

vals = ddadfsggfmnald(ns)
plt.plot(ns, vals, marker = "o", linestyle = "dashed")
plt.xlabel("$n$")
plt.ylabel("$EE_{AB}$")
plt.title("periodični robni pogoji")
plt.show()"""


## EE by split position (per)

"""ns = [2,3,4,5,6,7,8]
@njit(nopython = True, parallel = True)
def ddadfsggfmnald(ns):
    vals = np.zeros((len(ns), ns[-1]), dtype=np.float64)
    for i in range(len(ns)):
        n = ns[i]
        state = calculate_ground_state_per(-1,n)
        done = np.zeros(n-2, dtype=np.byte)
        for k in prange(1,n):
            A = [j for j in range(0,k,1)]
            B = [j for j in range(k,n,1)]    
            U,d,Vh = Schmidt(state, A, B)
            vals[i,k] = EE(d)
            done[k] = 1

            print(i)
            print(np.average(done)*100)
            print("#################")
    return vals

vals = ddadfsggfmnald(ns)

for i in range(len(ns)):
    ratios = np.array([_ for _ in range(1,ns[i])])/ns[i]
    plt.plot(ratios, vals[i,1:ns[i]], marker = "o", linestyle = "dashed", label = f"$n={ns[i]}$")

plt.legend()
plt.xlabel("Razmerje razpolovitve")
plt.ylabel("$EE_{AB}$")
plt.title("periodični robni pogoji")
plt.show()"""

## EE by split position (per) even

"""ns = [2,4,6,8,10]
@njit(nopython = True, parallel = True)
def ddadfsggfmnald(ns):
    vals = np.zeros((len(ns), ns[-1]), dtype=np.float64)
    for i in range(len(ns)):
        n = ns[i]
        state = calculate_ground_state_per(-1,n)
        done = np.zeros(n-2, dtype=np.byte)
        for k in prange(1,n):
            A = [j for j in range(0,k,1)]
            B = [j for j in range(k,n,1)]    
            U,d,Vh = Schmidt(state, A, B)
            vals[i,k] = EE(d)
            done[k] = 1

            print(i)
            print(np.average(done)*100)
            print("#################")
    return vals

vals = ddadfsggfmnald(ns)

for i in range(len(ns)):
    ratios = np.array([_ for _ in range(1,ns[i])])/ns[i]
    plt.plot(ratios, vals[i,1:ns[i]], marker = "o", linestyle = "dashed", label = f"$n={ns[i]}$")

plt.legend()
plt.xlabel("Razmerje razpolovitve")
plt.ylabel("$EE_{AB}$")
plt.title("periodični robni pogoji")
plt.show()"""


## EE by split position (per) odd

"""ns = [3,5,7,9]
@njit(nopython = True, parallel = True)
def ddadfsggfmnald(ns):
    vals = np.zeros((len(ns), ns[-1]), dtype=np.float64)
    for i in range(len(ns)):
        n = ns[i]
        state = calculate_ground_state_per(-1,n)
        done = np.zeros(n-2, dtype=np.byte)
        for k in prange(1,n):
            A = [j for j in range(0,k,1)]
            B = [j for j in range(k,n,1)]    
            U,d,Vh = Schmidt(state, A, B)
            vals[i,k] = EE(d)
            done[k] = 1

            print(i)
            print(np.average(done)*100)
            print("#################")
    return vals

vals = ddadfsggfmnald(ns)

for i in range(len(ns)):
    ratios = np.array([_ for _ in range(1,ns[i])])/ns[i]
    plt.plot(ratios, vals[i,1:ns[i]], marker = "o", linestyle = "dashed", label = f"$n={ns[i]}$")

plt.legend()
plt.xlabel("Razmerje razpolovitve")
plt.ylabel("$EE_{AB}$")
plt.title("periodični robni pogoji")
plt.show()"""
    

## EE by split position  even

"""ns = [2,4,6,8,10]
@njit(nopython = True, parallel = True)
def ddadfsggfmnald(ns):
    vals = np.zeros((len(ns), ns[-1]), dtype=np.float64)
    for i in range(len(ns)):
        n = ns[i]
        state = calculate_ground_state(-1,n)
        done = np.zeros(n-2, dtype=np.byte)
        for k in prange(1,n):
            A = [j for j in range(0,k,1)]
            B = [j for j in range(k,n,1)]    
            U,d,Vh = Schmidt(state, A, B)
            vals[i,k] = EE(d)
            done[k] = 1

            print(i)
            print(np.average(done)*100)
            print("#################")
    return vals

vals = ddadfsggfmnald(ns)

for i in range(len(ns)):
    ratios = np.array([_ for _ in range(1,ns[i])])/ns[i]
    plt.plot(ratios, vals[i,1:ns[i]], marker = "o", linestyle = "dashed", label = f"$n={ns[i]}$")

plt.legend()
plt.xlabel("Razmerje razpolovitve")
plt.ylabel("$EE_{AB}$")
plt.title("odprti robni pogoji")
plt.show()"""


## EE by split position  odd

"""ns = [3,5,7,9]
@njit(nopython = True, parallel = True)
def ddadfsggfmnald(ns):
    vals = np.zeros((len(ns), ns[-1]), dtype=np.float64)
    for i in range(len(ns)):
        n = ns[i]
        state = calculate_ground_state(-1,n)
        done = np.zeros(n-2, dtype=np.byte)
        for k in prange(1,n):
            A = [j for j in range(0,k,1)]
            B = [j for j in range(k,n,1)]    
            U,d,Vh = Schmidt(state, A, B)
            vals[i,k] = EE(d)
            done[k] = 1

            print(i)
            print(np.average(done)*100)
            print("#################")
    return vals

vals = ddadfsggfmnald(ns)

for i in range(len(ns)):
    ratios = np.array([_ for _ in range(1,ns[i])])/ns[i]
    plt.plot(ratios, vals[i,1:ns[i]], marker = "o", linestyle = "dashed", label = f"$n={ns[i]}$")

plt.legend()
plt.xlabel("Razmerje razpolovitve")
plt.ylabel("$EE_{AB}$")
plt.title("odprti robni pogoji")
plt.show()"""

#nekompaktno
"""ns = [2,3,4,5,6,7,8,9,10]
@njit(nopython = True, parallel = True)
def ddadfsggfmnald(ns):
    vals = np.zeros((len(ns), ns[-1]), dtype=np.float64)
    for i in range(len(ns)):
        n = ns[i]
        state = calculate_ground_state(-1,n)
        done = np.zeros(n-2, dtype=np.byte)
        for k in prange(2,n):
            A = [j for j in range(0,n,k)]
            ALL = [j for j in range(0,n,1)]
            B = []
            for index in ALL:
                if index not in A:
                    B.append(index)
            
            U,d,Vh = Schmidt(state, A, B)
            vals[i,k] = EE(d)
            done[k] = 1

            print(i)
            print(np.average(done)*100)
            print("#################")
    return vals

vals = ddadfsggfmnald(ns)

for i in range(len(ns)):
    ks = np.array([_ for _ in range(2,ns[i])])
    plt.plot(ks, vals[i,2:ns[i]], marker = "o", linestyle = "dashed", label = f"$n={ns[i]}$")

plt.legend()
plt.xlabel("Perioda razpolovitve")
plt.ylabel("$EE_{AB}$")
plt.title("odprti robni pogoji")
plt.show()
"""

#nekompaktno periodično
"""ns = [2,3,4,5,6,7,8,9,10]
@njit(nopython = True, parallel = True)
def ddadfsggfmnald(ns):
    vals = np.zeros((len(ns), ns[-1]), dtype=np.float64)
    for i in range(len(ns)):
        n = ns[i]
        state = calculate_ground_state_per(-1,n)
        done = np.zeros(n-2, dtype=np.byte)
        for k in prange(2,n):
            A = [j for j in range(0,n,k)]
            ALL = [j for j in range(0,n,1)]
            B = []
            for index in ALL:
                if index not in A:
                    B.append(index)
            
            U,d,Vh = Schmidt(state, A, B)
            vals[i,k] = EE(d)
            done[k] = 1

            print(i)
            print(np.average(done)*100)
            print("#################")
    return vals

vals = ddadfsggfmnald(ns)

for i in range(len(ns)):
    ks = np.array([_ for _ in range(2,ns[i])])
    plt.plot(ks, vals[i,2:ns[i]], marker = "o", linestyle = "dashed", label = f"$n={ns[i]}$")

plt.legend()
plt.xlabel("Perioda razpolovitve")
plt.ylabel("$EE_{AB}$")
plt.title("periodični robni pogoji")
plt.show()
"""


#print(calculate_ground_state(-1, 2))
