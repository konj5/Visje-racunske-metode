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


def create_A0(q, beta, J):
    A = np.zeros((q,q,q,q), dtype=np.float64)
    for l in range(q):
        for r in range(q):
            for t in range(q):
                for b in range(q):
                    A[l,r,t,b] = np.exp(beta*J/2 * (
                        int(t==r) + int(t==l) + int(b==r) + int(b==l)
                    ))
    return A

def create_grid(q, n, beta, J):
    N = 2**n
    grid = np.zeros((N,N), dtype=object)
    A0 = create_A0(q,beta,J)

    for i in range(N):
        for j in range(N):
            grid[i,j] = A0


def tensor_renormalize(A, q, Mlim):

    #diagonal pair split

    Ad = np.zeros((q*q, q*q), dtype=np.float64)
    for i in range(q*q):
        for j in range(q*q):
            Ad[q,q] = A[i//q, j//q, j%q, i%q]

    U,S,Vt = np.linalg.svd(Ad, full_matrices=False)
    





    
            

