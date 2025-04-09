import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import scipy.integrate
import scipy.interpolate
import scipy.special
from tqdm import tqdm
import re, sys
from matplotlib.animation import FuncAnimation
import time
import numba
from numba import jit, njit
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))


"""N = 20
gamma = 1
TL = 2
TR = 1

B = np.eye(N, k=0) * -3 + np.eye(N, k=1) * 1 + np.eye(N, k=-1) * 1
B[0,0], B[-1,-1] = -2, -2

X = np.zeros((2*N, 2*N))
X[N:,0:N] = np.eye(N)
X[N:,N:] = np.eye(N) * -gamma
X[0:N,N:] = B
X = X.T

print(X)

Y = np.zeros((2*N, 2*N))
Y[N,N] = TL * 2 * gamma
Y[-1,-1] = TR * 2 * gamma


print(Y)

from scipy.linalg import solve_lyapunov

C = solve_lyapunov(X,-Y)

plt.plot([C[i,i] for i in range(N,2*N)], marker = "o", linestyle = "dashed")
plt.ylabel("$\\langle p_j^2 \\rangle$")
plt.xlabel("Zaporedno mesto delca")
#plt.ylim(0,2)
plt.show()"""

Ns = []
vasl = []
for N in range(2,100):
    gamma = 1
    TL = 2
    TR = 1

    B = np.eye(N, k=0) * -3 + np.eye(N, k=1) * 1 + np.eye(N, k=-1) * 1
    B[0,0], B[-1,-1] = -2, -2

    X = np.zeros((2*N, 2*N))
    X[N:,0:N] = np.eye(N)
    X[N:,N:] = np.eye(N) * -gamma
    X[0:N,N:] = B
    X = X.T


    Y = np.zeros((2*N, 2*N))
    Y[N,N] = TL * 2 * gamma
    Y[-1,-1] = TR * 2 * gamma



    from scipy.linalg import solve_lyapunov

    C = solve_lyapunov(X,-Y)

    Ns.append(N)
    vasl.append(np.average([C[j,N+j] for j in range(N)]))


plt.plot(Ns, vasl, marker = "o", linestyle = "dashed")
plt.ylabel("$\\langle J \\rangle$")
plt.xlabel("N")
#plt.ylim(0,2)
plt.show()

