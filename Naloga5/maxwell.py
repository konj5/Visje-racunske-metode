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

from scipy.integrate import solve_ivp

#v = q1, p1, q2, p2, ..., qN, pN, KL, KR


def run(N,M, lamb, tau, TL, TR, tmax):

    @jit(nopython=True)
    def f(t,v):
        vdot = np.zeros(v.shape)
        for i in range(N):
            #Coordinate q
            vdot[2*i] += v[2*i+1]

            #Conjugate momentum
            vdot[2*i+1] += -v[2*i] - 4*lamb*v[2*i]**3
            if i != 0:
                vdot[2*i+1] += v[2*(i-1)] - v[2*i]
            if i != N-1:
                vdot[2*i+1] += v[2*(i+1)] - v[2*i]

        return vdot
    
    #startstate = np.zeros(2*N+2)
    startstate = np.random.rand(2*N)
    taus = np.arange(0,tmax,tau)

    states = np.zeros((len(startstate), 1))
    states[:,-1] = startstate
    ts = np.array([0])

    #@njit(nopython=True)
    def maxwelleval(taus, tau, TL, TR, startstate, ts, M, states):
        for i in range(len(taus)):
            sol = solve_ivp(f,(i*tau,(i+1)*tau), states[:,-1], "DOP853", max_step = 0.1)
            ts = np.append(ts, sol.t[1:])
            states = np.append(states, sol.y[:,1:], axis=1)

            #print(ts.shape)
            #print(states.shape)

            for j in range(M):
                states[2*j+1,-1] = np.random.normal(0,np.sqrt(TL))
                states[len(startstate)-2*j-1,-1] = np.random.normal(0,np.sqrt(TR))
        
        return ts, states


    ts, states = maxwelleval(taus, tau, TL, TR, startstate, ts, M, states)
    #print(ts.shape)
    #print(states.shape)
            
    return ts, states

"""t,y = run(10,1,0,1,1,1,1)
print(y.shape)
print(t)
print(y)"""


