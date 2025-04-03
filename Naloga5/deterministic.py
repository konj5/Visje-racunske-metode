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

from scipy.integrate import ode

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


            if i < M:
                vdot[2*i+1] += -v[-2]*v[2*i+1] 
            elif i > N-M-1:
                vdot[2*i+1] += -v[-1]*v[2*i+1]

        #Left bath
        #print(np.sum(v[0:2*M:2]))

        vdot[-2] = 1/tau * (np.sum(v[1:2*M:2]**2)-M*TL)
    
        #Right bath
        vdot[-1] = 1/tau * (np.sum(v[len(v)-2-M*2+1:len(v)-2+1:2]**2)-M*TR)

        return vdot
    
    #startstate = np.zeros(2*N+2)
    startstate = np.random.rand(2*N+2)*0.01
    #startstate = np.ones(2*N+2) * 0.001
    startstate[-2:] = [0,0]

    """r = ode(f).set_integrator("dopri5")
    r.set_initial_value(startstate, 0)

    sol=[]
    dt = 0.1
    ts = []

    sol.append(startstate)
    ts.append(0)

    while r.successful() and r.t < tmax:
        r.integrate(r.t+dt)
        sol.append(r.y)
        ts.append(r.t)
    return np.array(ts), np.array(sol).T"""

    sol = solve_ivp(f,(0,tmax), startstate, "DOP853", max_step = 0.1)
    return sol.t, sol.y

    

"""t,y = run(5,2,0,1,1,1,1)
print(y.shape)
print(t)
print(y)"""


