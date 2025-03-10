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
from numba import jit

# p = 1, k = 1
decomp = [
    [1],
    [1]
]

# p = 2, k = 2
decomp = [
    [1/2, 1/2],
    [1,0]
]


# p = 4, k = 4
r2 = 2**(1/3)
x1 = 1/(2-r2)
x0 = -r2*x1
decomp = [
    [x1/2,(x0+x1)/2 ,(x0+x1)/2 ,x1/2],
    [x1,x0,x1,0]
]

# p = 3, k = 3
p1 = (1 + 1j/np.sqrt(3))/4
p2 = 2*p1
p3 = 1/2
p4 = np.conj(p2)
p5 = np.conj(p1)

decomp = [
    [p1,p3,p5],
    [p2,p4,0]
]

# p = 4, k = 5
p1 = (1 + 1j/np.sqrt(3))/4
p2 = 2*p1
p3 = 1/2
p4 = np.conj(p2)
p5 = np.conj(p1)

decomp = [
    [p1,p3,p5,p4,p2],
    [p2,p4,p5,p3,p1]
]


def U(state, dt, lamb):
    q1,q2,p1,p2 = state
    ret = np.zeros([q1,q2,p1,p2])
    
    ###Kinetic part
    ret += np.array([p1, p2, 0, 0])*dt

    ###Potential part
    ret += -np.array([0, 0, q1, q2])*dt
    ret += -2*lamb*q1*q2*np.array([0, 0, q2, q1])*dt

    return ret



@jit
def integrate(startstate, dt, tmax, lamb):
    ts = np.arange(0,tmax,dt)

    states = np.zeros((len(startstate), len(ts)))
    states[:,0] = startstate


    print(1/0)
    #Here i forgot to use any cool step pattern (this is actively wrong)

    for i in range(1,len(ts)):
        states[:,i] += U(states[:,i-1], dt, lamb)

    return ts, states

from scipy.integrate import solve_ivp
def integrate_rk45(startstate, dt, tmax, lamb):
    ts = np.arange(0,tmax,dt)
    def f(y,t):
        return -np.array([y[0] + 2*lamb*y[0]*y[1]*y[0], y[1] + 2*lamb*y[0]*y[1]*y[0]]) 
    
    sol = solve_ivp(f, (0,tmax), startstate, "RK45", t_eval = ts)

    return sol.t, sol.y
    

        





