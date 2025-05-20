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


# p = 1, k = 1
decomp11 = [
    [1],
    [1]
]


# p = 2, k = 2
decomp22 = [
    [1/2, 1/2],
    [1,0]
]


# p = 4, k = 4
r2 = 2**(1/3)
x1 = 1/(2-r2)
x0 = -r2*x1
decomp44 = [
    [x1/2,(x0+x1)/2 ,(x0+x1)/2 ,x1/2],
    [x1,x0,x1,0]
]

# p = 3, k = 3
p1 = (1 + 1j/np.sqrt(3))/4
p2 = 2*p1
p3 = 1/2+0j
p4 = np.conj(p2)
p5 = np.conj(p1)

decomp33 = [
    [p1,p3,p5],
    [p2,p4,0]
]

# p = 4, k = 5
p1 = (1 + 1j/np.sqrt(3))/4
p2 = 2*p1
p3 = 1/2+0j
p4 = np.conj(p2)
p5 = np.conj(p1)

decomp45 = [
    [p1,p3,p5,p4,p2],
    [p2,p4,p5,p3,p1]
]


def kinetic(state, dt, coef, lamb):
    q1,q2,p1,p2 = state
    #ret = np.array([q1,q2,p1,p2])
    ret = np.zeros(4)
    
    ###Kinetic part
    ret += np.array([p1, p2, 0, 0])*dt*coef

    return ret

def kineticComplex(state, dt, coef, lamb):
    q1,q2,p1,p2 = state
    #ret = np.array([q1,q2,p1,p2])
    ret = np.zeros(4, dtype=complex)
    
    ###Kinetic part
    ret += np.array([p1, p2, 0, 0])*dt*coef

    return ret

def potential(state, dt, coef, lamb):
    q1,q2,p1,p2 = state
    #ret = np.array([q1,q2,p1,p2])
    ret = np.zeros(4)

    ###Potential part
    ret += -np.array([0, 0, q1, q2])*dt*coef
    ret += -2*lamb*q1*q2*np.array([0, 0, q2, q1])*dt*coef

    return ret

def potentialComplex(state, dt, coef, lamb):
    q1,q2,p1,p2 = state
    #ret = np.array([q1,q2,p1,p2])
    ret = np.zeros(4, dtype=complex)

    ###Potential part
    ret += -np.array([0, 0, q1, q2])*dt*coef
    ret += -2*lamb*q1*q2*np.array([0, 0, q2, q1])*dt*coef

    return ret



#@jit
def integrate(startstate, dt, tmax, lamb, decomp):
    ts = np.arange(0,tmax,dt)

    if np.real(decomp[0][0]) == decomp[0][0]:
        states = np.zeros((len(startstate), len(ts)), dtype=float)
    else:
        states = np.zeros((len(startstate), len(ts)), dtype=complex)
    states[:,0] = startstate


    for i in range(1,len(ts)):
        states[:,i] = states[:,i-1]
        for j in range(len(decomp[0])):
            if np.real(decomp[0][0]) == decomp[0][0]:
                states[:,i] += kinetic(states[:,i], dt, decomp[0][j], lamb)

                states[:,i] += potential(states[:,i], dt, decomp[1][j], lamb)
            else:
                states[:,i] += kineticComplex(states[:,i], dt, decomp[0][j], lamb)

                states[:,i] += potentialComplex(states[:,i], dt, decomp[1][j], lamb)

    return ts, np.float64(np.real(states))


from scipy.integrate import solve_ivp
def integrate_rk45(startstate, dt, tmax, lamb):
    ts = np.arange(0,tmax,dt)
    def f(t,y):
        return -np.array([y[2],y[3],-y[0] - 2*lamb*y[0]*y[1]*y[1], -y[1] - 2*lamb*y[0]*y[1]*y[0]]) 
    
    startstate[2:] = -np.array(startstate[2:])

    sol = solve_ivp(f, (0,tmax), startstate, "RK45", t_eval = ts, max_step = dt)

    return sol.t, sol.y



lamb = 0
a = 0.02

startstate = [0,1,1,0]
dt = 0.01
ts, states = integrate(startstate=startstate, dt = dt, tmax=20, lamb=lamb, decomp=decomp44)
plt.plot(states[0,:], states[1,:], )

startstate = [0,1-a,1,0]
dt = 0.01
ts, states = integrate(startstate=startstate, dt = dt, tmax=20, lamb=lamb, decomp=decomp44)
plt.plot(states[0,:], states[1,:], )

startstate = [0,1+a,1,0]
dt = 0.01
ts, states = integrate(startstate=startstate, dt = dt, tmax=20, lamb=lamb, decomp=decomp44)
plt.plot(states[0,:], states[1,:], )

startstate = [0,1,1+a,0]
dt = 0.01
ts, states = integrate(startstate=startstate, dt = dt, tmax=20, lamb=lamb, decomp=decomp44)
plt.plot(states[0,:], states[1,:], )

startstate = [0,1,1-a,0]
dt = 0.01
ts, states = integrate(startstate=startstate, dt = dt, tmax=20, lamb=lamb, decomp=decomp44)
plt.plot(states[0,:], states[1,:], )


plt.title(f"$\\lambda = {lamb}$")
plt.xlabel("x")
plt.ylabel("p")
plt.axis('equal')
plt.show()


tmax = 13
lamb = 10
startstate = [0,1,1,0]
dt = 0.01
ts, states = integrate(startstate=startstate, dt = dt, tmax=tmax, lamb=lamb, decomp=decomp44)
plt.plot(states[0,:], states[1,:], )

startstate = [0,1-a,1,0]
dt = 0.01
ts, states = integrate(startstate=startstate, dt = dt, tmax=tmax, lamb=lamb, decomp=decomp44)
plt.plot(states[0,:], states[1,:], )

startstate = [0,1+a,1,0]
dt = 0.01
ts, states = integrate(startstate=startstate, dt = dt, tmax=tmax, lamb=lamb, decomp=decomp44)
plt.plot(states[0,:], states[1,:], )

startstate = [0,1,1+a,0]
dt = 0.01
ts, states = integrate(startstate=startstate, dt = dt, tmax=tmax, lamb=lamb, decomp=decomp44)
plt.plot(states[0,:], states[1,:], )

startstate = [0,1,1-a,0]
dt = 0.01
ts, states = integrate(startstate=startstate, dt = dt, tmax=tmax, lamb=lamb, decomp=decomp44)
plt.plot(states[0,:], states[1,:], )

plt.title(f"$\\lambda = {lamb}$")
plt.xlabel("x")
plt.ylabel("p")
plt.axis('equal')
plt.show()