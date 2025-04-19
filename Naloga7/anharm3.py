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
from numba import jit, njit
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))



def kill_me(iters, beta, ratio, part, limit, lamb):
    epsilon = 0.3
    M = int(beta*ratio)
    if M<=2:
        raise ValueError("bad ratio")
    
    MM = M
    if beta <= 1:
        epsilon = np.sqrt(beta)
        MM = M * 100
    
    state = np.random.normal(0,1,M)
    E = np.zeros(iters)
    energy = 0

    arg = state - np.roll(state,1)
    gugu = np.multiply(state,state)
    E0 = ratio * np.dot(arg,arg)/2 + np.dot(state,state)/(2*ratio) + lamb * np.dot(gugu, gugu)/ratio
    

    K, V, t, counter, x, tt = 0, 0, 0, 0, 0, 0
    X = np.zeros(int((iters-limit)/(10*M)))
    S = np.zeros(int((iters-limit)/(10*M)))
    velika = np.zeros(M)
    baza_x = 0
    print(int((iters-limit)/(10*M)))

    for i in np.arange(iters):
        j = np.random.randint(0,M)
        q = state[j] + np.random.uniform(-epsilon, epsilon)


        D = q-state[j]
        d = q+state[j]
        D4 = q**4-state[j]**4
        dE = ratio*D*(d-state[(j+1)%M] - state[(j-1)%M]) + D * d / (2*ratio) + lamb * D4/ratio

        if dE<0 or np.random.rand() < np.exp(-dE): ####################################################################### Probable error
            state[j] = q
            E[i] = dE
            counter += 1

        epsilon = epsilon + (counter / (i+1) - part) / MM

        if i > limit and i % (M * 10) == 0:
            arg = state - np.roll(state,1)
            K = K - ratio * np.dot(arg,arg)/(2*beta)
            gugu = state * state
            V = V + np.dot(state, state)/(2*M) + lamb * np.dot(gugu,gugu)/M
            x = x + np.sum(state)/M
            t = t+1
            #X[t] = x/t
            #S[t] = i

    weird = velika/t

    #return state, E, E0, V/t, ratio/2 + K/t, ratio/2+(K+V)/t, x/t, baza_x, epsilon, counter/iters, velika/t, -np.multiply(ratio,np.log(np.divide(weird,np.roll(weird,1)))),X,S
    return state,E,E0,V/t,ratio/2+K/t,ratio/2+(K+V)/t,x/t,baza_x,epsilon,counter/iters


def BETA(betas,iters,ratio,part,limit,lamb):
    D=len(betas)
    S,T,V=np.zeros(D),np.zeros(D),np.zeros(D)
    for i in np.arange(D):
        print(len(kill_me(iters,betas[i],ratio,part,limit,lamb)))
        a,b,c,d,e,f,g,h,rg,hz=kill_me(iters,betas[i],ratio,part,limit,lamb)
        S[i]=f
        T[i],V[i]=d,e
        print(i)
        print(f,rg,hz)
    return betas,S,T,V

betas, S, T, V = BETA(betas = np.linspace(0.1,1000,20), iters = 1000, ratio = 100, part = 10, limit = -1, lamb = 0)
plt.plot(betas, S, label = "E")
plt.plot(betas, T, label = "T")
plt.plot(betas, V, label = "V")
plt.legend()
plt.tight_layout()
plt.show()

