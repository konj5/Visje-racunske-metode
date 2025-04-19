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

@njit(nopython = True)
def move(state, lamb, beta, ratio, M):
    while(True):
        i = np.random.randint(0,M)
        dq = np.random.normal(0,np.sqrt(beta))
        
        dE = 0
        dE += ratio/2 * ((state[(i+1)%M]-state[i]-dq)**2 - (state[(i+1)%M]-state[i])**2)
        dE += ratio/2 * ((state[(i-1)%M]-state[i]-dq)**2 - (state[(i-1)%M]-state[i])**2)
        dE += 1/ratio * ((state[i]+dq)**2-(state[i])**2)/2
        dE += 1/ratio * lamb * ((state[i]+dq)**4-(state[i])**4)

        if dE < 0 or np.exp(-beta*dE) > np.random.random():
            state[i] += dq
            return state


def attempt3(iters, lamb, beta, ratio):
    M=int(beta*ratio)
    if M<=2:
        raise ValueError("Bad ratio it seems!")
    state = np.zeros(M)

    for i in range(iters):
        state = move(state, lamb, beta, ratio, M)

    return state


def attempt2(lamb, iters, beta, ratio):
    eps = 0.3
    M = int(beta*ratio)
    if M <= 2:
        raise ValueError("Like bad params man!")
    
    MM=M
    if beta<=1:
        epsilon=np.sqrt(beta)
        MM=M*100
    
    qs=np.random.normal(0,1,M)
    E=np.zeros(iters)
    energy=0

    dq=np.subtract(qs,np.roll(qs,1))
    q2s=qs**2
    E0=ratio*np.dot(dq,dq)/2+np.dot(qs,qs)/(2*ratio)+lamb*np.dot(q2s,q2s)/ratio

    K,V,t,stevec,x,tt=0,0,0,0,0,0
    X,S=np.zeros(int((iters)/(10*M))),np.zeros(int((iters)/(10*M))) 
    velika=np.zeros(M)    
    baza_x=0

    for i in np.arange(iters):
        j=np.random.randint(0,M)
        q=qs[j]+np.random.uniform(-epsilon,epsilon)#normal(0,epsilon)
        
        D=q-qs[j]
        d=q+qs[j]
        D4=q**4-qs[j]**4
        dE=ratio*(D)*(d-qs[(j+1) % M] - qs[j-1])+D*d/(2*ratio)+lamb*D4/ratio
        
        if dE<0: qs[j],E[i],stevec=q,dE,stevec+1
        elif np.random.rand()<np.exp(-dE): qs[j],E[i],stevec=q,dE,stevec+1
        
        
        epsilon=epsilon+(stevec/(i+1))/MM
        
        if i%(M*10)==0:
            arg=np.subtract(qs,np.roll(qs,1))
            K=K-ratio*np.dot(arg,arg)/(2*beta)
            gugu=np.multiply(qs,qs)
            V=V+np.dot(qs,qs)/(2*M)+lamb*np.dot(gugu,gugu)/M
            x=x+np.sum(qs)/M
#                baza_x[t,:]=baza
#                velika=np.add(velika,np.multiply(baza[0],baza))
            t=t+1
            X[t]=x/t
            S[t]=i
    weird=np.divide(velika,t)
    return qs,E,E0,V/t,ratio/2+K/t,ratio/2+(K+V)/t,x/t,baza_x,epsilon,stevec/iters,np.divide(velika,t),-np.multiply(ratio,np.log(np.divide(weird,np.roll(weird,1)))),X,S
