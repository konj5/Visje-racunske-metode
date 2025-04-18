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



def heisenberg(N, iters, J, h, T, startstate = None):
    maxiters = 5000
    states = np.zeros((N,d,iters), dtype=np.byte)
    energies = np.zeros(iters, dtype=np.float32)

    if startstate is None:
        startstate = np.zeros((N,d), dtype=np.float32)

        for i in range(N):
            startstate[i,:] = np.random.rand(d)
            startstate[i,:] = startstate[i,:] / np.linalg.norm(startstate[i,:])
            
    E0 = np.float32(0)
    for i in range(N):
        E0 += -J * startstate[i,:].dot(startstate[(i+1)%N,:])
        E0 += -J * startstate[i,:].dot(startstate[(i-1)%N,:])

        E0 += -h * startstate[i,2]
    
    state = startstate
    leave_condition = False
    for k in trange(0,iters, leave=False):
        if leave_condition:
            break

        states[:,:,k] = state
        energies[k] = E0
        failsafe = 0
        while(True):
            failsafe += 1
            if failsafe > maxiters: 
                leave_condition = True
                break
            i = np.random.randint(0,N)
            diff = np.random.rand(d)
            diff = diff / np.linalg.norm(diff)
            diff = diff * 0.1
            original = state[i,:]
            new = original + diff
            new = new / np.linalg.norm(new)
            realdiff = new - original
            DeltaE = (J * state[(i+1)%N,:]+J * state[(i-1)%N,:]).dot(realdiff) + h * realdiff[2]
            ######################

            if  DeltaE < 0 or np.random.random() < np.exp(-DeltaE/T):
                state = new
                E0 += DeltaE
                break

    return states

from scipy.spatial.transform import Rotation

d = 3

def heisenberg_M0(N, iters, J, h, T, startstate = None):
    maxiters = 5000
    states = np.zeros((N,d,iters), dtype=np.float32)
    energies = np.zeros(iters, dtype=np.float32)

    if startstate is None:
        startstate = np.zeros((N,d), dtype=np.float32)

        ilist = np.arange(0,N,1)

        while(len(ilist) > 0):
            i = np.random.choice(len(ilist))
            ii = ilist[i]
            ilist = np.delete(ilist,i)
            i = ii
            j = np.random.choice(len(ilist))
            jj = ilist[j]
            ilist = np.delete(ilist,j)
            j = jj

            vec = np.random.random(d)
            vec = vec / np.linalg.norm(vec)

            startstate[i,:] = vec
            startstate[j,:] = -vec
    
       
    E0 = np.float32(0)
    for i in range(N):
        E0 += -J * startstate[i,:].dot(startstate[(i+1)%N,:])
        E0 += -J * startstate[i,:].dot(startstate[(i-1)%N,:])

        E0 += -h * startstate[i,-1]
    
    state = startstate
    leave_condition = False
    for k in trange(0,iters, leave=False):
        if leave_condition:
            break

        states[:,:,k] = state
        energies[k] = E0
        failsafe = 0
        while(True):
            failsafe += 1
            if failsafe > maxiters: 
                leave_condition = True
                break
            i = np.random.randint(0,N)
            j = np.random.choice([(i+1)%N,(i-1)%N])

            s = state[i,:]
            q = state[j,:]

            if (s == -q).all():
                s = -s
                q = -q

            else:
                phi = np.random.rand() * 2*np.pi/25
                rot = Rotation.from_rotvec(phi * (s+q)/np.linalg.norm(s+q))

                s = rot.apply(s)
                q = rot.apply(q)

            ######################

            realdiff = state[i,:] - s
            DeltaE = (J * state[(i+1)%N,:]+J * state[(i-1)%N,:]).dot(realdiff) + h * realdiff[2]

            realdiff = state[j,:] - q
            DeltaE += (J * state[(j+1)%N,:]+J * state[(j-1)%N,:]).dot(realdiff) + h * realdiff[2]

            new = state.copy()
            new[i,:] = s
            new[j,:] = q

            if  DeltaE < 0 or np.random.random() < np.exp(-DeltaE/T):
                #print(f"{DeltaE}") 
                state = new
                E0 += DeltaE
                break

    #plt.plot(energies)
    #plt.show()

    return states


def correlation(state):
    N = len(state[:,0])
    rs = np.arange(0,N//2,1)
    cs = np.zeros(len(rs), dtype=np.float32)
    #print(state)

    for j in range(len(rs)):
        r = rs[j]
        for i in range(N):
            cs[j] += state[i,:].dot((state[(i+r)%N,:]))
            #cs[j] += state[i,2]*(state[(i+r)%N,2])
        cs[j] = cs[j]/N

    return rs, cs

def correlationZ(state):
    N = len(state[:,0])
    rs = np.arange(0,N//2,1)
    cs = np.zeros(len(rs), dtype=np.float32)
    #print(state)

    for j in range(len(rs)):
        r = rs[j]
        for i in range(N):
            #cs[j] += state[i,:].dot((state[(i+r)%N,:]))
            cs[j] += state[i,2]*(state[(i+r)%N,2])
        cs[j] = cs[j]/N

    return rs, cs



h = 10
J = 1
N = 100
iters = 1000000
Ts = [0.001, 0.01, 0.1, 1, 5]
rss = []
css = []
cssz = []

for i in trange(0,len(Ts)):
    T = Ts[i]
    states = heisenberg_M0(N,iters,J,h,T)
    print(np.sum(states[:,:,-1], axis=0))

    rs, cs = correlation(states[:,:,-1])

    rss.append(rs)
    css.append(cs)

    rs, cs = correlationZ(states[:,:,-1])

    cssz.append(cs)
    
    #plt.plot(rs, cs, label = f"$T = {T}$")

fig, axs = plt.subplots(1,2)
ax1, ax2 = axs

for i in trange(0,len(Ts)):
    T = Ts[i]
    ax1.plot(rss[i], css[i], label = f"$T = {T}$")
    ax2.plot(rss[i], cssz[i], label = f"$T = {T}$")

ax2.legend()
ax1.set_xlabel("$r$")
ax1.set_ylabel("$\\langle \\vec{c}_0 \\vec{c}_r \\rangle$")

ax2.set_xlabel("$r$")
ax2.set_ylabel("$\\langle c_0^z c_r^z \\rangle$")

plt.tight_layout()
plt.show()

    


