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



def ising(N, iters, J, h, T, startstate = None):
    maxiters = 5000
    states = np.zeros((N,N,iters), dtype=np.byte)
    energies = np.zeros(iters, dtype=np.float32)

    if startstate is None:
        startstate = np.zeros((N,N), dtype=np.byte)

        for i in range(N):
            for j in range(N):
                startstate[i,j] = np.random.randint(0,2)

    def E_pair(s1,s2):
        return -J * (1 if s1 == 1 else -1) * (1 if s2 == 1 else -1)
                
    E0 = np.float32(0)
    for i in range(N):
        for j in range(N):
            try:
                E0 += E_pair(startstate[i,j], startstate[(i+1)%N,j])
            except IndexError:
                pass

            try:
                E0 += E_pair(startstate[i,j], startstate[(i-1)%N,j])
            except IndexError:
                pass

            try:
                E0 += E_pair(startstate[i,j], startstate[i,(j+1)%N])
            except IndexError:
                pass

            try:
                E0 += E_pair(startstate[i,j], startstate[i,(j-1)%N])
            except IndexError:
                pass

            E0 += -h * startstate[i,j]
    
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
            i,j = np.random.randint(0,N), np.random.randint(0,N)

            #################
            E00 = 0
            try:
                E00 += E_pair(state[i,j], state[(i+1)%N,j])
            except IndexError:
                pass

            try:
                E00 += E_pair(state[i,j], state[(i-1)%N,j])
            except IndexError:
                pass

            try:
                E00 += E_pair(state[i,j], state[i,(j+1)%N])
            except IndexError:
                pass

            try:
                E00 += E_pair(state[i,j], state[i,(j-1)%N])
            except IndexError:
                pass

            new_state = state.copy()
            new_state[i,j] = 0 if new_state[i,j] == 1 else 1

            Enew = 0
            try:
                Enew += E_pair(new_state[i,j], new_state[(i+1)%N,j])
            except IndexError:
                pass

            try:
                Enew += E_pair(new_state[i,j], new_state[(i-1)%N,j])
            except IndexError:
                pass

            try:
                Enew += E_pair(new_state[i,j], new_state[i,(j+1)%N])
            except IndexError:
                pass

            try:
                Enew += E_pair(new_state[i,j], new_state[i,(j-1)%N])
            except IndexError:
                pass

            DeltaE = Enew-E00
            ######################

            if  DeltaE < 0 or np.random.random() < np.exp(-DeltaE/T):
                state = new_state
                E0 += DeltaE
                break

    return states, energies


def Ising_M(state):
    return np.average(2*state-1)

#@njit(nopython = True)
def Ising_sus(state, beta):
    M2 = 0
    M1 = 0
    state = 2*state-1
    
    M2 += np.sum(state*state)/np.size(state)
    M1 += np.sum(state)/np.size(state)


    return beta * (M2-M1**2)

#@njit(nopython = True)
def Ising_heat(state, beta, J):
    E2 = 0
    E1 = 0
    state = 2*state-1
    N = len(state[0,:])
    for i in range(N):
        for j in range(N):
            E1 += np.float64(-J*(state[i][(j+1)%N]+state[i][(j-1)%N]+state[(i+1)%N][j]+state[(i-1)%N][j])*state[i][j])/np.size(state)
            E2 +=np.float64((-J*(state[i][(j+1)%N]+state[i][(j-1)%N]+state[(i+1)%N][j]+state[(i-1)%N][j])*state[i][j])**2)/np.size(state)

    return beta**2 * (E2-E1**2)
    


def potts(N, iters, J, T, q, startstate = None):
    maxiters = 100
    states = np.zeros((N,N,iters), dtype=np.byte)
    energies = np.zeros(iters, dtype=np.float32)

    if startstate is None:
        startstate = np.zeros((N,N), dtype=np.byte)

        for i in range(N):
            for j in range(N):
                startstate[i,j] = np.random.randint(1,q+1)

    def E_pair(s1,s2):
        return -J * (1 if s1 == s2 else 0)
    
    def random_not_old(s,q):
        k = s
        while k == s:
            k = np.random.randint(1,q+1)
        return k
                
    E0 = np.float32(0)
    for i in range(N):
        for j in range(N):
            try:
                E0 += E_pair(startstate[i,j], startstate[i+1,j])
            except IndexError:
                pass

            try:
                E0 += E_pair(startstate[i,j], startstate[i-1,j])
            except IndexError:
                pass

            try:
                E0 += E_pair(startstate[i,j], startstate[i,j+1])
            except IndexError:
                pass

            try:
                E0 += E_pair(startstate[i,j], startstate[i,j-1])
            except IndexError:
                pass

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
                lave_condition = True
                break
            i,j = np.random.randint(0,N), np.random.randint(0,N)

            #################
            E00 = 0
            try:
                E00 += E_pair(state[i,j], state[i+1,j])
            except IndexError:
                pass

            try:
                E00 += E_pair(state[i,j], state[i-1,j])
            except IndexError:
                pass

            try:
                E00 += E_pair(state[i,j], state[i,j+1])
            except IndexError:
                pass

            try:
                E00 += E_pair(state[i,j], state[i,j-1])
            except IndexError:
                pass

            new_state = state.copy()
            new_state[i,j] = random_not_old(new_state[i,j], q)

            Enew = 0
            try:
                Enew += E_pair(new_state[i,j], new_state[i+1,j])
            except IndexError:
                pass

            try:
                Enew += E_pair(new_state[i,j], new_state[i-1,j])
            except IndexError:
                pass

            try:
                Enew += E_pair(new_state[i,j], new_state[i,j+1])
            except IndexError:
                pass

            try:
                Enew += E_pair(new_state[i,j], new_state[i,j-1])
            except IndexError:
                pass

            DeltaE = Enew-E00
            ######################

            if DeltaE < 0 or np.random.random() < np.exp(-DeltaE/T):
                state = new_state
                E0 += DeltaE
                break

    return states, energies

def Potts_M(state, q):
    return np.sum(np.exp(2*np.pi*1j*(state-1)/q))/np.size(state)

def Potts_sus(state, beta, q):
    M2 = 0
    M1 = 0
    
    M2 += np.sum(np.exp(2*np.pi*1j*(state-1)/q) * np.exp(-2*np.pi*1j*(state-1)/q))/np.size(state)
    M1 += np.sum(np.exp(2*np.pi*1j*(state-1)/q))/np.size(state)

    return beta * (M2-M1**2)

def Potts_heat(state, beta, J):
    E2 = 0
    E1 = 0

    N = len(state[0,:])
    for i in range(N):
        for j in range(N):
            DE = np.float64(-J*(float(state[i,(j+1)%N] == state[i,j]) + float(state[i,(j-1)%N] == state[i,j]) + float(state[(i+1)%N,j] == state[i,j])+ float(state[(i-1)%N,j] == state[i,j])))
            E1 += DE/np.size(state)
            E2 += DE**2/np.size(state)

    return beta**2 * (E2-E1**2)

    