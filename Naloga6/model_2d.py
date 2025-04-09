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



def ising(N, iters, J, h, T):
    maxiters = 100
    states = np.zeros((N,N,iters), dtype=np.byte)
    energies = np.zeros(iters, dtype=np.float32)
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
                E0 += E_pair(startstate[i,j], startstate[i,-1])
            except IndexError:
                pass

            E0 += -h * startstate[i,j]
    
    state = startstate
    leave_condition = False
    for k in trange(0,iters):
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
                E00 += E_pair(state[i,j], state[i,-1])
            except IndexError:
                pass

            new_state = state.copy()
            new_state[i,j] = 0 if new_state[i,j] == 1 else 1

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
                Enew += E_pair(new_state[i,j], new_state[i,-1])
            except IndexError:
                pass

            DeltaE = Enew-E00
            ######################

            if DeltaE < 0 or np.random.random() < np.exp(-DeltaE/T):
                state = new_state
                E0 += DeltaE
                break

    return states, energies


def Ising_M(states):
    return np.average(2*states-1, axis=2)

def Ising_sus(mags, beta):
    mags = mags[len(mags)//2:]
    return beta * np.var(mags)

def Ising_heat(energs, beta):
    energs = energs[len(energs)//2:]
    return beta**2 * np.var(energs)
    


def potts(N, iters, J, T, q):
    maxiters = 100
    states = np.zeros((N,N,iters), dtype=np.byte)
    energies = np.zeros(iters, dtype=np.float32)
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
                E0 += E_pair(startstate[i,j], startstate[i,-1])
            except IndexError:
                pass

    state = startstate
    leave_condition = False
    for k in trange(0,iters):
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
                E00 += E_pair(state[i,j], state[i,-1])
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
                Enew += E_pair(new_state[i,j], new_state[i,-1])
            except IndexError:
                pass

            DeltaE = Enew-E00
            ######################

            if DeltaE < 0 or np.random.random() < np.exp(-DeltaE/T):
                state = new_state
                E0 += DeltaE
                break

    return states, energies

def Potts_M(states, q):
    return np.sum(np.exp(2*np.pi*1j*(states-1)/q), axis=2)

def Potts_sus(mags, beta):
    mags = mags[len(mags)//2:]
    return beta * np.var(mags)

def Potts_heat(energs, beta):
    energs = energs[len(energs)//2:]
    return beta**2 * np.var(energs)

    