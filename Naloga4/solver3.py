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
from numba import jit
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))


#IMPLEMENTED BASIS


s0 = np.eye(2, dtype=complex)
sx = np.array([[0,1],[1,0]], dtype=complex)
sy = np.array([[0,-1j],[1j,0]], dtype=complex)
sz = np.array([[1,0],[0,-1]], dtype=complex)

p1 = (1 + 1j/np.sqrt(3))/4
p2 = 2*p1
p3 = 1/2+0j
p4 = np.conj(p2)
p5 = np.conj(p1)

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
decomp22 = np.array(decomp22)

# p = 4, k = 4
r2 = 2**(1/3)
x1 = 1/(2-r2)
x0 = -r2*x1
decomp44 = [
    [x1/2,(x0+x1)/2 ,(x0+x1)/2 ,x1/2],
    [x1,x0,x1,0]
]
decomp44 = np.array(decomp44)

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
decomp33 = np.array(decomp33)

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
decomp45 = np.array(decomp45)

@jit
def e_zh01(z, Jx, Jy, Jz):
    mat = np.zeros((4,4), dtype=np.complex128)

    ez = np.exp(z*Jz)
    emz = np.exp(-z*Jz)

    csh_m = np.cosh(z*(Jx-Jy))
    csh = np.cosh(z*(Jx+Jy))

    snh_m = np.sinh(z*(Jx-Jy))
    snh = np.sinh(z*(Jx+Jy))

    mat[0,0] = ez * csh_m
    mat[0,3] = ez * snh_m
    mat[1,1] = emz * csh
    mat[1,2] = emz * snh
    mat[2,2] = emz * csh
    mat[2,1] = emz * snh
    mat[3,3] = ez * csh_m
    mat[3,0] = ez * snh_m

    return mat

@jit
def to_binary_state(k, n):
    """state = np.zeros(n,dtype=np.byte)
    dat = np.array(list(bin(k)[2:]))
    state[-len(dat):] = dat
    return state"""
    state = np.zeros(n, dtype=np.byte)
    if k == 0:
        return state

    binary = []
    while k != 0:
        bit = k % 2
        binary.insert(0, bit)
        k = k // 2
    state[-len(binary):] = binary
    return state


@jit
def to_decimal_state(binarr):
    val = 0
    for i in range(len(binarr)):
        val += binarr[-(i+1)] * 2**i
    return val

@jit
def applyA(state, n, U2, basis):
    N = len(state)

    for j in np.arange(0,n,2):
        endstate = np.zeros(len(state), dtype=np.complex128)
        
        for i in range(N):
            binrepr = basis[i,:]
            bits = binrepr[j:j+2]
            decibits = to_decimal_state(bits)

            for endstate_local in [[0,0], [0,1], [1,0],[1,1]]:
                end_binrepr =  binrepr.copy()
                end_binrepr[j:j+2] = np.array(endstate_local)

                end_deci_local = to_decimal_state(np.array(endstate_local))
                end_deci = to_decimal_state(end_binrepr)

                if state[i] != 0:
                    endstate[end_deci] += U2[end_deci_local,decibits] * state[i]

        state = endstate.copy()

        #(endstate)

    return state

@jit
def applyB(state, n, U2, basis):
    N = len(state)

    for j in np.arange(1,n-1,2):
        endstate = np.zeros(len(state), dtype=np.complex128)
        
        for i in range(N):
            binrepr = basis[i,:]
            bits = binrepr[j:j+2]
            decibits = to_decimal_state(bits)

            for endstate_local in [[0,0], [0,1], [1,0],[1,1]]:
                end_binrepr =  binrepr.copy()
                end_binrepr[j:j+2] = np.array(endstate_local)

                end_deci_local = to_decimal_state(np.array(endstate_local))
                end_deci = to_decimal_state(end_binrepr)

                if state[i] != 0:
                    endstate[end_deci] += U2[end_deci_local,decibits] * state[i]

        state = endstate.copy()

    #special case at periodic boundary

    endstate = np.zeros(len(state), dtype=np.complex128)
        
    for i in range(N):
        binrepr = basis[i,:]

        #change here
        bits = np.array([binrepr[-1], binrepr[0]])

        decibits = to_decimal_state(bits)

        for endstate_local in [[0,0], [0,1], [1,0],[1,1]]:
            end_binrepr =  binrepr.copy()

            #change here
            end_binrepr[-1] = np.array(endstate_local)[0]
            end_binrepr[0] = np.array(endstate_local)[1]

            end_deci_local = to_decimal_state(np.array(endstate_local))
            end_deci = to_decimal_state(end_binrepr)

            if state[i] != 0:
                endstate[end_deci] += U2[end_deci_local,decibits] * state[i]

    state = endstate.copy()



        #(endstate)

    return state

"""print(to_binary_state(2,4))
startstate = np.array([0 if i != 2 else 1 for i in range(2**4)], dtype=np.complex128)
print(startstate)
print(applyA(startstate, 4, np.eye(4)))
print(applyB(startstate, 4, np.eye(4)))"""




@jit
def evolve_state_once(state, n, dt, Jx, Jy, Jz, decomp, basis):
    for i in range(len(decomp[0,:])):
        U2 = e_zh01(-1j * dt * decomp[0,i], Jx, Jy, Jz)
        state = applyA(state, n, U2, basis)

        U2 = e_zh01(-1j * dt * decomp[1,i], Jx, Jy, Jz)
        state = applyB(state, n, U2, basis)

    return state



"""startstate = np.array([0 if i != 2 else 1 for i in range(2**4)], dtype=np.complex128)
print(startstate)
print(evolve_state_once(startstate,4,0.1,1,1,1,decomp45))"""

def evolve(state, n, dz, z, Jx, Jy, Jz, decomp):
    assert n % 2 == 0
    assert len(state) == 2**n

    basis = np.zeros((2**n, n), dtype=np.byte)
    for i in range(2**n):
        basis[i,:] = to_binary_state(i,n)

    ts = np.arange(0,z,dz)
    for i in range(len(ts)):
        state = evolve_state_once(state, n, dz, Jx, Jy, Jz, decomp, basis)
    return state

@jit
def build_base(n):
    basis = np.zeros((2**n, n), dtype=np.byte)
    for i in range(2**n):
        basis[i,:] = to_binary_state(i,n)

    print(f"basis constructed!")
    return basis

"""build_base(2)
stime = time.time()
build_base(14)
print(time.time()-stime)"""

@jit
def evolve_zs(state, n, dz, z, Jx, Jy, Jz, decomp, basis):
    assert n % 2 == 0
    assert len(state) == 2**n

    if np.imag(z) == 0:
        ts = np.arange(0,np.abs(z),dz, dtype=numba.complex128)
    else:
        ts = np.arange(0,np.abs(np.imag(z)),np.imag(dz), dtype=numba.complex128)

    states = np.zeros((len(ts), len(state)), dtype=numba.complex128)
    states[0,:] = state

    #print("\n")
    for i in range(1,len(ts)):
        states[i,:] = evolve_state_once(states[i-1,:], n, dz, Jx, Jy, Jz, decomp, basis)
        #print(f"{i}/{len(ts)}")
    return states

"""startstate = np.array([0 if i != 2 else 1 for i in range(2**4)], dtype=np.complex128)
print(startstate)
end = evolve(startstate,4,0.1,1,1,1,1,decomp45)
print(end)
print(np.linalg.norm(end))"""

@jit
def get_typicall_state(n):
    sigma = 0.5

    compstate = np.random.normal(0,sigma,2**n) + 1j * np.random.normal(0,sigma,2**n)

    compstate /= np.linalg.norm(compstate)

    return compstate


def Z(N_psi, n, db, b, Jx, Jy, Jz, decomp):

    basis = build_base(n)

    summed = 0
    for _ in tqdm(range(N_psi)):
        state = get_typicall_state(n)
        state = evolve(state, n, 1j*db, -1j*b/2, Jx, Jy, Jz, decomp, basis)
        summed += np.linalg.norm(state)

    return summed/N_psi


def Z_zs(N_psi, n, db, b, Jx, Jy, Jz, decomp):
    basis = build_base(n)
    summed = 0
    for _ in  tqdm(range(N_psi)):
        state = get_typicall_state(n)
        states = evolve_zs(state, n, 1j*db, -1j*b/2, Jx, Jy, Jz, decomp, basis)
        summed += np.linalg.norm(states, axis=1)

    return summed/N_psi

def F(N_psi, n, db, b, Jx, Jy, Jz, decomp):
    return -1/b * np.log(Z(N_psi, n, db, b, Jx, Jy, Jz, decomp))

def F_zs(N_psi, n, db, b, Jx, Jy, Jz, decomp):
    Zs = Z_zs(N_psi, n, db, b, Jx, Jy, Jz, decomp)
    bs = np.linspace(0,b,len(Zs))
    return -1/bs * np.log(Zs)

@jit
def H(state, n, basis, Jx, Jy, Jz):
    endstate = np.zeros((2**n), dtype=np.complex128)
    for j in range(n-1):
        for i in range(2**n):
            bitrepr = basis[i]
            bits = bitrepr[j:j+2]

            if np.all(bits == np.array([0,0])):
                bitrepr[j:j+2] = np.array([0,0])
                endstate[to_decimal_state(bitrepr)] += Jz * state[i]
                bitrepr[j:j+2] = np.array([1,1])
                endstate[to_decimal_state(bitrepr)] += (Jx-Jy) * state[i]
            elif np.all(bits == np.array([0,1])):
                bitrepr[j:j+2] = np.array([0,1])
                endstate[to_decimal_state(bitrepr)] += -Jz * state[i]
                bitrepr[j:j+2] = np.array([1,0])
                endstate[to_decimal_state(bitrepr)] += (Jx+Jy) * state[i]
            elif np.all(bits == np.array([1,0])):
                bitrepr[j:j+2] = np.array([0,1])
                endstate[to_decimal_state(bitrepr)] += (Jx+Jy) * state[i]
                bitrepr[j:j+2] = np.array([1,0])
                endstate[to_decimal_state(bitrepr)] += -Jz * state[i]
            elif np.all(bits == np.array([1,1])):
                bitrepr[j:j+2] = np.array([0,0])
                endstate[to_decimal_state(bitrepr)] += (Jx-Jz) * state[i]
                bitrepr[j:j+2] = np.array([1,1])
                endstate[to_decimal_state(bitrepr)] += Jz * state[i]

    for i in range(2**n):
            bitrepr = basis[i]
            bits = np.array([bitrepr[-1], bitrepr[0]])

            if np.all(bits == np.array([0,0])):
                bitrepr[-1], bitrepr[0] = np.array([0,0])
                endstate[to_decimal_state(bitrepr)] += Jz * state[i]
                bitrepr[-1], bitrepr[0] = np.array([1,1])
                endstate[to_decimal_state(bitrepr)] += (Jx-Jy) * state[i]
            elif np.all(bits == np.array([0,1])):
                bitrepr[-1], bitrepr[0] = np.array([0,1])
                endstate[to_decimal_state(bitrepr)] += -Jz * state[i]
                bitrepr[-1], bitrepr[0] = np.array([1,0])
                endstate[to_decimal_state(bitrepr)] += (Jx+Jy) * state[i]
            elif np.all(bits == np.array([1,0])):
                bitrepr[-1], bitrepr[0] = np.array([0,1])
                endstate[to_decimal_state(bitrepr)] += (Jx+Jy) * state[i]
                bitrepr[-1], bitrepr[0] = np.array([1,0])
                endstate[to_decimal_state(bitrepr)] += -Jz * state[i]
            elif np.all(bits == np.array([1,1])):
                bitrepr[-1], bitrepr[0] = np.array([0,0])
                endstate[to_decimal_state(bitrepr)] += (Jx-Jz) * state[i]
                bitrepr[-1], bitrepr[0] = np.array([1,1])
                endstate[to_decimal_state(bitrepr)] += Jz * state[i]  
    
    return endstate
    



def E_zs(N_psi, n, db, b, Jx, Jy, Jz, decomp):

    basis = build_base(n)

   

    summed = 0
    for _ in tqdm(range(N_psi)):
        states = evolve_zs(get_typicall_state(n), n, 1j*db, -1j*b/2, Jx, Jy, Jz, decomp, basis)
        Hstates = np.zeros(states.shape, dtype=np.complex128)
        for i in range(len(states[:,0])):
            Hstates[i,:] = H(states[i,:], n, basis, Jx, Jy, Jz)
        
        summed += np.einsum("ij,ij->i", states.conj(), Hstates)
    
    summed = summed/N_psi

    return summed/Z_zs(N_psi, n, db, b, Jx, Jy, Jz, decomp)





#korelator spina v z smeri na site1 ob času 0 in site2 ob času t
def correlator_zz(site1, site2, N_psi, n, dt, t, Jx, Jy, Jz, decomp):
    basis = build_base(n)
    summed = 0
    for _ in  tqdm(range(N_psi)):
        randstate = get_typicall_state(n)

        #prepare states 1:
        state1 = np.zeros(len(randstate), dtype=np.complex128)
        for i in range(len(randstate)):
            bin_repr = basis[i]
            state1[i] = randstate[i] * (-1)**bin_repr[site1]
        states1 = evolve_zs(state1,n,dt,t,Jx,Jy,Jz,decomp, basis)

        #prepare states 2:
        states2 = evolve_zs(randstate,n,dt,t,Jx,Jy,Jz,decomp, basis)
        for j in range(len(states2[:,0])):
            for i in range(len(randstate)):
                bin_repr = basis[i]
                states2[j,i] = states2[j,i] * (-1)**bin_repr[site2]
        states2 = states2.conj()

        #calculate corelators
        summed += np.einsum("ij,ij->i", states1, states2)
    
    return summed/N_psi

#@jit
def correlator_JJ_NOT_WORKING_RIGHT(N_psi, n, dt, t, Jx, Jy, Jz, decomp):
    basis = build_base(n)
    summed = 0
    #for _ in  tqdm(range(N_psi)):
    for _ in range(N_psi):
        randstate = get_typicall_state(n)

        for site in range(0,n-1,1):
            print(f"{_}/{N_psi}, {site}/{n-1}")

            #prepare states 1:
            state1 = np.zeros(len(randstate), dtype=np.complex128)
            for i in range(len(randstate)):
                bin_repr = basis[i]
                
                tempstate = np.zeros(len(randstate), dtype=np.complex128)
                tempstate[i] = 1

                #apply y2, x1
                if bin_repr[site+1] == 0:
                    new_bin_repr = bin_repr.copy()
                    new_bin_repr[site+1] = 1
                    new_i = to_decimal_state(new_bin_repr)
                    tempstate = np.zeros(len(randstate), dtype=np.complex128)
                    tempstate[new_i] = 1j
                else:
                    new_bin_repr = bin_repr.copy()
                    new_bin_repr[site+1] = 0
                    new_i = to_decimal_state(new_bin_repr)
                    tempstate = np.zeros(len(randstate), dtype=np.complex128)
                    tempstate[new_i] = -1j

                bin_repr = new_bin_repr.copy()
                
                if bin_repr[site] == 0:
                    new_bin_repr = bin_repr.copy()
                    new_bin_repr[site+1] = 1
                    new_i = to_decimal_state(new_bin_repr)
                    tempstate = np.zeros(len(randstate), dtype=np.complex128)
                    tempstate[new_i] = 1
                else:
                    new_bin_repr = bin_repr.copy()
                    new_bin_repr[site] = 0
                    new_i = to_decimal_state(new_bin_repr)
                    tempstate = np.zeros(len(randstate), dtype=np.complex128)
                    tempstate[new_i] = 1

                tempstate1 = tempstate.copy()

                tempstate = np.zeros(len(randstate), dtype=np.complex128)
                tempstate[i] = 1

                #apply y2, x1
                if bin_repr[site+1] == 0:
                    new_bin_repr = bin_repr.copy()
                    new_bin_repr[site+1] = 1
                    new_i = to_decimal_state(new_bin_repr)
                    tempstate = np.zeros(len(randstate), dtype=np.complex128)
                    tempstate[new_i] = 1
                else:
                    new_bin_repr = bin_repr.copy()
                    new_bin_repr[site+1] = 0
                    new_i = to_decimal_state(new_bin_repr)
                    tempstate = np.zeros(len(randstate), dtype=np.complex128)
                    tempstate[new_i] = 1

                bin_repr = new_bin_repr.copy()
                
                if bin_repr[site] == 0:
                    new_bin_repr = bin_repr.copy()
                    new_bin_repr[site+1] = 1
                    new_i = to_decimal_state(new_bin_repr)
                    tempstate = np.zeros(len(randstate), dtype=np.complex128)
                    tempstate[new_i] = 1j
                else:
                    new_bin_repr = bin_repr.copy()
                    new_bin_repr[site] = 0
                    new_i = to_decimal_state(new_bin_repr)
                    tempstate = np.zeros(len(randstate), dtype=np.complex128)
                    tempstate[new_i] = -1j

                tempstate2 = tempstate.copy()

                state1 += 2*(tempstate1-tempstate2)

            states1 = evolve_zs(state1,n,dt,t,Jx,Jy,Jz,decomp, basis)

            #prepare states 2:
            states2 = evolve_zs(randstate,n,dt,t,Jx,Jy,Jz,decomp, basis)
            for j in range(len(states2[:,0])):
                for i in range(len(randstate)):
                    bin_repr = basis[i]
                    
                    tempstate = np.zeros(len(randstate), dtype=np.complex128)
                    tempstate[i] = 1

                    #apply y2, x1
                    if bin_repr[site+1] == 0:
                        new_bin_repr = bin_repr.copy()
                        new_bin_repr[site+1] = 1
                        new_i = to_decimal_state(new_bin_repr)
                        tempstate = np.zeros(len(randstate), dtype=np.complex128)
                        tempstate[new_i] = 1j
                    else:
                        new_bin_repr = bin_repr.copy()
                        new_bin_repr[site+1] = 0
                        new_i = to_decimal_state(new_bin_repr)
                        tempstate = np.zeros(len(randstate), dtype=np.complex128)
                        tempstate[new_i] = -1j

                    bin_repr = new_bin_repr.copy()
                    
                    if bin_repr[site] == 0:
                        new_bin_repr = bin_repr.copy()
                        new_bin_repr[site+1] = 1
                        new_i = to_decimal_state(new_bin_repr)
                        tempstate = np.zeros(len(randstate), dtype=np.complex128)
                        tempstate[new_i] = 1
                    else:
                        new_bin_repr = bin_repr.copy()
                        new_bin_repr[site] = 0
                        new_i = to_decimal_state(new_bin_repr)
                        tempstate = np.zeros(len(randstate), dtype=np.complex128)
                        tempstate[new_i] = 1

                    tempstate1 = tempstate.copy()

                    tempstate = np.zeros(len(randstate), dtype=np.complex128)
                    tempstate[i] = 1

                    #apply y2, x1
                    if bin_repr[site+1] == 0:
                        new_bin_repr = bin_repr.copy()
                        new_bin_repr[site+1] = 1
                        new_i = to_decimal_state(new_bin_repr)
                        tempstate = np.zeros(len(randstate), dtype=np.complex128)
                        tempstate[new_i] = 1
                    else:
                        new_bin_repr = bin_repr.copy()
                        new_bin_repr[site+1] = 0
                        new_i = to_decimal_state(new_bin_repr)
                        tempstate = np.zeros(len(randstate), dtype=np.complex128)
                        tempstate[new_i] = 1

                    bin_repr = new_bin_repr.copy()
                    
                    if bin_repr[site] == 0:
                        new_bin_repr = bin_repr.copy()
                        new_bin_repr[site+1] = 1
                        new_i = to_decimal_state(new_bin_repr)
                        tempstate = np.zeros(len(randstate), dtype=np.complex128)
                        tempstate[new_i] = 1j
                    else:
                        new_bin_repr = bin_repr.copy()
                        new_bin_repr[site] = 0
                        new_i = to_decimal_state(new_bin_repr)
                        tempstate = np.zeros(len(randstate), dtype=np.complex128)
                        tempstate[new_i] = -1j

                    tempstate2 = tempstate.copy()

                    states2[j,:] += 2*(tempstate1-tempstate2)

            states2 = states2.conj()

            #calculate corelators
            summed += np.einsum("ij,ij->i", states1, states2)
    
    return summed/N_psi


#def magnetization(N_psi, n, dt, t, Jx, Jy, Jz, decomp):

J2 = [
    [0,0,0,0],
    [0,0,-2j,0],
    [0,2j,0,0],
    [0,0,0,0]
]
    
@jit
def J(state, n, basis):
    endstate = np.zeros(2**n, dtype=np.complex128)

    for j in range(n-1): 
        for i in range(2**n):
            binrepr = basis[i] 
            bits = binrepr[j:j+2]
            
            if np.all(bits == np.array([0,1])):
                endbin = binrepr.copy()
                endbin[j:j+2] = [1,0]
                endstate[to_decimal_state(endbin)] += 2j * state[j]

            elif np.all(bits ==  np.array([1,0])):
                endbin = binrepr.copy()
                endbin[j:j+2] = [0,1]
                endstate[to_decimal_state(endbin)] += -2j * state[j]
        
    for i in range(2**n):
        binrepr = basis[i] 
        bits = np.array([binrepr[-1], binrepr[0]])
        
        if np.all(bits == np.array([0,1])):
            endbin = binrepr.copy()
            endbin[-1], endbin[0] = [1,0]
            endstate[to_decimal_state(endbin)] += 2j * state[j]

        elif np.all(bits ==  np.array([1,0])):
            endbin = binrepr.copy()
            endbin[-1], endbin[0] = [0,1]
            endstate[to_decimal_state(endbin)] += -2j * state[j]


    return endstate



def correlator_JJ(N_psi, n, dt, t, Jx, Jy, Jz, decomp):
    basis = build_base(n)
    summed = 0
    for i in tqdm(range(N_psi)):
        randstate = get_typicall_state(n)
        
        #Prepare the 1st states
        state1 = randstate.copy()
        state1 = J(state1,n,basis)
        states1 = evolve_zs(state1,n,dt,t,Jx,Jy,Jz,decomp, basis)

        #Prepare the 2nd states
        states2 = evolve_zs(randstate,n,dt,t,Jx,Jy,Jz,decomp, basis)
        for j in range(len(states2[:,0])):
            states2[j,:] = J(states2[j,:], n, basis)

        states2 = states2.conj()

        #calculate corelators
        summed += np.einsum("ij,ij->i", states1, states2)

    return summed/N_psi

@jit
def get_typicall_state_T(n,Jx,Jy,Jz,T,basis):
    sigma = 0.5

    while(True):
        compstate = np.random.normal(0,sigma,2**n) + 1j * np.random.normal(0,sigma,2**n)

        compstate /= np.linalg.norm(compstate)

        E = np.real(compstate.conj().dot(H(compstate,n,basis,Jx,Jy,Jz)))
        if 1 < np.random.exponential(E/T):
            return compstate


def correlator_JJT(T,N_psi, n, dt, t, Jx, Jy, Jz, decomp):
    basis = build_base(n)
    summed = 0
    for i in tqdm(range(N_psi)):
        randstate = get_typicall_state_T(n, Jx, Jy, Jz, T, basis)
        
        #Prepare the 1st states
        state1 = randstate.copy()
        state1 = J(state1,n,basis)
        states1 = evolve_zs(state1,n,dt,t,Jx,Jy,Jz,decomp, basis)

        #Prepare the 2nd states
        states2 = evolve_zs(randstate,n,dt,t,Jx,Jy,Jz,decomp, basis)
        for j in range(len(states2[:,0])):
            states2[j,:] = J(states2[j,:], n, basis)

        states2 = states2.conj()

        #calculate corelators
        summed += np.einsum("ij,ij->i", states1, states2)

    return summed/N_psi
        


