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


import model_2d

"""N = 25
iters = 4000
J = 1
h = 0
beta = 10
T = 1/beta

states, energies = model_2d.ising(N,iters,J,h,T)

plt.plot(energies)
plt.show()

plt.imshow(states[:,:,-1], cmap=plt.get_cmap("binary"))

plt.show()"""


"""#### Solutions for different temps
N = 100
iters = 100000

h = 0

Ts = [0.1, 1, 2, 2.269, 3, 5]
datas = np.zeros((len(Ts), N, N), dtype=np.byte)
fig, axs = plt.subplots(2,len(Ts))

for i in trange(0, len(Ts)):
    T = Ts[i]

    J = 1
    states, energies = model_2d.ising(N,iters,J,h,T)
    if np.sum(states[:,:,-1]*2-1) < 0: states[:,:,-1] = -states[:,:,-1]
    if i == 0:
       axs[0,i].imshow(states[:,:,-1]*0 + 1, cmap=plt.get_cmap("binary")) 
    else:
        axs[0,i].imshow(states[:,:,-1], cmap=plt.get_cmap("binary"))
    axs[0,i].set_xticks([])
    axs[0,i].set_yticks([])

    J = -1
    states, energies = model_2d.ising(N,iters,J,h,T)
    axs[1,i].imshow(states[:,:,-1], cmap=plt.get_cmap("binary"))
    axs[1,i].set_xticks([])
    axs[1,i].set_yticks([])

    axs[0,i].set_title(f"$T = {T}$")

axs[0,0].set_ylabel("$J=1$")
axs[1,0].set_ylabel("$J=-1$")

plt.show()"""



#Susceptibilnost

betas = np.linspace(1/5, 1/0.5, 500)

N = 50
iters = 10000
J = 1
h = 0

magss = np.zeros_like(betas)
susceps = np.zeros_like(betas)
sptopls = np.zeros_like(betas)

startstate = None

for i in trange(0, len(betas)):
    beta = betas[i]
    T = 1/beta
    states, energies = model_2d.ising(N,iters,J,h,T,startstate)

    #Try if this works or not!
    startstate = states[:,:,-1]
    #print(startstate)

    mags = model_2d.Ising_M(states[:,:,-1])
    sus = model_2d.Ising_sus(states[:,:,-1], beta)
    sph = model_2d.Ising_heat(states[:,:,-1], beta, J)

    magss[i] = mags
    susceps[i] = sus
    sptopls[i] = sph


    
fig, axs = plt.subplots(1,3)


axs[0].scatter(betas, np.abs(magss), s = 10, marker = "+")
axs[1].scatter(betas, susceps, s = 10, marker = "+")
axs[2].scatter(betas, sptopls, s = 10, marker = "+")

plt.show()

fig, axs = plt.subplots(1,3)


axs[0].plot(1/betas, np.abs(magss), marker = "+", linestyle='None')

Ts = np.linspace(np.min(1/betas), np.max(1/betas), 10000)
Ms = (1-np.asinh(np.log(1+np.sqrt(2)) * Ts/(2/np.log(1+np.sqrt(2))))**4)**(1/8)

axs[0].plot(Ts, Ms, label = "analitično")
axs[0].legend()


axs[1].plot(1/betas, susceps, marker = "+", linestyle='None')
axs[2].plot(1/betas, sptopls, marker = "+", linestyle='None')


axs[0].set_ylabel("Magnetizacija")
axs[0].set_xlabel("T")

axs[1].set_ylabel("Susceptibilnost")
axs[1].set_xlabel("T")

axs[2].set_ylabel("Specifična toplota")
axs[2].set_xlabel("T")

plt.tight_layout()

plt.show()



"""#Susceptibilnost

betas = np.linspace(1/5, 1/0.5, 500)

N = 50
iters = 10000
J = 1
h = 0

magss = np.zeros_like(betas)
susceps = np.zeros_like(betas)
sptopls = np.zeros_like(betas)

startstate = None

for i in trange(0, len(betas)):
    beta = betas[i]
    T = 1/beta
    states, energies = model_2d.ising(N,iters,J,h,T,startstate)

    #Try if this works or not!
    startstate = states[:,:,-1]
    #print(startstate)

    mags = model_2d.Ising_M(states[:,:,-1])
    sus = model_2d.Ising_sus(states[:,:,-1], beta)
    sph = model_2d.Ising_heat(states[:,:,-1], beta, J)

    magss[i] = mags
    susceps[i] = sus
    sptopls[i] = sph

fig, axs = plt.subplots(1,3)


axs[0].scatter(1/betas, np.abs(magss), s = 10, marker = "+")
axs[1].scatter(1/betas, susceps, s = 10, marker = "+")
axs[2].scatter(1/betas, sptopls, s = 10, marker = "+", label = "$h = 0$")


h = 0.1

magss = np.zeros_like(betas)
susceps = np.zeros_like(betas)
sptopls = np.zeros_like(betas)

startstate = None

for i in trange(0, len(betas)):
    beta = betas[i]
    T = 1/beta
    states, energies = model_2d.ising(N,iters,J,h,T,startstate)

    #Try if this works or not!
    startstate = states[:,:,-1]
    #print(startstate)

    mags = model_2d.Ising_M(states[:,:,-1])
    sus = model_2d.Ising_sus(states[:,:,-1], beta)
    sph = model_2d.Ising_heat(states[:,:,-1], beta, J)

    magss[i] = mags
    susceps[i] = sus
    sptopls[i] = sph

axs[0].scatter(1/betas, np.abs(magss), s = 10, marker = "+")
axs[1].scatter(1/betas, susceps, s = 10, marker = "+")
axs[2].scatter(1/betas, sptopls, s = 10, marker = "+", label = "$h = 0.1$")

h = 1

magss = np.zeros_like(betas)
susceps = np.zeros_like(betas)
sptopls = np.zeros_like(betas)

startstate = None

for i in trange(0, len(betas)):
    beta = betas[i]
    T = 1/beta
    states, energies = model_2d.ising(N,iters,J,h,T,startstate)

    #Try if this works or not!
    startstate = states[:,:,-1]
    #print(startstate)

    mags = model_2d.Ising_M(states[:,:,-1])
    sus = model_2d.Ising_sus(states[:,:,-1], beta)
    sph = model_2d.Ising_heat(states[:,:,-1], beta, J)

    magss[i] = mags
    susceps[i] = sus
    sptopls[i] = sph



axs[0].scatter(1/betas, np.abs(magss), s = 10, marker = "+")
axs[1].scatter(1/betas, susceps, s = 10, marker = "+")
axs[2].scatter(1/betas, sptopls, s = 10, marker = "+", label = "$h = 1$")

axs[2].legend()

plt.show()"""


