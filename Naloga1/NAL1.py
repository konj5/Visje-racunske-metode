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


import solver

def V(x,lamb):
    return 1/2 * x**2 + lamb * x**4

def eigenstate(N, x):
    return 1/(np.pi**(1/4) * np.sqrt(2**N * scipy.special.factorial(N))) * scipy.special.hermite(N)(x) * np.exp(-x**2/2)

def coherent_state(d,x):
    return eigenstate(0,x-d)


L = 15
dx = 0.1
dt = 0.005

tmax = 2
nmax = int(tmax/dt)
duration = 10 #seconds

N = 0
lamb = 0

xs = np.arange(-L,L, dx)
Vmn = np.zeros((len(xs), nmax))
V0 = V(xs,lamb)
for i in range(nmax):
    Vmn[:,i] = V0

spacewise2 = [1/2, -1, 1/2]
spacewise4 = [-1/12, 4/3, -5/3, 4/3, -1/12]

spacewise = spacewise4


"""#solution = solver.finite_differencer(eigenstate(N,xs), Vmn, dt, dx, nmax, spacewise=spacewise)
#solution = solver.finite_propagatorer(eigenstate(N,xs), Vmn, dt, dx, nmax, spacewise=spacewise)
#solution = solver.implicinator(eigenstate(N,xs), Vmn, dt, dx, nmax, spacewise=spacewise)

#solution = solver.finite_propagatorer(coherent_state(1,xs), Vmn, dt, dx, nmax, spacewise=spacewise)
solution = solver.implicinator(coherent_state(1,xs), Vmn, dt, dx, nmax, spacewise=spacewise)


solution = scipy.interpolate.interp1d([i/(nmax-1) * (30*duration) for i in range(nmax)], solution, axis=1)([i for i in range(30 * duration)])


def animate(i):
    plt.cla()
    #plt.title(f"Metoda: Končne diference, $h={dx}$, $\\tau={dt}$, $t = {i/(30*10)*nmax * dt:0.3f}$")
    #plt.title(f"Metoda: Končni propagator, $h={dx}$, $\\tau={dt}$, $t = {i/(30*10)*nmax * dt:0.3f}$")
    plt.title(f"Metoda: Implicitna, $h={dx}$, $\\tau={dt}$, $t = {i/(30*duration)*nmax * dt:0.3f}$")

    #i = int(i * nmax/(30*duration))
    #print(f"animating: {int(i * nmax/(30*duration))}/{nmax}")

    print(f"animating: {i}/{30*duration}")
    

    #plt.plot(xs, np.real(solution[:,i])**2, c = "red", label = "Re$\\psi^2$")
    #plt.plot(xs, np.imag(solution[:,i])**2, c = "blue", label = "Im$\\psi^2$")
    plt.plot(xs, np.abs(solution[:,i])**2, c = "black", label = "$|\\psi|^2$")

    #plt.xlabel("Rdeča - Re$\\psi^2$, Modra - Im$\\psi^2$,  Črna - $|\\psi|^2$")
    


fig = plt.figure()

ani = FuncAnimation(fig, animate, frames=30*duration, )
ani.save('animacija_adsadasdasda.mp4',  
          writer = 'ffmpeg', fps = 30) 
print("done")
plt.show()"""


"""ts = np.linspace(0,tmax,nmax)
solution = solver.finite_propagatorer(eigenstate(N,xs), Vmn, dt, dx, nmax, spacewise=spacewise2)
plt.plot(ts,np.abs(1-np.trapz(np.abs(solution)**2,xs, axis=0)), label="Končni propagator 2.reda")

print(1)

solution = solver.finite_propagatorer(eigenstate(N,xs), Vmn, dt, dx, nmax, spacewise=spacewise4)
plt.plot(ts,np.abs(1-np.trapz(np.abs(solution)**2,xs, axis=0)), label="Končni propagator 4.reda")

print(2)

solution = solver.implicinator(eigenstate(N,xs), Vmn, dt, dx, nmax, spacewise=spacewise2)
plt.plot(ts,np.abs(1-np.trapz(np.abs(solution)**2,xs, axis=0)), label="implicitna 2.reda")

print(3)

solution = solver.implicinator(eigenstate(N,xs), Vmn, dt, dx, nmax, spacewise=spacewise4)
plt.plot(ts,np.abs(1-np.trapz(np.abs(solution)**2,xs, axis=0)), label="implicitna 4.reda")

print(4)

plt.ylabel("$|1-|\\psi|^2|$")
plt.xlabel("t")
plt.yscale("log")
plt.legend()
plt.show()"""

"""tprop = []
timp = []

import time
for h in 10**np.linspace(np.log10(0.1),np.log10(0.05),10):
    dt = h**2/2
    nmax = int(tmax/dt)
    print(dt)
    stime = time.time()
    solution = solver.finite_propagatorer(eigenstate(N,xs), Vmn, dt, dx, nmax, spacewise=spacewise2)
    tprop.append(time.time()-stime)

    stime = time.time()
    solution = solver.implicinator(eigenstate(N,xs), Vmn, dt, dx, nmax, spacewise=spacewise2)
    timp.append(time.time()-stime)


plt.plot(10**np.linspace(0,np.log10(0.05),10), tprop, label = "Končni propagator")
plt.plot(10**np.linspace(0,np.log10(0.05),10), timp, label = "Implcitna")

plt.xlabel("h ($\\tau = h^2/2$)")
plt.ylabel("Čas računanja[s]")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.show()

plt.plot(10**np.linspace(0,np.log10(0.05),10), tprop, label = "Končni propagator")
plt.plot(10**np.linspace(0,np.log10(0.05),10), timp, label = "Implcitna")

plt.xlabel("h ($\\tau = h^2/2$)")
plt.ylabel("Čas računanja[s]")
plt.yscale("log")
#plt.xscale("log")
plt.legend()
plt.show()

plt.plot(10**np.linspace(0,np.log10(0.05),10), tprop, label = "Končni propagator")
plt.plot(10**np.linspace(0,np.log10(0.05),10), timp, label = "Implcitna")

plt.xlabel("h ($\\tau = h^2/2$)")
plt.ylabel("Čas računanja[s]")
#plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.show()

plt.plot(10**np.linspace(0,np.log10(0.05),10), tprop, label = "Končni propagator")
plt.plot(10**np.linspace(0,np.log10(0.05),10), timp, label = "Implcitna")

plt.xlabel("h ($\\tau = h^2/2$)")
plt.ylabel("Čas računanja[s]")
#plt.yscale("log")
#plt.xscale("log")
plt.legend()
plt.show()"""


ts = np.linspace(0,tmax,nmax)

Ls = np.linspace(1,16,10)

solution1, solution2, solution3, solution4 = [],[],[],[]

for i in range(len(Ls)):
    print(f"{i}/{len(Ls)}")

    N = 0
    xs = np.arange(-Ls[i],Ls[i], dx)
    Vmn = np.zeros((len(xs), nmax))
    V0 = V(xs,lamb)
    for i in range(nmax):
        Vmn[:,i] = V0

    solution = solver.finite_propagatorer(eigenstate(N,xs), Vmn, dt, dx, nmax, spacewise=spacewise2)[:,-1]
    solution1.append(np.abs(1-np.trapz(np.abs(solution)**2,xs)))
    
    print(1)

    solution = solver.finite_propagatorer(eigenstate(N,xs), Vmn, dt, dx, nmax, spacewise=spacewise4)[:,-1]
    solution2.append(np.abs(1-np.trapz(np.abs(solution)**2,xs)))
    

    print(2)

    solution = solver.implicinator(eigenstate(N,xs), Vmn, dt, dx, nmax, spacewise=spacewise2)[:,-1]
    solution3.append(np.abs(1-np.trapz(np.abs(solution)**2,xs)))
    

    print(3)

    solution = solver.implicinator(eigenstate(N,xs), Vmn, dt, dx, nmax, spacewise=spacewise4)[:,-1]
    solution4.append(np.abs(1-np.trapz(np.abs(solution)**2,xs)))
    

    print(4)

plt.plot(Ls,solution1, label="Končni propagator 2.reda")
plt.plot(Ls,solution2, label="Končni propagator 4.reda")
plt.plot(Ls,solution3, label="implicitna 2.reda")
plt.plot(Ls,solution4, label="implicitna 4.reda")


plt.ylabel("$|1-|\\psi|^2|$")
plt.xlabel("L")
plt.yscale("log")
plt.legend()
plt.show()









