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

L = 15
dx = 0.1
dt = 0.001

tmax = 30
nmax = int(tmax/dt)
duration = 30 #seconds

N = 0
lamb = 2

xs = np.arange(-L,L, dx)
Vmn = np.zeros((len(xs), nmax))
V0 = V(xs,lamb)
for i in range(nmax):
    Vmn[:,i] = V0

spacewise2 = [1/2, -1, 1/2]
spacewise4 = [-1/12, 4/3, -5/3, 4/3, -1/12]

spacewise = spacewise4


#solution = solver.finite_differencer(eigenstate(N,xs), Vmn, dt, dx, nmax, spacewise=spacewise)
#solution = solver.finite_propagatorer(eigenstate(N,xs), Vmn, dt, dx, nmax, spacewise=spacewise)
solution = solver.implicinator(eigenstate(N,xs), Vmn, dt, dx, nmax, spacewise=spacewise)


solution = scipy.interpolate.interp1d([i/(nmax-1) * (30*duration) for i in range(nmax)], solution, axis=1)([i for i in range(30 * duration)])


def animate(i):
    plt.cla()
    #plt.title(f"Metoda: Končne diference, $h={dx}$, $\\tau={dt}$, $t = {i/(30*10)*nmax * dt:0.3f}$")
    #plt.title(f"Metoda: Končni propagator, $h={dx}$, $\\tau={dt}$, $t = {i/(30*10)*nmax * dt:0.3f}$")
    plt.title(f"Metoda: Implicitna, $h={dx}$, $\\tau={dt}$, $t = {i/(30*duration)*nmax * dt:0.3f}$")

    #i = int(i * nmax/(30*10))
    #print(f"{i}/{nmax}")
    print(f"{i}/{30*duration}")

    plt.plot(xs, np.real(solution[:,i])**2, c = "red", label = "Re$\\psi^2$")
    plt.plot(xs, np.imag(solution[:,i])**2, c = "blue", label = "Im$\\psi^2$")
    plt.plot(xs, np.abs(solution[:,i])**2, c = "black", label = "$|\\psi|^2$")

    plt.xlabel("Rdeča - Re$\\psi^2$, Modra - Im$\\psi^2$,  Črna - $|\\psi|^2$")


fig = plt.figure()

ani = FuncAnimation(fig, animate, frames=30*duration, )
ani.save('animacija.mp4',  
          writer = 'ffmpeg', fps = 30) 
print("done")
#plt.show()


"""ts = np.linspace(0,tmax,nmax)
solution = solver.finite_propagatorer(eigenstate(N,xs), Vmn, dt, dx, nmax, spacewise=spacewise2)
plt.plot(ts,np.abs(1-np.trapz(np.abs(solution)**2,xs, axis=0)), label="Končni propagator 2.reda")


solution = solver.finite_propagatorer(eigenstate(N,xs), Vmn, dt, dx, nmax, spacewise=spacewise4)
plt.plot(ts,np.abs(1-np.trapz(np.abs(solution)**2,xs, axis=0)), label="Končni propagator 4.reda")

solution = solver.implicinator(eigenstate(N,xs), Vmn, dt, dx, nmax, spacewise=spacewise2)
plt.plot(ts,np.abs(1-np.trapz(np.abs(solution)**2,xs, axis=0)), label="implicitna 2.reda")

solution = solver.implicinator(eigenstate(N,xs), Vmn, dt, dx, nmax, spacewise=spacewise4)
plt.plot(ts,np.abs(1-np.trapz(np.abs(solution)**2,xs, axis=0)), label="implicitna 4.reda")

plt.ylabel("$|1-|\\psi|^2|$")
plt.xlabel("t")
plt.yscale("log")
plt.legend()
plt.show()"""







