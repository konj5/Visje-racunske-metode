import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import scipy.integrate
import scipy.interpolate
import scipy.sparse
import scipy.special
from tqdm import tqdm
import re, sys
from matplotlib.animation import FuncAnimation

from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


h = 0.05
L = 8
lamb = 0

dt = h**2/4

tmax = 1

Nx = int(L/h)+1
Nt = int(tmax/dt)

#print(f"tmax = {Nt * dt}")


spacewise4 = [-1/12, 4/3, -5/3, 4/3, -1/12]
Hx = csc_matrix((Nx*Nx, Nx*Nx), dtype=complex)
Hy = csc_matrix((Nx*Nx, Nx*Nx), dtype=complex)
HV = csc_matrix((Nx*Nx, Nx*Nx), dtype=complex)

xblock =csc_matrix((Nx,Nx), dtype=complex)
for k in range(len(spacewise4)):
    xblock += spacewise4[k] * sparse.eye(Nx,Nx, k=-2+k)

    Hy += spacewise4[k] * sparse.eye(Nx*Nx,Nx*Nx, k = Nx * (-2+k))

Hx = scipy.sparse.block_diag([xblock for _ in range(Nx)])

#Hx = Hx * -1/2 * 1/h**2
#Hy = Hy * -1/2 * 1/h**2

Hx = Hx * -1/2
Hy = Hy * -1/2

V = np.fromfunction(lambda i, j: 1/2 * (i**2 + j**2) + lamb *(i*j)**2, (Nx,Nx))

diagonala = np.zeros(Nx*Nx)
for i in range(1,Nx+1):
    diagonala[(i-1)*Nx:i*Nx] = V[:,i-1]*h**4

HV += scipy.sparse.diags(diagonala)

H = Hx + Hy + HV

A = csc_matrix(sparse.eye(Nx*Nx) + 1j /2 * H)
B = csc_matrix(sparse.eye(Nx*Nx) - 1j /2 * H)

xs = np.linspace(-L, L, Nx)
xs, ys = np.meshgrid(xs, xs)

psis = []

def psi0(x, y, x0, y0):
    return 1/np.pi**(1/2) * np.exp(-(x-x0)**2/2-(y-y0)**2/2)

psi = psi0(xs, ys, 2, 0)
psi[0,:] = psi[-1,:] = psi[:,0] = psi[:,-1] = 0
psis.append(np.copy(psi))


for i in range(1,Nt):
    print(f"\rSolving {i}/{Nt}", end="      ")
    psi_vect = psi.reshape((Nx*Nx))

    
    b = B.dot(psi_vect)
    psi_vect = spsolve(A,b)
    psi = psi_vect.reshape((Nx,Nx)) 
    psis.append(np.copy(psi))


mod_psis = [] 
for wavefunc in psis:
    re = np.real(wavefunc) 
    im = np.imag(wavefunc) 
    mod = np.sqrt(re**2 + im**2) 
    mod_psis.append(mod) 

fig = plt.figure() 
ax = fig.add_subplot(111, xlim=(-L,L), ylim=(-L,L)) 

img = ax.imshow(mod_psis[0], extent=[-L,L,-L,L], cmap=plt.get_cmap("hot"), vmin=0, vmax=np.max(mod_psis), zorder=1) 

def animate(i):
    
    """
    Animation function. Paints each frame. Function for Matplotlib's 
    FuncAnimation.
    """
    
    img.set_data(mod_psis[i])
    img.set_zorder(1)
    print(f"\rAnimating {i}/{Nt}", end="      ")
    
    return img


anim = FuncAnimation(fig, animate, interval=1, frames =np.arange(0,Nt,2), repeat=False, blit=0)


anim.save('./animationsName.mp4', writer="ffmpeg", fps=60)

print("done")




