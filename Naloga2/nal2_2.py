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
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))


from solver import mat_v_bazi2 as basic_herm, Lanczos_method as lanczos

Ns = np.int32(np.linspace(5,200,20))
lambs = [0.05,0.1,0.5,1,2]

data = np.zeros((len(Ns), len(lambs)))


"""for j in tqdm(range(len(lambs))):
    lamb = lambs[j]
    exact_e, exact_v = lanczos(np.array([int(i == 0) for i in range(300)]), 300, lamb)
    #exact_e, exact_v = basic_herm(200, lamb)
    
    for i in tqdm(range(len(Ns)), leave=False):
        N = Ns[i]
    

        
        

        

        lanczos_e, lanczos_v = lanczos(np.array([int(i == 0) for i in range(N)]), N, lamb)

        #print(np.abs(exact_e - exact_v))

        data[i,j] = np.sum(np.abs(exact_e[0:len(lanczos_e)] - lanczos_e) <= 1e-3) / N


for j in range(len(lambs)):
    lamb = lambs[j]

    plt.plot(Ns, data[:,j], label = f"$\\lambda = {lamb}$")

plt.legend()
plt.xlabel("N")
plt.ylabel("Delež točnih stanj ($\\varepsilon < 0.001$)")
plt.show()
"""
    
Ns = np.int32(np.linspace(15,200,20))
lamb = 1

data = np.zeros((Ns[0], len(Ns)))

for i in tqdm(range(len(Ns)), leave=False):
    N = Ns[i]


    lanczos_e, lanczos_v = lanczos(np.array([int(i == 0) for i in range(N)]), N, lamb)

    data[:,i] = lanczos_e[:Ns[0]]

    
for i in range(Ns[0]):
    plt.plot(Ns, data[i,:], label = f"E{i}", color = plt.get_cmap("viridis")(colors.Normalize(0,Ns[0])(i)))

plt.legend()
plt.show()




