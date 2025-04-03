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
from numba import jit, njit
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))


import deterministic, maxwell
N = 20
M = 1
lamb = 1
tau = 1
TL=2
TR=1
tmax = 2000

"""print("kalkulieren")
ts, ys = deterministic.run(N,M,lamb,tau,TL,TR,tmax)

print("wiederkalkunieren schön wieder")
q = ys[0:-2:2, :]
p = ys[1:-2:2, :]
J = np.zeros_like(q)
for i in range(len(J)):
    try:
        J[i,:] = -0.5 * (q[i+1,:]-q[i-1,:]) * p[i,:]
    except IndexError:
        pass

J = J[1:-1]

J = np.trapz(J[:,len(ts)//10:],ts[len(ts)//10:])/tmax

plt.plot([i for i in range(len(J))], J, linestyle = "dashed", marker='o')
print(np.average(J))
plt.show()"""


"""tmaxs = 10**np.array([1,2,3,4])
for tmax in tmaxs:
    print("kalkulieren")
    ts, ys = deterministic.run(N,M,lamb,tau,TL,TR,tmax)

    print("wiederkalkunieren schön wieder")
    q = ys[0:-2:2, :]
    p = ys[1:-2:2, :]
    J = np.zeros_like(q)
    for i in range(len(J)):
        try:
            J[i,:] = -0.5 * (q[i+1,:]-q[i-1,:]) * p[i,:]
        except IndexError:
            pass

    J = J[M:-M]

    J = np.trapz(J[:,len(ts)//10:],ts[len(ts)//10:])/tmax

    plt.plot([i for i in range(len(J))], J, linestyle = "dashed", marker='o', label = "$t_{max} = $" + f"${tmax}$")
    print(np.average(J))
plt.legend()
plt.xlabel("Zaporedno število delca")
plt.ylabel("$\\langle \\text{J}_{i,i+1}\\rangle$")
plt.title(f"$N={N}$, $M = {M}$, $\\lambda = {lamb}$, $T_L = {TL}$, $T_R = {TR}$")
plt.show()"""

"""from tqdm import trange

taus = [0.01, 0.1, 1, 5, 10]
for i in trange(0, len(taus)):
    tau = taus[i]


    ts, ys = deterministic.run(N,M,lamb,tau,TL,TR,tmax)

    q = ys[0:-2:2, :]
    p = ys[1:-2:2, :]
    J = np.zeros_like(q)
    for i in range(len(J)):
        try:
            J[i,:] = -0.5 * (q[i+1,:]-q[i-1,:]) * p[i,:]
        except IndexError:
            pass

    J = J[M:-M]

    J = np.trapz(J[:,len(ts)//10:],ts[len(ts)//10:])/tmax

    plt.plot([i for i in range(len(J))], J, linestyle = "dashed", marker='o', label = "$\\tau = $" + f"${tau}$")
    print(np.average(J))

plt.legend(loc="upper right")
plt.xlabel("Zaporedno število delca")
plt.ylabel("$\\langle \\text{J}_{i,i+1}\\rangle$")
plt.title(f"$N={N}$, $M = {M}$, $\\lambda = {lamb}$, $T_L = {TL}$, $T_R = {TR}$")
plt.show()"""

"""from tqdm import trange

taus = np.linspace(0,10,30)
vals = []
for i in trange(0, len(taus)):
    tau = taus[i]


    ts, ys = deterministic.run(N,M,lamb,tau,TL,TR,tmax)

    q = ys[0:-2:2, :]
    p = ys[1:-2:2, :]
    J = np.zeros_like(q)
    for i in range(len(J)):
        try:
            J[i,:] = -0.5 * (q[i+1,:]-q[i-1,:]) * p[i,:]
        except IndexError:
            pass

    J = J[M:-M]

    J = np.trapz(J[:,len(ts)//10:],ts[len(ts)//10:])/tmax

    #plt.plot([i for i in range(len(J))], J, linestyle = "dashed", marker='o', label = "$\\tau = $" + f"${tau}$")
    vals.append(np.average(J))

plt.plot(taus, vals)

plt.xlabel("$\\tau$")
plt.ylabel("$\\langle J \\rangle$")
plt.title(f"$N={N}$, $M = {M}$, $\\lambda = {lamb}$, $T_L = {TL}$, $T_R = {TR}$")
plt.show()"""



"""from tqdm import trange

Ns = np.arange(5,50,(50-5)//10)
vals = []
for i in trange(0, len(Ns)):
    N = Ns[i]


    ts, ys = deterministic.run(N,M,lamb,tau,TL,TR,tmax)

    q = ys[0:-2:2, :]
    p = ys[1:-2:2, :]
    J = np.zeros_like(q)
    for i in range(len(J)):
        try:
            J[i,:] = -0.5 * (q[i+1,:]-q[i-1,:]) * p[i,:]
        except IndexError:
            pass

    J = J[M:-M]

    J = np.trapz(J[:,len(ts)//10:],ts[len(ts)//10:])/tmax

    #plt.plot([i for i in range(len(J))], J, linestyle = "dashed", marker='o', label = "$\\tau = $" + f"${tau}$")
    vals.append(np.average(J))

from scipy.optimize import curve_fit

def f(x,k):
    return k * (TR-TL)/x

popt, pcov = curve_fit(f, Ns, vals, p0 = 1)
kopt = popt[0]
NNs = np.linspace(Ns[0], Ns[-1], 100)
plt.plot(NNs, f(NNs,kopt), linestyle = "dashed", color = "red")

plt.plot(Ns, vals)

plt.xlabel("$N$")
plt.ylabel("$\\langle J \\rangle$")
plt.title(f"$N={N}$, $M = {M}$, $\\lambda = {lamb}$, $T_L = {TL}$, $T_R = {TR}$")
plt.show()"""


"""from tqdm import trange

lambs = np.round(10**np.linspace(np.log10(0.001), 0, 6), decimals=3)
lambs = np.append(lambs, 2.0)
vals = []
for i in trange(0, len(lambs)):
    lamb = lambs[i]


    ts, ys = deterministic.run(N,M,lamb,tau,TL,TR,tmax)

    q = ys[0:-2:2, :]
    p = ys[1:-2:2, :]
    J = np.zeros_like(q)
    for i in range(len(J)):
        try:
            J[i,:] = -0.5 * (q[i+1,:]-q[i-1,:]) * p[i,:]
        except IndexError:
            pass

    J = J[M:-M]

    J = np.trapz(J[:,len(ts)//10:],ts[len(ts)//10:])/tmax

    plt.plot([i for i in range(len(J))], J, linestyle = "dashed", marker='o', label = "$\\lambda = $" + f"${lamb}$")
    #vals.append(np.average(J))

plt.legend(loc="upper right")
plt.xlabel("Zaporedno število delca")
plt.ylabel("$\\langle J \\rangle$")
plt.title(f"$N={N}$, $M = {M}$, $T_L = {TL}$, $T_R = {TR}$"+", $t_{max}=$"+f"${tmax}$")
plt.show()"""


from tqdm import trange

tmax = 2000

lambs = np.round(np.linspace(0, 1, 6), decimals=3)
Ns = np.arange(5,50,(50-5)//10)

import seaborn as sns
pallete = sns.color_palette(n_colors=len(lambs))
for j in trange(0, len(lambs)):
    lamb = lambs[j]
    vals = []
    for i in trange(0, len(Ns), leave=False):
        N = Ns[i]


        ts, ys = maxwell.run(N,M,lamb,tau,TL,TR,tmax)

        q = ys[0::2, :]
        p = ys[1::2, :]
        J = np.zeros_like(q)
        for i in range(len(J)):
            try:
                J[i,:] = -0.5 * (q[i+1,:]-q[i-1,:]) * p[i,:]
            except IndexError:
                pass

        J = J[M:-M]

        J = np.trapezoid(J[:,len(ts)//10:],ts[len(ts)//10:])/tmax

        #plt.plot([i for i in range(len(J))], J, linestyle = "dashed", marker='o', label = "$\\tau = $" + f"${tau}$")
        vals.append(np.average(J)*N)

    #from scipy.optimize import curve_fit
    #
    #def f(x,k):
    #    return k * (TR-TL)/x
    #
    #popt, pcov = curve_fit(f, Ns, vals, p0 = 1)
    #kopt = popt[0]
    #NNs = np.linspace(Ns[0], Ns[-1], 100)
    #
    #plt.plot(NNs, f(NNs,kopt), linestyle = "dashed", c = pallete[j])


    plt.plot(Ns, vals, label = "$\\lambda = $" + f"${lamb}$", c = pallete[j], linestyle = "dashed", marker='o')


plt.legend(loc="upper right")
plt.xlabel("$N$")
plt.ylabel("$\\langle J \\rangle \\cdot \\frac{N}{\\Delta T}$")

#plt.xscale("log")
#plt.yscale("log")
plt.title(f"$M = {M}$, $T_L = {TL}$, $T_R = {TR}$")
plt.show()


"""from tqdm import trange

tmax = 2000

lambs = np.round(np.linspace(0, 1, 6), decimals=3)
Ns = np.arange(5,50,(50-5)//10)

import seaborn as sns
pallete = sns.color_palette(n_colors=len(lambs))
for j in trange(0, len(lambs)):
    lamb = lambs[j]
    vals = []
    for i in trange(0, len(Ns), leave=False):
        N = Ns[i]


        ts, ys = maxwell.run(N,M,lamb,tau,TL,TR,tmax)

        q = ys[0::2, :]
        p = ys[1::2, :]
        J = np.zeros_like(q)
        for i in range(len(J)):
            try:
                J[i,:] = -0.5 * (q[i+1,:]-q[i-1,:]) * p[i,:]
            except IndexError:
                pass

        J = J[M:-M]

        J = np.trapz(J[:,len(ts)//10:],ts[len(ts)//10:])/tmax

        #plt.plot([i for i in range(len(J))], J, linestyle = "dashed", marker='o', label = "$\\tau = $" + f"${tau}$")
        vals.append(np.average(J))

    from scipy.optimize import curve_fit
    
    def f(x,k):
        return k * (TR-TL)/x
    
    popt, pcov = curve_fit(f, Ns, vals, p0 = 1)
    kopt = popt[0]
    NNs = np.linspace(Ns[0], Ns[-1], 100)
    
    plt.plot(NNs, f(NNs,kopt), linestyle = "dashed", c = pallete[j])

    #plt.plot(Ns, vals, label = "$\\lambda = $" + f"${lamb}$", c = pallete[j], linestyle = "dashed", marker='o')
    plt.scatter(Ns, vals, label = "$\\lambda = $" + f"${lamb}$", color = pallete[j], s = 7)


plt.legend()
plt.xlabel("$N$")
plt.ylabel("$\\langle J \\rangle \\cdot \\frac{N}{\\Delta T}$")

plt.xscale("log")
plt.yscale("log")
plt.title(f"$M = {M}$, $T_L = {TL}$, $T_R = {TR}$")
plt.show()"""



"""from tqdm import trange

tmax = 2000
lamb = 1

taus = np.round(np.linspace(0.1, 2, 20), decimals=3)
Ns = np.arange(5,50,(50-5)//10)

import seaborn as sns
pallete = sns.color_palette(n_colors=len(taus))
ks = []
for j in trange(0, len(taus)):
    tau = taus[j]
    vals = []
    for i in trange(0, len(Ns), leave=False):
        N = Ns[i]


        ts, ys = maxwell.run(N,M,lamb,tau,TL,TR,tmax)

        q = ys[0::2, :]
        p = ys[1::2, :]
        J = np.zeros_like(q)
        for i in range(len(J)):
            try:
                J[i,:] = -0.5 * (q[i+1,:]-q[i-1,:]) * p[i,:]
            except IndexError:
                pass

        J = J[M:-M]

        J = np.trapz(J[:,len(ts)//10:],ts[len(ts)//10:])/tmax

        #plt.plot([i for i in range(len(J))], J, linestyle = "dashed", marker='o', label = "$\\tau = $" + f"${tau}$")
        vals.append(np.average(J)*N)

    ks.append(np.average(vals))

plt.plot(taus, ks, marker='o')

plt.xlabel("$\\tau$")
plt.ylabel("$\\kappa$")

plt.title(f"$M = {M}$, $T_L = {TL}$, $T_R = {TR}$, $\\lambda = {lamb}$")
plt.show()"""