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
import copy
from numba import jit, njit, prange
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))




class prefactor():

    def __init__(self, num, coefs, zorder):
        self.coefs (str) = sorted(coefs)
        self.num (complex) = num
        self.zorder (int) = zorder

    def __eq__(self, other):
        if self.zorder == other.zorder:
            if self.coefs == other.coefs:
                return True
        return False



class opstring():

    def __init__(self, string, num, coefs, zorder):
        self.opst (str) = string
        self.prefact (prefactor) = prefactor(num, coefs, zorder)

    def __add__(self, other):
        if self == other:
            return prefactor(self.num + other.num, self.coefs, self.zorder)
        
        raise Exception("adding opstrings with different prefactors")
    
    def __mul__(self, other):
        return opstring(self.opst+other.opst, self.num * other.num, sorted(self.coefs+other.coefs), self.zorder+other.zorder)

    def __eq__(self, other):
        return self.prefact == other.prefact and self.opst == other.opst

class opsum():

    def __init__(self):
        self.ops = []

    def __add__(self, other):
        done = False
        toadd = []
        for j, opstr2 in enumerate(other.ops):
            for i, opstr in enumerate(self.ops):
                if opstr == opstr2:
                    self.ops[i] = opstr + opstr2
                    done = True
                    break
               
            if not done:
                toadd.append(other)
    
        for x in toadd:
            self.ops.append(x)

    def __sub__(self, other):
        temp = copy.deepcopy(other)
       
        for i in range(temp.ops):
            temp.ops.prefact.num *= -1

        return self + temp

    def __mul__(self, other):
        new = opsum()

        for i,x1 in enumerate(self.ops):
            for j,x2 in enumerate(other.ops):
                temp = opsum()
                temp.ops = x1 * x2
                new = new + temp

        return new
    
    
    def scalarmul(self, scalar):
        temp = copy.deepcopy(self)
       
        for i in range(temp.ops):
            temp.ops.prefact.num *= scalar

        return temp
    


        

class nested_commutator():
    
    def __init__(self, string):
        #example string = "[A,[A,B]]"
        



        
                   
        
        