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


BCH_coefs = {
    "a":1,
    "b":1,

    "ab":1/2,

    "aab":1/12,
    "bba":1/12,
}


k = 1
p = 1

#define coef vector 
coefs = ["1_a1_[A]","1_b1_[B]"]

#do 1 step (to order p)
coefs = ["1_a1_[A] + 1_b1_[B] + 2_a1b1_[AB]"]


#coefs format: c1_coefs_operator, c1 --> 1/c1 v vsoti


def readEntry(entry):
    parts = entry.split(" + ")

    for part in parts:
        c, coefs, operator = part.split("_")


def mergeOperators(o1, o2, p):
    if o1 == 0 or o2 == 0:
        return 0

    o1 = o1[1:-1]
    o2 = o2[1:-1]

    o3 = o1 + o2

    if o3[-1] == o3[-2]:
        return 0
    
    elif len(o3)-2 > p-2:
        return 0
    
    else:
        return "[" + o3 + "]"



def mergeLast(stuff, p):
    s1 = stuff[-2]
    s2 = stuff[-1]
    stuff = stuff[:len(stuff)-1]

    c1, coefs1, operator1 = s1.split("_")

    parts = s2.split(" + ")

    newstring = ""
    for j in range(len(parts)):
        part = parts[j]
        c2, coefs2, operator2 = part.split("_")

        for i in range(p+1):
            if i == 0 and j == 0:
                newstring +=  s1 + " + " + s2

            elif i == 1:
                newOp = mergeOperators(operator1, operator2, p)
                if newOp != 0:
                    newstring += " + " + str(int(c1)*int(c2) * 2) + "_" + coefs1 + coefs2 + "_" + newOp

            elif i == 2:
                newOp1 = mergeOperators(operator1, mergeOperators(operator1, operator2, p), p)
                newOp2 = mergeOperators(operator2, mergeOperators(operator2, operator1, p), p)

                if newOp1 != 0:
                    newstring += " + " + str(int(c1)*int(c2) * 12) + "_" + coefs1 + coefs2 + "_" + newOp1

                if newOp2 != 0:
                    newstring += " + " + str(int(c1)*int(c2) * 12) + "_" + coefs1 + coefs2 + "_" + newOp2

            elif i == 3:
                newOp =   mergeOperators(operator1,mergeOperators(operator2,mergeOperators(operator2, operator1, p),p),p)
                if newOp != 0:
                    newstring += " + " + str(int(c1)*int(c2) * 24) + "_" + coefs1 + coefs2 + "_" + newOp

            ## TU DODAJ VIŠJE REDE BCH KOEFICIENTOV


    stuff[-1] = newstring
    return stuff

def merge(stuff, p):
    while(len(stuff) > 1):
        stuff = mergeLast(stuff,p)
    return stuff[0]

def get_equation_from_merged(merged_stuff):
    parts = merged_stuff.split(" + ")

    operators = []
    equations = []
    for part in parts:
        c, coefs, operator = part.split("_")

        swapped = operator
        last = swapped[-2]
        sectolast = swapped[-3]
        swapped = operator[:-3] + last + sectolast + "]"


        if operator in operators:
            i = operators.index(operator)
            equations[i] += " + " + c + "_" + coefs

        elif swapped in operators:
            i = operators.index(swapped)
            equations[i] += " + -" + c + "_" + coefs


        else:
            operators.append(operator)
            equations.append(c + "_" + coefs)

    return operators, equations

def toHumanReadable(operators, equations):
    for i in range(len(operators)):
        operator = operators[i]
        equation = equations[i]
        print(operator, end=":   ")

        parts = equation.split(" + ")

        for j in range(len(parts)):
            part = parts[j]
            num, vars = part.split("_")

            if num == "1":
                print(f"{vars}", end="")
            elif num[0] == "-":
                print(f"-1/{num[1:]} * {vars}", end="")
            else:
                print(f"1/{num} * {vars}", end="")


            if j != len(parts)-1:
                print("",end=" + ")

        if operator in ["[A]", "[B]"]:
            print(" = 1")
        else:
            print(" = 0")
    

def startstuff(n):
    ret = []
    for i in range(n):
        if i % 2 == 0:
            ret.append(f"1_a{i//2+1}_[A]")
        if i % 2 == 1:
            ret.append(f"1_b{i//2+1}_[B]")
    return ret




def get_equations(n,p):
    stuff = startstuff(n)
    ops, eqs = get_equation_from_merged(merge(stuff, p))
    toHumanReadable(ops, eqs)

        

#print(mergeLast(["1_a1_[A]", "1_b1_[B]", "1_a2_[A]"], p = 1))
#print(mergeLast(mergeLast(["1_a1_[A]", "1_b1_[B]", "1_a2_[A]"], p = 1), p = 1))


#ops, eqs = get_equation_from_merged(merge(["1_a1_[A]", "1_b1_[B]", "1_a2_[A]"], p = 2))

#print(ops)
#print(eqs)


#toHumanReadable(ops, eqs)

get_equations(3,2)

                











