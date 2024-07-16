# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 15:46:56 2021

@author: Omar.CHADAD
"""

from scipy.integrate import quad
from scipy.integrate import dblquad , tplquad
from scipy.special import binom
from Bmoment import *
import numpy as np



def integral(a,b,n,f):
    l=[]
    for i in range(n+1):
        def integrand(x):
            y=(b-x)/(b-a)
            return binom(n,i)*(y**i)*((1-y)**(n-i))*f(x)
        l.append(quad(integrand,a,b)[0])
    return l 


## 2D integral over the triangle <(1,0),(0,1),(0,0)>
def Chi(x,y):
    if x+y<=1 and x+y>=0:
        return 1
    else:
        return 0

def integral2D(f,n):
    l=[]
    for i in range(n,-1,-1):
        for j in range(n-i,-1,-1):
            def integrand2D(y,x):
                return Chi(x,y)*binom(n,i)*binom(n-i,j)*(x**i)*(y**j)*((1-x-y)**(n-i-j))*f(x,y)
            l.append(dblquad(integrand2D,0,1,lambda x:0,lambda x:1)[0])
    return l 


## 3D integrale over the tetrahedron <(1,0,0),(0,1,0),(0,0,1),(0,0,0)>
def Chi3(x,y,z):
    if x+y+z<=1 and x+y+z>=0:
        return 1
    else:
        return 0

def integral3D(f,n):
    l=[]
    for i in range(n,-1,-1):
        for j in range(n-i,-1,-1):
            for k in range(n-i-j,-1,-1):
                def integrand3D(y,x,z):
                    return Chi3(x,y,z)*binom(n,i)*binom(n-i,j)*binom(n-i-j,k)*(x**i)*(y**j)*(z**k)*((1-x-y-z)**(n-i-j-k))*f(x,y,z)
                l.append(tplquad(integrand3D,0,1,lambda x:0,lambda x:1,lambda x,y:0,lambda x,y:1)[0])
    return l 


""" 1d Test 
def f1d(x):
    return x*x*np.exp(x)

a=0
b=1
n=3
q=n+1

err1d= abs(integral(a,b,n,f1d)-M1D(a,b,f1d,n,q))

print("erreur 1d integration",err1d) """

""" 2d Test

def f2d(x,y):
    return x*y*np.exp(x+y)

L2d=[1,0,0,1,0,0]
err2d= abs(integral2D(f2d,n)-Moment2D(L2d,f2d,n,q))
print("erreur 2d",err2d) """


""" 3d Test 
def f3d(x,y,z):
    return x*y*z

L3d=[1,0,0,0,1,0,0,0,1,0,0,0]
err3d= abs(integral3D(f3d,n)-Moment3D(L3d,f3d,n,q))
print("erreur 3d",err3d)"""