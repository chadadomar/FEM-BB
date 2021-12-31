# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 15:46:56 2021

@author: Omar.CHADAD
"""

from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy.special import binom



def integral(a,b,n,f):
    l=[]
    for i in range(n+1):
        def integrand(x):
            y=(b-x)/(b-a)
            return binom(n,i)*(y**i)*((1-y)**(n-i))*f(x)
        l.append(quad(integrand,a,b)[0])
    return l 

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

