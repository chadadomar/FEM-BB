# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 13:55:16 2021

@author: Omar.CHADAD
"""

import numpy as np

import scipy.special 

n=110 

l=dict()

for i in range(1,n+1):
    l[i]=scipy.special.roots_jacobi(i,0,0)
np.save("jaccobi rule 0",l)

l2=dict()

for i in range(1,n+1):
    l[i]=scipy.special.roots_jacobi(i,1,0)
np.save("jaccobi rule 1",l)

l3=dict()

for i in range(1,n+1):
    l[i]=scipy.special.roots_jacobi(i,2,0)
np.save("jaccobi rule 2",l)


t=np.zeros((2*n+4,2*n+4),dtype="longdouble")
for i in range(2*n+4):
    t[i][0]=1
    t[i][i]=1
for i in range(2,2*n+4):
    for j in range(1,i):
        t[i][j]=t[i-1][j-1]+t[i-1][j]
np.save("Binomial coeff",t)

## precomputed array for 1 dimension
D=dict()
for k in range(n+1):
    [x,w]=scipy.special.roots_jacobi(k+1,0,0)
    M=np.zeros((k+1,k+1))
    for i in range(k+1):
        a=((1-x[i])/2)**k
        b=(1+x[i])/2
        t=1.0
        for j in range(k+1):
            M[i][j]=w[i]*a*t
            a/=((1-x[i])/2)
            t*=b
    D[k]=M
np.save("Precomp D",D)

D1=dict()
for k in range(n+1):
    M=np.zeros((k+1,k+1))
    [x,w]=scipy.special.roots_jacobi(k+1,1,0)
    for i in range(k+1):
        a=((1-x[i])/2)**k
        b=np.sqrt(w[i])
        for j in range(k+1):
            M[j][i]=b*a
            a/=((1-x[i])/2)
    D1[k]=M
np.save("Precomp D1",D1)


P1=dict()
for k in range(n+1):
    M=np.zeros((k+1,k+1))
    [x,w]=scipy.special.roots_jacobi(k+1,1,0)
    for i in range(k+1):
        a=(1+x[i])/2
        b=np.sqrt(w[i])
        t=1
        for j in range(k+1):
            M[j][i]=b*t
            t*=a
    P1[k]=M  

np.save("Precomp P1",P1)

D2=dict()
for k in range(n+1):
    M=np.zeros((k+1,k+1))
    [x,w]=scipy.special.roots_jacobi(k+1,0,0)
    for i in range(k+1):
        a=((1-x[i])/2)**k
        b=np.sqrt(w[i])
        for j in range(k+1):
            M[j][i]=b*a
            a/=((1-x[i])/2)
    D2[k]=M
    
np.save("Precomp D2",D2)

P2=dict()

for k in range(n+1):
    M=np.zeros((k+1,k+1))
    [x,w]=scipy.special.roots_jacobi(k+1,0,0)
    for i in range(k+1):
        a=(1+x[i])/2
        b=np.sqrt(w[i])
        t=1
        for j in range(k+1):
            M[j][i]=b*t
            t*=a
    P2[k]=M

np.save("Precomp P2",P2)

A1=dict()
for k in range(n+1):
    M=np.zeros((k+1,k+1))
    [x,w]=scipy.special.roots_jacobi(k+1,2,0)
    for i in range(k+1):
        a=((1-x[i])/2)**k
        b=np.sqrt(w[i])
        for j in range(k+1):
            M[j][i]=b*a
            a/=((1-x[i])/2)
    A1[k]=M

np.save("Precomp A1",A1)

B1=dict()

for k in range(n+1):
    M=np.zeros((k+1,k+1))
    [x,w]=scipy.special.roots_jacobi(k+1,2,0)
    for i in range(k+1):
        a=(1+x[i])/2
        b=np.sqrt(w[i])
        t=1
        for j in range(k+1):
            M[j][i]=b*t
            t*=a
    B1[k]=M

np.save("Precomp B1",B1)

