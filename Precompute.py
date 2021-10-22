# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 13:55:16 2021

@author: Omar.CHADAD
"""

import numpy as np

import scipy.special 

n=100

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