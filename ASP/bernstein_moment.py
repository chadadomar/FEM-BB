#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 12:11:52 2023

@author: omarch
"""
import numpy as np
import math as mth
from Whitney_condition_number import *
np.set_printoptions(linewidth=10000)


T=[0,1,2]
#Edge=[[0,1],[0,2],[1,2]]
Pos=[ [0,0],
      [1,0],
      [0,1] ] 


### 2D domaine points in lexicographic order
def indexes2D(n):
    return [(i,j, n-(i+j)) for i in range(n,-1, -1) for j in range(n-i, -1, -1)]


def unitVec(a,b):
    xa,ya=a
    xb,yb=b
    norm=np.sqrt((xb-xa)**2+(yb-ya)**2)
    x=(xb-xa)/norm
    y=(yb-ya)/norm
    return [x,y]


def moment(r):
    ndof=r*(r+2)
    if r>1:
        dofs=dof(r)[0]+dof(r)[1]
    else:
        dofs=dof(1)
    nedge=3*r
    nBern=int((r+1)*(r+2)/2)
    M=np.zeros((ndof,2*nBern))
    
    ind=indexes2D(r)
    
    # for e1=(1,0)
    for i in range(ndof):
        for j in range(nBern):
            alpha=ind[j]
            if i < nedge:
                [l,m,beta_l,beta_m]=dofs[i]
                alpha_n=sum(alpha) - alpha[l]-alpha[m]
                if alpha_n== 0:
                    cst= unitVec(Pos[l],Pos[m])[0] * mth.comb(r,alpha[l])
                    M[i][j]+= cst * fact(alpha[l]+beta_l) * fact(alpha[m]+beta_m) / fact(1+alpha[l]+beta_l+alpha[m]+beta_m)
            else:
                [l,m,beta_0,beta_1,beta_2]=dofs[i]
                cst= unitVec(Pos[l],Pos[m])[0] * fact(r)/ (fact(alpha[0])*fact(alpha[1])*fact(alpha[2]) )
                M[i][j]+=cst*2* fact(alpha[0]+beta_0) * fact(alpha[1]+beta_1) * fact(alpha[2]+beta_2) / fact(2+alpha[0]+beta_0+alpha[1]+beta_1+alpha[2]+beta_2)
    
    # for e2=(0,1)
    for i in range(ndof):
        for j in range(nBern):
            alpha=ind[j]
            if i < nedge:
                [l,m,beta_l,beta_m]=dofs[i]
                alpha_n=sum(alpha) - alpha[l]-alpha[m]
                if alpha_n== 0:
                    cst= unitVec(Pos[l],Pos[m])[1] * mth.comb(r,alpha[l])
                    M[i][j+nBern]+= cst * fact(alpha[l]+beta_l) * fact(alpha[m]+beta_m) / fact(1+alpha[l]+beta_l+alpha[m]+beta_m)
            else:
                [l,m,beta_0,beta_1,beta_2]=dofs[i]
                cst= unitVec(Pos[l],Pos[m])[1] * fact(r)/ (fact(alpha[0])*fact(alpha[1])*fact(alpha[2]) )
                M[i][j+nBern]+=cst*2* fact(alpha[0]+beta_0) * fact(alpha[1]+beta_1) * fact(alpha[2]+beta_2) / fact(2+alpha[0]+beta_0+alpha[1]+beta_1+alpha[2]+beta_2)
                
    return M
                    
                    