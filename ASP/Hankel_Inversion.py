#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 09:52:19 2023

@author: omarch
"""

import numpy as np
from scipy.linalg import hankel


"""Inverting Hankel matrix """

def recStep(H,B):
    """ H is the full hankel matrix"""
    n=len(B)-1
    Res=np.zeros((n+2,n+2))
    u=np.zeros(n+2)
    u[-1]=1
    for s in range(n+1):
        for j in range(n+1):
            u[s]-=H[n][j+1]*B[j][s]
    lam=0
    for s in range(n+2):
        lam+=H[n+1][s]*u[s]
    for r in range(n+2):
        for s in range(n+2):
            if r < n+1 and s < n+1:
                Res[r][s]=B[r][s]+u[r]*u[s]/lam
            else:
                Res[r][s]=u[r]*u[s]/lam
    return Res

H=np.array([[1, 2, 3, 4 ],
            [2, 3, 4, 7],
            [3, 4, 7, 7],
            [4, 7, 7, 8]])
#print(np.linalg.inv(H))

#W=recStep( H, np.array([[-3, 2],[2,-1]]) )


def InvHankel(H):
    n=len(H)-1
    B=np.array([[1/H[0][0]]])
    for k in range(n):
        res=recStep(H,B)
        B=res
    return B
                        
         
#print(InvHankel(H) - np.linalg.inv(H)) 
    
           

        