#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 20:22:27 2023

@author: omarch
"""
# Gradient matrix of order r
import numpy as np

def indexes2D(n):
    return [(i,j, n-(i+j)) for i in range(n,-1, -1) for j in range(n-i, -1, -1)]

def GradMatrix(r):
    ndof=r*(r+2)
    L=indexes2D(r)
    nBern=int((r+1)*(r+2)/2)
    
    G=np.zeros((ndof,nBern))
    
    for j in range(nBern):
        alpha=L[j]
        if alpha[0]==r:
            G[1][j]+=1
            G[2][j]+= -1
            for i in range(3,nBern+3):
                beta=L[i-3]
                if beta[0]>0 and beta[0]<r:
                    G[i][j]+= - beta[0]/r
        elif alpha[1]==r:
            G[2][j]+=1
            G[0][j]+= -1
            for i in range(3,nBern+3):
                beta=L[i-3]
                if beta[1]>0 and beta[1]<r:
                    G[i][j]+= - beta[1]/r
        elif alpha[2]==r:
            G[0][j]+=1
            G[1][j]+= -1
            for i in range(3,nBern+3):
                beta=L[i-3]
                if beta[2]>0 and beta[2]<r:
                    G[i][j]+= - beta[2]/r
        else:
            G[j+3][j]+=1
    return G
            
    