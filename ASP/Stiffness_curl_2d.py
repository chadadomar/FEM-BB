#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:02:52 2023

@author: omarch
"""

import numpy as np



## Return T: pascal triangle of ordre n

def fact(n):
    if n<0:
        raise Exception("Sorry, no numbers below zero")
    if n==0:
        return 1
    else:
        return n*fact(n-1)


### Aire triangle based on Heron Formulae
def AirT2D(L):
    [x1,y1,x2,y2,x3,y3]=L
    a=np.sqrt((x1-x2)**2+(y1-y2)**2)
    b=np.sqrt((x1-x3)**2+(y1-y3)**2)
    c=np.sqrt((x3-x2)**2+(y3-y2)**2)
    p=(a+b+c)/2
    return np.sqrt(p*(p - a) * (p - b) * (p - c))

def indexes2D(n):
    return [(i,j, n-(i+j)) for i in range(n,-1, -1) for j in range(n-i, -1, -1)]

def getIndex2D(n,t):
    #n : polynomial order
    #t : index of the domain point (also polynomial index) in lexicographical order
    (i,j,k)=t
    return int((n-i)*(n-i+1)//2+n-i-j)  


def Stiff2d(L,r):
    p=r-1
    T=AirT2D(L)
    ndof=r*(r+2)
    nBern=int((r+1)*(r+2)/2)
    
    S=np.zeros((ndof,ndof))

    S[-3:,-3:]=np.full((3,3),1/T)
    
    In=indexes2D(r-1)
    In.pop()
    m=len(In)
    for i in range(m):
        for j in range(m):
            alpha=In[i]
            beta=In[j]
            coef=2*T*(fact(p)**2)/(fact(2*r))
            for k in range(3):
                for l in range(3):
                    if alpha[l]==0:
                        continue
                    else:
                        if k!=l:
                            c_eta= -r * (alpha[k]+1) * alpha[l] /(2*T) 
                        else:
                            c_eta= r*( alpha[0]*alpha[1]+alpha[0]*alpha[2]+alpha[1]*alpha[2]+p)/T
                        for u in range(3):
                            for v in range(3):
                                if beta[v]==0:
                                    continue
                                else:
                                    if k!=l:
                                        c_pho= -r * (beta[k]+1) * beta[l] /(2*T) 
                                    else:
                                        c_pho= r*( beta[0]*beta[1]+beta[0]*beta[2]+beta[1]*beta[2]+p)/T
                                    eta=list(alpha)
                                    eta[k]+=1
                                    eta[l]-=1
                                    pho=list(beta)
                                    pho[u]+=1
                                    pho[v]-=1
                                    coefbin= fact(eta[0]+pho[0])/fact(eta[0])
                                    coefbin*=fact(eta[1]+pho[1])/fact(eta[1])
                                    coefbin*=fact(eta[2]+pho[2])/fact(eta[2])
                                    S[i+nBern-3][j+nBern-3]+=coefbin*c_eta*c_pho
            S[i+nBern-3][j+nBern-3]*=coef
    for k in range(m):
        alpha=In[k]
        for i in range(1,4):
            coef=2*fact(p)/(T*fact(p+2))
            for k in range(3):
                for l in range(3):
                    if k!=l:
                        c_eta= -r * (alpha[k]+1) * alpha[l] /(2*T) 
                    else:
                        c_eta= r*( alpha[0]*alpha[1]+alpha[0]*alpha[2]+alpha[1]*alpha[2]+p)/T
                        S[k+nBern-3][-i]+=c_eta

            S[k+nBern-3][-i]*=coef 
    S[-3:,nBern-3:]= np.transpose( S[nBern-3:,-3:] )

    return S                                                                                       