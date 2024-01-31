#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:02:52 2023

@author: omarch
"""

import numpy as np
import math as mt


## Return T: pascal triangle of ordre n

def fact(n):
    if n<0:
        raise Exception("Sorry, no numbers below zero")
    if n==0:
        return 1
    else:
        return n*fact(n-1)
    
def multifact(a,b):
    n=len(a)
    if len(b)!=n:
        raise Exception("Sorry, different size")
    else:
        p=1
        for i in range(n):
            p*=mt.comb(a[i],b[i])
        return p

def sumVect(u,v):
    m=len(u)
    n=len(v)
    if n!=m:
        print("the vectors have different size")
    else:
        w=[]
        for i in range(n):
            w.append(int(u[i]+v[i]))
        return w

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


def c_star(alpha,L):
    T=AirT2D(L)
    p=sum(alpha)
    c_bar=[]
    
    # k=l 
    eta=list(alpha)
    '''q=(alpha[0]+1)*(alpha[1]+alpha[2])
    q+=(alpha[1]+1)*(alpha[0]+alpha[2])
    q+=(alpha[2]+1)*(alpha[1]+alpha[0])
    q*=((p+1)/(2*T) )'''
    q=((p+1)/(T) )* (alpha[0]*alpha[1]+alpha[1]*alpha[2]+alpha[2]*alpha[0]+p)
    c_bar.append((eta,q))
    
    # k != l
    for k in range(3):
        for l in range(3):
            if alpha[l]==0:
                continue
            if k!=l:
                eta=list(alpha)
                eta[k]+=1
                eta[l]-=1
                q=(-(p+1)/(2*T) ) * (alpha[k]+1) * alpha[l]
                c_bar.append((eta,q))
    return c_bar

def Stiff2d(L,r):
    p=r-1
    T=AirT2D(L)
    ndof=r*(r+2)
    nBern=int((r+1)*(r+2)/2)
    
    S=np.zeros((ndof,ndof))
    
    # whitney whitney
    S[-3:,-3:]=np.full((3,3),1/T)
    
    
    
    # Gamma Gamma
    In=indexes2D(r-1)
    In.pop()
    m=len(In)
    for i in range(m):
        for j in range(m):
            alpha=In[i]
            beta=In[j]
            coef=2*T*(fact(p)**2)/(fact(2*r))
            star_alpha=c_star(alpha,L)
            star_beta=c_star(beta,L)
            '''if (i==0 and j==1):
                print("coef is ", coef)
                print("star_alpha ", star_alpha)
                print("star_beta ",star_beta)'''
            temp=0
            for x in star_alpha:
                for y in star_beta:
                    eta=x[0]
                    pho=y[0]
                    temp+=multifact(sumVect(eta,pho) , pho) *x[1] *y[1]
                    '''if (i==0 and j==1):
                        print("temps ",temp)'''
            S[nBern-3+i][nBern-3+j]+=coef*temp
            
    # Gamma Whitney
    for k in range(m):
        alpha=In[k]
        for i in range(1,4):
            coef=2*fact(p)/(T*fact(p+2))
            star_alpha=c_star(alpha,L)
            temp=0
            for x in star_alpha:
                temp+=x[1]
            S[k+nBern-3][-i]+=coef*temp
    S[-3:,nBern-3:]= np.transpose( S[nBern-3:,-3:] )

    return S                                                                                       