#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 19:24:45 2023

@author: omarch
"""

import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)



T=[0,1,2]
Edge=[[0,1],[0,2],[1,2]]



def fact(n):
    if n==0:
        return 1
    else:
        return n*fact(n-1)


def dof(r):
    # r: order of Whitney forms (r>=1)
    dofEdge=[]
    dofFace=[]
    for i in range(3):
        e=Edge[i]
        l=e[0]
        m=e[1]
        for beta in range(r):
            alpha=r-1-beta
            dofEdge.append([l,m,alpha,beta])
    if r==1:
        return dofEdge
    else:
        for gamma in range(r-1):
            for beta in range(r-1-gamma):
                alpha=r-2-gamma-beta
                dofFace.append([0,1,alpha,beta,gamma])
        for gamma in range(r-1):
            for beta in range(r-1-gamma):
                alpha=r-2-gamma-beta
                dofFace.append([0,2,alpha,beta,gamma])
    return dofEdge,dofFace
                
def edgeIntegral(edgeDof,w):
    [i,j,alpha,beta]=edgeDof
    q=len(w)
    if q==4:
        # edge type generator
        if [i,j]==w[:2]:
            k_1=alpha+w[2]
            #print("k_1: ",k_1)
            k_2=beta+w[3]
            #print("k_2: ",k_2)
            return fact(k_1)*fact(k_2)/ fact((1+k_1+k_2))
        else:
            return 0
    else:
        return 0

def faceIntegral(faceDof,w):
    [i,j,alpha,beta,gamma]=faceDof
    #L=[alpha,beta,gamma]
    q=len(w)
    if q==4:
        # edge type generator
        if i==w[0]:
            if j==w[1]:
                if j==1:
                    # edge w . tangent = lambda_i+lambda_j
                    k_1=alpha+w[2]
                    k_2=beta+w[3]
                    return 2*fact(k_1)*fact(k_2)*fact(gamma)*(k_1+k_2+2) / fact((3+k_1+k_2+gamma))
                else:
                    #j=2
                    k_1=alpha+w[2]
                    k_2=gamma+w[3]
                    return 2*fact(k_1)*fact(k_2)*fact(beta)*(k_1+k_2+2) / fact((3+k_1+k_2+beta))
            else:
                # edge w . tangent = lambda_k
                if j==1:
                    # k= 2
                    k_1=alpha+w[2]
                    k_2=gamma+w[3]+1
                    return 2*fact(k_1)*fact(k_2)*fact(beta) / fact((2+k_1+k_2+beta))
                else:
                    # j=2 , k=1
                    k_1=alpha+w[2]
                    k_2=beta+w[3]+1
                    return 2*fact(k_1)*fact(k_2)*fact(gamma) / fact((2+k_1+k_2+gamma))
        else:
            if j==1:
                # edge w . tangent = -lambda_2
                k_1=beta+w[2]
                k_2=gamma+w[3]+1
                return (-2)*fact(k_1)*fact(k_2)*fact(alpha) / fact((2+k_1+k_2+alpha))
            else:
                #j=2 , k=2
                # edge w . tangent = lambda_1
                k_1=beta+w[2]+1
                k_2=gamma+w[3]
                return 2*fact(k_1)*fact(k_2)*fact(alpha) / fact((2+k_1+k_2+alpha))
    
    else:
        # face type generator
        if i==w[0]:
            if j==1:
                if w[1]==1:
                    # edge w . tangent = lambda_0+lambda_1
                    k_1=alpha+w[2]
                    k_2=beta+w[3]
                    k_3=gamma+w[4]+1
                    return 2*fact(k_1)*fact(k_2)*fact(k_3)*(k_1+k_2+2) / fact((3+k_1+k_2+k_3))
                else:
                    #w[1]==2
                    # edge w . tangent = lambda_2
                    k_1=alpha+w[2]
                    k_2=beta+w[3]+1
                    k_3=gamma+w[4]+1
                    return 2*fact(k_1)*fact(k_2)*fact(k_3) / fact((2+k_1+k_2+k_3))
            else:
                # j = 2
                if w[1]==1: 
                    # edge w . tangent = lambda_1
                    k_1=alpha+w[2]
                    k_2=beta+w[3]+1
                    k_3=gamma+w[4]+1
                    return 2*fact(k_1)*fact(k_2)*fact(k_3) / fact((2+k_1+k_2+k_3))
                else:
                    #W[1]=2
                    # edge w . tangent = lambda_0+lambda_2
                    k_1=alpha+w[2]
                    k_2=beta+w[3]+1
                    k_3=gamma+w[4]
                    return 2*fact(k_1)*fact(k_2)*fact(k_3)*(k_1+k_3+2) / fact((3+k_1+k_2+k_3))



def Vandermond2d(r):
    # genrelaized vandermonde matrix Vij=sigm_i(w_j)
    ndof=r*(r+2)
    V=np.zeros((ndof,ndof))
    if r==1:
        dofEdge=dof(r)
        for i in range(ndof):
            sigma=dofEdge[i]
            for j in range(ndof):
                w=dofEdge[j]
                V[i][j]+=edgeIntegral(sigma,w)
        return V
    else:
        dofEdge,dofFace=dof(r)
        #print(dofEdge)
        #print(dofFace)
        nedge=3*r
        for i in range(ndof):
            if i<nedge:
                sigma=dofEdge[i]
                for j in range(ndof):
                    if j<nedge:
                        w=dofEdge[j]
                    else:
                        w=dofFace[j-nedge]
                    V[i][j]+=edgeIntegral(sigma,w)
            else:
                sigma=dofFace[i-nedge]
                for j in range(ndof):
                    if j<nedge:
                        w=dofEdge[j]
                    else:
                        w=dofFace[j-nedge]
                    V[i][j]=faceIntegral(sigma,w)
    return V
            

def cond(r):
    Le=np.zeros(r-1)
    Lf=np.zeros(r-1)
    for i in range(2,r+1):
        V=Vandermond2d(i)
        Ve=V[0:i,0:i]
        Vf=V[3*i:i*(i+2), 3*i:i*(i+2)]
        Le[i-2]=(np.linalg.cond(Ve,np.inf))
        Lf[i-2]=(np.linalg.cond(Vf,np.inf))
    return Le,Lf
            
        
        