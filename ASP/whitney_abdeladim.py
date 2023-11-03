#!/usr/bin/env python3
# -*- coding: utf-8 -*-


''' Here we change the choice of the basis face, we consider the tow vector edges [0,1] and [1,2] '''

import sympy as sy
import numpy as np
from fractions import Fraction


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
                dofFace.append([1,2,alpha,beta,gamma])
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
            return sy.Rational(fact(k_1)*fact(k_2), fact((1+k_1+k_2)))
        else:
            return 0
    else:
        return 0


def faceIntegral(faceDof,w):
    [i,j,alpha,beta,gamma]=faceDof
    L=[alpha,beta,gamma]
    q=len(w)
    if q==4:
        # edge type generator
        if i==w[0]:
            if j==w[1]:
                # edge w . tangent = lambda_i+lambda_j
                x=L[i]
                y=L[j]
                L.pop(i)
                L.pop(j-1)
                z=L[-1]
                k_1=x+w[2]
                #k_1=alpha+w[2]
                k_2=y+w[3]
                #k_2=beta+w[3]
                return sy.Rational(2*fact(k_1)*fact(k_2)*fact(z)*(k_1+k_2+2) , fact((3+k_1+k_2+z)))
            else:
                # edge w . tangent = lambda_k
                x=L[i]
                y=L[j]
                L.pop(i)
                L.pop(j-1)
                z=L[-1]
                k_1=x+w[2]
                k_2=z+1+w[3]
                return sy.Rational(2*fact(k_1)*fact(y)*fact(k_2),fact((2+k_1+y+k_2)))
        else:
            if j==w[0]:
                # edge w . tangent = -lambda_k
                x=L[i]
                y=L[j]
                L.pop(i)
                L.pop(j-1)
                z=L[-1]
                k_1=y+w[3]
                k_2=z+1+w[2]
                return sy.Rational( 2*fact(x)*fact(k_1)*fact(k_2) , fact((2+x+k_1+k_2)) )
            else:
                if i==0:
                    k_1=w[2]+beta
                    k_2=w[3]+1+gamma
                    return sy.Rational( (-2)*fact(alpha)*fact(k_1)*fact(k_2) , fact((2+alpha+k_1+k_2)) )
                else:
                    k_1=w[2]+alpha+1
                    k_2=w[3]+beta
                    return sy.Rational( (-2)*fact(gamma)*fact(k_1)*fact(k_2) , fact((2+gamma+k_1+k_2)) )
    else:
        # face dof & face generator
        if i==w[0]:
            H=[w[2],w[3],w[4]]
            x_1=L[i]
            y_1=L[j]
            L.pop(i)
            L.pop(j-1)
            z_1=L[-1]
            x_2=H[i]
            y_2=H[j]
            H.pop(i)
            H.pop(j-1)
            z_2=H[-1]
            # edge w . tangent = lambda_i+lambda_j
            k_1=x_1+x_2
            k_2=y_1+y_2
            k_3=z_1+z_2+1
            return sy.Rational( 2*fact(k_1)*fact(k_2)*fact(k_3)*(k_1+k_2+2) ,  fact((3+k_1+k_2+k_3)) )
        else:
            if i==0:
                # i = 0 , j=1 , w[0]=1 , w[1]=2
                k_1=alpha+w[2]+1
                k_2=beta+w[3]
                k_3=gamma+1+w[4]
                return sy.Rational( (-2)*fact(k_1)*fact(k_2)*fact(k_3) ,  fact((2+k_1+k_2+k_3)) )
            else:
                # i = 1 , j=2 , w[0]=0 , w[1]=1
                k_1=alpha+w[2]+1
                k_2=beta+w[3]
                k_3=gamma+1+w[4]
                return sy.Rational( (-2)*fact(k_1)*fact(k_2)*fact(k_3) ,  fact((2+k_1+k_2+k_3)) )

def Vandermond2d(r):
    # genrelaized vandermonde matrix Vij=sigm_i(w_j)
    ndof=r*(r+2)
    V=sy.zeros(ndof,ndof)
    if r==1:
        dofEdge=dof(r)
        for i in range(ndof):
            sigma=dofEdge[i]
            for j in range(ndof):
                w=dofEdge[j]
                #V[i][j]+=edgeIntegral(sigma,w)
                V[i,j]+=edgeIntegral(sigma,w)
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
                    #V[i][j]+=edgeIntegral(sigma,w)
                    V[i,j]+=edgeIntegral(sigma,w)
            else:
                sigma=dofFace[i-nedge]
                for j in range(ndof):
                    if j<nedge:
                        w=dofEdge[j]
                    else:
                        w=dofFace[j-nedge]
                    #V[i][j]+=faceIntegral(sigma,w)
                    V[i,j]+=faceIntegral(sigma,w)
    return V
    """lc=1
    for i in range(ndof):
        for j in range(ndof):
            lc=sy.ilcm(sy.denom(V[i,j]),lc)
    for i in range(ndof):
        for j in range(ndof):
            V[i,j]*=lc
    return lc,V
    #return V"""
            

