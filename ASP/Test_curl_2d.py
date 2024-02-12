#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:03:22 2024


Test projection L2 du problème H(curl)

Espace de discretisation: Nédelec first type

@author: omarch
"""
from mesh_curl_2D import *
import numpy as np
from Mass_curl_2d import *
from Stiffness_curl_2d import *
from Loadvector_curl_2d import *
from Evaluation_curl_BBform import *
from Quadratur_over_triangle import *
from scipy import integrate


np.set_printoptions(precision=5 ,linewidth=10000)




# Right hand side and solution
def f(x,y):
    return np.array([y*(1-y)+2,x*(1-x)+2])

def uf(x,y):
    return np.array([y*(1-y),x*(1-x)])

def curluf(x,y):
    return 2*(y-x)

def g(x,y):
    return np.array([-y,x])

# defining second memebre
def h(x,y):
    return np.array([ (1 - 2*y)**2*np.sin(y*(1 - y)) + np.sin(y*(1 - y)) + 2*np.cos(y*(1 - y)) , (1 - 2*x)**2*np.sin(x*(1 - x)) + np.sin(x*(1 - x)) + 2*np.cos(x*(1 - x)) ])  

def uh(x,y):
    return np.array([np.sin(y*(1-y)),np.sin(x*(1-x))]) 

def curluh(x,y):
    return (1 - 2*x)*np.cos(x*(1 - x)) - (1 - 2*y)*np.cos(y*(1 - y))

def qh(x,y):
    return np.array([np.sin(np.pi*y),np.sin(np.pi*x)])
def curlqh(x,y):
    return np.pi*np.cos(np.pi*x) - np.pi*np.cos(np.pi*y)
def q(x,y):
    return np.array([np.sin(np.pi*y) + np.pi**2*np.sin(np.pi*y) ,np.sin(np.pi*x) +  np.pi**2*np.sin(np.pi*x)])

mesh_points,mesh_tris,mesh_edges,tris_edges=mesh(2)
ntris=len(mesh_tris)
nedges=len(mesh_edges)


def global_matrices(r,h):
    ndof=nbr_globDof(nedges,ntris,r)
    M=np.zeros((ndof,ndof))
    S=np.zeros((ndof,ndof))
    F=np.zeros(ndof)
    
    for i in range(ntris):
        T=mesh_tris[i]
        
        # vertex of the triangle/elem tr
        p0=mesh_points[T[0]]
        p1=mesh_points[T[1]]
        p2=mesh_points[T[2]]
        
        # liste of vertices's coordinates
        Liste=[p0[0],p0[1],p1[0],p1[1], p2[0],p2[1]]
        #print("traingel vertices \n",Liste)
        
        # Local element 
        St=Stiff2d(Liste,r)
        Mt=mass2d(Liste,r)
        Ft=load2d(h,Liste,r)
        #print("S_T \n",St)
        #print("M_T \n",Mt)
        #print("F_T \n",Ft)
        
        local_ndof=r*(r+2)
        for j in range(local_ndof):
            glob_j,sign_j=local_to_global(nedges,T, tris_edges[i], i ,j,r)
            F[glob_j]+= sign_j * Ft[j]
            for k in range(local_ndof):
                glob_k,sign_k=local_to_global(nedges,T, tris_edges[i], i ,k,r)
                S[glob_j][glob_k]+= sign_j * sign_k * St[j][k]
                M[glob_j][glob_k]+= sign_j * sign_k* Mt[j][k]
                
    return S,M,F


def assemble_boundary(r,h):
    S,M,F=global_matrices(r,h)
    I=IndexToDelete(mesh_edges,mesh_points,r)
    S=np.delete(S, I,0)
    S=np.delete(S, I,1)
    M=np.delete(M, I,0)
    M=np.delete(M, I,1)
    F=np.delete(F, I,0)
    #print("global mass matrix \n",M)
    X=np.linalg.solve(S+M,F)  
    return X,I

def reconstruct(X,I):
    # creat new vector with newX:
        # lenth = sum of lengths of X and I
        # newX[i]=0 for i in I
    li=len(I)
    n=len(X)+li
    newX=np.zeros(n)
    x=0
    i=0
    flag=True
    for k in range(n):
        if flag and k==I[i]:
            i+=1
            if i==li:
                flag=False
        else:
            newX[k]=X[x]
            x+=1
    return newX


### Aribtrary order r:
def testCurl(r,f,uf):
    X,I=assemble_boundary(r,f)
    newX=reconstruct(X,I)
    #print("I",I)
    error=0
    for i in range(ntris):
        T=mesh_tris[i]
        
        p0=mesh_points[T[0]]
        p1=mesh_points[T[1]]
        p2=mesh_points[T[2]]
        
        Liste=[p0[0],p0[1],p1[0],p1[1],p2[0],p2[1]]
        ndof=r*(r+2)
        Coef=[]
        for j in range(ndof):
            k,sign=local_to_global(nedges, T, tris_edges[i], i, j, r)
            #print("i j k signe",i,j,k,sign)
            Coef.append(sign*newX[k])
        def errf(x,y):
            return (Eval_curl(Liste,r,Coef,x,y)[0]-uf(x,y)[0])**2 +   (Eval_curl(Liste,r,Coef,x,y)[1]-uf(x,y)[1])**2
        contribution=quad(Liste, errf, r+1)
        error+=contribution
        #print("triangle ", Liste ,"error ",contribution)
        #print(" BB form ", Coef)
    print("error", np.sqrt(error))
    #print("error without square", error)
    
def HcurlError(r,f,uf,curluf):
    X,I=assemble_boundary(r,f)
    newX=reconstruct(X,I)
    #print("I",I)
    error=0
    for i in range(ntris):
        T=mesh_tris[i]
        
        p0=mesh_points[T[0]]
        p1=mesh_points[T[1]]
        p2=mesh_points[T[2]]
        
        Liste=[p0[0],p0[1],p1[0],p1[1],p2[0],p2[1]]
        ndof=r*(r+2)
        Coef=[]
        for j in range(ndof):
            k,sign=local_to_global(nedges, T, tris_edges[i], i, j, r)
            #print("i j k signe",i,j,k,sign)
            Coef.append(sign*newX[k])
        def errf(x,y):
            return (Eval_curl(Liste,r,Coef,x,y)[0]-uf(x,y)[0])**2 +   (Eval_curl(Liste,r,Coef,x,y)[1]-uf(x,y)[1])**2 + (Eval_curlcurl(Liste,r,Coef,x,y)-curluf(x,y))**2
        contribution=quad(Liste, errf, r+1)
        error+=contribution
        #print("triangle ", Liste ,"error ",contribution)
        #print(" BB form ", Coef)
    print("error", np.sqrt(error))
    #print("error without square", error)
    
    
