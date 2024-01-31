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
    return np.array([1,1])

def g(x,y):
    return np.array([-y,x])

# defining second memebre
def f1(x,y):
    return np.array([-y*x,x*x*y])  # f=W_1 whitney form




mesh_points,mesh_tris,mesh_edges,tris_edges=mesh()
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
    #print("global mass matrix \n",M)
    X=np.linalg.solve(M,F)  
    return X


### Aribtrary order r:
def testarbitrary(r,g):
    X=assemble_boundary(r,g)
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
            Coef.append(sign*X[k])
        def errf(x,y):
            return (Eval_curl(Liste,r,Coef,x,y)[0]-g(x,y)[0])**2 +   (Eval_curl(Liste,r,Coef,x,y)[1]-g(x,y)[1])**2
        error+=quad(Liste, errf, 7)
    print("error", np.sqrt(error))
    
    
