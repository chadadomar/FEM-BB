#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 18:55:02 2024

@author: omarch
"""




# TO DO :  include boundary conditioon !!!!!!!!!!


from Gradient_projector import *
from Projection_matrix import *
from Mass_curl_2d import *
from Stiffness_curl_2d import *
from scipy.linalg import block_diag
from Assembly import *

def Glob_Stiff_Mass_curl(r,k):
    mesh_points,mesh_tris,mesh_edges,tris_edges=mesh(k)
    ntris=len(mesh_tris)
    nedges=len(mesh_edges)
    
    ndof=nbr_globDof(nedges,ntris,r)
    M=np.zeros((ndof,ndof))
    S=np.zeros((ndof,ndof))


    for i in range(ntris):
        T=mesh_tris[i]

        # vertex of the triangle/elem tr
        p0=mesh_points[T[0]]
        p1=mesh_points[T[1]]
        p2=mesh_points[T[2]]

        # liste of vertices's coordinates
        Liste=[p0[0],p0[1],p1[0],p1[1], p2[0],p2[1]]

        # Local element
        St=Stiff2d(Liste,r)
        Mt=mass2d(Liste,r)


        local_ndof=r*(r+2)
        for j in range(local_ndof):
            glob_j,sign_j=local_to_global(nedges,T, tris_edges[i], i ,j,r)
            for k in range(local_ndof):
                glob_k,sign_k=local_to_global(nedges,T, tris_edges[i], i ,k,r)
                S[glob_j][glob_k]+= sign_j * sign_k * St[j][k]
                M[glob_j][glob_k]+= sign_j * sign_k* Mt[j][k]
                
    I=IndexToDelete(mesh_edges,mesh_points,r)
    S=np.delete(S, I,0)
    S=np.delete(S, I,1)
    M=np.delete(M, I,0)
    M=np.delete(M, I,1)
    
    return S,M

def Glob_Stiff_Mass_H1(p,k):
    
    mesh_points,mesh_tris,mesh_edges,tris_edges=mesh(k)
    nvertices=len(mesh_points)          # number of domaine points  
    nedges= len(mesh_edges)
    ntris = len(mesh_tris)
    ndof=nbr_globDof_H1(nvertices,nedges,ntris,p) # global dof for H1



    M=np.zeros((ndof,ndof))             # Golbal mass matrix
    S=np.zeros((ndof,ndof))             # Golbal stiffness matrix
    w=(p+2)*(p+1)//2
    
    for ti in range(ntris):
        t=mesh_tris[ti]
        # vertex of the triangle/elem t
        v0=mesh_points[t[0]]
        v1=mesh_points[t[1]]
        v2=mesh_points[t[2]]
        ##liste of vertices's coordinates
        Trig=[v0[0],v0[1],v1[0],v1[1], v2[0],v2[1]] 
        Se=cst_StiffMat_2D(Trig,np.eye(2), p)       #local stifness matrix
        Me=MassMat2D(Trig,lambda x,y:1,p)           #local stifness matrix

        for i in range(w):
            I=local_to_global_H1(nvertices, nedges, t, tris_edges[ti], ti, i, p)
            for j in range(w):
                J=local_to_global_H1(nvertices, nedges, t, tris_edges[ti], ti, j, p)
                S[I][J]+=Se[i][j]
                M[I][J]+=Me[i][j]
    I=IndexToDelete_H1(mesh_edges,mesh_points,p)
    S=np.delete(S, I,0)
    S=np.delete(S, I,1)
    M=np.delete(M, I,0)
    M=np.delete(M, I,1)
    
    return S,M

def ASP_preconditioner(r,k,tau):
    #input:
        # r : order of FE
        # k : integer s.t the area of triangles does not exceed 1/2**k
        # Tau : parameter
    # Output:
        # ASP preconditioner
        
    G=G_glob0(r,k)
    P=Glob_Proj0(r,k)
    
    mesh_points,mesh_tris,mesh_edges,tris_edges=mesh(k)
    ntris=len(mesh_tris)
    nedges=len(mesh_edges)
    Stiff,Mass=Glob_Stiff_Mass_H1(r,k)
    Scurl,Mcurl=Glob_Stiff_Mass_curl(r,k)
    A=Scurl+tau*Mcurl
    
    H=block_diag(Stiff+Mass,Stiff+Mass)
    M=block_diag(Mass,Mass)
    L=Stiff
    S=np.diag(np.diag(A))
    
    Q= H + tau*M
    
    B=np.linalg.inv(S)
    B+= P @ np.linalg.inv(Q) @ np.transpose(P)
    B+= (1/tau) * G @ np.linalg.inv(L) @ np.transpose(G)
    
    return  B
    
    
    