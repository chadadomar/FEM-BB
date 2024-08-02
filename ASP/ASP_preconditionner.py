#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 18:55:02 2024

@author: omarch
"""






from Gradient_projector import *
from Projection_matrix import *
from Mass_curl_2d import *
from Stiffness_curl_2d import *
from scipy.linalg import block_diag , pinvh , lapack
from Assembly import *
import numpy as np



inds_cache = {}

def upper_triangular_to_symmetric(ut):
    n = ut.shape[0]
    try:
        inds = inds_cache[n]
    except KeyError:
        inds = np.tri(n, k=-1, dtype=bool)
        inds_cache[n] = inds
    ut[inds] = ut.T[inds]


def fast_positive_definite_inverse(m):
    cholesky, info = lapack.dpotrf(m)
    if info != 0:
        raise ValueError('dpotrf failed on input {}'.format(m))
    inv, info = lapack.dpotri(cholesky)
    if info != 0:
        raise ValueError('dpotri failed on input {}'.format(cholesky))
    upper_triangular_to_symmetric(inv)
    return inv

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

def Glob_Stiff_Mass_curl_bis(r,k):
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
                if glob_j >= glob_k:
                    S[glob_j][glob_k]+= sign_j * sign_k * St[j][k]
                    M[glob_j][glob_k]+= sign_j * sign_k* Mt[j][k]   
                    
    S+=np.transpose(S) -     np.diag(np.diag(S)) 
    M+=np.transpose(M) -     np.diag(np.diag(M)) 
    
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

def Glob_Stiff_Mass_H1_bis(p,k):
    
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
                if I >= J:
                    S[I][J]+=Se[i][j]
                    M[I][J]+=Me[i][j]
    S+=np.transpose(S) -     np.diag(np.diag(S)) 
    M+=np.transpose(M) -     np.diag(np.diag(M)) 
    
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
    Stiff,Mass=Glob_Stiff_Mass_H1_bis(r,k)
    Scurl,Mcurl=Glob_Stiff_Mass_curl_bis(r,k)
    A=Scurl+tau*Mcurl
    
    #H=block_diag(Stiff+Mass,Stiff+Mass)
    #M=block_diag(Mass,Mass)
    #Q= H + tau*M

    aux=fast_positive_definite_inverse(Stiff+(1+tau)*Mass)
    Q=block_diag(aux,aux)
    S=np.diag(1/np.diag(A))

    B= S
    B+= P @ Q @ np.transpose(P)
    B+= (1/tau) * G @ fast_positive_definite_inverse(Stiff) @ np.transpose(G)
    
    return  B

def Test_Sym(r,k,tau):
    #input:
        # r : order of FE
        # k : integer s.t the area of triangles does not exceed 1/2**k
        # Tau : parameter
    # Output:
        # ASP preconditioner

    Stiff,Mass=Glob_Stiff_Mass_H1_bis(r,k)
    #P=Glob_Proj0(r,k)
    H=block_diag(Stiff+Mass,Stiff+Mass)
    M=block_diag(Mass,Mass)
    L=np.linalg.inv(Stiff)
    Q=H + tau*M
    '''Scurl,Mcurl=Glob_Stiff_Mass_curl_bis(r,k)
    A=Scurl+tau*Mcurl'''
    
    aux=fast_positive_definite_inverse(Stiff+(1+tau)*Mass)
    Q2=block_diag(aux,aux)
    return Q , Q2

MaxB=0    

for k in range(2,8):
    for r in range(1,8):
        B=ASP_preconditioner(r,k,1)

        
        print( " loss of symmetry for r="+str(r)+" and k="+str(k))
        m=np.max(np.abs(B- np.transpose(B)))
        print("error is ",m)
        if m>MaxB:
            MaxB=m
        
        '''if np.allclose( I ,np.eye(n)) :
            print("valid for r="+str(r)+" and k="+str(k))
        else:
            print( " loss of symmetry for r="+str(r)+" and k="+str(k))
            m=np.max(np.abs(I- np.eye(n)))
            print("error is ",m)
            if m>MaxB:
                MaxB=m'''
        
            
print(MaxB)

