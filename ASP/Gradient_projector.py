#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:34:21 2024

@author: omarch
"""
import numpy as np
from mesh_curl_2D import *

def indexes2D(n):
    return [(i,j, n-(i+j)) for i in range(n,-1, -1) for j in range(n-i, -1, -1)]

def getIndex2D(n,t):
    #n : polynomial order
    #t : multi-index of the domain point (also polynomial index) in lexicographical order
    (i,j,k)=t
    if i+j+k !=n:
        raise Exception("not valide index")
    else:
        return int((n-i)*(n-i+1)//2+n-i-j)  

def getIndex2D_new(n,t):
    #Input:
        #n : polynomial order
        #t : multi-index of the domain point wich is not a vertex (also polynomial index) in lexicographical order
    #Output:
        # index of t
    i=getIndex2D(n,t)
    if i==0:
        raise Exception("index of vertex point")
    elif i < n*(n+1)//2:
        return i-1
    elif i==n*(n+1)//2:
        raise Exception("index of vertex point")
    elif i< (n+1)*(n+2)//2:
        return i-2
    else:
        raise Exception("index of vertex point")

def G_local(r):
    #input:
        # T: triangle vertices
        # r: order of approximation
    # Output:
        # G: local matrix of gradient operator 
    ndofH1=(r+1)*(r+2)//2
    ndofHcurl=r*(r+2)
    G=np.zeros((ndofHcurl,ndofH1))
    
    Ind=indexes2D(r)
    
    
    for j in range(ndofH1):
        if j==0:
            G[-2][j]+=1
            G[-1][j]+=-1
            for k in range(ndofH1):
                alpha=Ind[k]
                if alpha[0]>1 and alpha[0]<r:
                    i=getIndex2D_new(r,alpha)
                    G[i][j]+= - alpha[0]/r
        elif j== r*(r+1)//2:
            G[-1][j]+=1
            G[-3][j]+=-1
            for k in range(ndofH1):
                alpha=Ind[k]
                if alpha[1]>1 and alpha[1]<r:
                    i=getIndex2D_new(r,alpha)
                    G[i][j]+= - alpha[1]/r
        elif j==ndofH1-1:
            G[-3][j]+=1
            G[-2][j]+=-1
            for k in range(ndofH1):
                alpha=Ind[k]
                if alpha[2]>1 and alpha[2]<r:
                    i=getIndex2D_new(r,alpha)
                    G[i][j]+= - alpha[2]/r
        else:
            alpha=Ind[j]
            i=getIndex2D_new(r,alpha)
            G[i][j]+=1
    return G
            
        
def G_glob(r,k):
    #input:
        # k : integer s.t the area of triangles does not exceed 1/2**k
        # r : order of approximation
    #Output:
        # G: The matrix of gradient projection of scalar global H1-Bernstien basis into H curl global vectorial basis
    mesh_points,mesh_tris,mesh_edges,tris_edges=mesh(k)
    nvertices=len(mesh_points)  
    nedges= len(mesh_edges)
    ntris = len(mesh_tris)
    
    nh1 = nbr_globDof_H1(nvertices,nedges,ntris,r)
    nhcurl = nbr_globDof(nedges,ntris,r)
    nlocdofHcurl=r*(r+2)
    nlocdofH1=(r+1)*(r+2)//2
    tol=1e-17
    flag=True
    Bigflag=True
    Gglob=np.zeros((nhcurl,nh1))
    FlagMatrix=np.zeros((nhcurl,nh1))
    for ti in range(ntris):
        t=mesh_tris[ti]
        Gloc=G_local(r)
        for i in range(nlocdofHcurl):
            for j in range(nlocdofH1):
                I,sign=local_to_global(nedges,t,tris_edges[ti], ti,i,r)
                J=local_to_global_H1(nvertices, nedges, t , tris_edges[ti], ti,j,r)
                if FlagMatrix[I][J]==0:
                    Gglob[I][J]+=sign*Gloc[i][j]
                    #Gglob[I][J]+=Gloc[i][j]
                    FlagMatrix[I][J]=1
                    #print("True case")
                    #print("i j I J",i,j,I,J)
                else:
                    flag= (abs(Gglob[I][J] -sign*Gloc[i][j])<tol) 
                    #flag= ( abs(Gglob[I][J] -Gloc[i][j])<tol ) 
                    if flag==False:
                        Bigflag=False
                        print("False case")
                        print("i j I J",i,j,I,J)
                        print("Gglob[I][J] ",Gglob[I][J])
                        print("sign*Gloc[i][j] ",sign*Gloc[i][j])
                        #print("Gloc[i][j] ",Gloc[i][j])
                    #else:
                        #print("True case")
                        #print("i j I J",i,j,I,J)     
    #return Gglob,FlagMatrix,Bigflag
    #return Bigflag
    return Gglob

def G_glob0(r,k):
    #input:
        # k : integer s.t the area of triangles does not exceed 1/2**k
        # r : order of approximation
    #Output:
        # G: The matrix of gradient projection of scalar global H1-Bernstien basis into H curl global vectorial basis
    mesh_points,mesh_tris,mesh_edges,tris_edges=mesh(k)
    nvertices=len(mesh_points)  
    nedges= len(mesh_edges)
    ntris = len(mesh_tris)
    
    nh1 = nbr_globDof_H1(nvertices,nedges,ntris,r)
    nhcurl = nbr_globDof(nedges,ntris,r)
    nlocdofHcurl=r*(r+2)
    nlocdofH1=(r+1)*(r+2)//2
    tol=1e-17
    flag=True
    Bigflag=True
    Gglob=np.zeros((nhcurl,nh1))
    FlagMatrix=np.zeros((nhcurl,nh1))
    for ti in range(ntris):
        t=mesh_tris[ti]
        Gloc=G_local(r)
        for i in range(nlocdofHcurl):
            for j in range(nlocdofH1):
                I,sign=local_to_global(nedges,t,tris_edges[ti], ti,i,r)
                J=local_to_global_H1(nvertices, nedges, t , tris_edges[ti], ti,j,r)
                if FlagMatrix[I][J]==0:
                    Gglob[I][J]+=sign*Gloc[i][j]
                    #Gglob[I][J]+=Gloc[i][j]
                    FlagMatrix[I][J]=1
                    #print("True case")
                    #print("i j I J",i,j,I,J)
                else:
                    flag= (abs(Gglob[I][J] -sign*Gloc[i][j])<tol) 
                    #flag= ( abs(Gglob[I][J] -Gloc[i][j])<tol ) 
                    if flag==False:
                        Bigflag=False
                        print("False case")
                        print("i j I J",i,j,I,J)
                        print("Gglob[I][J] ",Gglob[I][J])
                        print("sign*Gloc[i][j] ",sign*Gloc[i][j])
                        #print("Gloc[i][j] ",Gloc[i][j])
                    #else:
                        #print("True case")
                        #print("i j I J",i,j,I,J)     
    #return Gglob,FlagMatrix,Bigflag
    #return Bigflag
    
    J=IndexToDelete_H1(mesh_edges,mesh_points,r)
    I=IndexToDelete(mesh_edges,mesh_points,r)

    Gglob=np.delete(Gglob, I,0)
    Gglob=np.delete(Gglob, J,1)
    
    return Gglob
    
    
    
    