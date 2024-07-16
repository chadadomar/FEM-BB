#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 18:01:13 2023

Projector matrix

@author: omarch
"""

import numpy as np
import math as mth
from bernstein_moment import *
from Vandermond_Inversion import *
from Passage_bernstein_to_whitney import *
from mesh_curl_2D import *
#np.set_printoptions(linewidth=10000)

def P(r):
    
    ndof=r*(r+2)
    nBern=int((r+1)*(r+2)/2)
    M=np.zeros((ndof,2*nBern))
    
    B=moment(r)
    
    Vinv=InvVandermond(r)
    Q=np.linalg.inv(PassageScr(r))
    
    for k in range(ndof):
        for l in range(2*nBern):
            for i in range(ndof):
                for j in range(ndof):
                    M[k][l]+=B[j][l]*Vinv[i][j]*Q[k][i]
    return M

def Projection(t,T,r):
    #input:
        # T : list of vertices coordinates, the order of vertices is important
        # r : order of approximation
    #Output:
        # M: The matrix of projection of vectorial H1-Bernstien basis into H curl vectorial basis in T
    ndof=r*(r+2)
    nBern=int((r+1)*(r+2)/2)
    M=np.zeros((ndof,2*nBern))
    
    B=momentbis(t,T,r)
    Vinv=InvVandermond(r)
    #Q=np.linalg.inv(PassageScr(r))
    Q=np.linalg.inv(PassageScr_newOrder(r))
    
    for k in range(ndof):
        for l in range(2*nBern):
            for i in range(ndof):
                for j in range(ndof):
                    M[k][l]+=B[j][l]*Vinv[i][j]*Q[k][i]
    return M


def Glob_Proj(r,k):
    #input:
        # k : integer s.t the area of triangles does not exceed 1/2**k
        # r : order of approximation
    #Output:
        # M: The matrix of projection of vectorial global H1-Bernstien basis into H curl global vectorial basis
    mesh_points,mesh_tris,mesh_edges,tris_edges=mesh(k)
    nvertices=len(mesh_points)  
    nedges= len(mesh_edges)
    ntris = len(mesh_tris)
    
    nh1 = nbr_globDof_H1(nvertices,nedges,ntris,r)
    nhcurl = nbr_globDof(nedges,ntris,r)
    nBern=(r+1)*(r+2)//2
    nlocdof=r*(r+2)
    
    Pglob=np.zeros((nhcurl,2*nh1))
    FlagMatrix=np.zeros((nhcurl,2*nh1))
    tol=1e-12
    flag=True
    Bigflag=True
    for ti in range(ntris):
        t=mesh_tris[ti]
        #print("triangle ",ti,t)
        # vertex of the triangle/elem t
        v0=mesh_points[t[0]]
        v1=mesh_points[t[1]]
        v2=mesh_points[t[2]]
        ##liste of vertices's coordinates
        T=[v0[0],v0[1],v1[0],v1[1], v2[0],v2[1]] 
        Ploc=Projection(t,T,r)
        #print("Ploc \n",Ploc)
        for i in range(nlocdof):
            for j in range(nBern):
                I,sign=local_to_global(nedges,t,tris_edges[ti], ti,i,r)
                J=local_to_global_H1(nvertices, nedges, t , tris_edges[ti], ti,j,r)
                if FlagMatrix[I][J]==0:
                    Pglob[I][J]+=sign*Ploc[i][j]
                    #Pglob[I][J]+=Ploc[i][j]
                    FlagMatrix[I][J]=1
                    #print("True case")
                    #print("i j I J",i,j,I,J)
                else:
                    flag= (abs(Pglob[I][J] -sign*Ploc[i][j])<tol) 
                    #flag= ( abs(Pglob[I][J] -Ploc[i][j])<tol ) 
                    if flag==False:
                        Bigflag=False
                        print("False case")
                        print("i j I J",i,j,I,J)
                        print("Pglob[I][J] ",Pglob[I][J])
                        print("sign*Ploc[i][j] ",sign*Ploc[i][j])
                        #print("Ploc[i][j] ",Ploc[i][j])
                    #else:
                        #print("True case")
                        #print("i j I J",i,j,I,J)
        for i in range(nlocdof):
            for j in range(nBern):
                I,sign=local_to_global(nedges,t,tris_edges[ti], ti,i,r)
                J=local_to_global_H1(nvertices, nedges, t , tris_edges[ti], ti,j,r)
                if FlagMatrix[I][J+nh1]==0:
                    Pglob[I][J+nh1]+=sign*Ploc[i][j+nBern]
                    #Pglob[I][J+nh1]+=Ploc[i][j+nBern]
                    FlagMatrix[I][J+nh1]=1
                    #print("True case")
                    #print("i j I J",i,j+nBern,I,J+nh1)
                else:
                    flag= (abs(Pglob[I][J+nh1] - sign*Ploc[i][j+nBern])<tol)
                    #flag= (abs(Pglob[I][J+nh1] - Ploc[i][j+nBern])<tol)
                    if flag==False:
                        Bigflag=False
                        print("False case")
                        print("i j I J",i,j+nBern,I,J+nh1)
                        print("Pglob[I][J] ",Pglob[I][J+nh1])
                        print("sign*Ploc[i][j] ",sign*Ploc[i][j+nBern])
                        #print("Ploc[i][j] ",Ploc[i][j+nBern])
                    #else:
                        #print("True case")
                        #print("i j I J",i,j+nBern,I,J+nh1)
                    
    #return Pglob,FlagMatrix,Bigflag
    #return Bigflag
    return Pglob


def Glob_Proj0(r,k):
    #input:
        # k : integer s.t the area of triangles does not exceed 1/2**k
        # r : order of approximation
    #Output:
        # Pglob: The matrix of projection of vectorial global H1-Bernstien basis into H curl global vectorial basis
    mesh_points,mesh_tris,mesh_edges,tris_edges=mesh(k)
    nvertices=len(mesh_points)  
    nedges= len(mesh_edges)
    ntris = len(mesh_tris)
    
    nh1 = nbr_globDof_H1(nvertices,nedges,ntris,r)
    nhcurl = nbr_globDof(nedges,ntris,r)
    nBern=(r+1)*(r+2)//2
    nlocdof=r*(r+2)
    
    Pglob=np.zeros((nhcurl,2*nh1))
    FlagMatrix=np.zeros((nhcurl,2*nh1))
    tol=1e-12
    flag=True
    Bigflag=True
    for ti in range(ntris):
        t=mesh_tris[ti]
        # vertex of the triangle/elem t
        v0=mesh_points[t[0]]
        v1=mesh_points[t[1]]
        v2=mesh_points[t[2]]
        ##liste of vertices's coordinates
        T=[v0[0],v0[1],v1[0],v1[1], v2[0],v2[1]] 
        Ploc=Projection(t,T,r)
        for i in range(nlocdof):
            for j in range(nBern):
                I,sign=local_to_global(nedges,t,tris_edges[ti], ti,i,r)
                J=local_to_global_H1(nvertices, nedges, t , tris_edges[ti], ti,j,r)
                if FlagMatrix[I][J]==0:
                    Pglob[I][J]+=sign*Ploc[i][j]
                    FlagMatrix[I][J]=1

                else:
                    flag= (abs(Pglob[I][J] -sign*Ploc[i][j])<tol) 
                    #flag= ( abs(Pglob[I][J] -Ploc[i][j])<tol ) 
                    if flag==False:
                        Bigflag=False
                        print("False case")
                        print("i j I J",i,j,I,J)
                        print("Pglob[I][J] ",Pglob[I][J])
                        print("sign*Ploc[i][j] ",sign*Ploc[i][j])
                        #print("Ploc[i][j] ",Ploc[i][j])
                    #else:
                        #print("True case")
                        #print("i j I J",i,j,I,J)
        for i in range(nlocdof):
            for j in range(nBern):
                I,sign=local_to_global(nedges,t,tris_edges[ti], ti,i,r)
                J=local_to_global_H1(nvertices, nedges, t , tris_edges[ti], ti,j,r)
                if FlagMatrix[I][J+nh1]==0:
                    Pglob[I][J+nh1]+=sign*Ploc[i][j+nBern]
                    #Pglob[I][J+nh1]+=Ploc[i][j+nBern]
                    FlagMatrix[I][J+nh1]=1
                    #print("True case")
                    #print("i j I J",i,j+nBern,I,J+nh1)
                else:
                    flag= (abs(Pglob[I][J+nh1] - sign*Ploc[i][j+nBern])<tol)
                    #flag= (abs(Pglob[I][J+nh1] - Ploc[i][j+nBern])<tol)
                    if flag==False:
                        Bigflag=False
                        print("False case")
                        print("i j I J",i,j+nBern,I,J+nh1)
                        print("Pglob[I][J] ",Pglob[I][J+nh1])
                        print("sign*Ploc[i][j] ",sign*Ploc[i][j+nBern])
                        #print("Ploc[i][j] ",Ploc[i][j+nBern])
                    #else:
                        #print("True case")
                        #print("i j I J",i,j+nBern,I,J+nh1)
                    
    # Taking into account boundary conditions
    
    J=IndexToDelete_H1(mesh_edges,mesh_points,r)
    o=len(J)
    for i in range(o):
        x=J[i]
        J.append(x+nh1)
    I=IndexToDelete(mesh_edges,mesh_points,r)

    Pglob=np.delete(Pglob, I,0)
    Pglob=np.delete(Pglob, J,1)

    return Pglob


'''f = open("Projection_test.txt", "a")
f.close()

f = open("Projection_test.txt", "a")
f.write("\n")
f.write("####    tolerance is set to e-12  ##### \n")
f.close()

for r in range(8,9):
    for k in range(1,5):
        if Glob_Proj(r,k):
            f = open("Projection_test.txt", "a")
            f.write("r="+str(r)+" and k="+str(k)+" valid \n")
            f.close()
        else:
            f = open("Projection_test.txt", "a")
            f.write("r="+str(r)+" and k="+str(k)+" not valid \n")
            f.close()'''
        
                
                