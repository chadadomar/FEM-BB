#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 17:35:11 2022

@author: omarch
"""
from Assembly import *
import numpy as np
import meshpy.triangle as triangle
from decimal import *
import matplotlib.pyplot as plt
from Quadratur_over_triangle import *
from Evaluation_curl_BBform import *
from scipy.sparse.linalg import cg, gmres, bicgstab, minres
from tabulate import tabulate
sprint = lambda x: '{:.2e}'.format(x)
np.set_printoptions(precision=5)


def Eval_BB(L,p,C,x,y):
    ndof=(p+2)*(p+1)//2
    I=indexes2D(p)
    res=0
    for i in rnage(ndof):
        alpha=I[i]
        res+=Bern(L,alpha,p,x,y)
    return res


def indexes2D(n):
    return [(i,j, n-(i+j)) for i in range(n,-1, -1) for j in range(n-i, -1, -1)]


# Generating meshes where the area of triangles are less than 1/2^k
def mesh(k):     
    #Vertices of the domaain
    #points = [(0,0), (1, 0),  (1, 1),  (0,1), (0,0)]
    points = [(0,0), (1, 0),  (1, 1),  (0,1)]
    
    #To creat edges
    def round_trip_connect(start, end):
        result = []
        for i in range(start, end):
            result.append((i, i + 1))
        result.append((end, start))
        return result
    
    info = triangle.MeshInfo()
    info.set_points(points)
    info.set_facets(round_trip_connect(0, len(points) - 1))

    #mesh = triangle.build(info, refinement_func=needs_refinement)
    mesh = triangle.build(info,max_volume=1/(2**k),)
    
    mesh_points = np.array(mesh.points)
    mesh_tris = np.array(mesh.elements)
    number_tris=len(mesh_tris)

    edges1=mesh_tris[:,[1,2]]
    edges2=mesh_tris[:,[2,0]]
    edges3=mesh_tris[:,[0,1]]

    v=np.zeros((number_tris*3,2),dtype=(int))
    v[::3]=edges1
    v[1::3]=edges2
    v[2::3]=edges3
    v.sort(axis=1)
    mesh_edges,Q,tris_edges=np.unique(v, axis=0, return_index=True, return_inverse=True)
    tris_edges=np.reshape(tris_edges,(number_tris,3))
    
    #for i in range(number_tris):
        #print(mesh_tris[i],mesh_edges[tris_edges[i]])

    '''plt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_tris)
    for i in range(len(mesh_points)):
      plt.text(mesh_points[i][0], mesh_points[i][1], str(i), fontsize=13 )
    plt.show()'''

    return mesh_points,mesh_tris,mesh_edges,tris_edges

def getIndex2D(n,t):
    #n : polynomial order
    #t : index of the domain point (also polynomial index)
    (i,j,k)=t
    return int((n-i)*(n-i+1)//2+n-i-j)  

def nbr_glob_Dof(nvertices,nedges,ntris,p):
    #Input:
        # nvertices: total number of nodes
        # nedges   : total number of edges
        # ntris    : total number of triangles
        # p        : polynomial order
    #Output:
        # total number of golable functions       
    ndof=(p+2)*(p+1)//2
    nglob_tris= ndof - 3*p
    s=nvertices
    s+=nedges*(p-1)  # number of global functions on edge
    s+=ntris* nglob_tris # number of global functions : interior gradient Bernstein , bubble gamma 
    return s

def local_to_globalH1(nvertices, nedges, node_tris , edge_tris, tris_ind,local_ind,p):
    # input:
        # nvertices: total number of nodes
        # nedges   : total number of edges
        # node_tris: global indices of the triangl's nodes
        # edge_tris: global indices of the triangl's edges
        # tris_ind : number of the triangle with respect to the mesh
        # local_ind: index of local basis function
        # p        : order of FE
    #Output:
        # global number of the correspondant basis function:
            # first global functions are those associated with vertices with same order
            # Second are edge bernstein:
                # For an edge, first function is the one corresponding to the point with lower index, and so on
            # Third are interior  bernstein:
                # Ordered in lexicographical way
    T=node_tris
    E=edge_tris
    I=indexes2D(p)
    ndof=(p+2)*(p+1)//2
    nglob_tris= ndof - 3*p # number of interior global functions per triangle
    
    if local_ind <0 or local_ind >=ndof:
        raise Exception("local inex of basis function is not valid")   
    elif local_ind==0 :
        global_ind=T[0]
    elif local_ind== ndof-1-p:
        global_ind=T[1]
    elif local_ind==ndof-1:
        global_ind=T[2]
    else:
        alpha=I[local_ind]
        if alpha[0]==0:
            # edge bernstein basis of the edge [T[1],T[2]]
            if T[1] < T[2]:
                global_ind= nvertices + E[0]*(p-1) + p-alpha[1]-1
            else:
                global_ind= nvertices + E[0]*(p-1) + alpha[1]-1
        elif alpha[1]==0:
            # edge bernstein basis of the edge [T[0],T[2]]
            if T[0]<T[2]:
                global_ind=nvertices+E[1]*(p-1) + p-alpha[0]-1
            else:
                global_ind=nvertices +E[1]*(p-1) + alpha[0]-1
        elif alpha[2]==0:
            # edge bernstein basis of the edge [T[0],T[1]]
            if T[0]<T[1]:
                global_ind=nvertices+E[2]*(p-1) + p-alpha[0]-1
            else:
                global_ind=nvertices+E[2]*(p-1) + alpha[0]-1
        else:
            beta=(alpha[0]-1,alpha[1]-1,alpha[2]-1)
            global_ind= nvertices + nedges*(p-1) + tris_ind * nglob_tris + getIndex2D(p-3, beta)     
    return global_ind


def domainp(n,k):
    #input:
        # n polynomial order
    # Output:
        # tuple : P, newT 
        ## P : 2d array xy-coordinates of all domain points P[i]=[x_i , y_i]
        ## newT : 2d array indices(in P) of domaines points orderes lexicographically
    P,T=mesh(k)[0],mesh(k)[1]
    nt=len(T)
    nv=len(P)
    plt.triplot(P[:, 0], P[:, 1], T)
    I=indexes2D(n)
    w=(n+2)*(n+1)//2
    newT=np.zeros((nt,w),dtype=np.uint64)
    for i in range(nt):
        t=T[i]
        v0=P[t[0]]
        v1=P[t[1]]
        v2=P[t[2]]
        newT[i][0]=t[0]
        newT[i][-1]=t[2]
        newT[i][w-1-n]=t[1]
        for k in range(w):
            ind=I[k]
            #if ind.count(0)<2:
            x=(ind[0]*v0[0]+ind[1]*v1[0]+ind[2]*v2[0])/n
            y=(ind[0]*v0[1]+ind[1]*v1[1]+ind[2]*v2[1])/n
            locatexyinP=np.where((P==[x,y]).all(axis=1))[0]
            if len(locatexyinP)==0:
                #print("len is ",len(locatexyinP))
                P=np.vstack((P,[[x,y]]))
                #print("nv is ",nv)
                newT[i][k]=nv
                nv+=1
            else:
                #print("len is ",len(locatexyinP))
                #print("location of P_x,y is ",locatexyinP[0])
                newT[i][k]=locatexyinP[0]
                
    # For ploting purposes
    lesx=[x[0] for x in P]
    lesy=[x[1] for x in P]
    # Local indices in the element q
    q=0
    elem=newT[q]
    print("T0 ",T[q]) # T contain only the sommet of the triangle
    print("elem ", elem) # elem contain all domain point indices
    newx=[P[int(i)][0] for i in elem]
    newy=[P[int(i)][1] for i in elem]
    #print("lesx ",lesx)
    #print("lesy ",lesy)
    for i in range(w):
        x=newx[i]
        y=newy[i]
        #print(i,"(", x,";",y,")")
        #plt.text(x, y, str(i), fontsize=20)
          
    for i in range(len(P)):
        if Onboundary(P[i]):
            plt.text(P[i][0], P[i][1], str(i), fontsize=13, color='red')
        else:
            plt.text(P[i][0], P[i][1], str(i), fontsize=13 )
            
    plt.scatter(lesx,lesy, color="red")
    plt.title("Domain points for n="+str(n))
    #plt.scatter(newx,newy, color="green")
    #print("points",P)
    plt.show()
    
    return P,newT



#check if a point is on the boundary
def Onboundary(P):
    x=P[0]
    y=P[1]
    if x*y*(1-x)*(1-y)==0:
        return True
    else:
        return False

#check if an edge is on the boundary of the unit square [0,1]^2
def EdgeOnboundary(v,u):
    flag=False
    if v[0]==u[0]:
        if v[0]==0 or v[0]==1:
            flag=True
    elif v[1]==u[1]:
        if v[1]==0 or v[1]==1:
            flag=True
    return flag

# Collect indices of globale functions non vanishing on boundary of [0,1]^2
def IndexToDelete(mesh_edges,mesh_points,p):
    # retrun sorted liste of indices of global functions non vanishing on boundary of [0,1]^2
    I=[]
    nedges=len(mesh_edges)
    nvertices=  len(mesh_points)
    for i in range(nvertices):
        if Onboundary(mesh_points[i]):
            I.append(i)
    for i in range(nedges):
        E=mesh_edges[i]
        p1=mesh_points[E[0]]
        p2=mesh_points[E[1]]
        if EdgeOnboundary(p1,p2):
            for j in range(p-1):
                ind=nvertices + i*(p-1)+j
                I.append(ind)
    return I

def reconstruct(X,I):
    # creat new vector newX:
        # lenth = sum of lengths of X and I
        # newX[i]=0 for all i in I
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

# Test L2 projection

def f(x,y):
    return 2*(np.pi**2)*np.sin(np.pi*x) * np.sin(np.pi*y)

def u(x,y):
    return np.sin(np.pi*y) * np.sin(np.pi*x)

def L2projection(p,k):
    mesh_points,mesh_tris,mesh_edges,tris_edges=mesh(k)
    nvertices=len(mesh_points)          # number of domaine points  
    nedges= len(mesh_edges)
    ntris = len(mesh_tris)
    ndof=nbr_glob_Dof(nvertices,nedges,ntris,p)
    M=np.zeros((ndof,ndof))             # Golbal mass matrix
    B=np.zeros(ndof)                    # Global load vector    
    w=(p+2)*(p+1)//2
    
    for ti in range(ntris):
        t=mesh_tris[ti]
        # vertex of the triangle/elem t
        v0=mesh_points[t[0]]
        v1=mesh_points[t[1]]
        v2=mesh_points[t[2]]
        ##liste of vertices's coordinates
        Trig=[v0[0],v0[1],v1[0],v1[1], v2[0],v2[1]] 
        Me=MassMat2D(Trig,lambda x,y:1,p) #local stifness matrix
        Be=Moment2D(Trig, f, p) #local load vector
        for i in range(w):
            I=local_to_globalH1(nvertices, nedges, t, tris_edges[ti], ti, i, p)
            B[I]+=Be[i]
            for j in range(w):
                J=local_to_globalH1(nvertices, nedges, t, tris_edges[ti], ti, j, p)
                M[I][J]+=Me[i][j]                   
    
    X=np.linalg.solve(M,B)
    error=0
    for ti in range(ntris):
        t=mesh_tris[ti]
        # vertex of the triangle/elem t
        v0=mesh_points[t[0]]
        v1=mesh_points[t[1]]
        v2=mesh_points[t[2]]
        ##liste of vertices's coordinates
        Trig=[v0[0],v0[1],v1[0],v1[1], v2[0],v2[1]] 
        def func(x,y):
            lam=BarCord2d(Trig,x,y)
            BB=[]
            for j in range(w):
                J=local_to_globalH1(nvertices, nedges, t,  tris_edges[ti], ti, j, p)
                BB.append(X[J])
            return (f(x,y)-deCasteljau2D(lam,BB,p))**2
        error+=quad(Trig, func, p+1)
    error=np.sqrt(error)
    print("{:.2e}".format(error))

#2D 
## Poisson equation  -div( grad u)=2(x(1-x)+y(1-y)), u=0 on the boundary of the 
## reference square
# matrix in the variational formula
def A(x,y):
    return np.array([[1,0],[1,0]])

#def v(x,y):
    #return np.sin(np.pi*x)*np.sin(np.pi*y)

def f2(x,y):
    return 2*(x*(1-x)+y*(1-y))
def u2(x,y):
    return x*(1-x)*y*(1-y)

def sol_poisson_2D(p,k):
    mesh_points,mesh_tris,mesh_edges,tris_edges=mesh(k)
    nvertices=len(mesh_points) # number of domaine points  
    nedges= len(mesh_edges)
    ntris = len(mesh_tris)
    ndof=nbr_glob_Dof(nvertices,nedges,ntris,p)
    S=np.zeros((ndof,ndof))             # Golbal stiffness matrix
    B=np.zeros(ndof)                    # Global load vector    
    w=(p+2)*(p+1)//2

    for ti in range(ntris):
        t=mesh_tris[ti]
        # vertex of the triangle/elem t
        v0=mesh_points[t[0]]
        v1=mesh_points[t[1]]
        v2=mesh_points[t[2]]
        ##liste of vertices's coordinates
        Trig=[v0[0],v0[1],v1[0],v1[1], v2[0],v2[1]] 
        Se=cst_StiffMat_2D(Trig,np.eye(2), p) #local stifness matrix
        Be=Moment2D(Trig, f , p) #local load vector
        #Be=Moment2D(Trig, lambda x,y:0 , p)
        for i in range(w):
            I=local_to_globalH1(nvertices, nedges, t, tris_edges[ti], ti, i, p)
            B[I]+=Be[i]
            for j in range(w):
                J=local_to_globalH1(nvertices, nedges, t, tris_edges[ti], ti, j, p)
                S[I][J]+=Se[i][j]   
                    
    Bound=IndexToDelete(mesh_edges, mesh_points, p) 
    S=np.delete(S,Bound,0)
    S=np.delete(S,Bound,1)
    B=np.delete(B,Bound,0)
    X=np.linalg.solve(S,B)
    C=reconstruct(X, Bound)
    error=0
    for ti in range(ntris):
        t=mesh_tris[ti]
        v0=mesh_points[t[0]]
        v1=mesh_points[t[1]]
        v2=mesh_points[t[2]]
        Trig=[v0[0],v0[1],v1[0],v1[1], v2[0],v2[1]] 
        def func(x,y):
            lam=BarCord2d(Trig,x,y)
            BB=[]
            for j in range(w):
                J=local_to_globalH1(nvertices, nedges, t,  tris_edges[ti], ti, j, p)
                BB.append(C[J])
            return (u(x,y)-deCasteljau2D(lam,BB,p))**2  
        partial=quad(Trig, func, p)
        error+=partial
    error=np.sqrt(error)
    return error

'''ps=(2,3,4,5,6,7)
ks=(1,2,3,4,5,6,7,8)

headers = ['grid/degree p']

for p in ps:
    headers.append(str(p))

# add table rows
rows = []
for k in ks:
    ncell = str(k)
    row = [ncell]
    for p in ps:
        value = sol_poisson_2D(p,k)
        v = "{:.2e}".format(value) 
        if isinstance(value, str):
            v = value
        elif isinstance(value, int):
            v = '$'+str(value) +'$'
        else:
            v =  '$'+sprint(value)+'$' 
        row.append(v)
    rows.append(row)

table = tabulate(rows, headers=headers,tablefmt ='fancy_grid')
f = open("results after correction.txt", "w")
f.write(str(table))   
f.close()'''