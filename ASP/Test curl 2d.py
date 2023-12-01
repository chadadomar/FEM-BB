# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 23:17:43 2023

Test 2d curl finite element

@author: Omar.CHADAD
"""

import numpy as np
from Mass_curl_2d import *
from Stiffness_curl_2d import *
from Loadvector_curl_2d import *
from Evaluation_curl_BBform import *
import meshpy.triangle as triangle
import matplotlib.pyplot as plt
from scipy import integrate

tau=1
psi=np.sqrt(tau)

L=[0,0,1,0,0,1]
r=3

c=- 1/ (tau * (np.exp(-psi*0.5) + np.exp(psi*0.5)) )

#solution for 2d curl elliptic problem, with f=(1,1)
def v1(x,y):
    return c* ( np.exp(-psi*(y-0.5)) + np.exp(psi*(y-0.5)) ) + 1/tau

def v2(x,y):
    return c* ( np.exp(-psi*(x-0.5)) + np.exp(psi*(x-0.5)) ) + 1/tau

def v(x,y):
    return np.array([v1(x,y),v2(x,y)])

# defining second memebre
def f(x,y):
    return np.array([1,1])



def mesh():
    
    #Vertices of the domaain
    points = [(0,0), (1, 0),  (1, 1),  (0,1), (0,0)]
    
    #To creta edges
    def round_trip_connect(start, end):
        result = []
        for i in range(start, end):
            result.append((i, i + 1))
        result.append((end, start))
        return result

    def needs_refinement(vertices, area):
        vert_origin, vert_destination, vert_apex = vertices
        bary_x = (vert_origin.x + vert_destination.x + vert_apex.x) / 3
        bary_y = (vert_origin.y + vert_destination.y + vert_apex.y) / 3

        dist_center = np.sqrt((bary_x - 1) ** 2 + (bary_y - 0) ** 2)
        max_area = np.fabs(0.7 * (dist_center - 0.5)) + 0.01
        return area > max_area

    info = triangle.MeshInfo()
    info.set_points(points)
    info.set_facets(round_trip_connect(0, len(points) - 1))

    #mesh = triangle.build(info, refinement_func=needs_refinement)
    mesh = triangle.build(info,max_volume=10e-2, min_angle=25)
    #mesh.write_neu(open("nico.neu", "w"))
    #triangle.write_gnuplot_mesh("triangles.dat", mesh)
    
    mesh_points = np.array(mesh.points)
    mesh_tris = np.array(mesh.elements)
    #mesh_facets= np.array(mesh.facets)

    #print(mesh_facets)
    #print("mesh points", mesh_points)
    #print("mesh elements", mesh_tris)
    

    plt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_tris)
    plt.show()
    
    return mesh_points,mesh_tris

#check if an edge is on the boundary of the unit square
def EdgeOnboundary(v,u):
    flag=False
    if v[0]==u[0]:
        if v[0]==0 or v[0]==1:
            flag=True
    elif v[1]==u[1]:
        if v[1]==0 or v[1]==1:
            flag=True
    return flag

# Ordre of the finite element
'''
r=2
ndof=r*(r+2)

P,T=mesh()

nt=len(T)
ng=nt*ndof
S=np.zeros((ng,ng))
M=np.zeros((ng,ng))
F=np.zeros(ng)

for i in range(nt):
    tr=T[i]
    tr.sort()
    
    # vertex of the triangle/elem t
    p0=P[tr[0]]
    p1=P[tr[1]]
    p2=P[tr[2]]
    
    ##liste of vertices's coordinates
    L=[p0[0],p0[1],p1[0],p1[1], p2[0],p2[1]] 
    St=Stiff2d(L,r)
    Mt=mass2d(L,r)
    Ft=load2d(f,L,r)
    
    for k in range(ndof):
        F[i*ndof+k]+=Ft[k]
        for l in range(ndof):
            S[i*ndof+k][i*ndof+l]+=St[k][l]
            M[i*ndof+k][i*ndof+l]+=Mt[k][l]

# Removin non vanishing, on boundary, shape functions 


IndexToDelete=[]
IndexTriangle=np.zeros((nt,ndof))
BBform=[ndof]*nt
H=indexes2D(r)
H.remove((r,0,0))
H.remove((0,r,0))
H.remove((0,0,r))    
for i in range(nt):
    tr=T[i]
    tr.sort()
    
    # vertex of the triangle/elem t
    p0=P[tr[0]]
    p1=P[tr[1]]
    p2=P[tr[2]]
    
    if EdgeOnboundary(p0,p1):
        IndexToDelete.append(i*ndof+ndof-1) # edge function w3
        IndexTriangle[i][ndof-1]+=1
        BBform[i]-=1
        for j in range(1,r):
            z=H.index( (j,r-j,0))
            IndexToDelete.append(i*ndof+z)
            IndexTriangle[i][z]+=1
            BBform[i]-=1
            
    if  EdgeOnboundary(p0,p2):
        IndexToDelete.append(i*ndof+ndof-2) # edge function w2
        IndexTriangle[i][ndof-2]+=1
        BBform[i]-=1
        for j in range(1,r):
            z=H.index( (j,0,r-j))
            IndexToDelete.append(i*ndof+z)
            IndexTriangle[i][z]+=1
            BBform[i]-=1
            
    if  EdgeOnboundary(p1,p2):
        IndexToDelete.append(i*ndof+ndof-3) # edge function w1
        IndexTriangle[i][ndof-3]+=1
        BBform[i]-=1
        for j in range(1,r):
            z=H.index( (0,j,r-j))
            IndexToDelete.append(i*ndof+z)
            IndexTriangle[i][z]+=1
            BBform[i]-=1


print("triangles", T)
print("Ponints", P)
print("triangles on boundary", IndexTriangle)
print("Index To delete", IndexToDelete )
Triangles=[T[i] for i in IndexTriangle]


plt.triplot(P[:, 0], P[:, 1], T)
plt.triplot(P[:, 0], P[:, 1], Triangles,'ro-')
plt.show()

S=np.delete(S,IndexToDelete,0)
S=np.delete(S,IndexToDelete,1)
M=np.delete(M,IndexToDelete,0)
M=np.delete(M,IndexToDelete,1)
F=np.delete(F,IndexToDelete)


X=np.linalg.solve(S+tau*M,F)


# Evaluate the L2 error
def dJ( u, v, p1, p2, p3 ):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    dxdu = ( (1-v)*x2 + v*x3 - x1 )
    dxdv = ( u*x3 - u*x2 )
    dydu = ( (1-v)*y2 + v*y3 - y1 )
    dydv = ( u*y3 - u*y2 )
    return np.abs( dxdu*dydv - dxdv*dydu )


def tridblquad( integrand, p1, p2, p3 ):
    
    #Perform double quadtrature integration on triangular domain.
    #Input: function to integrate, points of triangle as tuples.
    #Output: integral and estimated absolute error as a tuple.
    
    x1, y1 = p1 ; x2, y2 = p2 ; x3, y3 = p3
    # transformation to the unit square
    g = lambda u, v, c1, c2, c3: (1-u)*c1 + u*( (1-v)*c2 + v*c3 )
    # transformation for the integrand, 
    # including the Jacobian scaling factor
    def h( u, v ):
        x = g( u, v, x1, x2, x3 )
        y = g( u, v, y1, y2, y3 )
        I = integrand( x, y )
        I *= dJ( u, v, p1, p2, p3 )
        return I
    # perfrom the double integration using quadrature in the transformed space
    integral, error = integrate.dblquad( h, 0, 1, lambda x: 0, lambda x: 1, epsrel=1e-6, epsabs=0 )
    return integral



l2error=0
pos=0
for i in range(nt):
    tr=T[i]
    tr.sort()
    C=np.zeros(ndof)
    # vertex of the triangle/elem t
    p0=P[tr[0]]
    p1=P[tr[1]]
    p2=P[tr[2]]
    L=[p0[0],p0[1],p1[0],p1[1], p2[0],p2[1]]
    for k in range(ndof):
        if  IndexTriangle[i][k]!=1:
            C[k]+=X[pos]
            pos+=1
    #print("C for ",i,C)
    def integrand(x,y):
        return (Eval_curl(L,r,C,x,y)[0]-v1(x,y))**2 +   (Eval_curl(L,r,C,x,y)[1]-v2(x,y))**2
    #x=p0[0]
    #y=p0[1]
    #print( (Eval_curl(L,r,C,x,y)[0]-v1(x,y))**2 +   (Eval_curl(L,r,C,x,y)[1]-v2(x,y))**2  )
    l2error+=tridblquad( integrand, p0, p1, p2 )

print("the L2 norm error is", np.sqrt(l2error))'''
    
    
    
    
