#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:59:40 2024

@author: omarch
"""

import meshpy.triangle as triangle
import matplotlib.pyplot as plt
import numpy as np
import math as m


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

#check if a point is on the boundary of the unit square [0,1]^2
def PointOnboundary(P):
    x=P[0]
    y=P[1]
    if x*y*(1-x)*(1-y)==0:
        return True
    else:
        return False

# check if a point is inside an edge
def is_point_inside_segment(start, end, point):
    x, y = point
    x1, y1 = start
    x2, y2 = end

    # Check if the point is within the bounding box of the segment
    if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
        # Calculate the cross product to determine if the point is on the same line
        cross_product = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        
        # Check if the cross product is close to zero (considering floating-point precision)
        epsilon = 1e-10
        return abs(cross_product) < epsilon

    return False

# Generating meshes
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

    '''def needs_refinement(vertices, area):
        vert_origin, vert_destination, vert_apex = vertices
        bary_x = (vert_origin.x + vert_destination.x + vert_apex.x) / 3
        bary_y = (vert_origin.y + vert_destination.y + vert_apex.y) / 3

        dist_center = np.sqrt((bary_x - 1) ** 2 + (bary_y - 0) ** 2)
        max_area = np.fabs(0.7 * (dist_center - 0.5)) + 0.01
        return area > max_area'''

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

def nbr_globDof(nedges,ntris,r):
    s=0
    s+=nedges*r  # number of global functions on edges: whitney + gradient bernstein
    s+=ntris* ( m.comb(r-1,2) + m.comb(r+1,2)-1) # number of global functions : interior gradient Bernstein , bubble gamma 
    return s

def indexes2D(n):
    return [(i,j, n-(i+j)) for i in range(n,-1, -1) for j in range(n-i, -1, -1)]

def getIndex2D(n,t):
    #n : polynomial order
    #t : index of the domain point (also polynomial index)
    (i,j,k)=t
    return int((n-i)*(n-i+1)//2+n-i-j)  

def local_to_global(nedges,node_tris,edge_tris, tris_ind,local_ind,r):
    # input: 
        # nod_tris: global indices of the triangl's nodes
        # edge_tris: global indices of the triangl's edges
        # tris_ind : number of the triangle with respect to the mesh
        # local_ind: index of local basis function
        # r : order of FE
    #Output:
        # global number of the correspondant basis function:
            # first global functions are those associated with edges:
                # For an edge, first function is Whitney, after gradient bernstein
            # Second are interior gradient bernstein
            # Third are buble functions
        # signe = 1 if its in same direction , -1 if it's in opposite direction
        
    T=node_tris
    E=edge_tris
    
    I=indexes2D(r)
    I.remove((r,0,0))
    I.remove((0,r,0))
    I.remove((0,0,r))
    
    ndof=r*(r+2)
    nBern=(r+2)*(r+1)//2
    nBerngrad=nBern-3
    
    
    nglob_tris=m.comb(r-1,2)+ m.comb(r+1,2)-1 # number of global functions per triangle
    
    if local_ind <0 or local_ind >=ndof:
        raise Exception("local inex of basis function is not valid")
    
    if local_ind < nBerngrad:
        # basis function is gradient Bernstien
        alpha=I[local_ind]
        if alpha[0]==0:
            # edge gradient basis of the edge [T[1],T[2]]
            if T[1] < T[2]:
                global_ind=E[0]*r + r-alpha[1]
            else:
                global_ind=E[0]*r + alpha[1]
            signe=1
        elif alpha[1]==0:
            # edge gradient basis of the edge [T[0],T[2]]
            if T[0]<T[2]:
                global_ind=E[1]*r + r-alpha[0]
            else:
                global_ind=E[1]*r + alpha[0]
            signe=1
        elif alpha[2]==0:
            # edge gradient basis of the edge [T[0],T[1]]
            if T[0]<T[1]:
                global_ind=E[2]*r + r-alpha[0]
            else:
                global_ind=E[2]*r + alpha[0]
            signe=1
        else:
            # interior gradient bernstein
            #print("we are here")
            #print("nglob tris",nglob_tris)
            #global_ind= nedges*r + tris_ind*nglob_tris+local_ind
            beta=(alpha[0]-1,alpha[1]-1,alpha[2]-1)
            global_ind= nedges*r + tris_ind * nglob_tris +getIndex2D(r-3, beta)
            signe=1
            
    elif local_ind < ndof-3:
        # Bubble function
        global_ind= nedges*r + tris_ind * nglob_tris +  m.comb(r-1,2) + local_ind-nBerngrad
        signe=1
        
    else:
        # Whitney edge function
        u=local_ind-ndof
        if u==-3:
            # First Whitney edge function
            global_ind=E[0]*r 
            signe=(T[2]-T[1]) / abs( ( T[2]-T[1] ) )
        elif u==-2:
            # Second whitney edge function
            global_ind=E[1]*r 
            signe=(T[0]-T[2]) / abs( (T[0]-T[2] ) )
        else:
            # Third whitney edge function
            global_ind=E[2]*r 
            signe=(T[1]-T[0]) / abs( (T[1]-T[0] ) )
    return global_ind,signe

def IndexToDelete(mesh_edges,mesh_points,r):
    # retrun sorted liste of indices of global functions non vanishing on boundary of [0,1]^2
    I=[]
    nedges=len(mesh_edges)
    for i in range(nedges):
        E=mesh_edges[i]
        p1=mesh_points[E[0]]
        p2=mesh_points[E[1]]
        if EdgeOnboundary(p1,p2):
            for j in range(r):
                ind=i*r+j
                I.append(ind)
    return I