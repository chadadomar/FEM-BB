#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 17:35:11 2022

@author: omarch
"""
#from Assembly import StiffMat2D , Moment2D , indexes2D, BarCord2d
import meshpy.triangle as triangle
import numpy as np
from decimal import *
import matplotlib.pyplot as plt
import quadpy


# Tests:
    
## change font size in plot:
plt.rcParams.update({'font.size': 20})


#2D 
## Poisson equation  -div( grad u)=2(x(1-x)+y(1-y)), u=0 on the boundary of the 
## reference square

def indexes2D(n):
    return [(i,j, n-(i+j)) for i in range(n,-1, -1) for j in range(n-i, -1, -1)]


## Mesh of the reference triangle
def mesh():
    
    #Vertices of the reference triangle
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

        dist_center = np.sqrt((bary_x - 1) ** 2 + (bary_y - 1) ** 2)
        max_area = np.fabs(0.05 * (dist_center - 0.5)) + 0.01
        return area > max_area

    info = triangle.MeshInfo()
    info.set_points(points)
    info.set_facets(round_trip_connect(0, len(points) - 1))

    mesh = triangle.build(info, refinement_func=needs_refinement)

    #mesh.write_neu(open("nico.neu", "w"))
    #triangle.write_gnuplot_mesh("triangles.dat", mesh)
    
    mesh_points = np.array(mesh.points)
    mesh_tris = np.array(mesh.elements)
    
    #print("mesh points", mesh_points)
    #print("mesh elements", mesh_tris)


    #plt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_tris)
    #plt.show()
    
    return mesh_points,mesh_tris


def domainp(n):
    #input:
        # n polynomial order
    # Output:
        # tuple : P, newT 
        ## P : 2d array xy-coordinates of all domain points P[i]=[x_i , y_i]
        ## newT : 2d array indices(in P) of domaines points orderes lexicographically
    P,T=mesh()
    nt=len(T)
    nv=len(P)
    #plt.triplot(P[:, 0], P[:, 1], T)
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
    lesx=[x[0] for x in P]
    lesy=[x[1] for x in P]
    '''
    # Local indices in the element q
    q=3
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
        print(i,"(", x,";",y,")")
        #plt.text(x, y, str(i), fontsize=20)'''
        
    
    '''for i in range(len(P)):
        if Onboundary(P[i]):
            plt.text(P[i][0], P[i][1], str(i), fontsize=13, color='red')
        else:
            plt.text(P[i][0], P[i][1], str(i), fontsize=13 )'''
            
    #plt.scatter(lesx,lesy, color="red")
    #plt.title("Domain points for n="+str(n))
    #plt.scatter(newx,newy, color="green")
    #print("points",P)
    #plt.show()
    
    return P,newT

# matrix in the variational formula
def A(x,y):
    return np.array([[1,0],[1,0]])



#check if a point is on the boundary
def Onboundary(P):
    x=P[0]
    y=P[1]
    if x*y*(1-x)*(1-y)==0:
        return True
    else:
        return False
    
def locToglob(n,ind_loc,ind_elem):
    w=(n+2)*(n+1)//2
    if ind_loc > w-1 or ind_loc<0:
        print("the local index is out of range")
    else:
        P,T=domainp(n)
        t=T[ind_elem]  # Triangle containig the point
        I=t[ind_loc]   # Golbal inex of the point 
        
        # Visual verification
        '''#sommet of the traingle are:
        v0=P[t[0]]
        v1=P[t[w-1-n]]
        v2=P[t[-1]]
        #plt.plot([v0[0],v1[0],v2[0],v0[0]],[v0[1],v1[1],v2[1],v0[1]],color="green")
        (t0,t1,t2)=indexes2D(n)[ind_loc]
        resP=[(t0*v0[0]+t1*v1[0]+t2*v2[0])/n,(t0*v0[1]+t1*v1[1]+t2*v2[1])/n]
        plt.text(resP[0],resP[1],"He",color="green")
        plt.text(P[I][0],P[I][1],"Here",color="red")
        plt.show() '''
        
        return I

# Return The BB-vector of the FEM-solution
def sol2D(n):
    P,T=domainp(n)
    #print("P and its lenth is ",len(P),P)
    #print("T and its lenth is ",len(T),T)
    nv=len(P) # number of domaine points
    #print("number of domaine points :",nv)
    #print("number of elements :" ,len(T))

    K=np.zeros((nv,nv)) # Golbal stiffness matrix
    B=np.zeros(nv)      # Global load vector
    
    '''d=dict()
    for i in range(nv):
        x=Decimal(P[i][0])
        y=Decimal(P[i][1])
        d[(x,y)]=i
    
    bool= set(d.keys()) == set([(P[i][0],P[i][1]) for i in range(nv)])
    print(bool)
    print("d ",d)'''
    
    
    w=(n+2)*(n+1)//2  # numbre of domain points per element
    #Ind=indexes2D(n)
    
    for ti in range(len(T)):
        t=T[ti]
        
        # vertex of the triangle/elem t
        v0=P[t[0]]
        v1=P[t[w-1-n]]
        v2=P[t[-1]]

        ##liste of vertices's coordinates
        Trig=[v0[0],v0[1],v1[0],v1[1], v2[0],v2[1]] 
        #print("triang number ",ti," is ", Trig)
        
        Ke=StiffMat2D(Trig, A, n) #local stifness matrix
        Be=Moment2D(Trig, lambda x,y:2*(x*(1-x)+y*(1-y)), n) #local load vector
        #print("ke ",Ke)
        #print("Be ",Be)
        
        for i in range(w):
            for j in range(w):
                I=t[i] # global index of i
                J=t[j] # global index of j
                K[I][J]+=Ke[i][j]
                
        for i in range(w):
            I=t[i] # global index of i
            B[I]+=Be[i]
    
    #print("K ",len(K),"\n",K)
    #print("B n",len(B),"\n",B) 
       
    Bound=[]
    for i in range(nv-1,-1,-1):
        if Onboundary(P[i]):
            K=np.delete(K,i,axis=0)
            K=np.delete(K,i,axis=1)
            B=np.delete(B,i)
            Bound.append(i)
    
    X=np.linalg.solve(K,B)
    #print("boundary Points (size: ", len(Bound)," ) \n", Bound)
    #print("BB without boundary coeff vect (size:",len(X)," ) \n", X)
    
    #construct full BB coeff vector:
    BB=np.ones(nv)
    for i in Bound:
        BB[i]=0
    k=0
    for i in range(nv):
        if BB[i]==1.:
            BB[i]=X[k]
            k+=1
    #print(k)
    return BB
        


## The functions below need modification


def plotpoisson2d(n,m):
    C=sol2D(n)
    lesx=np.linspace(0,1,m)
    lesy=np.linspace(0,1,m)
    X,Y=np.meshgrid(lesx,lesy)
    Z=np.zeros((m,m))
    T=np.zeros((m,m))
    E=np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            x=X[i][j]
            y=Y[i][j]
            if x+y<=1:
                Z[i][j]+=deCasteljau2D((1-x-y,x,y),C,n)
                T[i][j]+=x*y*(x+y-1)
                E[i][j]+=abs(Z[i][j]-T[i][j])
    fig = plt.figure(figsize =(14, 9))
    ax = plt.axes(projection='3d')
    #ax.plot_surface(X, Y, Z)
    surf=ax.plot_surface(X, Y, Z,cmap='viridis')
    #fig.colorbar(surf, ax = ax,shrink = 0.5, aspect = 5)
    #ax.set_title('Erreur u''=2(x+y)')
    plt.show()
        

## Exact sol is given by u(x,y)=x*y*(x-1)*(y-1)

    
def l2_normP2d(n):
    C=sol2D(n)
    P,T=domainp(n)
    w=(n+2)*(n+1)//2  # numbre of domain points per element
    error=0
    for t in T:
        # vertex of the triangle/elem t
        v0=P[t[0]]
        v1=P[t[w-1-n]]
        v2=P[t[-1]]
        triangle=np.array([v0,v1,v2])
        def f(x,y):
            T=[v0[0],v0[1],v1[0],v1[1], v2[0],v2[1]] 
            lam=BarCord2d(T,x,y)
            BB=[C[t[i]] for i in range(w)]
            return (x*y*(x-1)*(y-1)-deCasteljau2D(lam,BB,n))**2
        scheme = quadpy.t2.get_good_scheme(12)
        error+=scheme.integrate(f,triangle)
    error=np.sqrt(error)
    print("{:.2e}".format(error))

