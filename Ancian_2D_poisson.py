#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 23:08:29 2022

@author: omarch
"""

## Poisson on the refernece rectangle -div(grad u)=2(x(1-x)+y(1-y)) , 
## u=0 on the boundary, the exact sol is u(x,y)=xy(1-x)(1-y)

def sol2D_2(n):
    t0=timeit.default_timer()
    K1=StiffMat2D([0,0,1,0,0,1], A, n)
    B1=Moment2D([0,0,1,0,0,1], lambda x,y:2*(x*(1-x)+y*(1-y)), n)
    K2=StiffMat2D([1,0,1,1,0,1], A, n)
    B2=Moment2D([1,0,1,1,0,1], lambda x,y:2*(x*(1-x)+y*(1-y)), n)
    L=indexes2D(n)
    w=(n+1)*(n+2)//2
    C=[]
    Q=[]
    for i in range(w-1,-1,-1):
        if L[i][0]*L[i][1]*L[i][2]==0:
            K1=np.delete(K1,i,axis=0)
            K1=np.delete(K1,i,axis=1)
            K2=np.delete(K2,i,axis=0)
            K2=np.delete(K2,i,axis=1)
            B1=np.delete(B1,i)
            B2=np.delete(B2,i)
            C.append(i)
        else:
            Q.append(i)
    #print('K ',K)
    #print('B ',B)
    #print(C)
    #print(Q)
    K=K1+K2
    B=B1+B2
    X=np.array(np.linalg.solve(K,B))
    #print(X)
    t1=timeit.default_timer()-t0
    #print("time elapsed ",t1))
    Q.reverse()
    f=len(Q)
    BB=np.zeros(w)
    for i in range(f):
        BB[Q[i]]=X[i]
    return BB

def ploterror2D_2(n,m):
    C=sol2D_2(n)
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
            Z[i][j]+=deCasteljau2D((1-x-y,x,y),C,n)
            T[i][j]+=x*y*(1-x)*(1-y)
            E[i][j]+=abs(Z[i][j]-T[i][j])
    fig=plt.figure(figsize=(6,5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax=fig.add_axes([left, bottom, width, height]) 
    cp = plt.contourf(X, Y, E)
    plt.colorbar(cp)
    ax.set_title('Error of poisson 2D where $n=$'+str(n))
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #fig = plt.figure(figsize =(14, 9))
    #ax = plt.axes(projection='3d')
    #ax.plot_surface(X, Y, Z)
    #surf=ax.plot_surface(X, Y, E,cmap='viridis')
    #fig.colorbar(surf, ax = ax,shrink = 0.5, aspect = 5)
    #ax.set_title('Erreur u''=2(x+y)')
    plt.show()