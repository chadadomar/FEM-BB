# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 00:38:46 2023

Load vector for 2d curl 

@author: Omar.CHADAD
"""

from Bmoment import Moment2D
import numpy as np

#auxiliar functions
def indexes2D(n):
    return [(i,j, n-(i+j)) for i in range(n,-1, -1) for j in range(n-i, -1, -1)]

def getIndex2D(n,t):
    #n : polynomial order
    #t : index of the domain point (also polynomial index)
    (i,j,k)=t
    return int((n-i)*(n-i+1)//2+n-i-j)  

def cross2D(u,v):
    return u[0]*v[1]-u[1]*v[0]

def AirT2D(L):
    [x1,y1,x2,y2,x3,y3]=L
    a=np.sqrt((x1-x2)**2+(y1-y2)**2)
    b=np.sqrt((x1-x3)**2+(y1-y3)**2)
    c=np.sqrt((x3-x2)**2+(y3-y2)**2)
    p=(a+b+c)/2
    return np.sqrt(p*(p - a) * (p - b) * (p - c))


# vertices are orderd counter-clockwise
# Find orientation of 3 points
def orientation(p1, p2, p3):
	
	# to find the orientation of 
	# an ordered triplet (p1,p2,p3)
	val = (float(p2[1] - p1[1]) * (p3[0] - p2[0])) - (float(p2[0] - p1[0]) * (p3[1] - p2[1]))
	if val >= 0:
		# Clockwise orientation or collinear
		return True
	else: 
		# Counterclockwise orientation
		return False




def grad2D(L):
    [x1,y1,x2,y2,x3,y3]=L
    p1=[x1,y1]
    p2=[x2,y2]
    p3=[x3,y3]
    '''if orientation(p1,p2,p3):
        raise Exception('Triangle vertices not oriented counterclock-wise')'''
    T=AirT2D(L)
    
    v1=np.array([y3-y2,x2-x3])
    if cross2D([x3-x2,y3-y2],v1)<0:
        v1=-v1
    v2=np.array([y3-y1,x1-x3])
    if cross2D([x1-x3,y1-y3],v2)<0:
        v2=-v2
    v3=np.array([y1-y2,x2-x1])
    if cross2D([x2-x1,y2-y1],v3)<0:
        v3=-v3
    
    v1=v1/T/2
    v2=v2/T/2
    v3=v3/T/2
    return np.array([v1,v2,v3])


"""
Moment2D(L,f,n,q)
L: vertices of triangle ordere in counter-clock wise
f: bivariate scalar function
n: order of Bernstein
q: number of quadrature points
"""

def load2d(F,L,r):
    
    p=r-1
    ndof=r*(r+2)
    load=np.zeros(ndof)

    G=grad2D(L)
    
    index=indexes2D(r)
    #print("indexes 2D are:",index)
    nBern=len(index)
    In=index
    In.remove((r,0,0))
    In.remove((0,r,0))
    In.remove((0,0,r))
    nBerngrad=nBern-3
    #print("liste of index for gradient functions", In)
    
    
    def f1(x,y):
        return F(x,y)[0]
    def f2(x,y):
        return F(x,y)[1]
    
    Mp1=Moment2D(L,f1,p,r+1)
    Mp2=Moment2D(L,f2,p,r+1)
    
    for i in range(nBerngrad):
        for k in range(3):
            alpha=list(In[i])
            if alpha[k]==0:
                continue
            else:
                alpha[k]-=1
                beta=tuple(alpha)
                j=getIndex2D(p,beta)
                load[i]+=Mp1[j]*G[k][0]
                load[i]+=Mp2[j]*G[k][1]
        load[i]*=r

    Mr1=Moment2D(L,f1,r,r+1)
    Mr2=Moment2D(L,f2,r,r+1)
    
    Gamma=indexes2D(p)
    Gamma.pop()
    nGamma=len(Gamma)
    #print("liste of index for bubble functions", Gamma)

    for i in range(nGamma):
        alpha=list(Gamma[i])
        #print("first assignement alpha", alpha)
        for k in range(3):
            quantity1=alpha[(k-1)%3] *G[(k+1)%3][0] - alpha[(k+1)%3]*G[(k-1)%3][0]
            quantity1*=(alpha[k]+1)

            quantity2=alpha[(k-1)%3]*G[(k+1)%3][1] - alpha[(k+1)%3]*G[(k-1)%3][1]
            quantity2*=(alpha[k]+1)

            zeta=alpha.copy()
            zeta[k]+=1
            #print("alpha & zeta",alpha,zeta)
            beta=tuple(zeta)
            j=getIndex2D(r,beta)
            quantity1*=Mr1[j]
            quantity2*=Mr2[j]

            load[nBerngrad+i]+=quantity1+quantity2

    M1=Moment2D(L,f1,1,r+1)
    M2=Moment2D(L,f2,1,r+1)

    for i in range(3):
        for r in [-1,1]:
            e=[0,0,0]
            e[(i+r)%3]=1
            beta=tuple(e)
            j=getIndex2D(1,beta)
            load[nBerngrad+nGamma+i]+=r*( M1[j]* G[(i-r)%3][0] +  M2[j]* G[(i-r)%3][1])
    return load
    
    