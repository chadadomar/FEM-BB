# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 17:13:43 2023

Mass matrix 2D curl

@author: Omar.CHADAD
"""



import numpy as np
import math as m


## Return factorial: pascal triangle of ordre n

def fact(n):
    if n<0:
        raise Exception("Sorry, no numbers below zero")
    if n==0:
        return 1
    else:
        return n*fact(n-1)


## needed for orientation
def ScalarProd(u,v):
    m=len(u)
    n=len(v)
    if n!=m:
        raise Exception("the vectors have different size")
    else:
        w=0
        for i in range(n):
            w+=u[i]*v[i]
        return w

def sumVect(u,v):
    m=len(u)
    n=len(v)
    if n!=m:
        print("the vectors have different size")
    else:
        w=[]
        for i in range(n):
            w.append(int(u[i]+v[i]))
        return w

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
def grad2D(L):
    T=AirT2D(L)
    [x1,y1,x2,y2,x3,y3]=L
    
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
    G=np.array([v1,v2,v3])
    gradMat=np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            gradMat[i][j]+=ScalarProd(G[i],G[j])
    return gradMat

def indexes2D(n):
    return [(i,j, n-(i+j)) for i in range(n,-1, -1) for j in range(n-i, -1, -1)]

def getIndex2D(n,t):
    #n : polynomial order
    #t : index of the domain point (also polynomial index) in lexicographical order
    (i,j,k)=t
    return int((n-i)*(n-i+1)//2+n-i-j)  

def multifact(a,b):
    n=len(a)
    if len(b)!=n:
        raise Exception("Sorry, different size")
    else:
        p=1
        for i in range(n):
            p*=m.comb(a[i],b[i])
        return p


def mass2d(L,r):
    p=r-1
    T=AirT2D(L)
    grad=grad2D(L)
    ndof=r*(r+2)
    
    index=indexes2D(r)
    nBern=len(index)
    In=index
    In.remove((r,0,0))
    In.remove((0,r,0))
    In.remove((0,0,r))
    nBerngrad=nBern-3

    Gamma=indexes2D(p)
    Gamma.pop()
    nGamma=len(Gamma)

    
    M=np.zeros((ndof,ndof))
    
    for i in range(nBerngrad):
        for j in range(nBerngrad):
            coef=2*(r**2)*(fact(p)**2)*T/ fact(2*r)
            for k in range(3):
                alpha=list(In[i])
                if alpha[k]==0:
                    continue
                else:
                    alpha[k]-=1
                    for l in range(3):
                        beta=list(In[j])
                        if beta[l]==0:
                            continue
                        else:
                            beta[l]-=1
                            M[i][j]+= multifact(sumVect(alpha,beta), alpha)* grad[k][l]
            M[i][j]*=coef    
    for i in range(nBerngrad):
        for j in range(nGamma):
            coef=2*(r**2)*(fact(p)**2)*T/ fact(2*r+1)
            for k in range(3):
                alpha=list(In[i])
                if alpha[k]==0:
                    continue
                else:
                    alpha[k]-=1
                    for l in range(3):
                        beta=list(Gamma[j])
                        

                        M[i][j+nBerngrad]+=beta[l-1]*grad[k][((l+1)%3)] 
                        M[i][j+nBerngrad]-=beta[((l+1)%3)]*grad[k][l-1]

                        M[i][j+nBerngrad]*=(beta[l]+1)
                        
                        beta[l]+=1
                        M[i][j+nBerngrad]*=multifact(sumVect(alpha,beta), alpha)
            M[i][j+nBerngrad]*=coef
    
    for a in range(nBerngrad):
        for i in range(3):
            coef=T/m.comb(r+2,2)
            for k in range(3):
                alpha=list(In[a])
                if alpha[k]==0:
                    continue
                else:
                    alpha[k]-=1
                    for r in [-1,1]:
                        e=[0,0,0]
                        e[(i+r)%3]=1
                        M[a][nBerngrad+nGamma+i]+=r*multifact(sumVect(alpha,e), alpha)*grad[k][(i-r)%3]
            M[a][nBerngrad+nGamma+i]*=coef


    for i in range(nGamma):
        for j in range(nGamma):
            coef=T/ ( m.comb(2*r,r) * m.comb(2*r+2,2)  )
            for k in range(3):
                for l in range(3):
                    alpha=list(Gamma[i])
                    beta=list(Gamma[j])
                    M[nBerngrad+i][nBerngrad+j]+=alpha[k-1]*beta[l-1]* grad[(k+1)%3][(l+1)%3]
                    M[nBerngrad+i][nBerngrad+j]+=alpha[k-1]*beta[(l+1)%3]* grad[(k+1)%3][l-1]
                    M[nBerngrad+i][nBerngrad+j]+=alpha[(k+1)%3]*beta[l-1]* grad[k-1][(l+1)%3]
                    M[nBerngrad+i][nBerngrad+j]+=alpha[(k+1)%3]*beta[(k+1)%3] *grad[k-1][l-1]
                    M[nBerngrad+i][nBerngrad+j]*=(alpha[k]+1)*(beta[l]+1)

                    alpha[k]+=1
                    beta[l]+=1
                    M[nBerngrad+i][nBerngrad+j]*=multifact(sumVect(alpha,beta),alpha)
            M[nBerngrad+i][nBerngrad+j]*=coef

    for a in range(nGamma):
        for i in range(3):
            coef=T/ ( (r+1) * m.comb(r+3,2)  )
            for k in range(3):
                for r in [-1,1]:
                    alpha=list(Gamma[a])
                    e=[0,0,0]
                    e[(i+r)%3]=1
                    M[nBerngrad+a][nBerngrad+nGamma+i]+=alpha[k-1]*grad[(k+1)%3][(i-r)%3]
                    M[nBerngrad+a][nBerngrad+nGamma+i]+=alpha[(k+1)%3]*grad[k-1][(i-r)%3]
                    M[nBerngrad+a][nBerngrad+nGamma+i]*=(alpha[k]+1)

                    alpha[k]+=1
                    M[nBerngrad+a][nBerngrad+nGamma+i]*=r*multifact(sumVect(alpha,e), alpha)
            M[nBerngrad+a][nBerngrad+nGamma+i]*=coef

    for i in range(3):
        for j in range(3):
            coef=T/6
            for r in [-1,1]:
                for s in [-1,1]:
                    ei=[0,0,0]
                    ei[(i+r)%3]=1
                    ej=[0,0,0]
                    ej[(j+s)%3]=1
                    
                    E=sumVect(ei,ej)
                    bino=2/( fact(E[0])*fact(E[1])*fact(E[2]) )

                    M[nBerngrad+nGamma+i][nBerngrad+nGamma+j]+=r*s*grad[(i-r)%3][(j-s)%3]/bino

            M[nBerngrad+nGamma+i][nBerngrad+nGamma+j]*=coef
    M[nBerngrad:nBerngrad+nGamma,:nBerngrad]=np.transpose(M[:nBerngrad,nBerngrad:nBerngrad+nGamma])
    M[-3:,:nBerngrad]=np.transpose(M[:nBerngrad,-3:])
    M[-3:,nBerngrad:nBerngrad+nGamma]=np.transpose(M[nBerngrad:nBerngrad+nGamma,-3:])
    return M