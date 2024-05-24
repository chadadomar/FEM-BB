# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 23:53:03 2023

Evaluation of 2d 

@author: Omar.CHADAD
"""
import numpy as np
import math as m


#auxiliar functions
def indexes2D(n):
    return [(i,j, n-(i+j)) for i in range(n,-1, -1) for j in range(n-i, -1, -1)]

def getIndex2D(n,t):
    #n : polynomial order
    #t : index of the domain point (also polynomial index)
    (i,j,k)=t
    return int((n-i)*(n-i+1)//2+n-i-j)  

def AirT2D(L):
    [x1,y1,x2,y2,x3,y3]=L
    a=np.sqrt((x1-x2)**2+(y1-y2)**2)
    b=np.sqrt((x1-x3)**2+(y1-y3)**2)
    c=np.sqrt((x3-x2)**2+(y3-y2)**2)
    p=(a+b+c)/2
    return np.sqrt(p*(p - a) * (p - b) * (p - c))


def SignedAirT2D(L):
    [x1,y1,x2,y2,x3,y3]=L
    M=np.array([[1,1,1],[x1,x2,x3],[y1,y2,y3]])
    return np.linalg.det(M)

# function of barycentric cooordinate
def lambda1(L,x,y):
    T=SignedAirT2D(L)
    [x1,y1,x2,y2,x3,y3]=L
    return ((x2*y3-y2*x3)-x*(y3-y2)+y*(x3-x2))/(T)

def lambda2(L,x,y):
    T=SignedAirT2D(L)
    [x1,y1,x2,y2,x3,y3]=L
    return ((x3*y1-y3*x1)-x*(y1-y3)+y*(x1-x3))/(T)

def lambda3(L,x,y):
    T=SignedAirT2D(L)
    [x1,y1,x2,y2,x3,y3]=L
    return ((x1*y2-y1*x2)-x*(y2-y1)+y*(x2-x1))/(T)

# Gradient of barycentric coordinate 
# vertices are orderd counter-clockwise
def cross2D(u,v):
    return u[0]*v[1]-u[1]*v[0]

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
    return np.array([v1,v2,v3])

# Whitney edge element of low order
def w1(L,x,y):
    G=grad2D(L)
    return lambda2(L,x,y)*G[2] - lambda3(L, x, y)*G[1]

def w2(L,x,y):
    G=grad2D(L)
    return lambda3(L,x,y)*G[0] - lambda1(L, x, y)*G[2]

def w3(L,x,y):
    G=grad2D(L)
    return lambda1(L,x,y)*G[1] - lambda2(L, x, y)*G[0]

# Gradient function
def Bern(L,alpha,r,x,y):
    if sum(alpha) != r:
        raise Exception("index of bernstein polynomial not valid")
    else:
        res=m.comb(r,alpha[0])* m.comb(r-alpha[0],alpha[1]) * m.comb(r-alpha[0]-alpha[1],alpha[2])
        res*=lambda1(L, x, y)**alpha[0]
        res*=lambda2(L, x, y)**alpha[1]
        res*=lambda3(L, x, y)**alpha[2]
        return res

def Bern2dbis(L,BB,p,x,y):
    I=indexes2D(p)
    ndof=(p+1)*(p+2)//2
    res=0
    for i in range(ndof):
        alpha=I[i]
        res+=BB[i]*Bern(L,alpha,p,x,y)
    return res


def gradBern(L,alpha,r,x,y):
    if sum(alpha) != r:
        raise Exception("index of bernstein polynomial not valid")
    else:
        v=np.array([0.,0.])
        G=grad2D(L)
        if alpha[0]!=0:
            beta=list(alpha)
            beta[0]-=1
            v+= Bern(L,beta,r-1,x,y) * G[0]
        if alpha[1]!=0:
            beta=list(alpha)
            beta[1]-=1
            v+= Bern(L,beta,r-1,x,y) * G[1]
        if alpha[2]!=0:
            beta=list(alpha)
            beta[2]-=1
            v+= Bern(L,beta,r-1,x,y) * G[2]
        return r*v
        
# Buble functions
def Gamma(L,alpha,r,x,y):
    if sum(alpha) != r-1:
        raise Exception("index of bernstein polynomial not valid")
    res=alpha[0]*w1(L,x,y)+alpha[1]*w2(L,x,y)+alpha[2]*w3(L,x,y)
    res*=r * Bern(L, alpha, r-1, x, y)
    return res

# Evaluation of p=C[alpha]*B[alpha]
# C: vector of coefficient in the 2d curl bernstein

def Eval_curl(L,r,C,x,y):
    ndof=r*(r+2)
    if len(C)!= ndof:
        raise Exception("the size of vector of coefficient is not valid")
    else:
        res=np.array([0.,0.])
        
        index=indexes2D(r)
        nBern=len(index)
        In=index
        In.remove((r,0,0))
        In.remove((0,r,0))
        In.remove((0,0,r))
        nBerngrad=nBern-3

        Gammalist=indexes2D(r-1)
        Gammalist.pop()
        nGamma=len(Gammalist)
        
        for i in range(nBerngrad):
            alpha=In[i]
            res+=C[i]*gradBern(L,alpha,r,x,y)
        for i in range(nGamma):
            alpha=Gammalist[i]
            j=nBerngrad+i
            res+=C[j]*Gamma(L,alpha,r,x,y)
        res+= C[-3]*w1(L,x,y)
        res+= C[-2]*w2(L,x,y)
        res+= C[-1]*w3(L,x,y)
        
        return res

def c_star(alpha,L):
    T=AirT2D(L)
    p=sum(alpha)
    c_bar=[]
    
    # k=l 
    eta=list(alpha)
    '''q=(alpha[0]+1)*(alpha[1]+alpha[2])
    q+=(alpha[1]+1)*(alpha[0]+alpha[2])
    q+=(alpha[2]+1)*(alpha[1]+alpha[0])
    q*=((p+1)/(2*T) )'''
    q=((p+1)/(T) )* (alpha[0]*alpha[1]+alpha[1]*alpha[2]+alpha[2]*alpha[0]+p)
    c_bar.append((eta,q))
    
    # k != l
    for k in range(3):
        for l in range(3):
            if alpha[l]==0:
                continue
            if k!=l:
                eta=list(alpha)
                eta[k]+=1
                eta[l]-=1
                q=(-(p+1)/(2*T) ) * (alpha[k]+1) * alpha[l]
                c_bar.append((eta,q))
    return c_bar

def Eval_curlcurl(L,r,C,x,y):
    T=AirT2D(L)
    ndof=r*(r+2)
    if len(C)!= ndof:
        raise Exception("the size of vector of coefficient is not valid")
    else:
        res=0
        
        nBern=(r+2)*(r+1)//2
        nBerngrad=nBern-3

        Gammalist=indexes2D(r-1)
        Gammalist.pop()
        nGamma=len(Gammalist)
        
        for i in range(nGamma):
            alpha=Gammalist[i]
            j=nBerngrad+i
            star=c_star(alpha,L)
            temp=0
            for elem in star:
                eta,q=elem
                temp+=q*Bern(L, eta, r-1, x, y)
            res+=C[j]*temp
            
        # curl whitney contribution
        res+= (C[-3]+ C[-2] + C[-1]) /T
        
        return res
    
    
###################################################################################################


# Get barycentrc coordinates from cartesian cordinate
def BarCord2d(T,x,y):
    #In:
        # T: triangle(non degenerate)
        # x,y : Cartesian cordinate of the point
    #Out:
        # [lambda1,lambda2,lambda3]: list of barycentric coordinate of the point
    [x0,y0,x1,y1,x2,y2]=T
    M=np.array([[x0,x1,x2],[y0,y1,y2],[1,1,1]])
    if np.linalg.det(M)==0:
        print("Error: triangle is degenretaed")
    else:
        return np.linalg.solve(M,np.array([x,y,1]))
    
    
## 2 dimension
### lam: barycentrique cordinate of the point
### C vector of the BB form ordered in lexicographiqe
### l step index

def deCasteljau_step_2D(lam,BB,l):
    C=[x for x in BB]
    i=0
    j=1
    for r in range(1,l+1):
        for k in range(r):
            C[i]=lam[0]*C[i]+lam[1]*C[j]+lam[2]*C[j+1]
            i+=1
            j+=1
        j+=1
    return C[:-(l+1)]

### n degree of BB-polynom

def deCasteljau2D(lam,BB,n):
    C=[x for x in BB]
    if len(C)>1:
        return deCasteljau2D(lam,deCasteljau_step_2D(lam, C, n),n-1)
    else:
        return C[0]
    
def Bern2d(L,BB,n,x,y):
    lam=BarCord2d(L,x,y)
    return  deCasteljau2D(lam,BB,n)