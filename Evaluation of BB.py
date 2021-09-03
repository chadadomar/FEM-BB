import numpy as np
import scipy.special



# de Casteljau

## dimension 1
### evaluate p= c_i*B_{i,n} at point t

def deCasteljau1D(t,c):
    n=len(c)
    for i in range(n-1):
        for j in range(n-i):
            c[j]=c[j]*(1-t)+c[j+1]*t
    return c[0]

## dimension 2
### lam: barycentrique cordinate
### C vector of the BB form ordered in lexicographiqe
### l step index

def deCasteljau_step_2D(lam,C,l):
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

def deCasteljau2D(lam,C,n):
    if len(C)>1:
        return deCasteljau2D(lam,deCasteljau_step_2D(lam, C, n),n-1)
    else:
        return C[0]
       
# Sum factorisation technique 2D
## C matrix of the BB form ordered in lexicographiqe
## q: number of Gauss-Jaccobi quadrature point

def Eval(C,n,q):
    [c1,w]=scipy.special.roots_jacobi(q,1,0)
    [c2,w]=scipy.special.roots_jacobi(q,0,0)
    d=max(n+1,q)
    U=np.zeros((d,d))
    
    for a1 in range(n+1):
        for a2 in range(n+1-a1):
            a3=n-a1-a2
            U[a1,a2]=C[a1+a3*(2(a1+a2)+a3+3)/2]
    
    P=np.zeros((d,d))
    for i2 in range(q):
            x=c2[i2]
            s=1-x
            r=x/s
            for a1 in range(n+1):
                w=s**(n-a1)
                for a2 in range(n+1-a1):
                    P[a1,i2]+=w*U[a1,a2]
                    w*=r*(n-a1-a2)/(1+a2)
                    
    F=np.zeros((q,q))
    for i2 in range(q):
        for i1 in range(q):
            x=c1[i1]
            s=1-x
            r=x/s
            w=s**n
            for a1 in range(n+1):
                F[i1,i2]+=w*P[a1,i2]
                w*=r*(n-a1)/(1+a1)
    return F
    
                