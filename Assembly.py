import numpy as np
#import scipy.special
import timeit
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#np.seterr(all='warn')


# Polynomial order n is limited at 100

#Auxilary functions


##Binomial coeff, limited at 203

Bin=np.load("Binomial coeff.npy")



# Storing points and weights of Gauss-Jacobi quadrature rule
quad0=np.load("jaccobi rule 0.npy",allow_pickle=True)
quad1=np.load("jaccobi rule 1.npy",allow_pickle=True)
quad2=np.load("jaccobi rule 2.npy",allow_pickle=True)
## quad is a dictionnary, keys are quadrature ordre values are weights 
## and nodes, size of quad is 100


## matrice of scalar prod of gradient vectors
def grad1D(L):
    [a,b]=L  # a =< b
    T=abs(b-a)
    x=1/T/T
    s=np.array([[x, -x],[-x,x]])
    return s
    

## needed for orientation
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
    return np.array([v1,v2,v3])

def sumVect(u,v):
    m=len(u)
    n=len(v)
    if n!=m:
        print("the vectors have different size")
    else:
        w=np.zeros(n)
        for i in range(n):
            w[i]=int(u[i]+v[i])
        return w

def Sub(u,v):
    m=len(u)
    n=len(v)
    if n!=m:
        print("the vectors have different size")
    else:
        w=np.zeros(n)
        for i in range(n):
            w[i]=u[i]-v[i]
        return w

def ScalarProd(u,v):
    m=len(u)
    n=len(v)
    if n!=m:
        print("the vectors have different size")
    else:
        w=0
        for i in range(n):
            w+=u[i]*v[i]
        return w

def CrossProd3D(u,v):
    w=np.zeros(3)
    w[0]=u[1]*v[2]-u[2]*v[1]
    w[1]=u[2]*v[0]-u[0]*v[2]
    w[2]=u[0]*v[1]-u[1]*v[0]
    return w
    
def Inter(u,v,w):
    sub1=Sub(u,v)
    sub2=Sub(v,w)
    return 0.5*CrossProd3D(sub1, sub2)

def AirT3D(L):
    [x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4]=L
    u=np.array([x1-x4,y1-y4,z1-z4])
    v=np.array([x2-x4,y2-y4,z2-z4])
    w=np.array([x3-x4,y3-y4,z3-z4])
    cross=CrossProd3D(v, w)
    T=ScalarProd(u, cross)
    return abs(T)/6 

def grad3D(L):
    T=AirT3D(L)
    [x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4]=L
    v1=np.array([x1,y1,z1])
    v2=np.array([x2,y2,z2])
    v3=np.array([x3,y3,z3])
    v4=np.array([x4,y4,z4])
    
    Res1=-Inter(v3,v4,v2)/T/3
    Res2=Inter(v4,v1,v3)/T/3
    Res3=-Inter(v1,v2,v4)/T/3
    Res4=Inter(v2,v3,v1)/T/3
    
    return np.array([Res1,Res2,Res3,Res4])

### 2D domaine points in lexicographic order

def indexes2D(n):
    return [(i,j, n-(i+j)) for i in range(n,-1, -1) for j in range(n-i, -1, -1)]

### 3D domaine points in lexicographic order

def indexes3D(n):
    return [(i,j,k, n-(i+j+k)) for i in range(n,-1, -1) for j in range(n-i, -1, -1) for k in range(n-i-j,-1,-1)]


def getIndex2D(n,t):
    #n : number of domain point (also polynomial order)
    #t : index of the domain point (also polynomial index)
    (i,j,k)=t
    return int((n-i)*(n-i+1)//2+n-i-j)

def getIndex3D(n,t):
    #n : number of domain point (also polynomial order)
    #t : index of the domain point (also polynomial index)
    (i,j,k,l)=t
    if i==n:
        return 0
    else:
        s=n-i
        return int(0.5*((s-1)*s*(2*s-1)/6+3*(s-1)*s/2+2*s) + getIndex2D(s,(j,k,l)))
        
    

# Bernestein-Bézier moment


## One dimentional setting
###  Vector D: neded after Gauss_Jaccobi quadrature rule

D=np.load("Precomp D.npy",allow_pickle=True)

### Moment approximation

def Moment1D(a,b,f,n):
    M=D.item()[n]
    [x,w]=quad0.item()[n+1]
    F=np.zeros(n+1)
    P=Bin
    for j in range(n+1):
        for i in range(n+1):
            xi=a*((1+x[i])/2)+b*((1-x[i])/2)
            F[j]+=M[i][j]*f(xi)
        F[j]*=np.abs(b-a)*P[n][j]/2
    return F
    

## Tow dimensional sitting

### precomputed array 
D1=np.load("Precomp D1.npy",allow_pickle=True)

P1=np.load("Precomp P1.npy",allow_pickle=True)

D2=np.load("Precomp D2.npy",allow_pickle=True)

P2=np.load("Precomp P2.npy",allow_pickle=True)


### Evaluation of f at q² point quadrature

def Eval2D(f,q,L):
    [x1,y1,x2,y2,x3,y3]=L
    [c1,w]=quad1.item()[q]
    [c2,w]=quad0.item()[q]
    F=np.zeros((q,q))
    for i1 in range(q):
        for i2 in range(q):
            
            x=x1*(1+c1[i1])/2
            x+=x2*(1-c1[i1])*(1+c2[i2])/4
            x+=x3*(1-c1[i1])*(1-c2[i2])/4
            
            y=y1*(1+c1[i1])/2
            y+=y2*(1-c1[i1])*(1+c2[i2])/4
            y+=y3*(1-c1[i1])*(1-c2[i2])/4
            
            F[i1][i2]+=f(x,y)
    return F
    
### Moment approximation

def Moment2D(L,f,n):
    t0= timeit.default_timer()
    T=AirT2D(L)
    F=Eval2D(f, n+1, L)
    P=Bin
    A1=D1.item()[n]
    A2=D2.item()[n]
    B1=P1.item()[n]
    B2=P2.item()[n]
    In=indexes2D(n)
    l=(n+2)*(n+1)//2
    Aux=np.zeros((n+1,n+1))
    
    for b1 in range(n+1):
        for i1 in range(n+1):
            for i2 in range(n+1):
                Aux[b1][i2]+=A1[b1][i1]*B1[b1][i1]*F[i1][i2]
    
    M=np.zeros(l)
    for j in range(l):
        b1=In[j][0]
        b2=In[j][1]
        b3=In[j][2]
        for i2 in range(n+1):
            M[j]+=A2[b1+b2][i2]*B2[b2][i2]*Aux[b1][i2]
        M[j]*=T*P[b1+b2][b2]*P[n][b3]/4
    t1=timeit.default_timer()-t0
    #print("Time elapsed: ", t1)
    return M

## 3D Moment

### Evaluation of f at q^3 point quadrature

def Eval3D(f,q,L):
    [x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4]=L
    [c1,w]=quad2.item()[q]
    [c2,w]=quad1.item()[q]
    [c3,w]=quad0.item()[q]
    F=np.zeros((q,q,q))
    for i1 in range(q):
        for i2 in range(q):
            for i3 in range(q):
                
                x=x1*(1+c1[i1])/2
                x+=x2*(1-c1[i1])*(1+c2[i2])/4
                x+=x3*(1-c1[i1])*(1-c2[i2])*(1+c3[i3])/8
                x+=x4*(1-c1[i1])*(1-c2[i2])*(1-c3[i3])/8
                
                y=y1*(1+c1[i1])/2
                y+=y2*(1-c1[i1])*(1+c2[i2])/4
                y+=y3*(1-c1[i1])*(1-c2[i2])*(1+c3[i3])/8
                y+=y4*(1-c1[i1])*(1-c2[i2])*(1-c3[i3])/8
                
                z=z1*(1+c1[i1])/2
                z+=z2*(1-c1[i1])*(1+c2[i2])/4
                z+=z3*(1-c1[i1])*(1-c2[i2])*(1+c3[i3])/8
                z+=z4*(1-c1[i1])*(1-c2[i2])*(1-c3[i3])/8
                
                F[i1][i2][i3]+=f(x,y,z)
    return F

  
## precomputed array
A1=np.load("Precomp A1.npy",allow_pickle=True)

B1=np.load("Precomp B1.npy",allow_pickle=True)


def Moment3D(L,f,n):
    t0=timeit.default_timer()
    T=AirT3D(L)
    F=Eval3D(f, n+1, L)
    P=Bin
    Z1=A1.item()[n]
    Z2=D1.item()[n]
    Z3=D2.item()[n]
    Y1=B1.item()[n]
    Y2=P1.item()[n]
    Y3=P2.item()[n]
    In=indexes3D(n)
    l=len(In)
    
    H=np.zeros((n+1,n+1,n+1))  
    
    for b1 in range(n+1):
        for i1 in range(n+1):
            w=Z1[b1][i1]*Y1[b1][i1]
            for i2 in range(n+1):
                for i3 in range(n+1):
                    H[b1][i2][i3]+=w*F[i1][i2][i3]
                    
    U=np.zeros((n+1,n+1,n+1))
    
    for b1 in range(n+1):
        for b2 in range(n+1-b1):
            for i2 in range(n+1):
                w=Z2[b1+b2][i2]*Y2[b2][i2]
                for i3 in range(n+1):
                    U[b1][b2][i3]+=w*H[b1][i2][i3]
    
    M=np.zeros(l)
    for j in range(l):
        b1=In[j][0]
        b2=In[j][1]
        b3=In[j][2]
        b4=In[j][3]
        for i3 in range(n+1):
            M[j]+=Z3[b1+b2+b3][i3]*Y3[b3][i3]*U[b1][b2][i3]
        M[j]*=3*T*P[b1+b2][b2]*P[b1+b2+b3][b3]*P[n][b4]/32
    t1=timeit.default_timer()-t0
    #print("time elapsed ",t1)
    return M

# Constant Data
## mass matrix
### 1D

def CstMassMat1D(n,T):
    # input: 
        ## n polynomial degree
        ## T interval lenth
    # Output:
        # Mass matrix M_{i,j}=(Bn,i;Bn,j)
    P=Bin
    M=np.zeros((n+1,n+1))
    for a1 in range(n+1):
        for a2 in range(n+1):
            M[a1][a2]+=T*P[a1+a2][a1]*P[n-a1+n-a2][n-a1]/(P[2*n][n]*(2*n+1))
    return M


## 2D

def CstMassMat2D(n,T):
    t0=timeit.default_timer()
    In=indexes2D(n)
    l=len(In)
    P=Bin
    M=np.zeros((l,l))
    
    for i in range(l):
        for j in range(l):
            a1=In[i][0]
            a2=In[i][1]
            a3=In[i][2]
            
            b1=In[j][0]
            b2=In[j][1]
            b3=In[j][2]
            
            M[i][j]+=T*P[a1+b1][a1]/(P[2*n][n]*P[2*n+2][2])
            M[i][j]*=P[a2+b2][a2]*P[a3+b3][a3]
    t1=timeit.default_timer()-t0
    #print("time elapsed ",t1)
    return M

## 3D

def CstMassMat3D(n,T):
    t0=timeit.default_timer()
    In=indexes3D(n)
    l=len(In)
    P=Bin
    M=np.zeros((l,l))
    
    for i in range(l):
        for j in range(l):
            a1=In[i][0]
            a2=In[i][1]
            a3=In[i][2]
            a4=In[i][3]
            
            b1=In[j][0]
            b2=In[j][1]
            b3=In[j][2]
            b4=In[j][3]
            
            M[i][j]+=T*P[a1+b1][a1]/(P[2*n][n]*P[2*n+3][3])
            M[i][j]*=P[a2+b2][a2]*P[a3+b3][a3]*P[a4+b4][a4]
    t1=timeit.default_timer()-t0
    #print("time elapsed ",t1)
    return M
    
    
#Stiffness

## 1D

def cst_StiffMat_1D(L,n,A):
    T=abs(L[0]-L[1])
    M=CstMassMat1D(n-1, T)
    s=n*n*A*grad1D(L)
    S=np.zeros((n+1,n+1))
    for i in range(n):
        for j in range(n):
            k=M[i][j]
            S[i+1][j+1]+=k*s[0][0]
            S[i+1][j]+=k*s[0][1]
            S[i][j+1]+=k*s[1][0]
            S[i][j]+=k*s[1][1]
    return S
            
##2D
    
def cst_StiffMat_2D(L,A,n):
    t0=timeit.default_timer()
    T=AirT2D(L)
    M=CstMassMat2D(n-1, T)
    G=grad2D(L)
    s=np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            w=np.dot(A,G[i])
            o=np.vdot(G[j],w)
            s[i][j]+=n*n*o
    w=(n+2)*(n+1)//2
    S=np.zeros((w,w))
    e=np.array([(1,0,0),(0,1,0),(0,0,1)])
    In=indexes2D(n-1)
    L=(n+1)*n//2
    for p in range(L):
        for q in range(L):
            y=M[p][q]
            a=In[p]
            b=In[q]
            for i in range(3):
                for j in range(3):
                    u=tuple(sumVect(a, e[i]))
                    v=tuple(sumVect(b, e[j]))
                    k=getIndex2D(n, u)
                    l=getIndex2D(n, v)
                    S[k][l]+=y*s[i][j]
    t2=timeit.default_timer()-t0
    #print("time elapsed ",t2)
    return S

## 3D  

def cst_StiffMat_3D(L,A,n):
    t0=timeit.default_timer()
    T=AirT3D(L)
    M=CstMassMat3D(n-1, T)
    G=grad3D(L)
    s=np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            w=np.dot(A,G[i])
            o=np.vdot(G[j],w)
            s[i][j]+=n*n*o
            
    w=((n+1)*(n+2)*(n+3) )//6
    S=np.zeros((w,w))
    e=np.array([(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)])
    In=indexes3D(n-1)
    L=len(In)
    for p in range(L):
        for q in range(L):
            y=M[p][q]
            a=In[p]
            b=In[q]
            for i in range(4):
                for j in range(4):
                    u=tuple(sumVect(a, e[i]))
                    v=tuple(sumVect(b, e[j]))
                    k=getIndex3D(n, u)
                    l=getIndex3D(n, v)
                    S[k][l]+=y*s[i][j]
    t2=timeit.default_timer()-t0
    #print("time elapsed ",t2)
    return S


#Convective matrix

## 1D
def cst_ConvMat_1D(L,n,b):
    P=Bin
    [u,v]=L
    T=abs(u-v)
    r=0.5*T/P[2*n-1][n]
    v1=r*b/(u-v)
    v2=r*b/(v-u)
    V=np.zeros((n+1,n+1))
    for i in range(n):
        for j in range(n+1):
            w=P[i+j][i]
            i2=n-1-i
            j2=n-j
            w*=P[i2+j2][i2]
            V[i+1][j]+=w*v1
            V[i][j]+=w*v2
    return V
    

##2D 

def cst_ConvMat_2D(L,n,b):
    P=Bin
    T=AirT2D(L)
    G=grad2D(L)
    r=n*T/(P[2*n-1][n]*P[2*n+1][2])
    s=np.zeros(3)
    for i in range(3):
            s[i]+=r*ScalarProd(b, G[i])
    
    H1=indexes2D(n)
    H2=indexes2D(n-1)
    h1=(n+2)*(n+1)//2
    h2=n*(n+1)//2
    V=np.zeros((h1,h1))
    e=np.array([(1,0,0),(0,1,0),(0,0,1)])
    for i in range(h2):
        for j in range(h1):
            a=H2[i]
            b=H1[j]
            w=P[a[0]+b[0]][a[0]]*P[a[1]+b[1]][a[1]]*P[a[2]+b[2]][a[2]]
            for k in range(3):
                alpha=tuple(sumVect(a, e[k]))
                i2=getIndex2D(n, alpha)
                V[i2][j]+=w*s[k]
    return V
                      
## 3D

def cst_ConvMat_3D(L,n,b):
    P=Bin
    T=AirT3D(L)
    G=grad3D(L)
    r=n*T/(P[2*n-1][n]*P[2*n+2][3])
    s=np.zeros(4)
    for i in range(3):
            s[i]+=r*ScalarProd(b, G[i])
    
    H1=indexes3D(n)
    H2=indexes3D(n-1)
    h1=(n+3)*(n+2)*(n+1)//6
    h2=n*(n+2)*(n+1)//6
    V=np.zeros((h1,h1))
    e=np.array([(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)])
    for i in range(h2):
        for j in range(h1):
            a=H2[i]
            b=H1[j]
            w=P[a[0]+b[0]][a[0]]*P[a[1]+b[1]][a[1]]*P[a[2]+b[2]][a[2]]*P[a[3]+b[3]][a[3]]
            for k in range(4):
                alpha=tuple(sumVect(a, e[k]))
                i2=getIndex3D(n, alpha)
                V[i2][j]+=w*s[k]
    return V
    
    
# Variable Data

## Mass Matrix

### 1D

def MassMat1D(a,b,f,n):
    H=Moment1D(a,b,f,2*n)
    P=Bin
    M=np.zeros((n+1,n+1))
    for a1 in range(n+1):
        for b1 in range(n+1):
             w=P[a1+b1][a1]*P[n-a1+n-b1][n-a1]/P[2*n][n]
             M[a1][b1]=w*H[a1+b1]
    return M

### 2D

def MassMat2D(L,f,n):
    H=Moment2D(L,f,2*n)
    In=indexes2D(n)
    l=(n+2)*(n+1)//2
    P=Bin
    M=np.zeros((l,l))
    
    for i in range(l):
        for j in range(l):
            a1=In[i][0]
            a2=In[i][1]
            a3=In[i][2]
            
            b1=In[j][0]
            b2=In[j][1]
            b3=In[j][2]

            M[i][j]+=P[a1+b1][a1]/P[2*n][n]
            M[i][j]*=P[a2+b2][a2]*P[a3+b3][a3]
            M[i][j]*=H[i+j]
    return M

### 3D

def MassMat3D(L,f,n):
    H=Moment3D(L,f,2*n)
    In=indexes3D(n)
    l=len(In)
    P=Bin
    M=np.zeros((l,l))
    
    for i in range(l):
        for j in range(l):
            a1=In[i][0]
            a2=In[i][1]
            a3=In[i][2]
            a4=In[i][3]
            
            b1=In[j][0]
            b2=In[j][1]
            b3=In[j][2]
            b4=In[j][3]
            
            M[i][j]+=P[a1+b1][a1]/P[2*n][n]
            M[i][j]*=P[a2+b2][a2]*P[a3+b3][a3]*P[a4+b4][a4]
            M[i][j]*=H[i+j]
    return M
    
## Stiffness Matrix
### 1D

def StiffMat1D(a,b,f,n):
    H=Moment1D(a, b, f, 2*n-2)
    v=np.array([1/(a-b),1/(b-a)])
    M=np.zeros((2*n-1,2,2))
    for b in range(2*n-1):
        for i in range(2):
            for j in range(2):
                M[b][i][j]=v[i]*H[b]*v[j]
    P=Bin
    S=np.zeros((n+1,n+1))
    for a1 in range(n):
        for a2 in range(n):
            w=n*n*P[a1+a2][a1]
            w*=P[n-1-a1+n-1-a2][n-1-a1]
            w/=P[2*n-2][n-1]
            k=w
            S[a1+1][a2+1]+=k*M[a1+a2][0][0]
            S[a1+1][a2]+=k*M[a1+a2][0][1]
            S[a1][a2+1]+=k*M[a1+a2][1][0]
            S[a1][a2]+=k*M[a1+a2][1][1]
    return S        

### 2D
 
def StiffMat2D(L,A,n):
    t0=timeit.default_timer()
    G=grad2D(L)
    l=n*(2*n-1)
    H=np.zeros((l,2,2))
    for i in range(2):
        for j in range(2):
            O=Moment2D(L, lambda x,y: A(x,y)[i][j], 2*n-2)
            for b in range(l):
                H[b][i][j]=O[b]
    s=np.zeros((l,3,3))
    for b in range(l):
        for i in range(3):
            for j in range(3):
                w=np.dot(H[b],G[i])
                s[b][i][j]+=np.vdot(G[j],w)
    P=Bin
    w=(n+2)*(n+1)//2
    S=np.zeros((w,w))
    e=np.array([(1,0,0),(0,1,0),(0,0,1)])
    E=indexes2D(n-1)
    t=n*(n+1)//2
    t1=timeit.default_timer()-t0
    #print("consumed time befor the loop ",t1)
    for i in range(t):
        for j in range(t):
            a=E[i]
            b=E[j]
            w=n*n*P[a[0]+b[0]][a[0]]/P[2*n-2][n-1]
            w*=P[a[1]+b[1]][a[1]]*P[a[2]+b[2]][a[2]]
            for i2 in range(3):
                for j2 in range(3):
                    u=tuple(sumVect(a, e[i2]))
                    v=tuple(sumVect(b, e[j2]))
                    Z=tuple(sumVect(a, b))
                    k=getIndex2D(n, u)
                    f=getIndex2D(n, v)
                    z=getIndex2D(2*n-2, Z)
                    S[k][f]+=w*s[z][i2][j2]
    t2= timeit.default_timer()-t1
    #print("time elapsed evaluating the moment is ",t2)
    return S

### 3D

def StiffMat3D(L,A,n):
    # A is a 3*3 matrix valued function
    G=grad3D(L)
    l=n*(2*n+1)*(2*n-1)//3
    H=np.zeros((l,3,3))
    for i in range(3):
        for j in range(3):
            V=Moment3D(L, lambda x,y,z: A(x,y,z)[i][j], 2*n-2)
            for b in range(l):
                H[b][i][j]=V[b]
    s=np.zeros((l,4,4))
    for b in range(l):
        for i in range(3):
            for j in range(3):
                w=np.dot(H[b],G[j])
                s[b][i][j]+=np.vdot(G[i],w)
    
    L=((n+1)*(n+2)*(n+3) )//6
    S=np.zeros((L,L))
    
    e=np.array([(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)])
    In=indexes3D(n-1)
    y=len(In)
    P=Bin
    for i in range(y):
        for j in range(y):
            a=In[i]
            a1=a[0]
            a2=a[1]
            a3=a[2]
            a4=a[3]
            
            b=In[j]
            b1=b[0]
            b2=b[1]
            b3=b[2]
            b4=b[3]
            
            w=P[a1+b1][a1]*n*n/P[2*n-2][n-1]
            w*=P[a2+b2][a2]*P[a3+b3][a3]*P[a4+b4][a4]
            for i2 in range(4):
                for j2 in range(4):
                    u=tuple(sumVect(a, e[i2]))
                    v=tuple(sumVect(b, e[j2]))
                    Z=tuple(sumVect(a, b))
                    k=getIndex3D(n, u)
                    o=getIndex3D(n, v)
                    z=getIndex3D(2*n-2, Z)
                    S[k][o]+=w*s[z][i2][j2]
    return S

## Convectiv matrix

### 1D 

def Conv1D(a,b,f,n,q):
    v=[1/(a-b),1/(b-a)]
    P=Bin
    H=np.zeros((2*n,2))
    for b in range(2*n):
        for i in range(2):
            H[b][i]+=v[i]*(Moment1D(a, b, f, n, q)[b])
            
    V=np.zeros((n+1,n+1))
    for i in range(n):
        for j in range(n+1):
            w=P[i+j][i]*n/P[2*n-1][n]
            i2=n-1-i
            j2=n-j
            w*=P[i2+j2][i2]
            V[i+1][j]+=w*H[i+j][0]
            V[i][j]+=w*H[i+j][1]
    return V

### 2D

def Conv2D(L,f,n):
    # f is a 2dimensional vector valued function
    P=Bin
    G=grad2D(L)
    l=n*(2*n+1)
    
    M1=np.zeros((l,2))
    
    for j in range(2):
        V=Moment2D(L, lambda x,y:f(x,y)[j], 2*n-1)
        for b in range(l):
            M1[b][j]+=V[b]
            
    M2=np.zeros((l,3))
    for b in range(l):
        for j in range(3):
            M2[b][j]+=np.vdot(G[j],M1[b])
    
    H1=indexes2D(n)
    H2=indexes2D(n-1)
    h1=(n+2)*(n+1)//2
    h2=n*(n+1)//2
    V=np.zeros((h1,h1))
    e=np.array([(1,0,0),(0,1,0),(0,0,1)])
    for i in range(h2):
        for j in range(h1):
            a=H2[i]
            b=H1[j]
            w=P[a[0]+b[0]][a[0]]*n/P[2*n-1][n]
            w*=P[a[1]+b[1]][a[1]]*P[a[2]+b[2]][a[2]]
            for k in range(3):
                alpha=tuple(sumVect(a, e[k]))
                s=tuple(sumVect(a,b))
                i2=getIndex2D(n, alpha)
                x=getIndex2D(2*n-1, s)
                V[i2][j]+=w*M2[x][k]
    return V
    
### 3D 

def Conv3D(L,f,n):
    # f is a 2dimensional vector valued function
    P=Bin
    G=grad3D(L)
    l=n*(2*n+2)*(2*n+1)//3
    
    M1=np.zeros((l,3))
    
    for j in range(3):
        V=Moment3D(L, lambda x,y,z:f(x,y,z)[j], 2*n-1)
        for b in range(l):
            M1[b][j]+=V[b]
            
    M2=np.zeros((l,4))
    for b in range(l):
        for j in range(3):
            M2[b][j]+=np.vdot(G[j],M1[b])
    
    H1=indexes3D(n)
    H2=indexes3D(n-1)
    h1=(n+3)*(n+2)*(n+1)//6
    h2=n*(n+1)*(n+2)//6
    V=np.zeros((h1,h1))
    e=np.array([(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)])
    for i in range(h2):
        for j in range(h1):
            a=H2[i]
            b=H1[j]
            w=P[a[0]+b[0]][a[0]]*n/P[2*n-1][n]
            w*=P[a[1]+b[1]][a[1]]*P[a[2]+b[2]][a[2]]*P[a[3]+b[3]][a[3]]
            for k in range(4):
                alpha=tuple(sumVect(a, e[k]))
                s=tuple(sumVect(a, b))
                i2=getIndex3D(n, alpha)
                x=getIndex3D(2*n-1, s)
                V[i2][j]+=w*M2[x][k]
    return V

#Evaluation of BB

## 1 dimension

def deCasteljau1D(t, coefs):
    beta = [c for c in coefs]
    n = len(beta)
    for j in range(1, n):
        for k in range(n - j):
            beta[k] = beta[k] * (1 - t) + beta[k + 1] * t
    return beta[0]

## 2 dimension
### lam: barycentrique cordinate of the point
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



# Tests:

# Poisson 1D :  -u"=1 on [0,1], u(0)=u(1)=0 
def poisson1D(n):
    # return the BB vector of the solution, taking in account boundary condition
    K=cst_StiffMat_1D([0,1],n,1)
    K=np.delete(K,n,axis=0)
    K=np.delete(K,n,axis=1)
    K=np.delete(K,0,axis=0)
    K=np.delete(K,0,axis=1)
    B=Moment1D(0, 1, lambda x: 1, n)
    B=np.delete(B,n)
    B=np.delete(B,0)
    X=np.linalg.solve(K,B)
    return X

def plotpoisson1D(n,m):
    lesx=np.linspace(0,1,m)
    C=np.zeros(n+1)
    C[1:n]=poisson1D(n)
    print(C)
    lesy=[]
    In=[]
    for x in lesx:
        lesy.append(deCasteljau1D(x,C))
        In.append((x-x*x)/2)
    plt.title("poisson -u''=1")
    plt.plot(lesx,lesy,'r',label="approx")
    plt.plot(lesx,In, 'g', label="exact")
    plt.legend()
    plt.show()
 
## Exemple u'=1 , u(0)=0, sol exacte u(x)=x

def Ex1(n):
    V=cst_ConvMat_1D([0,1],n,1)
    V=np.delete(V,n,axis=0)
    V=np.delete(V,n,axis=1)
    B=Moment1D(0, 1, lambda x: 1, n)
    B=np.delete(B,n)
    X=np.linalg.solve(V,B)
    return X

def plotEx1(n,m):
    lesx=np.linspace(0,1,m)
    C=np.zeros(n+1)
    C[1:n+1]=Ex1(n)
    print(C)
    lesy=[]
    for x in lesx:
        lesy.append(deCasteljau1D(x,C))
    plt.title("u'=x")
    plt.plot(lesx,lesy)
    plt.show()

## Autre exemple: -u"+u=1 on (0,1), u(0)=u(1)=0

def Ex2(n):
    K=cst_StiffMat_1D([0,1],n,1)
    K=np.delete(K,n,axis=0)
    K=np.delete(K,n,axis=1)
    K=np.delete(K,0,axis=0)
    K=np.delete(K,0,axis=1)
    V=cst_ConvMat_1D([0,1],n,1)
    V=np.delete(V,n,axis=0)
    V=np.delete(V,n,axis=1)
    V=np.delete(V,0,axis=0)
    V=np.delete(V,0,axis=1)
    B=Moment1D(0, 1, lambda x: 1, n)
    B=np.delete(B,n)
    B=np.delete(B,0)
    X=np.linalg.solve(K+V,B)
    return X

# the exacte solution of the above equation
def exact(x):
    A=1/(1-np.exp(1))
    return A*(np.exp(x)-1)+x

def plotEx2(n,m):
    lesx=np.linspace(0,1,m)
    C=np.zeros(n+1)
    C[1:n]=Ex2(n)
    print(C)
    lesy=[]
    In=[]
    for x in lesx:
        lesy.append(deCasteljau1D(x,C))
        In.append(exact(x))
    plt.title("-u''+u=1")
    plt.plot(lesx,lesy,"r",label="aprox")
    plt.plot(lesx,In,"g",label="exact")
    plt.legend()
    plt.show()


## Poisson equation  div( grad u)=2(x+y), u=0 on the boundary of the reference triangle

def A(x,y):
    return np.array([[1,0],[1,0]])

def sol2D(n):
    t0=timeit.default_timer()
    K=StiffMat2D([0,0,1,0,0,1], A, n)
    B=Moment2D([0,0,1,0,0,1], lambda x,y:2*(x+y), n)
    L=indexes2D(n)
    w=(n+1)*(n+2)//2
    C=[]
    Q=[]
    for i in range(w-1,-1,-1):
        if L[i][0]*L[i][1]*L[i][2]==0:
            K=np.delete(K,i,axis=0)
            K=np.delete(K,i,axis=1)
            B=np.delete(B,i)
            C.append(i)
        else:
            Q.append(i)
    #print('K ',K)
    #print('B ',B)
    #print(C)
    #print(Q)
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

    

def plotpoisson2D(n,m):
    C=sol2D(n)
    l=indexes2D(m)
    w=(m+2)*(m+1)//2
    lesx=[]
    lesy=[]
    lesz=[]
    for i in range(w):
        lesx.append(l[i][1]/m)
        lesy.append(l[i][2]/m)
        lesz.append(deCasteljau2D(l[i],C,n))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(lesx, lesy, lesz, cmap='Blues')
        





    