import numpy as np
import scipy.special


#Auxilary functions


##Binomial coeff
def Pascal(n):
    t=np.zeros((n+1,n+1))
    for i in range(n+1):
        t[i][0]=1
        t[i][i]=1
    for i in range(2,n+1):
        for j in range(1,i):
            t[i][j]=t[i-1][j-1]+t[i-1][j]
    return t

## matrice of scalar prod of gradient vect
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
            w[i]=u[i]+v[i]
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

# Bernestein-Bézier moment


## One dimentional setting
###  Vector D: neded after Gauss_Jaccobi quadrature rule

def D(n,q):
    [x,w]=scipy.special.roots_jacobi(q,0,0)
    M=np.zeros((q,n+1))
    for i in range(q):
        a=((1-x[i])/2)**n
        b=(1+x[i])/2
        t=1.0
        for j in range(n+1):
            M[i][j]=w[i]*a*t
            a/=((1-x[i])/2)
            t*=b
    return M

### Moment approximation

def Moment1D(a,b,f,n,q):
    M=D(n,q)
    [x,w]=scipy.special.roots_jacobi(q,0,0)
    F=np.zeros(n+1)
    P=Pascal(n)
    for j in range(n+1):
        for i in range(q):
            xi=a*((1+x[i])/2)+b*((1-x[i])/2)
            F[j]+=M[i][j]*f(xi)
        F[j]*=np.abs(b-a)*P[n][j]/2
    return F
    

## Tow dimensional sitting

### precomputed array 

def D1(n,q):
    M=np.zeros((n+1,q))
    [x,w]=scipy.special.roots_jacobi(q,1,0)
    for i in range(q):
        a=((1-x[i])/2)**n
        b=np.sqrt(w[i])
        for j in range(n+1):
            M[j][i]=b*a
            a/=((1-x[i])/2)
    return M

def P1(n,q):
    M=np.zeros((n+1,q))
    [x,w]=scipy.special.roots_jacobi(q,1,0)
    for i in range(q):
        a=(1+x[i])/2
        b=np.sqrt(w[i])
        t=1
        for j in range(n+1):
            M[j][i]=b*t
            t*=a
    return M

def D2(n,q):
    M=np.zeros((n+1,q))
    [x,w]=scipy.special.roots_jacobi(q,0,0)
    for i in range(q):
        a=((1-x[i])/2)**n
        b=np.sqrt(w[i])
        for j in range(n+1):
            M[j][i]=b*a
            a/=((1-x[i])/2)
    return M

def P2(n,q):
    M=np.zeros((n+1,q))
    [x,w]=scipy.special.roots_jacobi(q,0,0)
    for i in range(q):
        a=(1+x[i])/2
        b=np.sqrt(w[i])
        t=1
        for j in range(n+1):
            M[j][i]=b*t
            t*=a
    return M


### Evaluation of f at q² point quadrature

def Eval2D(f,q,L):
    [x1,y1,x2,y2,x3,y3]=L
    [c1,w]=scipy.special.roots_jacobi(q,1,0)
    [c2,w]=scipy.special.roots_jacobi(q,0,0)
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

def Moment2D(L,f,n,q):
    T=AirT2D(L)
    F=Eval2D(f, q, L)
    P=Pascal(n)
    A1=D1(n,q)
    A2=D2(n,q)
    B1=P1(n,q)
    B2=P2(n,q)
    In=indexes2D(n)
    l=len(In)
    Aux=np.zeros((n+1,q))
    
    for b1 in range(n+1):
        for i1 in range(q):
            for i2 in range(q):
                Aux[b1][i2]+=A1[b1][i1]*B1[b1][i1]*F[i1][i2]
    
    M=np.zeros(l)
    for j in range(l):
        b1=In[j][0]
        b2=In[j][1]
        b3=In[j][2]
        for i2 in range(q):
            M[j]+=A2[b1+b2][i2]*B2[b2][i2]*Aux[b1][i2]
        M[j]*=T*P[b1+b2][b2]*P[n][b3]/4
    return M

## 3D Moment



### Evaluation of f at q^3 point quadrature


def Eval3D(f,q,L):
    [x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4]=L
    [c1,w]=scipy.special.roots_jacobi(q,2,0)
    [c2,w]=scipy.special.roots_jacobi(q,1,0)
    [c3,w]=scipy.special.roots_jacobi(q,0,0)
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

def A1(n,q):
    M=np.zeros((n+1,q))
    [x,w]=scipy.special.roots_jacobi(q,2,0)
    for i in range(q):
        a=((1-x[i])/2)**n
        b=np.sqrt(w[i])
        for j in range(n+1):
            M[j][i]=b*a
            a/=((1-x[i])/2)
    return M
def B1(n,q):
    M=np.zeros((n+1,q))
    [x,w]=scipy.special.roots_jacobi(q,2,0)
    for i in range(q):
        a=(1+x[i])/2
        b=np.sqrt(w[i])
        t=1
        for j in range(n+1):
            M[j][i]=b*t
            t*=a
    return M

def A2(n,q):
    M=np.zeros((n+1,q))
    [x,w]=scipy.special.roots_jacobi(q,1,0)
    for i in range(q):
        a=((1-x[i])/2)**n
        b=np.sqrt(w[i])
        for j in range(n+1):
            M[j][i]=b*a
            a/=((1-x[i])/2)
    return M
def B2(n,q):
    M=np.zeros((n+1,q))
    [x,w]=scipy.special.roots_jacobi(q,1,0)
    for i in range(q):
        a=(1+x[i])/2
        b=np.sqrt(w[i])
        t=1
        for j in range(n+1):
            M[j][i]=b*t
            t*=a
    return M

def A3(n,q):
    M=np.zeros((n+1,q))
    [x,w]=scipy.special.roots_jacobi(q,0,0)
    for i in range(q):
        a=((1-x[i])/2)**n
        b=np.sqrt(w[i])
        for j in range(n+1):
            M[j][i]=b*a
            a/=((1-x[i])/2)
    return M
def B3(n,q):
    M=np.zeros((n+1,q))
    [x,w]=scipy.special.roots_jacobi(q,0,0)
    for i in range(q):
        a=(1+x[i])/2
        b=np.sqrt(w[i])
        t=1
        for j in range(n+1):
            M[j][i]=b*t
            t*=a
    return M


def Moment3D(L,f,n,q):
    
    T=AirT3D(L)
    F=Eval3D(f, q, L)
    P=Pascal(n)
    D1=A1(n,q)
    D2=A2(n,q)
    D3=A3(n,q)
    P1=B1(n,q)
    P2=B2(n,q)
    P3=B3(n,q)
    In=indexes3D(n)
    l=len(In)
    
    H=np.zeros((n+1,q,q))  
    
    for b1 in range(n+1):
        for i1 in range(q):
            w=D1[b1][i1]*P1[b1][i1]
            for i2 in range(q):
                for i3 in range(q):
                    H[b1][i2][i3]+=w*F[i1][i2][i3]
                    
    U=np.zeros((n+1,n+1,q))
    
    for b1 in range(n+1):
        for b2 in range(n+1-b1):
            for i2 in range(q):
                w=D2[b1+b2][i2]*P2[b2][i2]
                for i3 in range(q):
                    U[b1][b2][i3]+=w*H[b1][i2][i3]
    
    M=np.zeros(l)
    for j in range(l):
        b1=In[j][0]
        b2=In[j][1]
        b3=In[j][2]
        b4=In[j][3]
        for i3 in range(q):
            M[j]+=D3[b1+b2+b3][i3]*P3[b3][i3]*U[b1][b2][i3]
        M[j]*=3*T*P[b1+b2][b2]*P[b1+b2+b3][b3]*P[n][b4]/32
    
    return M

# Constant Data
#mass matrix with x
## 1D

def CstMassMat1D(n,T):
    # input: 
        ## n polynomial degree
        ## T simplex area
    # Output:
        # Mass matrix M_{i,j}=(Bn,i;Bn,j)
    P=Pascal(2*n)
    M=np.zeros((n+1,n+1))
    for a1 in range(n+1):
        for a2 in range(n+1):
            M[a1][a2]+=T*P[a1+a2][a1]*P[n-a1+n-a2][n-a1]/(P[2*n][n]*(2*n+1))
    return M


## 2D

def CstMassMat2D(n,T):
    
    In=indexes2D(n)
    l=len(In)
    P=Pascal(2*n+2)
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
    return M

## 3D

def CstMassMat3D(n,T):
    
    In=indexes3D(n)
    l=len(In)
    P=Pascal(2*n+3)
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

def positionIndex2D(b):
    return b[1]+b[2]*(2*(b[0]+b[1])+b[2]+3)//2

def cst_StiffMat_2D(L,n,A):
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
    L=len(In)
    for p in range(L):
        for q in range(L):
            y=M[p][q]
            a=In[p]
            b=In[q]
            for i in range(3):
                for j in range(3):
                    u=sumVect(a, e[i])
                    v=sumVect(b, e[j])
                    k=int(positionIndex2D(u))
                    l=int(positionIndex2D(v))
                    S[k][l]+=y*s[i][j]
    return S
    
## 3D  
   
def cst_StiffMat_3D(L,n,A):
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
    Ind=indexes3D(n)
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
                    k=Ind.index(u)
                    l=Ind.index(v)
                    S[k][l]+=y*s[i][j]
    return S

#Convective matrix

## 1D
def cst_ConvMat_1D(L,n,b):
    P=Pascal(2*n-1)
    [u,v]=L
    T=abs(u-v)
    r=0.5*T/P[2*n-1][2]
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
    P=Pascal(2*n+1)
    T=AirT2D(L)
    G=grad2D(L)
    r=n*T/(P[2*n-1][n]*P[2*n+1][2])
    s=np.zeros(3)
    for i in range(3):
            s[i]+=r*ScalarProd(b, G[i])
    
    H1=indexes2D(n)
    H2=indexes2D(n-1)
    h1=len(H1)
    h2=len(H2)
    V=np.zeros((h1,h1))
    e=np.array([(1,0,0),(0,1,0),(0,0,1)])
    for i in range(h2):
        for j in range(h1):
            a=H2[i]
            b=H1[j]
            w=P[a[0]+b[0]][a[0]]*P[a[1]+b[1]][a[1]]*P[a[2]+b[2]][a[2]]
            for k in range(3):
                alpha=tuple(sumVect(a, e[i]))
                i2=H1.index(alpha)
                V[i2][j]+=w*s[i]
    return V
                      
## 3D

def cst_ConvMat_3D(L,n,b):
    P=Pascal(2*n+2)
    T=AirT3D(L)
    G=grad3D(L)
    r=n*T/(P[2*n-1][n]*P[2*n+2][3])
    s=np.zeros(4)
    for i in range(3):
            s[i]+=r*ScalarProd(b, G[i])
    
    H1=indexes3D(n)
    H2=indexes3D(n-1)
    h1=len(H1)
    h2=len(H2)
    V=np.zeros((h1,h1))
    e=np.array([(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)])
    for i in range(h2):
        for j in range(h1):
            a=H2[i]
            b=H1[j]
            w=P[a[0]+b[0]][a[0]]*P[a[1]+b[1]][a[1]]*P[a[2]+b[2]][a[2]]*P[a[3]+b[3]][a[3]]
            for k in range(4):
                alpha=tuple(sumVect(a, e[i]))
                i2=H1.index(alpha)
                V[i2][j]+=w*s[i]
    return V
    
    
# Variable Data

## Mass Matrix

### 1D

def MassMat1D(a,b,f,n,q):
    H=Moment1D(a,b,f,2*n,q)
    P=Pascal(2*n)
    M=np.zeros((n+1,n+1))
    for a1 in range(n+1):
        for b1 in range(n+1):
             w=P[a1+b1][a1]*P[n-a1+n-b1][n-a1]/P[2*n][n]
             M[a1][b1]=w*H[a1+b1]
    return M

### 2D

def MassMat2D(L,f,n,q):
    H=Moment2D(L,f,2*n,q)
    In=indexes2D(n)
    l=len(In)
    P=Pascal(2*n+2)
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

def MassMat3D(L,f,n,q):
    H=Moment3D(L,f,2*n,q)
    In=indexes3D(n)
    l=len(In)
    P=Pascal(2*n+3)
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

def StiffMat1D(a,b,f,n,q):
    H=Moment1D(a, b, f, 2*n-2, q)
    v=np.array([1/(a-b),1/(b-a)])
    M=np.zeros((2*n-1,2,2))
    for b in range(2*n-1):
        for i in range(2):
            for j in range(2):
                M[b][i][j]=v[i]*H[b]*v[j]
    P=Pascal(2*n-2)
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

def StiffMat2D(L,A,n,q):
    G=grad2D(L)
    In=indexes2D(2*n-2)
    l=len(In)
    H=np.zeros((l,2,2))
    for b in range(l):
        for i in range(2):
            for j in range(2):
                H[b][i][j]=Moment2D(L, lambda x,y: A(x,y)[i][j], 2*n-2, q)[b]

    s=np.zeros((l,3,3))
    for b in range(l):
        for i in range(3):
            for j in range(3):
                w=np.dot(H[b],G[i])
                s[b][i][j]+=np.vdot(G[j],w)
                
    E=indexes2D(n-1)
    t=len(E)
    P=Pascal(2*n)
    w=(n+2)*(n+1)//2
    S=np.zeros((w,w))
    e=np.array([(1,0,0),(0,1,0),(0,0,1)])
    for i in range(t):
        for j in range(t):
            a=E[i]
            b=E[j]
            w=n*n*P[a[0]+b[0]][a[0]]/P[2*n-2][n-1]
            w*=P[a[1]+b[1]][a[1]]*P[a[2]+b[2]][a[2]]
            for i in range(3):
                for j in range(3):
                    u=sumVect(a, e[i])
                    v=sumVect(b, e[j])
                    Z=sumVect(a, b)
                    k=int(positionIndex2D(u))
                    f=int(positionIndex2D(v))
                    z=int(positionIndex2D(Z))
                    S[k][f]+=w*s[z][i][j]
    return S
            
def StiffMat3D(L,A,n,q):
    G=grad3D(L)
    IZ=indexes3D(2*n-2)
    l=len(IZ)
    H=np.zeros((l,3,3))
    for b in range(l):
        for i in range(3):
            for j in range(3):
                H[b][i][j]=Moment3D(L, lambda x,y,z: A(x,y,z)[i][j], 2*n-2, q)[b]

    s=np.zeros((l,4,4))
    for b in range(l):
        for i in range(3):
            for j in range(3):
                w=np.dot(H[b],G[j])
                s[b][i][j]+=np.vdot(G[i],w)
    
    q=((n+1)*(n+2)*(n+3) )//6
    S=np.zeros((q,q))
    e=np.array([(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)])
    In=indexes3D(n-1)
    Ind=indexes3D(n)
    y=len(In)
    P=Pascal(2*n+1) 
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
            for i in range(4):
                for j in range(4):
                    u=tuple(sumVect(a, e[i]))
                    v=tuple(sumVect(b, e[j]))
                    Z=tuple(sumVect(a, b))
                    k=Ind.index(u)
                    o=Ind.index(v)
                    z=IZ.index(Z)
                    S[k][o]+=w*s[z][i][j]
    return S
    