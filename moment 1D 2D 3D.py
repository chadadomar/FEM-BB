import numpy as np
import scipy.special

# Bernestein-Bézier moment
quads=dict()
for q in range(1,20):
    quads[q]=scipy.special.roots_jacobi(q,0,0)




### Binomial coefficient


## Return T: pascal triangle of ordre n

def Pascal(n):
    t=np.zeros((n+1,n+1))
    for i in range(n+1):
        t[i][0]=1
        t[i][i]=1
    for i in range(2,n+1):
        for j in range(1,i):
            t[i][j]=t[i-1][j-1]+t[i-1][j]
    return t




## One dimentional setting
###  Vector D: neded after Gauss_Jaccobi quadrature rule

def D(n,q):
    [x,w]=quads[q]
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


def M1D(a,b,f,n,q):
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

### 2D domaine points in lexicographic order

def indexes2D(n):
    return [(i,j, n-(i+j)) for i in range(n,-1, -1) for j in range(n-i, -1, -1)]


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

### Aire triangle based on Heron Formulae

def AirT2D(L):
    [x1,y1,x2,y2,x3,y3]=L
    a=np.sqrt((x1-x2)**2+(y1-y2)**2)
    b=np.sqrt((x1-x3)**2+(y1-y3)**2)
    c=np.sqrt((x3-x2)**2+(y3-y2)**2)
    p=(a+b+c)/2
    return np.sqrt(p*(p - a) * (p - b) * (p - c))

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
### 3D domaine points in lexicographic order

def indexes3D(n):
    return [(i,j,k, n-(i+j+k)) for i in range(n,-1, -1) for j in range(n-i, -1, -1) for k in range(n-i-j,-1,-1)]


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

## Air of a tetrahedron

## we define cross and scalar product

def ScalarProd3D(u,v):
    return u[0]*v[0]+u[1]*v[1]+u[2]*v[2]

def CrossProd3D(u,v):
    w=np.zeros(3)
    w[0]=u[1]*v[2]-u[2]*v[1]
    w[1]=u[2]*v[0]-u[0]*v[2]
    w[2]=u[0]*v[1]-u[1]*v[0]
    return w

def AirT3D(L):
    [x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4]=L
    u=np.array([x1-x4,y1-y4,z1-z4])
    v=np.array([x2-x4,y2-y4,z2-z4])
    w=np.array([x3-x4,y3-y4,z3-z4])
    cross=CrossProd3D(v, w)
    T=ScalarProd3D(u, cross)
    return abs(T)/6    


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




