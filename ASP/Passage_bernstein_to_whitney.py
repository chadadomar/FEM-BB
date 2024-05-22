#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:29:48 2023

@author: omarch

Matrix of Bernstien curl basis expressed in Whitney high order basis
"""

import numpy as np
import math as m
import sympy as sy
import scipy.sparse


T=[0,1,2]
Edge=[[0,1],[0,2],[1,2]]

### 2D domaine points in lexicographic order
def indexes2D(n):
    return [(i,j, n-(i+j)) for i in range(n,-1, -1) for j in range(n-i, -1, -1)]


def getIndex2D(n,t):
    #n : polynomial order
    #t : index of the domain point (also polynomial index) in lexicographical order
    (i,j,k)=t
    return int((n-i)*(n-i+1)//2+n-i-j)  



def indexes2Dbis(r):
    L=[]
    for gamma in range(r+1):
        for beta in range(r+1-gamma):
            alpha=r-gamma-beta
            L.append((alpha,beta,gamma))
    return L

def getIndex2Dbis(n,t):
    #n : polynomial order
    #t : index of the domain point (also polynomial index) in lexicographical order
    (i,j,k)=t
    if i<0 or j<0 or k<0 or i+j+k !=n:
        raise Exception("invalid multiindex")
    else:
        return int((n+1)*(n+2)/2) - 1 - int((n-k)*(n-k+1)//2+n-k-j)  

def fact(n):
    if n==0:
        return 1
    else:
        return n*fact(n-1)


def dof(r):
    # r: order of Whitney forms (r>=1)
    dofEdge=[]
    dofFace=[]
    for i in range(3):
        e=Edge[i]
        l=e[0]
        m=e[1]
        for beta in range(r):
            alpha=r-1-beta
            dofEdge.append([l,m,alpha,beta])
    if r==1:
        return dofEdge
    else:
        for gamma in range(r-1):
            for beta in range(r-1-gamma):
                alpha=r-2-gamma-beta
                dofFace.append([0,1,alpha,beta,gamma])
        for gamma in range(r-1):
            for beta in range(r-1-gamma):
                alpha=r-2-gamma-beta
                dofFace.append([0,2,alpha,beta,gamma])
    return dofEdge,dofFace


def Passage(r):
    l=r*(r+2)
    l1=3*r
    l2=l1+int(r*(r-1)/2)
    P=np.zeros((l,l))
    d=dof(r)
    Dof=d[0]+d[1]
    #firts column w_3=W^{1,2}=w^{0,1}
    for i in range(r):
        P[i][0]+=m.comb(r-1,i)
    for i in range(l1, l2):
        alpha=Dof[i][2:]
        P[i][0]+=fact(r-1)/(fact(alpha[0]) * fact(alpha[1]) * fact(alpha[2]+1))
    
    #Second column w_2=w^{1,3}=w^{0,2}
    for i in range(r,2*r):
        P[i][1]+=m.comb(r-1,i-r)
    for i in range(l2, l):
        alpha=Dof[i][2:]
        P[i][1]+=fact(r-1)/(fact(alpha[0]) * fact(alpha[1]+1) * fact(alpha[2]))
        
    
    #Third column w_1=w^{2,3}=w^{1,2}
    for i in range(2*r,3*r):
        P[i][2]+=m.comb(r-1,i-2*r)
    for i in range(l1, l2):
        alpha=Dof[i][2:]
        P[i][2]+=-fact(r-1)/(fact(alpha[0]+1) * fact(alpha[1]) * fact(alpha[2]))
    for i in range(l2, l):
        alpha=Dof[i][2:]
        P[i][2]+=fact(r-1)/(fact(alpha[0]+1) * fact(alpha[1]) * fact(alpha[2]))
    
    # Gradient Bernstein     
    L=indexes2D(r) 
    """ list of gradinet functions indices """
    
    L.remove((r,0,0))
    L.remove((0,r,0))
    L.remove((0,0,r))
    
    #print("L est",L)
    gr=len(L)
    
    for j in range(3,gr+3):
        alpha=L[j-3]
        alpha0=alpha[0]
        alpha1=alpha[1]
        alpha2=alpha[2]
        
        if alpha0==0:
            t=(0, alpha1-1, alpha2-1)
            ind= getIndex2Dbis(r-2,t)+ 3*r
            P[ind][j]+= r* m.comb(r-1,alpha1-1)
            
            t=(0, alpha1-1, alpha2-1)
            ind= getIndex2Dbis(r-2,t)+ l2
            P[ind][j] -= r* m.comb(r-1,alpha1)
            
            ind=2*r+ alpha2-1
            P[ind][j] += r*m.comb(r-1,alpha1)
            
            ind=2*r+ alpha2
            P[ind][j] -= r*m.comb(r-1,alpha1-1)
            
        elif alpha1==0:
            t=(alpha0-1, 0 , alpha2-1)
            ind= getIndex2Dbis(r-2,t)+ 3*r
            P[ind][j]-= r* ( m.comb(r-1,alpha0-1) + m.comb(r-1,alpha0))
            
            ind= r+ alpha2
            P[ind][j]+= r*m.comb(r-1,alpha0-1)
            
            ind= r+ alpha2-1
            P[ind][j]-= r*m.comb(r-1,alpha0)
            
            t=(alpha0-1, 0 , alpha2-1)
            ind= getIndex2Dbis(r-2,t)+ l2
            P[ind][j]+= r*  m.comb(r-1,alpha0) 
        
        elif alpha2==0:
            ind=alpha1-1
            P[ind][j]+= r* m.comb(r-1,alpha0)
            
            ind=alpha1
            P[ind][j]-= r*m.comb(r-1, alpha0-1)
            
            t=(alpha0-1, alpha1-1 , 0)
            ind= getIndex2Dbis(r-2,t)+ l2
            P[ind][j]+= r* ( m.comb(r-1,alpha0-1) -  m.comb(r-1,alpha0) )
            
            t=(alpha0-1, alpha1-1 , 0)
            ind= getIndex2Dbis(r-2,t)+ 3*r
            P[ind][j]+= r*  m.comb(r-1,alpha0)
        else:
            # alpha0, alpha1, alpha2 all non zero
            t=(alpha0, alpha1-1 , alpha2-1)
            ind= getIndex2Dbis(r-2,t)+ 3*r
            P[ind][j]+= r*  fact(r-1) / ( fact(alpha0)*fact(alpha1-1)*fact(alpha2) )
            
            t=(alpha0-1, alpha1 , alpha2-1)
            ind= getIndex2Dbis(r-2,t)+ 3*r
            P[ind][j]-= r* (fact(r-1)/fact(alpha1)) * ( 1 / ( fact(alpha0-1)*fact(alpha2) ) + 1 / ( fact(alpha0)*fact(alpha2-1)) )
            
            t=(alpha0-1, alpha1-1 , alpha2)
            ind= getIndex2Dbis(r-2,t)+ l2
            P[ind][j]+= r* (fact(r-1)/fact(alpha2)) * ( 1 / ( fact(alpha0-1)*fact(alpha1) ) - 1 / ( fact(alpha0)*fact(alpha1-1)) )
            
            t=(alpha0, alpha1-1 , alpha2-1)
            ind= getIndex2Dbis(r-2,t)+ l2
            P[ind][j]-= r*  fact(r-1) / ( fact(alpha0)*fact(alpha1)*fact(alpha2-1) )
            
            t=(alpha0-1, alpha1 , alpha2-1)
            ind= getIndex2Dbis(r-2,t)+ l2
            P[ind][j]+= r*  fact(r-1) / ( fact(alpha0)*fact(alpha1)*fact(alpha2-1) )
            
            t=(alpha0-1, alpha1-1 , alpha2)
            ind= getIndex2Dbis(r-2,t)+ 3*r
            P[ind][j]+= r*  fact(r-1) / ( fact(alpha0)*fact(alpha1-1)*fact(alpha2) )
            
   # Bubble gamma function 
    Gamma=indexes2D(r-1)
    Gamma.remove((0,0,r-1))
    
    
    for j in range(gr+3,l):
        
        alpha=Gamma[j-gr-3]
        alpha0=alpha[0]
        alpha1=alpha[1]
        alpha2=alpha[2]
        
        if alpha0==0:
            t=(0, alpha1 , alpha2-1)
            ind= getIndex2Dbis(r-2,t)+ 3*r
            P[ind][j]+= r* m.comb(r-1,alpha1)*alpha2
            
            t=(0, alpha1-1 , alpha2)
            ind= getIndex2Dbis(r-2,t)+ l2
            P[ind][j]+= r* m.comb(r-1,alpha1)*alpha1
        else:
            coef= r* fact(r-1)/ ( fact(alpha0) * fact(alpha1) * fact(alpha2) )
            
            t=(alpha0, alpha1 , alpha2-1)
            ind= getIndex2Dbis(r-2,t)+ 3*r
            P[ind][j]+= coef*alpha2
            
            t=(alpha0, alpha1-1 , alpha2)
            ind= getIndex2Dbis(r-2,t)+ l2
            P[ind][j]+= coef*alpha1
            
            t=(alpha0-1, alpha1 , alpha2)
            ind= getIndex2Dbis(r-2,t)+ l2
            P[ind][j]+= coef*alpha0
            
            t=(alpha0-1, alpha1 , alpha2)
            ind= getIndex2Dbis(r-2,t)+ 3*r
            P[ind][j]-= coef*alpha0
            
    return P


def PassageScr(r):
    l=r*(r+2)
    l1=3*r
    l2=l1+int(r*(r-1)/2)
    d=dof(r)
    Dof=d[0]+d[1]
    
    row=[]
    col=[]
    data=[]
    
    #firts column w_3=W^{1,2}=w^{0,1}
    for i in range(r):
        row.append(i)
        col.append(0)
        data.append(m.comb(r-1,i))

    for i in range(l1, l2):
        row.append(i)
        col.append(0)     
        alpha=Dof[i][2:]
        data.append(fact(r-1)/(fact(alpha[0]) * fact(alpha[1]) * fact(alpha[2]+1)))
    
    #Second column w_2=w^{1,3}=w^{0,2}
    for i in range(r,2*r):
        row.append(i)
        col.append(1)
        data.append(m.comb(r-1,i-r))
    for i in range(l2, l):
        alpha=Dof[i][2:]
        row.append(i)
        col.append(1)
        data.append(fact(r-1)/(fact(alpha[0]) * fact(alpha[1]+1) * fact(alpha[2])))
        
    
    #Third column w_1=w^{2,3}=w^{1,2}
    for i in range(2*r,3*r):
        row.append(i)
        col.append(2)
        data.append(m.comb(r-1,i-2*r))
    for i in range(l1, l2):
        alpha=Dof[i][2:]
        row.append(i)
        col.append(2)
        data.append(-fact(r-1)/(fact(alpha[0]+1) * fact(alpha[1]) * fact(alpha[2])))
    for i in range(l2, l):
        alpha=Dof[i][2:]
        row.append(i)
        col.append(2)
        data.append(fact(r-1)/(fact(alpha[0]+1) * fact(alpha[1]) * fact(alpha[2])))
    
        
    L=indexes2D(r) 
    """ list of gradinet functions indices """
    
    L.remove((r,0,0))
    L.remove((0,r,0))
    L.remove((0,0,r))
    
    #print("L est",L)
    gr=len(L)
    
    for j in range(3,gr+3):
        alpha=L[j-3]
        alpha0=alpha[0]
        alpha1=alpha[1]
        alpha2=alpha[2]
        
        if alpha0==0:
            t=(0, alpha1-1, alpha2-1)
            ind= getIndex2Dbis(r-2,t)+ 3*r
            row.append(ind)
            col.append(j)
            data.append(r* m.comb(r-1,alpha1-1))
            
            t=(0, alpha1-1, alpha2-1)
            ind= getIndex2Dbis(r-2,t)+ l2
            row.append(ind)
            col.append(j)
            data.append(-r* m.comb(r-1,alpha1))
            
            ind=2*r+ alpha2-1
            row.append(ind)
            col.append(j)
            data.append(r*m.comb(r-1,alpha1))
            
            ind=2*r+ alpha2
            row.append(ind)
            col.append(j)
            data.append(-r*m.comb(r-1,alpha1-1))
            
        elif alpha1==0:
            t=(alpha0-1, 0 , alpha2-1)
            ind= getIndex2Dbis(r-2,t)+ 3*r
            row.append(ind)
            col.append(j)
            data.append(-r* ( m.comb(r-1,alpha0-1) + m.comb(r-1,alpha0)))

            
            ind= r+ alpha2
            row.append(ind)
            col.append(j)
            data.append(r*m.comb(r-1,alpha0-1))

            
            ind= r+ alpha2-1
            row.append(ind)
            col.append(j)
            data.append(-r*m.comb(r-1,alpha0))
            
            t=(alpha0-1, 0 , alpha2-1)
            ind= getIndex2Dbis(r-2,t)+ l2
            row.append(ind)
            col.append(j)
            data.append(r*  m.comb(r-1,alpha0) )
        
        elif alpha2==0:
            ind=alpha1-1
            row.append(ind)
            col.append(j)
            data.append(r* m.comb(r-1,alpha0))
            
            ind=alpha1
            row.append(ind)
            col.append(j)
            data.append(-r*m.comb(r-1, alpha0-1))

            
            t=(alpha0-1, alpha1-1 , 0)
            ind= getIndex2Dbis(r-2,t)+ l2
            row.append(ind)
            col.append(j)
            data.append(r* ( m.comb(r-1,alpha0-1) -  m.comb(r-1,alpha0) ))
            
            t=(alpha0-1, alpha1-1 , 0)
            ind= getIndex2Dbis(r-2,t)+ 3*r
            row.append(ind)
            col.append(j)
            data.append(r*  m.comb(r-1,alpha0))

        else:
            # alpha0, alpha1, alpha2 all non zero
            t=(alpha0, alpha1-1 , alpha2-1)
            ind= getIndex2Dbis(r-2,t)+ 3*r
            row.append(ind)
            col.append(j)
            data.append(r*  fact(r-1) / ( fact(alpha0)*fact(alpha1-1)*fact(alpha2) ))
            
            
            t=(alpha0-1, alpha1 , alpha2-1)
            ind= getIndex2Dbis(r-2,t)+ 3*r
            row.append(ind)
            col.append(j)
            data.append(-r* (fact(r-1)/fact(alpha1)) * ( 1 / ( fact(alpha0-1)*fact(alpha2) ) + 1 / ( fact(alpha0)*fact(alpha2-1)) ))
            
            
            t=(alpha0-1, alpha1-1 , alpha2)
            ind= getIndex2Dbis(r-2,t)+ l2
            row.append(ind)
            col.append(j)
            data.append(r* (fact(r-1)/fact(alpha2)) * ( 1 / ( fact(alpha0-1)*fact(alpha1) ) - 1 / ( fact(alpha0)*fact(alpha1-1)) ))
        
            
            t=(alpha0, alpha1-1 , alpha2-1)
            ind= getIndex2Dbis(r-2,t)+ l2
            row.append(ind)
            col.append(j)
            data.append(-r*  fact(r-1) / ( fact(alpha0)*fact(alpha1)*fact(alpha2-1) ))
            
            t=(alpha0-1, alpha1 , alpha2-1)
            ind= getIndex2Dbis(r-2,t)+ l2
            row.append(ind)
            col.append(j)
            data.append(r*  fact(r-1) / ( fact(alpha0)*fact(alpha1)*fact(alpha2-1) ))
            
            
            t=(alpha0-1, alpha1-1 , alpha2)
            ind= getIndex2Dbis(r-2,t)+ 3*r
            row.append(ind)
            col.append(j)
            data.append(r*  fact(r-1) / ( fact(alpha0)*fact(alpha1-1)*fact(alpha2) ))
            
            
        
    Gamma=indexes2D(r-1)
    Gamma.remove((0,0,r-1))
    
    
    for j in range(gr+3,l):
        
        alpha=Gamma[j-gr-3]
        alpha0=alpha[0]
        alpha1=alpha[1]
        alpha2=alpha[2]
        
        if alpha0==0:
            if alpha2!=0:
                t=(0, alpha1 , alpha2-1)
                ind= getIndex2Dbis(r-2,t)+ 3*r
                row.append(ind)
                col.append(j)
                data.append(r* m.comb(r-1,alpha1)*alpha2)

            
            if alpha1 !=0:
                t=(0, alpha1-1 , alpha2)
                ind= getIndex2Dbis(r-2,t)+ l2
                row.append(ind)
                col.append(j)
                data.append(r* m.comb(r-1,alpha1)*alpha1)

        else:
            coef= r* fact(r-1)/ ( fact(alpha0) * fact(alpha1) * fact(alpha2) )
            
            if alpha2 != 0:
                t=(alpha0, alpha1 , alpha2-1)
                ind= getIndex2Dbis(r-2,t)+ 3*r
                row.append(ind)
                col.append(j)
                data.append(coef*alpha2)

            if alpha1 !=0:
                t=(alpha0, alpha1-1 , alpha2)
                ind= getIndex2Dbis(r-2,t)+ l2
                row.append(ind)
                col.append(j)
                data.append(coef*alpha1)
            
            
            t=(alpha0-1, alpha1 , alpha2)
            ind= getIndex2Dbis(r-2,t)+ l2
            row.append(ind)
            col.append(j)
            data.append(coef*alpha0)

            
            t=(alpha0-1, alpha1 , alpha2)
            ind= getIndex2Dbis(r-2,t)+ 3*r
            row.append(ind)
            col.append(j)
            data.append(-coef*alpha0)
    row=np.array(row)
    col=np.array(col)
    data=np.array(data)
    P=scipy.sparse.csr_array((data, (row, col)), shape=(l, l)).toarray()
    return P

        

def PassageSy(r):
    P=Passage(r)
    ndof=r*(r+2)
    V=sy.zeros(ndof,ndof)
    for i in range(ndof):
        for j in range(ndof):
            V[i,j]+=int(P[i][j])
    return V


def PassageScr_newOrder(r):
    l=r*(r+2)
    l1=3*r
    l2=l1+int(r*(r-1)/2)
    d=dof(r)
    Dof=d[0]+d[1]
    
    row=[]
    col=[]
    data=[]
    
    #last column w_3=W^{1,2} (w_2=w^{0,1})
    for i in range(r):
        row.append(i)
        col.append(l-1)
        data.append(m.comb(r-1,i))

    for i in range(l1, l2):
        row.append(i)
        col.append(l-1)     
        alpha=Dof[i][2:]
        data.append(fact(r-1)/(fact(alpha[0]) * fact(alpha[1]) * fact(alpha[2]+1)))
    
    #Second last column w_2=w^{3,1} (w_1=w^{2,0})
    for i in range(r,2*r):
        row.append(i)
        col.append(l-2)
        data.append(-m.comb(r-1,i-r))
    for i in range(l2, l):
        alpha=Dof[i][2:]
        row.append(i)
        col.append(l-2)
        data.append(-fact(r-1)/(fact(alpha[0]) * fact(alpha[1]+1) * fact(alpha[2])))
        
    
    #Third last column w_1=w^{2,3} (w_0=w^{1,2})
    for i in range(2*r,3*r):
        row.append(i)
        col.append(l-3)
        data.append(m.comb(r-1,i-2*r))
    for i in range(l1, l2):
        alpha=Dof[i][2:]
        row.append(i)
        col.append(l-3)
        data.append(-fact(r-1)/(fact(alpha[0]+1) * fact(alpha[1]) * fact(alpha[2])))
    for i in range(l2, l):
        alpha=Dof[i][2:]
        row.append(i)
        col.append(l-3)
        data.append(fact(r-1)/(fact(alpha[0]+1) * fact(alpha[1]) * fact(alpha[2])))
    
        
    L=indexes2D(r) 
    """ list of gradinet functions indices """
    
    L.remove((r,0,0))
    L.remove((0,r,0))
    L.remove((0,0,r))
    
    #print("L est",L)
    #gr=len(L)
    gr=(r+1)*(r+2)//2 -3
    

    
    for j in range(gr):
        alpha=L[j]
        alpha0=alpha[0]
        alpha1=alpha[1]
        alpha2=alpha[2]
        
        # IMPORTANT: note that in the following case discussion we don't have the case wher tow alph_i's
                    # are null, cuz this correspond to one of the removed indices (vertex indexe)
        
        if alpha0==0:
            t=(0, alpha1-1, alpha2-1)
            ind= getIndex2Dbis(r-2,t)+ 3*r
            row.append(ind)
            col.append(j)
            data.append(r* m.comb(r-1,alpha1-1))
            
            t=(0, alpha1-1, alpha2-1)
            ind= getIndex2Dbis(r-2,t) + l2
            row.append(ind)
            col.append(j)
            data.append(r* m.comb(r-1,alpha1))
            
            ind=2*r+ alpha2-1
            row.append(ind)
            col.append(j)
            data.append(r*m.comb(r-1,alpha1))   
            
            ind=2*r+ alpha2
            row.append(ind)
            col.append(j)
            data.append(-r*m.comb(r-1,alpha1-1))    
            
        elif alpha1==0:
            t=(alpha0-1, 0 , alpha2-1)
            ind= getIndex2Dbis(r-2,t)+ 3*r
            row.append(ind)
            col.append(j)
            data.append(-r* ( m.comb(r-1,alpha0-1) + m.comb(r-1,alpha0) ) )

            
            ind= r+ alpha2-1
            row.append(ind)
            col.append(j)
            data.append(r*m.comb(r-1,alpha0))

            
            ind= r+ alpha2
            row.append(ind)
            col.append(j)
            data.append(-r*m.comb(r-1,alpha0-1))
            
            t=(alpha0-1, 0 , alpha2-1)
            ind= getIndex2Dbis(r-2,t)+ l2
            row.append(ind)
            col.append(j)
            data.append( r* m.comb(r-1,alpha0) )
            
        
        elif alpha2==0:
            ind=alpha1-1
            row.append(ind)
            col.append(j)
            data.append(r* m.comb(r-1,alpha0))
            
            ind=alpha1
            row.append(ind)
            col.append(j)
            data.append(-r*m.comb(r-1, alpha0-1))

            
            t=(alpha0-1, alpha1-1 , 0)
            ind= getIndex2Dbis(r-2,t)+ l2
            row.append(ind)
            col.append(j)
            data.append( r* ( -m.comb(r-1,alpha0-1) -  m.comb(r-1,alpha0) ) )
            
            t=(alpha0-1, alpha1-1 , 0)
            ind= getIndex2Dbis(r-2,t)+ 3*r
            row.append(ind)
            col.append(j)
            data.append(r*  m.comb(r-1,alpha0))

        else:
            # alpha0, alpha1, alpha2 all non zero
            t=(alpha0, alpha1-1 , alpha2-1)
            ind= getIndex2Dbis(r-2,t)+ 3*r
            row.append(ind)
            col.append(j)
            data.append(r*  fact(r-1) / ( fact(alpha0)*fact(alpha1-1)*fact(alpha2) ))
            
            
            t=(alpha0-1, alpha1 , alpha2-1)
            ind= getIndex2Dbis(r-2,t)+ 3*r
            row.append(ind)
            col.append(j)
            #data.append(-r* (fact(r-1)/fact(alpha1)) * ( 1 / ( fact(alpha0-1)*fact(alpha2) ) + 1 / ( fact(alpha0)*fact(alpha2-1)) ))
            data.append(-r* ( 1 / alpha2  + 1 / alpha0 )* fact(r-1) / ( fact(alpha0-1) * fact(alpha1) * fact(alpha2-1)))
            
            
            t=(alpha0, alpha1-1 , alpha2-1)
            ind= getIndex2Dbis(r-2,t)+ l2
            row.append(ind)
            col.append(j)
            data.append(r*  fact(r-1) / ( fact(alpha0)*fact(alpha1)*fact(alpha2-1) ))
            
            
            t=(alpha0-1, alpha1-1 , alpha2)
            ind= getIndex2Dbis(r-2,t)+ l2
            row.append(ind)
            col.append(j)
            #data.append(- r* (fact(r-1)/fact(alpha2)) * ( 1 / ( fact(alpha0-1)*fact(alpha1) ) +  1 / ( fact(alpha0)*fact(alpha1-1)) ))
            data.append(- r *(1/alpha0 + 1/alpha1) * (fact(r-1)/(fact(alpha0-1)*fact(alpha1-1)*fact(alpha2))) )
        
            
            t=(alpha0-1, alpha1 , alpha2-1)
            ind= getIndex2Dbis(r-2,t)+ l2
            row.append(ind)
            col.append(j)
            data.append(r*  fact(r-1) / ( fact(alpha0)*fact(alpha1)*fact(alpha2-1) ))
            
            
            t=(alpha0-1, alpha1-1 , alpha2)
            ind= getIndex2Dbis(r-2,t)+ 3*r
            row.append(ind)
            col.append(j)
            data.append(r*  fact(r-1) / ( fact(alpha0)*fact(alpha1-1)*fact(alpha2) ))
            
            
        
    Gamma=indexes2D(r-1)
    Gamma.remove((0,0,r-1))
    
    # Gamma^P_alpha=
    for j in range(gr,l-3):
        
        alpha=Gamma[j-gr]
        alpha0=alpha[0]
        alpha1=alpha[1]
        alpha2=alpha[2]
        
        coef= r* fact(r-1)/ ( fact(alpha0) * fact(alpha1) * fact(alpha2) )
        
        if alpha2!=0:
            t=(alpha0, alpha1 , alpha2-1)
            ind= getIndex2Dbis(r-2,t)+ 3*r
            row.append(ind)
            col.append(j)
            data.append(coef*alpha2)
        
        if alpha1!=0:
            t=(alpha0, alpha1-1 , alpha2)
            ind= getIndex2Dbis(r-2,t)+ l2
            row.append(ind)
            col.append(j)
            data.append(-coef*alpha1)
        
        if alpha0!=0:
            
            t=(alpha0-1, alpha1 , alpha2)
            ind= getIndex2Dbis(r-2,t)+ l2
            row.append(ind)
            col.append(j)
            data.append(coef*alpha0)
            
            t=(alpha0-1, alpha1 , alpha2)
            ind= getIndex2Dbis(r-2,t)+ 3*r
            row.append(ind)
            col.append(j)
            data.append(-coef*alpha0)            
            
    row=np.array(row)
    col=np.array(col)
    data=np.array(data)
    P=scipy.sparse.csr_array((data, (row, col)), shape=(l, l)).toarray()
    return P

        

