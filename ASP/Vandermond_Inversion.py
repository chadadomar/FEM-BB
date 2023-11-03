#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 21:43:54 2023

@author: omarch
"""

from Whitney_condition_number import *
from Hankel_Inversion import *
import numpy as np
from scipy.linalg import block_diag



def SchurComp(A,B,C,D):
    """ Matrix block [[A,B],[C,D]] where A and D are square matrices  """
    
    Ainv=np.linalg.inv(A)
    M=D - C @ Ainv @ B
    Minv= np.linalg.inv(M)
    
    A1= Ainv + Ainv @ B @ Minv @ C @ Ainv
    A2= - Ainv @ B @ Minv
    A3= - Minv @ C @ Ainv
    
    Res1= np.concatenate((A1, A2), axis=1)
    Res2= np.concatenate((A3, Minv), axis=1)
    Res= np.concatenate((Res1, Res2), axis=0)
    
    return Res


def InvVandermond(r):
    V=Vandermond2d(r)
    
    a=3*r
    b=3*r+int(r*(r-1)/2)
    c=r*(r+2)
    
    Ve=V[0:r,0:r]
    
    C=V[a:c,0:a]
    
    Vf1=V[a:b  , a:b ]
    
    Vf2=V[b:c  , b:c]
    
    B=V[a:b , b :c ]
    
    Ve1=InvHankel(Ve)
    A1=block_diag(Ve1, Ve1, Ve1)
    A2=SchurComp(Vf1,B,B,Vf2)
    
    A3= - A2 @ C @ A1
    
    """
    print("V \n", V)
    print("Ve \n", Ve)
    print("C \n", C)
    print("Vf1 \n", Vf1)
    print("Vf2 \n", Vf2)
    print("B \n", B)"""
    
    return np.block([  [A1, np.zeros( (3*r,r*(r-1)) )] ,[A3 , A2] ])
    
    
    