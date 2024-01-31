#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 17:30:20 2024

@author: omarch

Numerical quadrature over a given triangle T

Reference: D.A. Dunavant,
      High degree efficient symmetrical gaussian
      quadrature rules for the triangle,
      International Journal for Num. Methods in Eng.,
      21, 1129-1148, 1985.
"""


from Gauss_nodes_wights_2D import *
from affine_transformation_2D import *
import numpy as np

def quad(T,f,p):
    # Input:
        # T [x_1,y_1,x_2,y_2,x_3,y_3] coordinates of vertices
        # f : the function we want to integrate over T
        # p : is the degree of polynomial which is integrated exactly
    # Output:
        # Approximation of integral of f over T
    X,W,n=NodesWeights(p)
    B_T,b_T=affine_trans(T)
    I=0
    for i in range(n):
        [x,y]=X[i]
        x_i=np.array([[x],[y]])
        x_T=np.dot(B_T,x_i)+b_T
        a=x_T[0][0]
        b=x_T[1][0]
        I+=W[i]*f(a,b)
    return I