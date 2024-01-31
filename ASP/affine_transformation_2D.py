#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 17:58:31 2024

@author: omarch

Afffine transformation from reference trinagle to a given trangle T
"""
import numpy as np

def affine_trans(T):
    # Input: 
        #T = [x_1,y_1,x_2,y_2,x_3,y_3] coordinates of vertices
    # Output :
        # B_T , b_T such that F_T(x)=B_T * x + b_t
    [x_1,y_1,x_2,y_2,x_3,y_3]=T
    b_T=np.array([[x_1],[y_1]])
    
    B_T=np.array( [ [x_2 - x_1, x_3 - x_1],[y_2 - y_1, y_3 - y_1] ] )
    
    return B_T,b_T