#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 18:01:13 2023

Projector matrix

@author: omarch
"""

import numpy as np
import math as mth
from bernstein_moment import *
from Vandermond_Inversion import *
from Passage_bernstein_to_whitney import *
np.set_printoptions(linewidth=10000)

def P(r):
    
    ndof=r*(r+2)
    nBern=int((r+1)*(r+2)/2)
    M=np.zeros((ndof,2*nBern))
    
    B=moment(r)
    
    Vinv=InvVandermond(r)
    Q=np.linalg.inv(PassageScr(r))
    
    for k in range(ndof):
        for l in range(2*nBern):
            for i in range(ndof):
                for j in range(ndof):
                    M[k][l]+=B[j][l]*Vinv[i][j]*Q[k][i]
    return M