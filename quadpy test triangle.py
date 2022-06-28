#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 19:44:12 2022

@author: omarch
"""

import numpy as np
import quadpy


scheme = quadpy.t2.get_good_scheme(12)
scheme.show()
val = scheme.integrate(lambda x: np.exp(x[0]), [[0.0, 0.0], [1.0, 0.0], [0.5, 0.7]])
print(val)